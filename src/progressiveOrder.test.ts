import { describe, expect, it } from "vitest";

import { orderTileForProgressiveDrawing } from "./progressiveOrder";
import type { TileData } from "./tileSource";

const gridTile = (): TileData => {
  const side = 64;
  const pointCount = side * side;
  const positions = new Float32Array(pointCount * 3);
  const rgb = new Uint8Array(pointCount * 3);
  for (let id = 0; id < pointCount; id += 1) {
    const offset = id * 3;
    positions[offset] = id % side;
    positions[offset + 1] = Math.floor(id / side);
    rgb[offset] = id & 0xff;
    rgb[offset + 1] = id >>> 8;
  }
  return { origin: [0, 0, 0], positions, rgb, pointCount };
};

const PATCH = 16;
const PATCHES_PER_SIDE = 16;
const PATCH_SIDE = PATCH * PATCHES_PER_SIDE;

/**
 * A node whose record order walks 16x16 patches, so 256 consecutive points —
 * one reorder block — are one compact piece of it. That is what makes record
 * order unusable as a progressive prefix, and it is the property the reorder
 * has to defeat: a prefix that misses whole blocks misses whole patches.
 */
const patchTile = (): TileData => {
  const pointCount = PATCH_SIDE * PATCH_SIDE;
  const positions = new Float32Array(pointCount * 3);
  const rgb = new Uint8Array(pointCount * 3);
  for (let id = 0; id < pointCount; id += 1) {
    const patch = Math.floor(id / 256);
    const within = id % 256;
    const offset = id * 3;
    positions[offset] = (patch % PATCHES_PER_SIDE) * PATCH + (within % PATCH);
    positions[offset + 1] =
      Math.floor(patch / PATCHES_PER_SIDE) * PATCH + Math.floor(within / PATCH);
    rgb[offset] = id & 0xff;
    rgb[offset + 1] = id >>> 8;
  }
  return { origin: [0, 0, 0], positions, rgb, pointCount };
};

const ids = (tile: TileData): number[] => {
  const result: number[] = [];
  for (let index = 0; index < tile.pointCount; index += 1) {
    result.push(tile.rgb![index * 3]! | (tile.rgb![index * 3 + 1]! << 8));
  }
  return result;
};

type Coverage = {
  readonly fraction: string;
  readonly emptyPatches: number;
  readonly leastDrawn: number;
  readonly mostDrawn: number;
};

const coverageOf = (
  tile: TileData,
  prefixPoints: number,
): Coverage & { readonly cells: number; readonly extent: number[] } => {
  const drawn = new Int32Array(PATCHES_PER_SIDE * PATCHES_PER_SIDE);
  const cells = new Set<string>();
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let index = 0; index < prefixPoints; index += 1) {
    const x = tile.positions[index * 3]!;
    const y = tile.positions[index * 3 + 1]!;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    const patch =
      Math.floor(y / PATCH) * PATCHES_PER_SIDE + Math.floor(x / PATCH);
    drawn[patch] = drawn[patch]! + 1;
    cells.add(`${Math.floor(x / 8)}-${Math.floor(y / 8)}`);
  }
  return {
    fraction: "",
    emptyPatches: drawn.reduce((total, count) => total + (count ? 0 : 1), 0),
    leastDrawn: Math.min(...drawn),
    mostDrawn: Math.max(...drawn),
    cells: cells.size,
    extent: [minX, minY, maxX, maxY],
  };
};

/**
 * Fractions the guard covers, with the half-resolution cell counts the current
 * ordering reaches. The bounds are a ratchet, not a fingerprint: an ordering
 * that spreads a prefix further may exceed them, one that spreads it less may
 * not. 1024 cells exist; a prefix of 1/64 draws 1024 points, so the cell count
 * is capped by the prefix long before it is capped by the ordering.
 */
const FRACTIONS = [
  { label: "1/64", fraction: 1 / 64, cells: 353 },
  { label: "1/32", fraction: 1 / 32, cells: 484 },
  { label: "1/16", fraction: 1 / 16, cells: 541 },
  { label: "1/8", fraction: 1 / 8, cells: 608 },
  { label: "1/4", fraction: 1 / 4, cells: 760 },
  { label: "1/2", fraction: 1 / 2, cells: 990 },
  // The band the view governor actually occupies: MIN_VIEW_QUALITY_FRACTION
  // is 0.05 and steady-state quality runs 10-40%.
  { label: "0.05", fraction: 0.05, cells: 513 },
  { label: "0.10", fraction: 0.1, cells: 590 },
] as const;

describe("orderTileForProgressiveDrawing", () => {
  it("is deterministic, keeps attributes paired, and preserves every point", () => {
    const first = orderTileForProgressiveDrawing(gridTile(), "3-4-2-1");
    const second = orderTileForProgressiveDrawing(gridTile(), "3-4-2-1");
    const firstIds = ids(first);

    expect(firstIds).toEqual(ids(second));
    expect([...firstIds].sort((a, b) => a - b)).toEqual(
      Array.from({ length: first.pointCount }, (_, id) => id),
    );
    for (let index = 0; index < first.pointCount; index += 1) {
      const id = firstIds[index]!;
      expect(first.positions[index * 3]).toBe(id % 64);
      expect(first.positions[index * 3 + 1]).toBe(Math.floor(id / 64));
    }
  });

  it.each([1, 255, 256, 257, 300, 319, 320, 511, 513, 100_003])(
    "preserves every point of a %i-point tile",
    (pointCount) => {
      const positions = new Float32Array(pointCount * 3);
      const rgb = new Uint8Array(pointCount * 3);
      for (let id = 0; id < pointCount; id += 1) {
        const offset = id * 3;
        positions[offset] = id;
        positions[offset + 1] = id * 2;
        positions[offset + 2] = id * 3;
        rgb[offset] = id & 0xff;
        rgb[offset + 1] = (id >>> 8) & 0xff;
        rgb[offset + 2] = (id >>> 16) & 0xff;
      }
      const tile = orderTileForProgressiveDrawing(
        { origin: [0, 0, 0], positions, rgb, pointCount },
        `9-1-2-${pointCount}`,
      );

      // A mapper uploads these directly and derives its element count on its
      // own, so a short or offset array would draw garbage.
      expect(tile.positions.length).toBe(pointCount * 3);
      expect(tile.rgb!.length).toBe(pointCount * 3);
      expect(tile.positions.byteOffset).toBe(0);
      expect(tile.rgb!.byteOffset).toBe(0);

      // Accumulated rather than asserted per point: 100,003 points is 300,009
      // assertions, and the matcher overhead alone outruns the test timeout.
      const seen = new Uint8Array(pointCount);
      const faults: string[] = [];
      for (let index = 0; index < pointCount; index += 1) {
        const id =
          tile.rgb![index * 3]! |
          (tile.rgb![index * 3 + 1]! << 8) |
          (tile.rgb![index * 3 + 2]! << 16);
        if (seen[id] !== 0) faults.push(`point ${id} emitted twice`);
        seen[id] = 1;
        if (
          tile.positions[index * 3] !== id ||
          tile.positions[index * 3 + 1] !== id * 2 ||
          tile.positions[index * 3 + 2] !== id * 3
        ) {
          faults.push(`point ${id} lost its position at slot ${index}`);
        }
      }

      expect(faults.slice(0, 4)).toEqual([]);
      expect(seen.indexOf(0)).toBe(-1);
    },
  );

  it("draws every part of the node at every prefix, not just at one crossover", () => {
    const tile = orderTileForProgressiveDrawing(patchTile(), "0-0-0-0");
    const measured = FRACTIONS.map(({ label, fraction }) => {
      const coverage = coverageOf(tile, Math.round(tile.pointCount * fraction));
      return { ...coverage, fraction: label };
    });

    // Every block of record order is represented at every fraction, and no
    // block is ever more than one doubling round — 2x — deeper than another.
    expect(
      measured.map(({ fraction, emptyPatches, leastDrawn, mostDrawn }) => ({
        fraction,
        emptyPatches,
        depthRatio: mostDrawn / leastDrawn,
      })),
    ).toEqual([
      { fraction: "1/64", emptyPatches: 0, depthRatio: 1 },
      { fraction: "1/32", emptyPatches: 0, depthRatio: 1 },
      { fraction: "1/16", emptyPatches: 0, depthRatio: 1 },
      { fraction: "1/8", emptyPatches: 0, depthRatio: 1 },
      { fraction: "1/4", emptyPatches: 0, depthRatio: 1 },
      { fraction: "1/2", emptyPatches: 0, depthRatio: 1 },
      { fraction: "0.05", emptyPatches: 0, depthRatio: 2 },
      { fraction: "0.10", emptyPatches: 0, depthRatio: 2 },
    ]);

    for (const [index, { label, cells }] of FRACTIONS.entries()) {
      const coverage = measured[index]!;
      expect(`${label}: ${coverage.extent.join(",")}`).toBe(
        `${label}: 0,0,${PATCH_SIDE - 1},${PATCH_SIDE - 1}`,
      );
      expect(`${label}: ${coverage.cells >= cells}`).toBe(`${label}: true`);
    }
  });

  it("keeps a short trailing block in the same rounds as a full one", () => {
    // 40,163 points is 157 blocks and a 163-point remainder: the trailing
    // block runs the rounds scaled to what it holds rather than dropping out.
    const pointCount = 157 * 256 + 163;
    const positions = new Float32Array(pointCount * 3);
    const rgb = new Uint8Array(pointCount * 3);
    for (let id = 0; id < pointCount; id += 1) {
      rgb[id * 3] = id & 0xff;
      rgb[id * 3 + 1] = (id >>> 8) & 0xff;
      rgb[id * 3 + 2] = (id >>> 16) & 0xff;
    }
    const tile = orderTileForProgressiveDrawing(
      { origin: [0, 0, 0], positions, rgb, pointCount },
      "5-1-1-1",
    );

    const prefix = Math.round(pointCount * 0.05);
    const drawn = new Int32Array(158);
    for (let index = 0; index < prefix; index += 1) {
      const id =
        tile.rgb![index * 3]! |
        (tile.rgb![index * 3 + 1]! << 8) |
        (tile.rgb![index * 3 + 2]! << 16);
      const block = Math.floor(id / 256);
      drawn[block] = drawn[block]! + 1;
    }

    expect(drawn.reduce((total, count) => total + (count ? 0 : 1), 0)).toBe(0);
    expect(drawn[157]).toBeGreaterThanOrEqual(
      Math.floor((Math.min(...drawn.subarray(0, 157)) * 163) / 256),
    );
  });
});
