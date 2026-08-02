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

const ids = (tile: TileData): number[] => {
  const result: number[] = [];
  for (let index = 0; index < tile.pointCount; index += 1) {
    result.push(tile.rgb![index * 3]! | (tile.rgb![index * 3 + 1]! << 8));
  }
  return result;
};

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

  it("makes an early prefix cover the complete spatial extent", () => {
    const tile = orderTileForProgressiveDrawing(gridTile(), "0-0-0-0");
    const prefixPoints = tile.pointCount / 4;
    const occupied = new Set<string>();
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
      occupied.add(`${Math.floor(x / 8)}-${Math.floor(y / 8)}`);
    }

    expect([minX, minY, maxX, maxY]).toEqual([0, 0, 63, 63]);
    expect(occupied.size).toBe(64);
  });
});
