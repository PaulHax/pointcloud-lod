import { describe, expect, it } from "vitest";

import { keyToString } from "./octree";
import { createPayloadResidency, type TileBatch } from "./payloadResidency";
import type { TileData } from "./tileSource";

/** A position-only tile of `pointCount` points costs 12 bytes each plus 64. */
const tile = (pointCount: number): TileData => ({
  origin: [0, 0, 0],
  positions: new Float32Array(pointCount * 3),
  pointCount,
});

const keysOf = (batch: TileBatch | null) =>
  batch && {
    added: batch.added.map((entry) => keyToString(entry.key)),
    removed: batch.removed.map(keyToString),
  };

describe("createPayloadResidency", () => {
  it("keeps one owner per payload as it moves between residency and the cache", () => {
    const residency = createPayloadResidency({ cacheBytes: 1 << 20 });
    residency.hold("1-0-0-0", tile(10));
    expect(residency.totals()).toEqual({
      residentTiles: 1,
      residentPoints: 10,
      residentBytes: 184,
      cachedTiles: 0,
      cachedBytes: 0,
    });

    residency.releaseExcept(new Set());
    expect(residency.isResident("1-0-0-0")).toBe(false);
    expect(residency.isDecoded("1-0-0-0")).toBe(true);
    expect(residency.totals()).toMatchObject({
      residentTiles: 0,
      residentPoints: 0,
      residentBytes: 0,
      cachedTiles: 1,
      cachedBytes: 184,
    });

    expect(residency.promote("1-0-0-0")).toBe(true);
    expect(residency.promote("2-0-0-0")).toBe(false);
    expect(residency.totals()).toMatchObject({
      residentTiles: 1,
      cachedTiles: 0,
    });

    // A resident key needs no cached copy.
    residency.park("1-0-0-0", tile(10));
    residency.park("1-1-0-0", tile(20));
    expect(residency.totals()).toMatchObject({
      residentTiles: 1,
      cachedTiles: 1,
      cachedBytes: 304,
    });
    expect([...residency.residentKeys()]).toEqual(["1-0-0-0"]);
  });

  it("releases only what the kept set does not name", () => {
    const residency = createPayloadResidency({ cacheBytes: 1 << 20 });
    residency.hold("0-0-0-0", tile(1));
    residency.hold("1-0-0-0", tile(1));
    residency.hold("1-1-0-0", tile(1));
    residency.releaseExcept(new Set(["1-0-0-0"]));
    expect([...residency.residentKeys()]).toEqual(["1-0-0-0"]);
    expect(residency.totals().cachedTiles).toBe(2);
  });

  it("evicts the least recently used cached payloads past its byte bound", () => {
    const residency = createPayloadResidency({ cacheBytes: 400 });
    residency.park("1-0-0-0", tile(10));
    residency.park("1-1-0-0", tile(10));
    residency.park("1-0-1-0", tile(10));
    expect(residency.isDecoded("1-0-0-0")).toBe(false);
    expect(residency.isDecoded("1-1-0-0")).toBe(true);
    expect(residency.isDecoded("1-0-1-0")).toBe(true);
  });

  it("hands the consumer the change since the last flush", () => {
    const residency = createPayloadResidency({ cacheBytes: 1 << 20 });
    residency.hold("0-0-0-0", tile(1));
    residency.hold("1-0-0-0", tile(1));
    expect(keysOf(residency.flush())).toEqual({
      added: ["0-0-0-0", "1-0-0-0"],
      removed: [],
    });
    expect(residency.flush()).toBeNull();
    expect([...residency.submitted().keys()]).toEqual(["0-0-0-0", "1-0-0-0"]);

    residency.releaseExcept(new Set(["0-0-0-0"]));
    expect(keysOf(residency.flush())).toEqual({
      added: [],
      removed: ["1-0-0-0"],
    });
  });

  it("never hands over a key made resident and released between flushes", () => {
    const residency = createPayloadResidency({ cacheBytes: 1 << 20 });
    residency.hold("0-0-0-0", tile(1));
    residency.flush();
    residency.hold("1-0-0-0", tile(1));
    residency.releaseExcept(new Set(["0-0-0-0"]));
    expect(residency.flush()).toBeNull();
  });

  it("reports a replaced payload as an addition only", () => {
    const residency = createPayloadResidency({ cacheBytes: 1 << 20 });
    residency.hold("0-0-0-0", tile(1));
    residency.flush();
    residency.releaseExcept(new Set());
    const replacement = tile(2);
    residency.hold("0-0-0-0", replacement);
    const batch = residency.flush()!;
    expect(batch.removed).toEqual([]);
    expect(batch.added).toHaveLength(1);
    expect(batch.added[0]!.tile).toBe(replacement);
  });

  it("keeps what the consumer holds across a clear, until the next flush removes it", () => {
    const residency = createPayloadResidency({ cacheBytes: 1 << 20 });
    residency.hold("0-0-0-0", tile(1));
    residency.park("1-0-0-0", tile(1));
    residency.flush();
    residency.clear();
    expect(residency.totals()).toEqual({
      residentTiles: 0,
      residentPoints: 0,
      residentBytes: 0,
      cachedTiles: 0,
      cachedBytes: 0,
    });
    expect(keysOf(residency.flush())).toEqual({
      added: [],
      removed: ["0-0-0-0"],
    });
  });

  it("gives back everything the consumer holds, once", () => {
    const residency = createPayloadResidency({ cacheBytes: 1 << 20 });
    residency.hold("0-0-0-0", tile(1));
    residency.hold("1-0-0-0", tile(1));
    residency.flush();
    expect(residency.takeBack().map(keyToString)).toEqual([
      "0-0-0-0",
      "1-0-0-0",
    ]);
    expect(residency.takeBack()).toEqual([]);
    residency.clear();
    expect(residency.flush()).toBeNull();
  });
});
