/**
 * Where each decoded point tile lives once its read has delivered it: in
 * renderer residency, or in a byte-bounded CPU cache for cheap reselection.
 * A payload has one owner at a time. A second copy would double-count
 * decoded bytes and spend cache capacity on a tile the renderer already
 * holds, so every move goes through `hold`, `releaseExcept`, `park` and
 * `promote`.
 *
 * The consumer sees residency as deltas: `flush` hands it the difference
 * between what is resident now and what it was last handed, so a key made
 * resident and released again between two flushes never reaches it, and no
 * batch both adds and removes one key.
 */

import { createLruCache } from "./lru";
import { keyFromString, type VoxelKey } from "./octree";
import { tileBytes, type TileData } from "./tileSource";

export type TileBatch = {
  readonly added: readonly { key: VoxelKey; tile: TileData }[];
  readonly removed: readonly VoxelKey[];
};

export type ResidencyTotals = {
  readonly residentTiles: number;
  readonly residentPoints: number;
  readonly residentBytes: number;
  readonly cachedTiles: number;
  readonly cachedBytes: number;
};

export type PayloadResidency = {
  /** Make a payload resident; the cache gives up any copy of the key. */
  hold(key: string, tile: TileData): void;
  /** Move every resident payload the set does not name into the cache. */
  releaseExcept(keep: ReadonlySet<string>): void;
  /** Cache a payload, unless residency already owns the key. */
  park(key: string, tile: TileData): void;
  /** Make a cached payload resident; false when the cache has none. */
  promote(key: string): boolean;
  isResident(key: string): boolean;
  /** Whether a payload for the key is resident or cached. */
  isDecoded(key: string): boolean;
  residentKeys(): IterableIterator<string>;
  /** What the consumer was last handed. */
  submitted(): ReadonlyMap<string, TileData>;
  /**
   * The change since the last flush, or null when there is none. The
   * consumer is then taken to hold exactly what is resident.
   */
  flush(): TileBatch | null;
  totals(): ResidencyTotals;
  /**
   * Drop every payload. The consumer keeps what it was handed until the next
   * flush takes it back.
   */
  clear(): void;
  /** Stop tracking what the consumer holds, and return it for removal. */
  takeBack(): VoxelKey[];
};

export const createPayloadResidency = (options: {
  readonly cacheBytes: number;
}): PayloadResidency => {
  const resident = new Map<string, TileData>();
  let residentPoints = 0;
  let residentBytes = 0;
  const cache = createLruCache<string, TileData>({
    maxBytes: options.cacheBytes,
  });
  let submitted = new Map<string, TileData>();

  const hold = (key: string, tile: TileData): void => {
    cache.delete(key);
    resident.set(key, tile);
    residentPoints += tile.pointCount;
    residentBytes += tileBytes(tile);
  };

  return {
    hold,

    releaseExcept(keep) {
      // Deleting the key being visited is safe while iterating a Map.
      for (const [key, tile] of resident) {
        if (keep.has(key)) continue;
        resident.delete(key);
        residentPoints -= tile.pointCount;
        residentBytes -= tileBytes(tile);
        cache.set(key, tile, tileBytes(tile));
      }
    },

    park(key, tile) {
      if (!resident.has(key)) cache.set(key, tile, tileBytes(tile));
    },

    promote(key) {
      const cached = cache.get(key);
      if (cached === undefined) return false;
      hold(key, cached);
      return true;
    },

    isResident: (key) => resident.has(key),

    isDecoded: (key) => resident.has(key) || cache.has(key),

    residentKeys: () => resident.keys(),

    submitted: () => submitted,

    flush() {
      const added: { key: VoxelKey; tile: TileData }[] = [];
      const removed: VoxelKey[] = [];
      for (const [key, tile] of resident) {
        // A key whose payload was replaced is an addition, never a
        // remove/add pair.
        if (submitted.get(key) === tile) continue;
        added.push({ key: keyFromString(key), tile });
      }
      for (const key of submitted.keys()) {
        if (!resident.has(key)) removed.push(keyFromString(key));
      }
      if (added.length === 0 && removed.length === 0) return null;
      submitted = new Map(resident);
      return { added, removed };
    },

    totals: () => ({
      residentTiles: resident.size,
      residentPoints,
      residentBytes,
      cachedTiles: cache.count(),
      cachedBytes: cache.totalBytes(),
    }),

    clear() {
      resident.clear();
      residentPoints = 0;
      residentBytes = 0;
      cache.clear();
    },

    takeBack() {
      const keys = [...submitted.keys()].map(keyFromString);
      submitted = new Map();
      return keys;
    },
  };
};
