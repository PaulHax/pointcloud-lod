import type { TileData } from "./tileSource";

/** FNV-1a keeps the same tile on the same deterministic permutation. */
const seedFrom = (key: string): number => {
  let hash = 0x811c9dc5;
  for (let index = 0; index < key.length; index += 1) {
    hash ^= key.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
};

/** Small deterministic generator used only to build a point permutation. */
const randomFrom = (seed: number): (() => number) => {
  let state = seed;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 0x1_0000_0000;
  };
};

const swapTriplet = (
  values: Float32Array | Uint8Array,
  leftPoint: number,
  rightPoint: number,
): void => {
  const left = leftPoint * 3;
  const right = rightPoint * 3;
  for (let component = 0; component < 3; component += 1) {
    const held = values[left + component]!;
    values[left + component] = values[right + component]!;
    values[right + component] = held;
  }
};

/**
 * Put a tile into deterministic random order in place.
 *
 * Progressive drawing uses a prefix of each VBO, so every prefix needs to be
 * a representative sample of the complete node. COPC/LAS record order makes
 * no such promise. A one-time Fisher-Yates pass inside the tile source
 * supplies that promise while keeping positions and RGB paired and without
 * allocating a second point payload. Sources apply it once, at decode, where
 * a worker-backed one is already off the main thread; the controller trusts
 * what it is handed.
 */
export const orderTileForProgressiveDrawing = (
  tile: TileData,
  key: string,
): TileData => {
  const random = randomFrom(seedFrom(key));
  for (let right = tile.pointCount - 1; right > 0; right -= 1) {
    const left = Math.floor(random() * (right + 1));
    if (left === right) continue;
    swapTriplet(tile.positions, left, right);
    if (tile.rgb !== undefined) swapTriplet(tile.rgb, left, right);
  }
  return tile;
};
