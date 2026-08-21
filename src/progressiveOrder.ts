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

const BLOCK_POINTS = 256;
const PREFIX_POINTS = 64;

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
 * Put a tile into deterministic, progressively representative block order.
 *
 * Progressive drawing uses a prefix of each VBO, so every prefix needs to be
 * a representative sample of the complete node. COPC/LAS record order makes
 * no such promise. Shuffled 256-point blocks distribute a random 64-point
 * slice from every block at the front, then append each block's remainder.
 * That keeps early prefixes broad while typed-array bulk copies avoid the
 * random reads and six scalar writes per point of a full Fisher-Yates pass.
 * Sources apply this once at decode, where a worker-backed one is already off
 * the main thread; the controller trusts what it is handed.
 */
export const orderTileForProgressiveDrawing = (
  tile: TileData,
  key: string,
): TileData => {
  const random = randomFrom(seedFrom(key));
  if (tile.pointCount <= BLOCK_POINTS) {
    for (let right = tile.pointCount - 1; right > 0; right -= 1) {
      const left = Math.floor(random() * (right + 1));
      if (left === right) continue;
      swapTriplet(tile.positions, left, right);
      if (tile.rgb !== undefined) swapTriplet(tile.rgb, left, right);
    }
    return tile;
  }

  const blockCount = Math.ceil(tile.pointCount / BLOCK_POINTS);
  const order = Uint32Array.from({ length: blockCount }, (_, index) => index);
  for (let right = blockCount - 1; right > 0; right -= 1) {
    const left = Math.floor(random() * (right + 1));
    const held = order[left]!;
    order[left] = order[right]!;
    order[right] = held;
  }

  const prefixStarts = new Uint8Array(blockCount);
  for (let block = 0; block < blockCount; block += 1) {
    const available = Math.min(
      BLOCK_POINTS,
      tile.pointCount - block * BLOCK_POINTS,
    );
    const maximumStart = Math.max(0, available - PREFIX_POINTS);
    prefixStarts[block] =
      block === 0
        ? 0
        : block === blockCount - 1
          ? maximumStart
          : Math.floor(random() * (maximumStart + 1));
  }

  const positions = new Float32Array(tile.positions.length);
  const rgb =
    tile.rgb === undefined ? undefined : new Uint8Array(tile.rgb.length);
  let outputValue = 0;
  const copyRange = (
    block: number,
    firstWithinBlock: number,
    requestedPoints: number,
  ): void => {
    const firstPoint = block * BLOCK_POINTS + firstWithinBlock;
    const pointCount = Math.min(requestedPoints, tile.pointCount - firstPoint);
    if (pointCount <= 0) return;
    const firstValue = firstPoint * 3;
    const endValue = firstValue + pointCount * 3;
    positions.set(tile.positions.subarray(firstValue, endValue), outputValue);
    rgb?.set(tile.rgb!.subarray(firstValue, endValue), outputValue);
    outputValue += pointCount * 3;
  };

  for (const block of order) {
    copyRange(block, prefixStarts[block]!, PREFIX_POINTS);
  }
  for (const block of order) {
    const prefixStart = prefixStarts[block]!;
    copyRange(block, 0, prefixStart);
    copyRange(
      block,
      prefixStart + PREFIX_POINTS,
      BLOCK_POINTS - prefixStart - PREFIX_POINTS,
    );
  }

  return { ...tile, positions, rgb };
};
