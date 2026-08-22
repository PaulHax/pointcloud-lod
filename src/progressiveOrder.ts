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

/**
 * Points a full block has contributed once each round has been emitted.
 *
 * A prefix is only representative if the granularity it is quantized to is
 * small compared to the prefix itself, so the round a prefix lands in has to
 * shrink as the prefix shrinks. Doubling gives that at every scale for the
 * cost of one bulk copy per round: a prefix ending inside a round leaves the
 * blocks it has not reached yet exactly one round behind the rest, so no
 * block is ever more than 2x the depth of another, and a prefix ending on a
 * round boundary draws every block to the same depth. The first round is 4
 * points rather than 1 so the smallest copies are still bulk copies.
 */
const ROUND_ENDS = [4, 8, 16, 32, 64, 128, BLOCK_POINTS];

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
 * no such promise. Points are cut into 256-point blocks — a block is a short
 * run of record order, so it is a small piece of the node — and every block
 * contributes to every {@link ROUND_ENDS} round, in a shuffled block order,
 * from a per-block rotated start. Any prefix therefore holds a comparable
 * slice of every block instead of all of some blocks. Typed-array bulk copies
 * avoid the random reads and six scalar writes per point of a full
 * Fisher-Yates pass. Sources apply this once at decode, where a worker-backed
 * one is already off the main thread; the controller trusts what it is
 * handed.
 *
 * What that buys stops at block granularity, deliberately. A round takes a
 * contiguous run from each block, so a prefix samples the node at as many
 * distinct places as there are blocks, not at as many places as it has
 * points; going finer would mean scattering points individually, which costs
 * more than the bulk copies save. Below `4 * blockCount` points — the first
 * round — a prefix cannot reach every block at all, which for a 100k-point
 * node is under 1.6% and well below the 5% floor a view ever asks for.
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
  const lastBlockPoints = tile.pointCount - (blockCount - 1) * BLOCK_POINTS;
  const order = Uint32Array.from({ length: blockCount }, (_, index) => index);
  for (let right = blockCount - 1; right > 0; right -= 1) {
    const left = Math.floor(random() * (right + 1));
    const held = order[left]!;
    order[left] = order[right]!;
    order[right] = held;
  }

  // Where each block starts contributing, so a round is not the same slice of
  // record order in every block. The block's points wrap around this start.
  const rotations = new Uint8Array(blockCount);
  for (let block = 0; block < blockCount; block += 1) {
    const available = block === blockCount - 1 ? lastBlockPoints : BLOCK_POINTS;
    rotations[block] = Math.floor(random() * available);
  }

  const positions = new Float32Array(tile.positions.length);
  const rgb =
    tile.rgb === undefined ? undefined : new Uint8Array(tile.rgb.length);
  let outputValue = 0;
  const copyPoints = (firstPoint: number, points: number): void => {
    if (points <= 0) return;
    const firstValue = firstPoint * 3;
    const endValue = firstValue + points * 3;
    positions.set(tile.positions.subarray(firstValue, endValue), outputValue);
    rgb?.set(tile.rgb!.subarray(firstValue, endValue), outputValue);
    outputValue += points * 3;
  };

  // A short last block runs the same rounds scaled to what it holds, so every
  // block reaches the same fraction of itself at the same round.
  const scaledEnd = (roundEnd: number): number =>
    Math.round((lastBlockPoints * roundEnd) / BLOCK_POINTS);

  let fullFrom = 0;
  let lastFrom = 0;
  for (const roundEnd of ROUND_ENDS) {
    const lastEnd = scaledEnd(roundEnd);
    // Which blocks lead varies by round, so a prefix cutting a round short
    // does not starve the same blocks every time.
    const lead = Math.floor(random() * blockCount);
    for (let index = 0; index < blockCount; index += 1) {
      const block = order[(index + lead) % blockCount]!;
      const isLast = block === blockCount - 1;
      const available = isLast ? lastBlockPoints : BLOCK_POINTS;
      const points =
        (isLast ? lastEnd : roundEnd) - (isLast ? lastFrom : fullFrom);
      if (points <= 0) continue;
      const blockFirst = block * BLOCK_POINTS;
      const start =
        ((isLast ? lastFrom : fullFrom) + rotations[block]!) % available;
      const beforeWrap = Math.min(points, available - start);
      copyPoints(blockFirst + start, beforeWrap);
      copyPoints(blockFirst, points - beforeWrap);
    }
    fullFrom = roundEnd;
    lastFrom = lastEnd;
  }

  return { ...tile, positions, rgb };
};
