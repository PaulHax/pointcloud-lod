/**
 * Pure per-tile draw allocation.
 *
 * A tile becomes eligible only after its parent received its complete prefix.
 * This preserves coarse coverage while allowing the most important visible
 * branch to refine ahead of peripheral branches. The last admitted tile may
 * consume a partial progressive prefix, so the draw budget has no tile-sized
 * discontinuity.
 */

export type DrawCandidate = {
  readonly key: string;
  readonly pointCount: number;
  readonly priority: number;
  readonly children: readonly string[];
};

export type PointPrefixAllocation = {
  readonly prefixes: ReadonlyMap<string, number>;
  readonly plannedPoints: number;
  readonly fullTiles: number;
  readonly partialTiles: number;
  readonly skippedTiles: number;
};

export type AllocatePointPrefixesOptions = {
  readonly root: string;
  readonly pointBudget: number;
  readonly getCandidate: (key: string) => DrawCandidate | undefined;
};

/** A total order on distinct keys, so admission is fully deterministic. */
const betterCandidate = (left: DrawCandidate, right: DrawCandidate): number =>
  right.priority - left.priority || left.key.localeCompare(right.key);

/**
 * The admission frontier as a binary heap under {@link betterCandidate}.
 *
 * Each admission takes the best tile and puts that tile's children back, so
 * the frontier changes by a handful of entries per step. Re-sorting the whole
 * list every time made the pass quadratic in the tiles a plan holds, which on
 * a dense view is thousands, for an order a heap maintains incrementally.
 */
const createFrontier = () => {
  const heap: DrawCandidate[] = [];

  const swap = (left: number, right: number): void => {
    const held = heap[left]!;
    heap[left] = heap[right]!;
    heap[right] = held;
  };

  return {
    push(candidate: DrawCandidate): void {
      heap.push(candidate);
      let index = heap.length - 1;
      while (index > 0) {
        const parent = (index - 1) >> 1;
        if (betterCandidate(heap[parent]!, heap[index]!) <= 0) break;
        swap(parent, index);
        index = parent;
      }
    },

    /** The best remaining candidate, or undefined once the frontier is spent. */
    pop(): DrawCandidate | undefined {
      const best = heap[0];
      const last = heap.pop();
      if (last !== undefined && heap.length > 0) {
        heap[0] = last;
        let index = 0;
        for (;;) {
          const left = index * 2 + 1;
          const right = left + 1;
          let next = index;
          if (
            left < heap.length &&
            betterCandidate(heap[left]!, heap[next]!) < 0
          ) {
            next = left;
          }
          if (
            right < heap.length &&
            betterCandidate(heap[right]!, heap[next]!) < 0
          ) {
            next = right;
          }
          if (next === index) break;
          swap(index, next);
          index = next;
        }
      }
      return best;
    },
  };
};

export const allocatePointPrefixes = (
  options: AllocatePointPrefixesOptions,
): PointPrefixAllocation => {
  const pointBudget = Math.max(0, Math.floor(options.pointBudget));
  // One pass over the reachable tree, for both the candidates admission walks
  // and the tile count its skips are measured against. Asking the host again
  // during admission made it rebuild every candidate's key, children and
  // priority a second time.
  const candidates = new Map<string, DrawCandidate>();
  const seen = new Set<string>();
  let drawableTiles = 0;
  const pending = [options.root];
  while (pending.length > 0) {
    const key = pending.pop()!;
    if (seen.has(key)) continue;
    seen.add(key);
    const candidate = options.getCandidate(key);
    if (candidate === undefined) continue;
    candidates.set(key, candidate);
    if (candidate.pointCount > 0) drawableTiles += 1;
    pending.push(...candidate.children);
  }

  const prefixes = new Map<string, number>();
  const frontier = createFrontier();
  const admit = (key: string): void => {
    const candidate = candidates.get(key);
    if (candidate === undefined) return;
    if (candidate.pointCount <= 0) {
      for (const child of candidate.children) admit(child);
      return;
    }
    frontier.push(candidate);
  };
  admit(options.root);

  let remaining = pointBudget;
  let fullTiles = 0;
  let partialTiles = 0;
  while (remaining > 0) {
    const candidate = frontier.pop();
    if (candidate === undefined) break;
    const count = Math.min(candidate.pointCount, remaining);
    prefixes.set(candidate.key, count);
    remaining -= count;
    if (count < candidate.pointCount) {
      partialTiles += 1;
      break;
    }
    fullTiles += 1;
    for (const child of candidate.children) admit(child);
  }

  const plannedPoints = [...prefixes.values()].reduce(
    (sum, count) => sum + count,
    0,
  );
  return {
    prefixes,
    plannedPoints,
    fullTiles,
    partialTiles,
    skippedTiles: Math.max(0, drawableTiles - prefixes.size),
  };
};
