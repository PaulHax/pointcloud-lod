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
  /** Priority-weighted points in this allocation. */
  readonly weightedPoints: number;
  /** Same point total distributed uniformly across the same tiles. */
  readonly uniformWeightedPoints: number;
};

export type AllocatePointPrefixesOptions = {
  readonly root: string;
  readonly pointBudget: number;
  readonly getCandidate: (key: string) => DrawCandidate | undefined;
};

const betterCandidate = (left: DrawCandidate, right: DrawCandidate): number =>
  right.priority - left.priority || left.key.localeCompare(right.key);

const uniformPrefixes = (
  candidates: readonly DrawCandidate[],
  pointBudget: number,
): ReadonlyMap<string, number> => {
  const totalPoints = candidates.reduce(
    (sum, candidate) => sum + candidate.pointCount,
    0,
  );
  const budget = Math.min(pointBudget, totalPoints);
  if (budget <= 0 || totalPoints <= 0) return new Map();

  const fraction = budget / totalPoints;
  const shares = candidates.map((candidate) => {
    const exact = candidate.pointCount * fraction;
    return {
      candidate,
      count: Math.floor(exact),
      remainder: exact - Math.floor(exact),
    };
  });
  let assigned = shares.reduce((sum, share) => sum + share.count, 0);
  shares.sort(
    (left, right) =>
      right.remainder - left.remainder ||
      left.candidate.key.localeCompare(right.candidate.key),
  );
  for (const share of shares) {
    if (assigned >= budget) break;
    share.count += 1;
    assigned += 1;
  }
  return new Map(
    shares
      .filter((share) => share.count > 0)
      .map((share) => [share.candidate.key, share.count]),
  );
};

export const allocatePointPrefixes = (
  options: AllocatePointPrefixesOptions,
): PointPrefixAllocation => {
  const pointBudget = Math.max(0, Math.floor(options.pointBudget));
  const reachable: DrawCandidate[] = [];
  const seen = new Set<string>();
  const pending = [options.root];
  while (pending.length > 0) {
    const key = pending.pop()!;
    if (seen.has(key)) continue;
    seen.add(key);
    const candidate = options.getCandidate(key);
    if (candidate === undefined) continue;
    if (candidate.pointCount > 0) reachable.push(candidate);
    pending.push(...candidate.children);
  }

  const prefixes = new Map<string, number>();
  const frontier: DrawCandidate[] = [];
  const admit = (key: string): void => {
    const candidate = options.getCandidate(key);
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
  while (remaining > 0 && frontier.length > 0) {
    frontier.sort(betterCandidate);
    const candidate = frontier.shift()!;
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
  const uniform = uniformPrefixes(reachable, plannedPoints);
  const weighted = (allocation: ReadonlyMap<string, number>): number =>
    reachable.reduce(
      (sum, candidate) =>
        sum + candidate.priority * (allocation.get(candidate.key) ?? 0),
      0,
    );

  return {
    prefixes,
    plannedPoints,
    fullTiles,
    partialTiles,
    skippedTiles: Math.max(0, reachable.length - prefixes.size),
    weightedPoints: weighted(prefixes),
    uniformWeightedPoints: weighted(uniform),
  };
};
