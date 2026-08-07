/**
 * Pure per-tile draw allocation.
 *
 * A tile becomes eligible only after its parent received its complete prefix.
 * Eligible tiles are admitted coarse-level-first, preserving broad coverage
 * before the most important visible branch refines. The last admitted tile
 * may consume a partial progressive prefix, so the draw budget has no
 * tile-sized discontinuity.
 */

export type DrawCandidate = {
  readonly key: string;
  readonly pointCount: number;
  readonly priority: number;
  readonly secondaryPriority?: number;
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
  right.priority - left.priority ||
  (right.secondaryPriority ?? 0) - (left.secondaryPriority ?? 0) ||
  left.key.localeCompare(right.key);

export const allocatePointPrefixes = (
  options: AllocatePointPrefixesOptions,
): PointPrefixAllocation => {
  const pointBudget = Math.max(0, Math.floor(options.pointBudget));
  // One pass over the reachable tree, for both the candidates admission walks
  // and the tile count its skips are measured against.
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
  let remaining = pointBudget;
  let fullTiles = 0;
  let partialTiles = 0;
  let frontier = [options.root];
  allocation: while (remaining > 0 && frontier.length > 0) {
    const ranked = frontier
      .map((key) => candidates.get(key))
      .filter((candidate) => candidate !== undefined)
      .sort(betterCandidate);
    const next: string[] = [];
    for (const candidate of ranked) {
      if (candidate.pointCount <= 0) {
        next.push(...candidate.children);
        continue;
      }
      const count = Math.min(candidate.pointCount, remaining);
      prefixes.set(candidate.key, count);
      remaining -= count;
      if (count < candidate.pointCount) {
        partialTiles += 1;
        break allocation;
      }
      fullTiles += 1;
      if (remaining === 0) break allocation;
      next.push(...candidate.children);
    }
    frontier = next;
  }

  return {
    prefixes,
    plannedPoints: pointBudget - remaining,
    fullTiles,
    partialTiles,
    skippedTiles: Math.max(0, drawableTiles - prefixes.size),
  };
};
