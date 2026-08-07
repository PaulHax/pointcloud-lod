/**
 * Pure point-budget node selection.
 *
 * Walks the hierarchy breadth-first (level by level), trying candidates in
 * descending priority within each level. Selection stops at the first
 * optional node that does not fit instead of filling the remainder with a
 * lower-priority sibling. COPC node sizes vary widely; backfilling would let
 * one cheap geographic region win repeatedly even when it is farther from the
 * camera. A node is a candidate only if its parent was selected, so the result
 * is always a parent-closed set — the basis for hole-free progressive
 * refinement (a child never appears without every ancestor available to cover
 * the gaps around it).
 */

import { keyToString, type VoxelKey } from "./octree";

export type HierarchyNode = {
  /** Points stored in this node (not cumulative over the subtree). */
  readonly pointCount: number;
  /** Children known to exist in the hierarchy. */
  readonly children: readonly VoxelKey[];
};

export type SelectNodesOptions = {
  /** Root of the traversal (usually the octree root). */
  root: VoxelKey;
  /**
   * Hierarchy accessor. Return undefined for unknown keys (e.g. a hierarchy
   * page that has not loaded yet); such nodes are skipped.
   */
  getNode: (key: VoxelKey) => HierarchyNode | undefined;
  /**
   * Priority of a node — higher is more important (e.g. screen-space error).
   * Only compared between candidates of the same level.
   */
  priority: (key: VoxelKey) => number;
  /** Secondary descending priority used only when primary priorities tie. */
  secondaryPriority?: (key: VoxelKey) => number;
  /** Maximum total points across all selected nodes. */
  pointBudget: number;
  /**
   * Parent-closed selection to retain while a larger budget adds detail. Its
   * cost is reserved before optional candidates compete, measured against the
   * hierarchy this call sees: a seeded key the view no longer offers reserves
   * nothing, so it cannot starve the new selection.
   */
  seed?: ReadonlySet<string>;
};

export type NodeSelection = {
  /** Selected node keys, as `keyToString` strings. */
  readonly selected: ReadonlySet<string>;
  /** Sum of `pointCount` over the selection; never exceeds the budget. */
  readonly totalPoints: number;
  /** Candidates examined, including entries unavailable from the hierarchy. */
  readonly consideredNodes: number;
  /** Candidates for which `getNode` returned hierarchy data. */
  readonly availableNodes: number;
  /** Available nodes admitted to the parent-closed selection. */
  readonly selectedNodes: number;
  /** Available nodes rejected only because their points did not fit. */
  readonly budgetSkippedNodes: number;
  /** Points stored in the nodes rejected only by the point budget. */
  readonly budgetSkippedPoints: number;
  /** Keys rejected only because their points did not fit. */
  readonly budgetSkipped: ReadonlySet<string>;
};

type ResolveNode = (
  key: VoxelKey,
  keyString: string,
) => HierarchyNode | undefined;

/**
 * Cost of the seed under the hierarchy this call sees: a breadth-first walk of
 * the seeded keys from the root. An unavailable one — culled, or on a page that
 * has gone away — ends the walk there, so neither it nor its descendants (which
 * are unreachable in a parent-closed set) reserve any points.
 */
const seedPoints = (
  root: VoxelKey,
  seed: ReadonlySet<string>,
  resolve: ResolveNode,
): number => {
  let total = 0;
  let frontier: VoxelKey[] = [root];
  while (frontier.length > 0) {
    const next: VoxelKey[] = [];
    for (const key of frontier) {
      const keyString = keyToString(key);
      if (!seed.has(keyString)) continue;
      const node = resolve(key, keyString);
      if (node === undefined) continue;
      total += node.pointCount;
      next.push(...node.children);
    }
    frontier = next;
  }
  return total;
};

export const selectNodes = (options: SelectNodesOptions): NodeSelection => {
  const {
    root,
    getNode,
    priority,
    secondaryPriority = () => 0,
    pointBudget,
    seed,
  } = options;

  // The seed walk visits keys the main walk visits again; `getNode` is allowed
  // to be observed once per key.
  const resolved = new Map<string, HierarchyNode | undefined>();
  const resolve: ResolveNode = (key, keyString) => {
    if (resolved.has(keyString)) return resolved.get(keyString);
    const node = getNode(key);
    resolved.set(keyString, node);
    return node;
  };

  const selected = new Set<string>();
  const budgetSkipped = new Set<string>();
  let totalPoints = 0;
  let consideredNodes = 0;
  let availableNodes = 0;
  let budgetSkippedPoints = 0;
  let reservedSeedPoints = seed ? seedPoints(root, seed, resolve) : 0;
  let candidates: VoxelKey[] = [root];
  // Once one node in breadth-first priority order cannot fit, selection has
  // reached this budget's spatial boundary. Do not spend the leftover on
  // deeper foreground descendants: that would overshoot the foreground, then
  // remove it on a later pass when the next horizon tile becomes affordable.
  // Required seed nodes remain admissible so budget growth stays additive.
  let optionalBoundaryReached = false;

  while (candidates.length > 0) {
    consideredNodes += candidates.length;
    const ranked: {
      keyString: string;
      node: HierarchyNode;
      priority: number;
      secondaryPriority: number;
    }[] = [];
    for (const key of candidates) {
      const keyString = keyToString(key);
      const node = resolve(key, keyString);
      if (node === undefined) continue;
      ranked.push({
        keyString,
        node,
        priority: priority(key),
        secondaryPriority: secondaryPriority(key),
      });
    }
    ranked.sort(
      (a, b) =>
        b.priority - a.priority ||
        b.secondaryPriority - a.secondaryPriority ||
        a.keyString.localeCompare(b.keyString),
    );
    availableNodes += ranked.length;

    const nextCandidates: VoxelKey[] = [];
    for (const { keyString, node } of ranked) {
      const required = seed?.has(keyString) ?? false;
      const reservedAfter = required
        ? Math.max(0, reservedSeedPoints - node.pointCount)
        : reservedSeedPoints;
      if (
        (!required && optionalBoundaryReached) ||
        totalPoints + node.pointCount + reservedAfter > pointBudget
      ) {
        // A seeded selection remains required even after an earlier optional
        // candidate hits the boundary. Everything else stays a priority
        // prefix: a smaller, farther sibling must not exploit the remainder.
        budgetSkippedPoints += node.pointCount;
        budgetSkipped.add(keyString);
        if (!required) optionalBoundaryReached = true;
        continue;
      }
      selected.add(keyString);
      totalPoints += node.pointCount;
      reservedSeedPoints = reservedAfter;
      nextCandidates.push(...node.children);
    }
    candidates = nextCandidates;
  }

  return {
    selected,
    totalPoints,
    consideredNodes,
    availableNodes,
    selectedNodes: selected.size,
    budgetSkippedNodes: budgetSkipped.size,
    budgetSkippedPoints,
    budgetSkipped,
  };
};
