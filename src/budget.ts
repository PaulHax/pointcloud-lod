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
  /** Maximum total points across all selected nodes. */
  pointBudget: number;
  /**
   * Parent-closed selection to retain while a larger budget adds detail.
   * `totalPoints` reserves its exact cost before optional candidates compete.
   */
  seed?: {
    readonly selected: ReadonlySet<string>;
    readonly totalPoints: number;
  };
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

export const selectNodes = (options: SelectNodesOptions): NodeSelection => {
  const { root, getNode, priority, pointBudget, seed } = options;

  const selected = new Set<string>();
  const budgetSkipped = new Set<string>();
  let totalPoints = 0;
  let consideredNodes = 0;
  let availableNodes = 0;
  let budgetSkippedPoints = 0;
  let reservedSeedPoints = seed?.totalPoints ?? 0;
  let candidates: VoxelKey[] = [root];
  // Once one node in breadth-first priority order cannot fit, selection has
  // reached this budget's spatial boundary. Do not spend the leftover on
  // deeper foreground descendants: that would overshoot the foreground, then
  // remove it on a later pass when the next horizon tile becomes affordable.
  // Required seed nodes remain admissible so budget growth stays additive.
  let optionalBoundaryReached = false;

  while (candidates.length > 0) {
    consideredNodes += candidates.length;
    const ranked: { key: VoxelKey; node: HierarchyNode; priority: number }[] =
      [];
    for (const key of candidates) {
      const node = getNode(key);
      if (node === undefined) continue;
      ranked.push({ key, node, priority: priority(key) });
    }
    ranked.sort((a, b) => b.priority - a.priority);
    availableNodes += ranked.length;

    const nextCandidates: VoxelKey[] = [];
    for (const { key, node } of ranked) {
      const keyString = keyToString(key);
      const required = seed?.selected.has(keyString) ?? false;
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
