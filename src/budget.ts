/**
 * Pure point-budget node selection.
 *
 * Walks the hierarchy breadth-first (level by level), trying candidates in
 * descending priority within each level, and selects a node only while its
 * points fit in the remaining budget. A node is a candidate only if its
 * parent was selected, so the result is always a parent-closed set — the
 * basis for hole-free progressive refinement (a child never appears without
 * every ancestor available to cover the gaps around it).
 */

import { childKeys, keyToString, type VoxelKey } from './octree';

export interface HierarchyNode {
  /** Points stored in this node (not cumulative over the subtree). */
  readonly pointCount: number;
  /** Children known to exist in the hierarchy. */
  readonly children: readonly VoxelKey[];
}

export interface SelectNodesOptions {
  /** Root of the traversal (usually the octree root). */
  root: VoxelKey;
  /**
   * Hierarchy accessor. Return undefined for unknown keys (e.g. a hierarchy
   * page that has not loaded yet); such nodes are skipped.
   */
  getNode: (key: VoxelKey) => HierarchyNode | undefined;
  /**
   * Priority of a node — higher is more important (e.g. inverse screen-space
   * error). Only compared between candidates of the same level.
   */
  priority: (key: VoxelKey) => number;
  /** Maximum total points across all selected nodes. */
  pointBudget: number;
}

export interface NodeSelection {
  /** Selected node keys, as `keyToString` strings. */
  readonly selected: ReadonlySet<string>;
  /** Sum of `pointCount` over the selection; never exceeds the budget. */
  readonly totalPoints: number;
}

export const selectNodes = (options: SelectNodesOptions): NodeSelection => {
  const { root, getNode, priority, pointBudget } = options;

  const selected = new Set<string>();
  let totalPoints = 0;
  let candidates: VoxelKey[] = [root];

  while (candidates.length > 0) {
    const ranked = candidates
      .map((key) => ({ key, node: getNode(key), priority: priority(key) }))
      .filter(
        (c): c is { key: VoxelKey; node: HierarchyNode; priority: number } =>
          c.node !== undefined,
      )
      .sort((a, b) => b.priority - a.priority);

    const nextCandidates: VoxelKey[] = [];
    for (const { key, node } of ranked) {
      if (totalPoints + node.pointCount > pointBudget) {
        // Skipped: its subtree stays out (parent invariant), but cheaper
        // siblings later in the ranking may still fit.
        continue;
      }
      selected.add(keyToString(key));
      totalPoints += node.pointCount;
      nextCandidates.push(...node.children);
    }
    candidates = nextCandidates;
  }

  return { selected, totalPoints };
};

/** Convenience: a `HierarchyNode.children` list of all 8 octree children. */
export const allChildren = (key: VoxelKey): readonly VoxelKey[] =>
  childKeys(key);
