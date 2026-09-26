/**
 * Pure draw planning: how many of each selected tile's points to draw, and
 * the coarsest spacing that leaves on screen.
 */

import { boundsIntersectsFrustum, type PreparedView } from "./camera";
import { ROOT_KEY, keyToString, type VoxelKey } from "./octree";
import { projectedSpacingScale } from "./pointDensity";
import {
  ROOT_KEY_STRING,
  childrenOf,
  type Hierarchy,
  type HierarchyEntry,
} from "./pointSelection";

export type DrawPlan = {
  /** Points each selected tile draws: a prefix of its progressive order. */
  readonly prefixes: ReadonlyMap<string, number>;
  readonly plannedPoints: number;
};

export const EMPTY_DRAW_PLAN: DrawPlan = {
  prefixes: new Map(),
  plannedPoints: 0,
};

/**
 * Thin every selected tile to the same fraction of its points, rounding a
 * partial point up. Selection and residency stay unchanged: only the
 * progressive prefixes move, and they move together, so a smaller draw
 * allowance reads as a uniformly sparser cloud rather than as whole tiles
 * dropping out of the deepest levels while their neighbours stay dense.
 * Structural nodes carry no points and get no prefix.
 */
export const planDraw = (
  target: Iterable<string>,
  nodes: ReadonlyMap<string, HierarchyEntry>,
  densityFraction: number,
): DrawPlan => {
  const prefixes = new Map<string, number>();
  let plannedPoints = 0;
  for (const keyString of target) {
    const entry = nodes.get(keyString);
    if (entry === undefined || entry.pointCount === 0) continue;
    const prefix = Math.min(
      entry.pointCount,
      Math.ceil(entry.pointCount * densityFraction),
    );
    prefixes.set(keyString, prefix);
    plannedPoints += prefix;
  }
  return { prefixes, plannedPoints };
};

export const samePrefixes = (
  left: ReadonlyMap<string, number>,
  right: ReadonlyMap<string, number>,
): boolean => {
  if (left.size !== right.size) return false;
  for (const [key, count] of left) {
    if (right.get(key) !== count) return false;
  }
  return true;
};

/**
 * The coarsest projected spacing any terminal of the selection leaves on
 * screen, or null when nothing is drawn. A terminal is a selected node where
 * refinement stopped: a leaf, a node under the refinement cutoff, or one
 * with a visible child the selection could not take (an unread page, or one
 * the point budget turned away). Thinning a terminal to its prefix spreads
 * its points, so its error is scaled by the prefix's density.
 *
 * The walk follows the selection rather than what has landed, so the answer
 * moves once, with the selection. Fine tiles arriving under a coarse parent
 * would otherwise stop it being a terminal one arrival at a time, and every
 * point would shrink while the selection stood still.
 */
export const largestTerminalSpacing = (input: {
  readonly target: ReadonlySet<string>;
  readonly hierarchy: Hierarchy;
  readonly view: PreparedView;
  readonly refinementCutoffPx: number;
  readonly plan: DrawPlan;
}): number | null => {
  const { target, hierarchy, view, refinementCutoffPx, plan } = input;
  if (plan.plannedPoints <= 0 || !target.has(ROOT_KEY_STRING)) return null;
  const { nodes, loadedPages } = hierarchy;
  let largest: number | null = null;

  const terminal = (keyString: string, entry: HierarchyEntry): void => {
    const prefix = plan.prefixes.get(keyString) ?? 0;
    if (entry.pointCount === 0 || prefix <= 0) return;
    const spacing =
      view.nodeScreenSpaceError(entry) *
      projectedSpacingScale(prefix / entry.pointCount);
    largest = Math.max(largest ?? 0, spacing);
  };

  const walk = (key: VoxelKey): void => {
    const keyString = keyToString(key);
    if (!target.has(keyString)) return;
    const entry = nodes.get(keyString);
    if (entry === undefined) return;

    const children = childrenOf(nodes, key, entry);
    if (
      children.length === 0 ||
      view.nodeScreenSpaceError(entry) < refinementCutoffPx
    ) {
      terminal(keyString, entry);
      return;
    }

    let blocked = false;
    const openChildren: VoxelKey[] = [];
    for (const child of children) {
      const childString = keyToString(child);
      const childEntry = nodes.get(childString);
      if (childEntry === undefined) {
        blocked = true;
        continue;
      }
      if (!boundsIntersectsFrustum(view.planes, childEntry.bounds)) continue;
      // Matches selection: an invisible page reference is not requested, so
      // it is not blocking anything either.
      if (childEntry.pageRef && !loadedPages.has(childString)) {
        blocked = true;
        continue;
      }
      // A visible, available child of a selected parent can only be absent
      // because the breadth-first point budget rejected it.
      if (!target.has(childString)) {
        blocked = true;
        continue;
      }
      openChildren.push(child);
    }

    if (blocked) terminal(keyString, entry);
    for (const child of openChildren) walk(child);
  };

  walk(ROOT_KEY);
  return largest;
};
