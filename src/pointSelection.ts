/**
 * Pure point selection: which octree nodes a view draws within a point
 * budget, which hierarchy pages it is blocked on, and the order requests go
 * out in. Everything reads the controller's hierarchy through a read-only
 * view and a prepared camera, so a pass never copies the hierarchy.
 *
 * COPC hierarchies are additive: children add detail while parents keep
 * rendering, so there is no parent/child swap and no hole risk. The
 * parent-closed selection is the whole hole-free story.
 */

import { selectNodes } from "./budget";
import {
  boundsIntersectsFrustum,
  type CameraView,
  type PreparedView,
} from "./camera";
import {
  ROOT_KEY,
  childKeys,
  keyToString,
  levelFromString,
  type Bounds,
  type VoxelKey,
} from "./octree";

export type HierarchyEntry = {
  readonly pointCount: number;
  readonly bounds: Bounds;
  readonly spacing: number;
  /** Children the page named, or null to derive them from sibling entries. */
  readonly children: readonly VoxelKey[] | null;
  /** The entry stands in for a hierarchy page that has to be read first. */
  readonly pageRef: boolean;
  /** Actual decoded payload cost, retained after its payload is evicted. */
  readonly tileBytes?: number;
};

/** RGB is the largest TileData payload: 12 position bytes + 3 color bytes. */
export const nodePayloadBytes = (entry: HierarchyEntry): number =>
  entry.pointCount === 0 ? 0 : (entry.tileBytes ?? entry.pointCount * 15 + 64);

export type Hierarchy = {
  readonly nodes: ReadonlyMap<string, HierarchyEntry>;
  readonly loadedPages: ReadonlySet<string>;
};

export const ROOT_KEY_STRING = keyToString(ROOT_KEY);

export const childrenOf = (
  nodes: ReadonlyMap<string, HierarchyEntry>,
  key: VoxelKey,
  entry: HierarchyEntry,
): readonly VoxelKey[] =>
  entry.children ??
  childKeys(key).filter((child) => nodes.has(keyToString(child)));

/** An unknown key has no error. */
const nodeSse = (
  nodes: ReadonlyMap<string, HierarchyEntry>,
  view: PreparedView,
  keyString: string,
): number => {
  const entry = nodes.get(keyString);
  return entry === undefined ? 0 : view.nodeScreenSpaceError(entry);
};

/** An unknown key sorts after every known one of its level. */
const centerOffset = (
  nodes: ReadonlyMap<string, HierarchyEntry>,
  view: PreparedView,
  keyString: string,
): number => {
  const entry = nodes.get(keyString);
  if (entry === undefined) return Number.POSITIVE_INFINITY;
  const { min, max } = entry.bounds;
  return view.centerRayOffset([
    (min[0] + max[0]) / 2,
    (min[1] + max[1]) / 2,
    (min[2] + max[2]) / 2,
  ]);
};

/** Screen-centre improvement required to replace a selected boundary tile. */
const SELECTION_CENTER_HYSTERESIS_CSS_PX = 8;

const centerOffsetHysteresis = (view: CameraView): number => {
  const normalized =
    (2 * SELECTION_CENTER_HYSTERESIS_CSS_PX) / view.viewportHeightCssPx;
  return view.projection === "perspective"
    ? Math.atan(normalized * Math.tan(view.fovY / 2))
    : normalized;
};

export type PointSelection = {
  readonly target: ReadonlySet<string>;
  /** Visible page references the walk stopped at, in the order it met them. */
  readonly neededPages: readonly string[];
  readonly targetPoints: number;
  readonly consideredNodes: number;
  readonly availableNodes: number;
  readonly sseStoppedNodes: number;
  readonly budgetSkippedNodes: number;
  readonly budgetSkippedPoints: number;
  /**
   * The root's screen-space error, or 0 when the root is culled or unknown.
   * It says whether the cloud is in view, not whether the budget admitted
   * anything: a root larger than the budget selects nothing, and reporting
   * that as culled would earn the cloud no quality, and so no budget that
   * could ever admit its root.
   */
  readonly rootSseCssPx: number;
};

export const EMPTY_SELECTION: PointSelection = {
  target: new Set(),
  neededPages: [],
  targetPoints: 0,
  consideredNodes: 0,
  availableNodes: 0,
  sseStoppedNodes: 0,
  budgetSkippedNodes: 0,
  budgetSkippedPoints: 0,
  rootSseCssPx: 0,
};

/**
 * Frustum cull, then centre-ray cone priority, then the parent-closed
 * point-budget walk. A selected tile keeps its place against a challenger
 * that is less than `SELECTION_CENTER_HYSTERESIS_CSS_PX` closer to the view
 * centre, so a small camera move does not trade one boundary tile for its
 * neighbour.
 */
export const selectPoints = (input: {
  readonly hierarchy: Hierarchy;
  readonly view: PreparedView;
  readonly pointBudget: number;
  readonly memoryBudgetBytes?: number;
  readonly refinementCutoffPx: number;
  /** The selection being replaced, which the hysteresis favours. */
  readonly previous: ReadonlySet<string>;
  /** A selection to keep while a larger budget adds detail. */
  readonly seed?: ReadonlySet<string>;
}): PointSelection => {
  const { hierarchy, view, previous } = input;
  const { nodes, loadedPages } = hierarchy;
  const hysteresis = centerOffsetHysteresis(view.view);
  const neededPages: string[] = [];
  let sseStoppedNodes = 0;
  const selection = selectNodes({
    root: ROOT_KEY,
    pointBudget: input.pointBudget,
    memoryBudgetBytes: input.memoryBudgetBytes,
    priority: (key) => {
      const keyString = keyToString(key);
      return (
        -centerOffset(nodes, view, keyString) +
        (previous.has(keyString) ? hysteresis : 0)
      );
    },
    secondaryPriority: (key) => nodeSse(nodes, view, keyToString(key)),
    seed: input.seed,
    getNode: (key) => {
      const keyString = keyToString(key);
      const entry = nodes.get(keyString);
      if (entry === undefined) return undefined;
      // Culling comes first: a page reference carries the bounds of the
      // subtree it stands for, so an invisible one must not be requested at
      // all. Reading it would spend a hierarchy slot on a region no
      // selection can use, which is the fan-out the page queue bounds.
      if (!boundsIntersectsFrustum(view.planes, entry.bounds)) return undefined;
      if (entry.pageRef && !loadedPages.has(keyString)) {
        neededPages.push(keyString);
        return undefined;
      }
      const children = childrenOf(nodes, key, entry);
      if (
        children.length > 0 &&
        view.nodeScreenSpaceError(entry) < input.refinementCutoffPx
      ) {
        sseStoppedNodes += 1;
        return {
          pointCount: entry.pointCount,
          children: [],
          memoryBytes: nodePayloadBytes(entry),
        };
      }
      return {
        pointCount: entry.pointCount,
        children,
        memoryBytes: nodePayloadBytes(entry),
      };
    },
  });
  const root = nodes.get(ROOT_KEY_STRING);
  return {
    target: selection.selected,
    neededPages,
    targetPoints: selection.totalPoints,
    consideredNodes: selection.consideredNodes,
    availableNodes: selection.availableNodes,
    sseStoppedNodes,
    budgetSkippedNodes: selection.budgetSkipped.size,
    budgetSkippedPoints: selection.budgetSkippedPoints,
    rootSseCssPx:
      root !== undefined && boundsIntersectsFrustum(view.planes, root.bounds)
        ? view.nodeScreenSpaceError(root)
        : 0,
  };
};

/**
 * Request order for both queues: coarse levels first, then the smallest 3D
 * centre-ray offset, then screen-space error. The selected frontier then
 * grows as concentric cones through the octree, so a hierarchy page is read
 * in the order the pages it unblocks would be.
 *
 * Each key's level and measures are read once and sorted alongside it rather
 * than inside the comparator, which a comparison sort calls O(n log n) times,
 * on every selection pass while the camera moves.
 */
export const frontierOrder = (
  keys: Iterable<string>,
  nodes: ReadonlyMap<string, HierarchyEntry>,
  view: PreparedView,
): string[] =>
  Array.from(keys, (keyString) => ({
    keyString,
    level: levelFromString(keyString),
    centerOffset: centerOffset(nodes, view, keyString),
    sse: nodeSse(nodes, view, keyString),
  }))
    .sort((a, b) =>
      a.level !== b.level
        ? a.level - b.level
        : a.centerOffset - b.centerOffset ||
          b.sse - a.sse ||
          a.keyString.localeCompare(b.keyString),
    )
    .map((entry) => entry.keyString);
