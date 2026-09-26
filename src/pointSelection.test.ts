import { describe, expect, it } from "vitest";

import { perspectiveView } from "../test/helpers";
import { prepareView } from "./camera";
import { keyFromString, type Bounds } from "./octree";
import {
  frontierOrder,
  selectPoints,
  type Hierarchy,
  type HierarchyEntry,
} from "./pointSelection";

// The identity view-projection shows the NDC cube from an eye at the origin
// looking down +z, so the centre ray is the +z axis.
const VIEW = prepareView(perspectiveView());

/** A cube of half size `half` centred at `center`. */
const box = (center: [number, number, number], half = 0.05): Bounds => ({
  min: [center[0] - half, center[1] - half, center[2] - half],
  max: [center[0] + half, center[1] + half, center[2] + half],
});

const node = (
  pointCount: number,
  bounds: Bounds,
  extra: Partial<HierarchyEntry> = {},
): HierarchyEntry => ({
  pointCount,
  bounds,
  spacing: 0.01,
  children: null,
  pageRef: false,
  ...extra,
});

const hierarchyOf = (
  nodes: Record<string, HierarchyEntry>,
  loadedPages: readonly string[] = [],
): Hierarchy => ({
  nodes: new Map(Object.entries(nodes)),
  loadedPages: new Set(loadedPages),
});

const children = (...keys: string[]) => ({
  children: keys.map((key) => keyFromString(key)),
});

/** A root over two children: one near the centre ray, one off to the side. */
const NEAR_AND_FAR = hierarchyOf({
  "0-0-0-0": node(100, box([0, 0, 0.5], 0.4), children("1-0-0-0", "1-1-0-0")),
  "1-0-0-0": node(60, box([0.1, 0, 0.5])),
  "1-1-0-0": node(60, box([-0.6, 0, 0.5])),
});

const select = (
  hierarchy: Hierarchy,
  pointBudget: number,
  extra: {
    readonly refinementCutoffPx?: number;
    readonly previous?: ReadonlySet<string>;
    readonly seed?: ReadonlySet<string>;
  } = {},
) =>
  selectPoints({
    hierarchy,
    view: VIEW,
    pointBudget,
    refinementCutoffPx: extra.refinementCutoffPx ?? 0,
    previous: extra.previous ?? new Set(),
    seed: extra.seed,
  });

describe("selectPoints", () => {
  it("takes the root, then the child nearest the centre ray, within the budget", () => {
    const selection = select(NEAR_AND_FAR, 200);
    expect([...selection.target].sort()).toEqual(["0-0-0-0", "1-0-0-0"]);
    expect(selection).toMatchObject({
      targetPoints: 160,
      consideredNodes: 3,
      availableNodes: 3,
      budgetSkippedNodes: 1,
      budgetSkippedPoints: 60,
      sseStoppedNodes: 0,
      neededPages: [],
    });
  });

  it("reports a visible root's error even when the budget admits nothing", () => {
    const selection = select(NEAR_AND_FAR, 50);
    expect(selection.target.size).toBe(0);
    expect(selection.budgetSkippedPoints).toBe(100);
    const root = NEAR_AND_FAR.nodes.get("0-0-0-0")!;
    expect(selection.rootSseCssPx).toBe(VIEW.nodeScreenSpaceError(root));
    expect(selection.rootSseCssPx).toBeGreaterThan(0);
  });

  it("reports no error for a culled root", () => {
    const selection = select(
      hierarchyOf({ "0-0-0-0": node(10, box([5, 0, 0.5])) }),
      1000,
    );
    expect(selection.target.size).toBe(0);
    expect(selection.rootSseCssPx).toBe(0);
  });

  it("stops at the unread pages it can see and asks for them in walk order", () => {
    const hierarchy = hierarchyOf({
      "0-0-0-0": node(
        10,
        box([0, 0, 0.5], 0.4),
        children("1-0-0-0", "1-1-0-0", "1-0-1-0"),
      ),
      "1-0-0-0": node(0, box([0.1, 0, 0.5]), { pageRef: true }),
      "1-1-0-0": node(0, box([-0.3, 0, 0.5]), { pageRef: true }),
      // Outside the frustum: its page would buy nothing this view can use.
      "1-0-1-0": node(0, box([5, 0, 0.5]), { pageRef: true }),
    });
    const selection = select(hierarchy, 1000);
    expect([...selection.target]).toEqual(["0-0-0-0"]);
    expect(selection.neededPages).toEqual(["1-0-0-0", "1-1-0-0"]);

    const read = select(
      hierarchyOf(Object.fromEntries(hierarchy.nodes), ["1-0-0-0"]),
      1000,
    );
    expect([...read.target].sort()).toEqual(["0-0-0-0", "1-0-0-0"]);
    expect(read.neededPages).toEqual(["1-1-0-0"]);
  });

  it("keeps a node under the refinement cutoff without its children", () => {
    const selection = select(NEAR_AND_FAR, 1000, {
      refinementCutoffPx: 1_000_000,
    });
    expect([...selection.target]).toEqual(["0-0-0-0"]);
    expect(selection.sseStoppedNodes).toBe(1);
    expect(selection.consideredNodes).toBe(1);
  });

  it("keeps a selected tile until a challenger is materially nearer the centre", () => {
    // 8 css px of hysteresis on a 100 px, 90° view is about 0.159 rad.
    const nearlyTied = hierarchyOf({
      "0-0-0-0": node(
        100,
        box([0, 0, 0.5], 0.4),
        children("1-0-0-0", "1-1-0-0"),
      ),
      "1-0-0-0": node(60, box([0.05, 0, 0.5])),
      "1-1-0-0": node(60, box([-0.1, 0, 0.5])),
    });
    const fresh = select(nearlyTied, 160);
    expect(fresh.target.has("1-0-0-0")).toBe(true);
    const held = select(nearlyTied, 160, {
      previous: new Set(["0-0-0-0", "1-1-0-0"]),
    });
    expect(held.target.has("1-1-0-0")).toBe(true);

    const challenged = select(NEAR_AND_FAR, 160, {
      previous: new Set(["0-0-0-0", "1-1-0-0"]),
    });
    expect(challenged.target.has("1-0-0-0")).toBe(true);
  });

  it("keeps a seeded selection that a larger budget still covers", () => {
    const hierarchy = hierarchyOf({
      "0-0-0-0": node(
        10,
        box([0, 0, 0.5], 0.4),
        children("1-0-0-0", "1-1-0-0"),
      ),
      "1-0-0-0": node(80, box([0.1, 0, 0.5])),
      "1-1-0-0": node(30, box([-0.6, 0, 0.5])),
    });
    expect([...select(hierarchy, 90).target].sort()).toEqual([
      "0-0-0-0",
      "1-0-0-0",
    ]);
    const seeded = select(hierarchy, 90, {
      seed: new Set(["0-0-0-0", "1-1-0-0"]),
    });
    expect([...seeded.target].sort()).toEqual(["0-0-0-0", "1-1-0-0"]);
  });
});

describe("frontierOrder", () => {
  it("orders by level, then centre-ray offset, then error, then key", () => {
    const nodes = hierarchyOf({
      "0-0-0-0": node(1, box([0, 0, 0.5], 0.4)),
      "1-0-0-0": node(1, box([0.4, 0, 0.5])),
      // On the centre ray: both have no offset, and the nearer one has the
      // larger error.
      "1-1-0-0": node(1, box([0, 0, 0.5])),
      "1-0-1-0": node(1, box([0, 0, 0.25])),
      "1-1-1-0": node(1, box([0, 0, 0.5])),
      "2-0-0-0": node(1, box([0, 0, 0.5])),
    }).nodes;
    expect(
      frontierOrder(
        [
          "2-0-0-0",
          "1-0-0-0",
          "1-1-0-1",
          "1-1-1-0",
          "1-1-0-0",
          "1-0-1-0",
          "0-0-0-0",
        ],
        nodes,
        VIEW,
      ),
    ).toEqual([
      "0-0-0-0",
      "1-0-1-0",
      "1-1-0-0",
      "1-1-1-0",
      "1-0-0-0",
      // Unknown to the hierarchy: last of its level.
      "1-1-0-1",
      "2-0-0-0",
    ]);
  });
});
