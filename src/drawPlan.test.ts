import { describe, expect, it } from "vitest";

import { perspectiveView } from "../test/helpers";
import { prepareView } from "./camera";
import {
  EMPTY_DRAW_PLAN,
  largestTerminalSpacing,
  planDraw,
  samePrefixes,
} from "./drawPlan";
import { keyFromString, type Bounds } from "./octree";
import type { Hierarchy, HierarchyEntry } from "./pointSelection";

// The identity view-projection shows the NDC cube from an eye at the origin:
// a spacing s at distance d projects to 50 s / d css px on its 100 px height.
const VIEW = prepareView(perspectiveView());

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

describe("planDraw", () => {
  it("thins every tile by the same fraction, rounding a partial point up", () => {
    const nodes = hierarchyOf({
      "0-0-0-0": node(10, box([0, 0, 0.5])),
      "1-0-0-0": node(7, box([0, 0, 0.5])),
      "1-1-0-0": node(0, box([0, 0, 0.5])),
    }).nodes;
    const plan = planDraw(
      ["0-0-0-0", "1-0-0-0", "1-1-0-0", "2-0-0-0"],
      nodes,
      0.4,
    );
    // Structural and unknown nodes draw nothing and get no prefix.
    expect([...plan.prefixes]).toEqual([
      ["0-0-0-0", 4],
      ["1-0-0-0", 3],
    ]);
    expect(plan.plannedPoints).toBe(7);
    expect(planDraw(["0-0-0-0"], nodes, 1).plannedPoints).toBe(10);
  });
});

describe("samePrefixes", () => {
  it("compares every key and count", () => {
    const plan = new Map([
      ["a", 1],
      ["b", 2],
    ]);
    expect(samePrefixes(plan, new Map(plan))).toBe(true);
    expect(samePrefixes(plan, new Map([["a", 1]]))).toBe(false);
    expect(
      samePrefixes(
        plan,
        new Map([
          ["a", 1],
          ["b", 3],
        ]),
      ),
    ).toBe(false);
  });
});

describe("largestTerminalSpacing", () => {
  const spacingOf = (
    hierarchy: Hierarchy,
    target: readonly string[],
    densityFraction = 1,
    refinementCutoffPx = 0,
  ) =>
    largestTerminalSpacing({
      target: new Set(target),
      hierarchy,
      view: VIEW,
      refinementCutoffPx,
      plan: planDraw(target, hierarchy.nodes, densityFraction),
    });

  // The root is 0.4 from the eye and projects to 12.5 px; both children are
  // 0.2 from it, at 0.5 px and 50 px.
  const TREE = hierarchyOf({
    "0-0-0-0": node(100, box([0, 0, 0.5], 0.1), {
      spacing: 0.1,
      ...children("1-0-0-0", "1-1-0-0"),
    }),
    "1-0-0-0": node(40, box([0.05, 0, 0.25]), { spacing: 0.002 }),
    "1-1-0-0": node(100, box([-0.05, 0, 0.25]), { spacing: 0.2 }),
  });

  it("is null when nothing is planned or the root is not selected", () => {
    expect(spacingOf(TREE, [])).toBeNull();
    expect(spacingOf(TREE, ["1-0-0-0"])).toBeNull();
    expect(spacingOf(TREE, ["0-0-0-0"], 0)).toBeNull();
    expect(
      largestTerminalSpacing({
        target: new Set(["0-0-0-0"]),
        hierarchy: TREE,
        view: VIEW,
        refinementCutoffPx: 0,
        plan: EMPTY_DRAW_PLAN,
      }),
    ).toBeNull();
  });

  it("takes the coarsest leaf of a fully selected tree", () => {
    expect(spacingOf(TREE, ["0-0-0-0", "1-0-0-0", "1-1-0-0"])).toBeCloseTo(50);
  });

  it("counts a parent whose visible child the budget turned away", () => {
    expect(spacingOf(TREE, ["0-0-0-0", "1-0-0-0"])).toBeCloseTo(12.5);
  });

  it("counts a parent under the refinement cutoff, and not its children", () => {
    expect(
      spacingOf(TREE, ["0-0-0-0", "1-0-0-0", "1-1-0-0"], 1, 1_000),
    ).toBeCloseTo(12.5);
  });

  it("counts a parent blocked on an unread page, but not on a culled one", () => {
    const paged = (loadedPages: readonly string[], x: number) =>
      hierarchyOf(
        {
          "0-0-0-0": TREE.nodes.get("0-0-0-0")!,
          "1-0-0-0": TREE.nodes.get("1-0-0-0")!,
          "1-1-0-0": node(0, box([x, 0, 0.25]), { pageRef: true }),
        },
        loadedPages,
      );
    expect(spacingOf(paged([], 0), ["0-0-0-0", "1-0-0-0"])).toBeCloseTo(12.5);
    expect(spacingOf(paged([], 5), ["0-0-0-0", "1-0-0-0"])).toBeCloseTo(0.5);
  });

  it("spreads a thinned terminal by the square root of its density", () => {
    expect(
      spacingOf(TREE, ["0-0-0-0", "1-0-0-0", "1-1-0-0"], 0.25),
    ).toBeCloseTo(100);
  });
});
