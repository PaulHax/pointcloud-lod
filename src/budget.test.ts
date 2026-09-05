import { describe, expect, it } from "vitest";

import { selectNodes, type HierarchyNode } from "./budget";
import { keyFromString, keyToString, ROOT_KEY, type VoxelKey } from "./octree";

/** Build a hierarchy accessor from a { 'l-x-y-z': pointCount } table. */
const hierarchyOf = (
  counts: Record<string, number>,
): ((key: VoxelKey) => HierarchyNode | undefined) => {
  const known = new Set(Object.keys(counts));
  return (key) => {
    const s = keyToString(key);
    const pointCount = counts[s];
    if (pointCount === undefined) return undefined;
    // Children = the known entries one level down within this node's octant.
    const children = [...known]
      .map((k) => keyFromString(k))
      .filter(
        (c) =>
          c.level === key.level + 1 &&
          c.x >> 1 === key.x &&
          c.y >> 1 === key.y &&
          c.z >> 1 === key.z,
      );
    return { pointCount, children };
  };
};

const flatPriority = (): number => 0;

/** `selectNodes` from the root over a { 'l-x-y-z': pointCount } table. */
const select = (
  counts: Record<string, number>,
  pointBudget: number,
  priority: (key: VoxelKey) => number = flatPriority,
) =>
  selectNodes({
    root: ROOT_KEY,
    getNode: hierarchyOf(counts),
    priority,
    pointBudget,
  });

describe("selectNodes", () => {
  it("selects just the root when it has no children", () => {
    const result = select({ "0-0-0-0": 100 }, 1000);

    expect([...result.selected]).toEqual(["0-0-0-0"]);
    expect(result.totalPoints).toBe(100);
    expect(result).toMatchObject({
      consideredNodes: 1,
      availableNodes: 1,
      selectedNodes: 1,
      budgetSkippedNodes: 0,
      budgetSkippedPoints: 0,
    });
  });

  it("selects nothing when the root alone exceeds the budget", () => {
    const result = select({ "0-0-0-0": 100, "1-0-0-0": 10 }, 50);

    expect(result.selected.size).toBe(0);
    expect(result.totalPoints).toBe(0);
    expect(result.budgetSkippedNodes).toBe(1);
    expect(result.budgetSkippedPoints).toBe(100);
  });

  it("selects a whole small tree that fits the budget", () => {
    const counts = {
      "0-0-0-0": 100,
      "1-0-0-0": 50,
      "1-1-0-0": 50,
      "2-0-0-0": 25,
    };
    const result = select(counts, 1000);

    expect(result.selected).toEqual(new Set(Object.keys(counts)));
    expect(result.totalPoints).toBe(225);
  });

  it("on budget exhaustion mid-level, keeps higher-priority siblings", () => {
    const result = select(
      {
        "0-0-0-0": 100,
        "1-0-0-0": 60,
        "1-1-0-0": 60,
        "1-0-1-0": 60,
      },
      230, // root(100) + two children(120) = 220; third would hit 280
      // Priority by x+y: '1-1-0-0' > '1-0-1-0'? both = 1; make it distinct:
      (key) => key.x * 2 + key.y,
    );

    // Ranked children: '1-1-0-0' (2), '1-0-1-0' (1), '1-0-0-0' (0).
    expect(result.selected).toEqual(new Set(["0-0-0-0", "1-1-0-0", "1-0-1-0"]));
    expect(result.totalPoints).toBe(220);
    expect(result.budgetSkippedNodes).toBe(1);
    expect(result.budgetSkippedPoints).toBe(60);
  });

  it("uses secondary priority only after the primary priority ties", () => {
    const counts = {
      "0-0-0-0": 10,
      "1-0-0-0": 20,
      "1-1-0-0": 20,
      "1-0-1-0": 20,
    };
    const result = selectNodes({
      root: ROOT_KEY,
      getNode: hierarchyOf(counts),
      pointBudget: 50,
      priority: (key) => (key.y === 0 ? 1 : 0),
      secondaryPriority: (key) => key.x,
    });

    expect(result.selected).toEqual(new Set(["0-0-0-0", "1-1-0-0", "1-0-0-0"]));
  });

  it("does not backfill the remainder with a lower-priority sibling", () => {
    const result = select(
      {
        "0-0-0-0": 100,
        "1-1-1-1": 500, // highest priority but too big — skipped
        "1-0-0-0": 40, // fits, but is below the priority boundary
      },
      200,
      (key) => key.x + key.y + key.z,
    );

    expect(result.selected).toEqual(new Set(["0-0-0-0"]));
    expect(result.budgetSkipped).toEqual(new Set(["1-1-1-1", "1-0-0-0"]));
    expect(result.totalPoints).toBe(100);
  });

  it("does not descend deeper after reaching a same-level priority boundary", () => {
    const result = select(
      {
        "0-0-0-0": 50,
        "1-0-0-0": 500, // exceeds remaining budget — skipped
        "2-0-0-0": 1, // tiny grandchild, but its parent is out
        "1-1-0-0": 10,
        "2-2-0-0": 5, // child of the selected '1-1-0-0'
      },
      100,
      (key) => key.x,
    );

    expect(result.selected).toEqual(new Set(["0-0-0-0", "1-1-0-0"]));
    expect(result.selected.has("2-0-0-0")).toBe(false);
    expect(result.selected.has("2-2-0-0")).toBe(false);
    expect(result.totalPoints).toBe(60);
  });

  it("refines level by level (breadth-first), not depth-first", () => {
    // Deep chain under one child vs a shallow sibling: with a budget that
    // only fits three nodes, breadth-first must take both level-1 nodes
    // before any level-2 node, regardless of priority.
    const result = select(
      {
        "0-0-0-0": 10,
        "1-0-0-0": 10,
        "2-0-0-0": 10, // very high priority, but level 2
        "1-1-0-0": 10, // low priority, level 1
      },
      30,
      (key) => (keyToString(key) === "2-0-0-0" ? 1000 : key.x),
    );

    expect(result.selected).toEqual(new Set(["0-0-0-0", "1-0-0-0", "1-1-0-0"]));
    expect(result.totalPoints).toBe(30);
  });

  it("uses a larger budget only to extend a seeded selection", () => {
    const counts = {
      "0-0-0-0": 10,
      "1-0-0-0": 80,
      "1-1-0-0": 30,
    };
    const priority = (key: VoxelKey): number =>
      keyToString(key) === "1-0-0-0" ? 100 : 1;
    // This parent-closed seed can come from an earlier view in which the
    // cheaper child had higher priority.
    const seed = new Set(["0-0-0-0", "1-1-0-0"]);

    const grown = selectNodes({
      root: ROOT_KEY,
      getNode: hierarchyOf(counts),
      priority,
      pointBudget: 90,
      seed,
    });
    expect(grown.selected).toEqual(seed);
    expect(grown.totalPoints).toBe(40);

    const enoughForBoth = selectNodes({
      root: ROOT_KEY,
      getNode: hierarchyOf(counts),
      priority,
      pointBudget: 120,
      seed,
    });
    expect(enoughForBoth.selected).toEqual(new Set(Object.keys(counts)));
    expect(enoughForBoth.totalPoints).toBe(120);
  });

  it("a seeded node the view no longer offers reserves nothing", () => {
    const counts = {
      "0-0-0-0": 10,
      "1-0-0-0": 80,
    };
    // '1-1-0-0' (30 points) was seeded from an earlier view that still showed
    // it; this view culls it, so its cost must not be held back.
    const result = selectNodes({
      root: ROOT_KEY,
      getNode: hierarchyOf(counts),
      priority: flatPriority,
      pointBudget: 90,
      seed: new Set(["0-0-0-0", "1-1-0-0"]),
    });

    expect(result.selected).toEqual(new Set(["0-0-0-0", "1-0-0-0"]));
    expect(result.totalPoints).toBe(90);
  });

  it("a seeded node behind a culled ancestor reserves nothing", () => {
    const counts = {
      "0-0-0-0": 10,
      "1-0-0-0": 80,
      "1-1-0-0": 30,
      "2-2-0-0": 40,
    };
    // The whole '1-1-0-0' branch is gone from this view: the seeded grandchild
    // is unreachable even though the hierarchy still knows its point count.
    const visible = hierarchyOf(counts);
    const result = selectNodes({
      root: ROOT_KEY,
      getNode: (key) =>
        keyToString(key) === "1-1-0-0" ? undefined : visible(key),
      priority: flatPriority,
      pointBudget: 90,
      seed: new Set(["0-0-0-0", "1-1-0-0", "2-2-0-0"]),
    });

    expect(result.selected).toEqual(new Set(["0-0-0-0", "1-0-0-0"]));
    expect(result.totalPoints).toBe(90);
  });

  it("skips nodes the hierarchy does not know (unloaded pages)", () => {
    const getNode = hierarchyOf({ "0-0-0-0": 10 });
    const result = selectNodes({
      root: ROOT_KEY,
      // Root claims a child that has no hierarchy entry yet.
      getNode: (key) =>
        keyToString(key) === "0-0-0-0"
          ? {
              pointCount: 10,
              children: [{ level: 1, x: 0, y: 0, z: 0 }],
            }
          : getNode(key),
      priority: flatPriority,
      pointBudget: 1000,
    });

    expect(result.selected).toEqual(new Set(["0-0-0-0"]));
    expect(result.totalPoints).toBe(10);
    expect(result.consideredNodes).toBe(2);
    expect(result.availableNodes).toBe(1);
  });

  it("returns empty for an unknown root", () => {
    const result = selectNodes({
      root: ROOT_KEY,
      getNode: () => undefined,
      priority: flatPriority,
      pointBudget: 1000,
    });

    expect(result.selected.size).toBe(0);
    expect(result.totalPoints).toBe(0);
  });

  it("zero-point structural nodes are selectable without consuming budget", () => {
    const result = select({ "0-0-0-0": 0, "1-0-0-0": 100 }, 100);

    expect(result.selected).toEqual(new Set(["0-0-0-0", "1-0-0-0"]));
    expect(result.totalPoints).toBe(100);
  });

  /**
   * Every child of a selected node is either selected or budget-skipped —
   * there is no third way for one to be absent. The controller's terminal walk
   * reads a missing child as budget-blocked without consulting the skip set,
   * which is only sound while this holds.
   */
  it("accounts for every child of a selected node", () => {
    const counts: Record<string, number> = {
      "0-0-0-0": 100,
      "1-0-0-0": 60,
      "1-1-0-0": 60,
      "1-0-1-0": 60,
      "1-1-1-0": 60,
    };
    // A budget that takes the root and only some of its children.
    const result = select(counts, 220);

    expect(result.selected.size).toBeGreaterThan(0);
    expect(result.budgetSkipped.size).toBeGreaterThan(0);
    for (const key of Object.keys(counts)) {
      expect(
        result.selected.has(key) || result.budgetSkipped.has(key),
        `${key} is neither selected nor budget-skipped`,
      ).toBe(true);
    }
  });
});
