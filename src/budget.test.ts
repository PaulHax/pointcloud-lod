import { describe, expect, it } from 'vitest';

import { selectNodes, type HierarchyNode } from './budget';
import { keyFromString, keyToString, ROOT_KEY, type VoxelKey } from './octree';

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

describe('selectNodes', () => {
  it('selects just the root when it has no children', () => {
    const result = selectNodes({
      root: ROOT_KEY,
      getNode: hierarchyOf({ '0-0-0-0': 100 }),
      priority: flatPriority,
      pointBudget: 1000,
    });

    expect([...result.selected]).toEqual(['0-0-0-0']);
    expect(result.totalPoints).toBe(100);
  });

  it('selects nothing when the root alone exceeds the budget', () => {
    const result = selectNodes({
      root: ROOT_KEY,
      getNode: hierarchyOf({ '0-0-0-0': 100, '1-0-0-0': 10 }),
      priority: flatPriority,
      pointBudget: 50,
    });

    expect(result.selected.size).toBe(0);
    expect(result.totalPoints).toBe(0);
  });

  it('selects a whole small tree that fits the budget', () => {
    const counts = {
      '0-0-0-0': 100,
      '1-0-0-0': 50,
      '1-1-0-0': 50,
      '2-0-0-0': 25,
    };
    const result = selectNodes({
      root: ROOT_KEY,
      getNode: hierarchyOf(counts),
      priority: flatPriority,
      pointBudget: 1000,
    });

    expect(result.selected).toEqual(new Set(Object.keys(counts)));
    expect(result.totalPoints).toBe(225);
  });

  it('on budget exhaustion mid-level, keeps higher-priority siblings', () => {
    const result = selectNodes({
      root: ROOT_KEY,
      getNode: hierarchyOf({
        '0-0-0-0': 100,
        '1-0-0-0': 60,
        '1-1-0-0': 60,
        '1-0-1-0': 60,
      }),
      // Priority by x+y: '1-1-0-0' > '1-0-1-0'? both = 1; make it distinct:
      priority: (key) => key.x * 2 + key.y,
      pointBudget: 230, // root(100) + two children(120) = 220; third would hit 280
    });

    // Ranked children: '1-1-0-0' (2), '1-0-1-0' (1), '1-0-0-0' (0).
    expect(result.selected).toEqual(
      new Set(['0-0-0-0', '1-1-0-0', '1-0-1-0']),
    );
    expect(result.totalPoints).toBe(220);
  });

  it('a skipped node cannot fit later even if siblings left room', () => {
    const result = selectNodes({
      root: ROOT_KEY,
      getNode: hierarchyOf({
        '0-0-0-0': 100,
        '1-1-1-1': 500, // highest priority but too big — skipped
        '1-0-0-0': 40, // still fits afterwards
      }),
      priority: (key) => key.x + key.y + key.z,
      pointBudget: 200,
    });

    expect(result.selected).toEqual(new Set(['0-0-0-0', '1-0-0-0']));
    expect(result.totalPoints).toBe(140);
  });

  it('never selects a child whose parent was skipped (hole-free invariant)', () => {
    const result = selectNodes({
      root: ROOT_KEY,
      getNode: hierarchyOf({
        '0-0-0-0': 50,
        '1-0-0-0': 500, // exceeds remaining budget — skipped
        '2-0-0-0': 1, // tiny grandchild, but its parent is out
        '1-1-0-0': 10,
        '2-2-0-0': 5, // child of the selected '1-1-0-0'
      }),
      priority: flatPriority,
      pointBudget: 100,
    });

    expect(result.selected).toEqual(
      new Set(['0-0-0-0', '1-1-0-0', '2-2-0-0']),
    );
    expect(result.selected.has('2-0-0-0')).toBe(false);
    expect(result.totalPoints).toBe(65);
  });

  it('refines level by level (breadth-first), not depth-first', () => {
    // Deep chain under one child vs a shallow sibling: with a budget that
    // only fits three nodes, breadth-first must take both level-1 nodes
    // before any level-2 node, regardless of priority.
    const result = selectNodes({
      root: ROOT_KEY,
      getNode: hierarchyOf({
        '0-0-0-0': 10,
        '1-0-0-0': 10,
        '2-0-0-0': 10, // very high priority, but level 2
        '1-1-0-0': 10, // low priority, level 1
      }),
      priority: (key) => (keyToString(key) === '2-0-0-0' ? 1000 : key.x),
      pointBudget: 30,
    });

    expect(result.selected).toEqual(
      new Set(['0-0-0-0', '1-0-0-0', '1-1-0-0']),
    );
    expect(result.totalPoints).toBe(30);
  });

  it('skips nodes the hierarchy does not know (unloaded pages)', () => {
    const getNode = hierarchyOf({ '0-0-0-0': 10 });
    const result = selectNodes({
      root: ROOT_KEY,
      // Root claims a child that has no hierarchy entry yet.
      getNode: (key) =>
        keyToString(key) === '0-0-0-0'
          ? {
              pointCount: 10,
              children: [{ level: 1, x: 0, y: 0, z: 0 }],
            }
          : getNode(key),
      priority: flatPriority,
      pointBudget: 1000,
    });

    expect(result.selected).toEqual(new Set(['0-0-0-0']));
    expect(result.totalPoints).toBe(10);
  });

  it('returns empty for an unknown root', () => {
    const result = selectNodes({
      root: ROOT_KEY,
      getNode: () => undefined,
      priority: flatPriority,
      pointBudget: 1000,
    });

    expect(result.selected.size).toBe(0);
    expect(result.totalPoints).toBe(0);
  });

  it('zero-point structural nodes are selectable without consuming budget', () => {
    const result = selectNodes({
      root: ROOT_KEY,
      getNode: hierarchyOf({
        '0-0-0-0': 0,
        '1-0-0-0': 100,
      }),
      priority: flatPriority,
      pointBudget: 100,
    });

    expect(result.selected).toEqual(new Set(['0-0-0-0', '1-0-0-0']));
    expect(result.totalPoints).toBe(100);
  });
});
