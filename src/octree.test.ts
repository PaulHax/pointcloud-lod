import { describe, expect, it } from 'vitest';

import {
  ROOT_KEY,
  childKeys,
  keyFromString,
  keyToString,
  nodeBounds,
  nodeCube,
  parentKey,
  pointSpacing,
  type Cube,
  type VoxelKey,
} from './octree';

describe('key string codec', () => {
  it('encodes as l-x-y-z', () => {
    expect(keyToString({ level: 3, x: 5, y: 0, z: 7 })).toBe('3-5-0-7');
    expect(keyToString(ROOT_KEY)).toBe('0-0-0-0');
  });

  it('round-trips', () => {
    const keys: VoxelKey[] = [
      ROOT_KEY,
      { level: 1, x: 1, y: 0, z: 1 },
      { level: 10, x: 1023, y: 512, z: 0 },
    ];
    for (const key of keys) {
      expect(keyFromString(keyToString(key))).toEqual(key);
    }
  });

  it('rejects malformed strings', () => {
    for (const bad of ['', '1-2-3', '1-2-3-4-5', 'a-0-0-0', '1--0-0', '-1-0-0-0']) {
      expect(() => keyFromString(bad)).toThrow();
    }
  });

  it('rejects coordinates out of range for the level', () => {
    expect(() => keyFromString('0-1-0-0')).toThrow();
    expect(() => keyFromString('2-4-0-0')).toThrow();
    expect(keyFromString('2-3-3-3')).toEqual({ level: 2, x: 3, y: 3, z: 3 });
  });
});

describe('parent / children derivation', () => {
  it('root has no parent', () => {
    expect(parentKey(ROOT_KEY)).toBeNull();
  });

  it('derives the parent by halving coordinates', () => {
    expect(parentKey({ level: 2, x: 3, y: 2, z: 1 })).toEqual({
      level: 1,
      x: 1,
      y: 1,
      z: 0,
    });
  });

  it('produces 8 distinct children, all mapping back to the parent', () => {
    const key: VoxelKey = { level: 2, x: 1, y: 3, z: 0 };
    const children = childKeys(key);
    expect(children).toHaveLength(8);
    expect(new Set(children.map(keyToString)).size).toBe(8);
    for (const child of children) {
      expect(child.level).toBe(3);
      expect(parentKey(child)).toEqual(key);
    }
  });
});

describe('node bounds (COPC root cube halved per level)', () => {
  const root: Cube = { center: [10, 20, 30], halfSize: 8 };

  it('root node bounds equal the root cube', () => {
    expect(nodeBounds(root, ROOT_KEY)).toEqual({
      min: [2, 12, 22],
      max: [18, 28, 38],
    });
  });

  it('level-1 node is one octant of the root', () => {
    expect(nodeBounds(root, { level: 1, x: 1, y: 0, z: 1 })).toEqual({
      min: [10, 12, 30],
      max: [18, 20, 38],
    });
  });

  it('children exactly tile their parent', () => {
    const parent: VoxelKey = { level: 2, x: 1, y: 2, z: 3 };
    const pb = nodeBounds(root, parent);
    const children = childKeys(parent).map((c) => nodeBounds(root, c));

    for (const cb of children) {
      for (let axis = 0; axis < 3; axis += 1) {
        expect(cb.min[axis]).toBeGreaterThanOrEqual(pb.min[axis]!);
        expect(cb.max[axis]).toBeLessThanOrEqual(pb.max[axis]!);
      }
    }
    const parentVolume = (pb.max[0] - pb.min[0]) ** 3;
    const childVolume = children.reduce(
      (sum, cb) => sum + (cb.max[0] - cb.min[0]) ** 3,
      0,
    );
    expect(childVolume).toBeCloseTo(parentVolume, 10);
  });

  it('nodeCube matches nodeBounds', () => {
    const key: VoxelKey = { level: 3, x: 4, y: 0, z: 7 };
    const cube = nodeCube(root, key);
    const bounds = nodeBounds(root, key);
    expect(cube.halfSize).toBeCloseTo(root.halfSize / 2 ** 3, 12);
    for (let axis = 0; axis < 3; axis += 1) {
      expect(cube.center[axis]! - cube.halfSize).toBeCloseTo(
        bounds.min[axis]!,
        12,
      );
      expect(cube.center[axis]! + cube.halfSize).toBeCloseTo(
        bounds.max[axis]!,
        12,
      );
    }
  });
});

describe('point spacing', () => {
  it('halves per level from the root spacing', () => {
    expect(pointSpacing(1.28, 0)).toBe(1.28);
    expect(pointSpacing(1.28, 1)).toBe(0.64);
    expect(pointSpacing(1.28, 5)).toBeCloseTo(0.04, 12);
  });
});
