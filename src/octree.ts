/**
 * Octree voxel addressing following the COPC convention: a cubic root node
 * enclosing the whole dataset, split into 8 equal children per level. A node
 * is addressed by `{level, x, y, z}` where `x`, `y`, `z` index the
 * `2^level` grid cells along each axis at that level.
 */

export type VoxelKey = {
  readonly level: number;
  readonly x: number;
  readonly y: number;
  readonly z: number;
};

export type Vec3 = readonly [number, number, number];

/** Axis-aligned cube, the COPC root-node shape. */
export type Cube = {
  readonly center: Vec3;
  readonly halfSize: number;
};

export type Bounds = {
  readonly min: Vec3;
  readonly max: Vec3;
};

export const ROOT_KEY: VoxelKey = { level: 0, x: 0, y: 0, z: 0 };

/** Encode a key as `'l-x-y-z'` (e.g. `'2-1-0-3'`). */
export const keyToString = (key: VoxelKey): string =>
  `${key.level}-${key.x}-${key.y}-${key.z}`;

const KEY_RE = /^(\d+)-(\d+)-(\d+)-(\d+)$/;

/** Decode `'l-x-y-z'` back into a key. Throws on malformed input. */
export const keyFromString = (s: string): VoxelKey => {
  const m = KEY_RE.exec(s);
  if (m === null) {
    throw new Error(`Invalid voxel key string: '${s}'`);
  }
  const [, level, x, y, z] = m;
  const key: VoxelKey = {
    level: Number(level),
    x: Number(x),
    y: Number(y),
    z: Number(z),
  };
  const extent = 1 << key.level;
  if (key.x >= extent || key.y >= extent || key.z >= extent) {
    throw new Error(`Voxel key out of range for its level: '${s}'`);
  }
  return key;
};

/**
 * The level of an encoded key, without decoding the rest of it.
 *
 * Request order is by level first, and both queues sort on every selection
 * pass, so this runs a few times per key per pass — enough that building a
 * whole key object and running a regex to read one number off it is worth
 * avoiding. Malformed input reads as `NaN`, which sorts as equal rather than
 * throwing: an ordering is not the place to reject a key `keyFromString`
 * refuses anyway.
 */
export const levelFromString = (s: string): number => {
  const end = s.indexOf("-");
  return end === -1 ? Number.NaN : Number(s.slice(0, end));
};

/** The 8 children of a node, in z-major bit order (dx fastest). */
export const childKeys = (key: VoxelKey): VoxelKey[] => {
  const children: VoxelKey[] = [];
  for (let dz = 0; dz <= 1; dz += 1) {
    for (let dy = 0; dy <= 1; dy += 1) {
      for (let dx = 0; dx <= 1; dx += 1) {
        children.push({
          level: key.level + 1,
          x: key.x * 2 + dx,
          y: key.y * 2 + dy,
          z: key.z * 2 + dz,
        });
      }
    }
  }
  return children;
};

/**
 * World-space bounds of a node: the root cube halved per level
 * (COPC convention — every node is itself a cube).
 */
export const nodeBounds = (root: Cube, key: VoxelKey): Bounds => {
  const size = (root.halfSize * 2) / (1 << key.level);
  const rootMin: Vec3 = [
    root.center[0] - root.halfSize,
    root.center[1] - root.halfSize,
    root.center[2] - root.halfSize,
  ];
  const min: Vec3 = [
    rootMin[0] + key.x * size,
    rootMin[1] + key.y * size,
    rootMin[2] + key.z * size,
  ];
  return {
    min,
    max: [min[0] + size, min[1] + size, min[2] + size],
  };
};

/** Node bounds expressed as a cube (center + half size). */
export const nodeCube = (root: Cube, key: VoxelKey): Cube => {
  const halfSize = root.halfSize / (1 << key.level);
  const { min } = nodeBounds(root, key);
  return {
    center: [min[0] + halfSize, min[1] + halfSize, min[2] + halfSize],
    halfSize,
  };
};

/**
 * World point spacing at a level: `rootSpacing / 2^level`
 * (each level doubles the sampling density).
 */
export const pointSpacing = (rootSpacing: number, level: number): number =>
  rootSpacing / 2 ** level;
