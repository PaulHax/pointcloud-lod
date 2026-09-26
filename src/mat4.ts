/**
 * Column-major 4x4 matrix math in the OpenGL layout, translation in indices
 * 12..14. Only `validateAffineMatrix` validates: a matrix is checked once,
 * where it enters the library, and trusted after that.
 */

import { finitePositive } from "./numeric";
import type { Vec3 } from "./octree";

/** Column-major 4x4 matrix. */
export type Mat16 = ArrayLike<number>;

/** Column-major affine matrix as a tuple, the 3D Tiles and glTF convention. */
export type Mat4 = [
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
];

/** Frozen: it is handed out as the default placement, not a scratch buffer. */
export const IDENTITY: readonly number[] = Object.freeze([
  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
]);

/** Column-major product `a × b`. */
export const multiply = (a: Mat16, b: Mat16): number[] => {
  // prettier-ignore
  const out: number[] = [
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
  ];
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      out[column * 4 + row] =
        a[row]! * b[column * 4]! +
        a[4 + row]! * b[column * 4 + 1]! +
        a[8 + row]! * b[column * 4 + 2]! +
        a[12 + row]! * b[column * 4 + 3]!;
    }
  }
  return out;
};

/** Cofactor inverse, or null when the matrix is singular or not finite. */
export const invert = (m: Mat16): number[] | null => {
  const a00 = m[0]!,
    a01 = m[1]!,
    a02 = m[2]!,
    a03 = m[3]!;
  const a10 = m[4]!,
    a11 = m[5]!,
    a12 = m[6]!,
    a13 = m[7]!;
  const a20 = m[8]!,
    a21 = m[9]!,
    a22 = m[10]!,
    a23 = m[11]!;
  const a30 = m[12]!,
    a31 = m[13]!,
    a32 = m[14]!,
    a33 = m[15]!;

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  const det =
    b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!Number.isFinite(det) || det === 0) return null;
  const d = 1 / det;

  const inverse = [
    (a11 * b11 - a12 * b10 + a13 * b09) * d,
    (a02 * b10 - a01 * b11 - a03 * b09) * d,
    (a31 * b05 - a32 * b04 + a33 * b03) * d,
    (a22 * b04 - a21 * b05 - a23 * b03) * d,
    (a12 * b08 - a10 * b11 - a13 * b07) * d,
    (a00 * b11 - a02 * b08 + a03 * b07) * d,
    (a32 * b02 - a30 * b05 - a33 * b01) * d,
    (a20 * b05 - a22 * b02 + a23 * b01) * d,
    (a10 * b10 - a11 * b08 + a13 * b06) * d,
    (a01 * b08 - a00 * b10 - a03 * b06) * d,
    (a30 * b04 - a31 * b02 + a33 * b00) * d,
    (a21 * b02 - a20 * b04 - a23 * b00) * d,
    (a11 * b07 - a10 * b09 - a12 * b06) * d,
    (a00 * b09 - a01 * b07 + a02 * b06) * d,
    (a31 * b01 - a30 * b03 - a32 * b00) * d,
    (a20 * b03 - a21 * b01 + a22 * b00) * d,
  ];
  for (const value of inverse) {
    if (!Number.isFinite(value)) return null;
  }
  return inverse;
};

/** Apply an affine matrix to a point. */
export const transformPoint = (m: Mat16, point: Vec3): Vec3 => [
  m[0]! * point[0] + m[4]! * point[1] + m[8]! * point[2] + m[12]!,
  m[1]! * point[0] + m[5]! * point[1] + m[9]! * point[2] + m[13]!,
  m[2]! * point[0] + m[6]! * point[1] + m[10]! * point[2] + m[14]!,
];

/** Apply an affine matrix's linear part to a direction. */
export const transformVector = (m: Mat16, vector: Vec3): Vec3 => [
  m[0]! * vector[0] + m[4]! * vector[1] + m[8]! * vector[2],
  m[1]! * vector[0] + m[5]! * vector[1] + m[9]! * vector[2],
  m[2]! * vector[0] + m[6]! * vector[1] + m[10]! * vector[2],
];

/**
 * `base · translate(origin)`: only the last column differs from `base`. Tile
 * geometry is stored relative to a tile origin, so a tile actor's matrix is
 * its member's placement with that origin folded in.
 */
export const translatedMatrix = (base: Mat16, origin: Vec3): number[] => {
  const out = Array.from(base);
  for (let row = 0; row < 4; row += 1) {
    out[12 + row] =
      base[row]! * origin[0] +
      base[4 + row]! * origin[1] +
      base[8 + row]! * origin[2] +
      base[12 + row]!;
  }
  return out;
};

/** Element-wise equality, treating null as its own value. */
export const sameMatrix = (a: Mat16 | null, b: Mat16 | null): boolean => {
  if (a === null || b === null) return a === null && b === null;
  if (a.length !== b.length) return false;
  for (let index = 0; index < a.length; index += 1) {
    if (a[index] !== b[index]) return false;
  }
  return true;
};

/**
 * The uniform scale of a similarity transform: finite, an affine bottom row,
 * and orthogonal columns of equal length. Null when the matrix is not one.
 */
export const similarityScale = (m: Mat16): number | null => {
  if (m.length !== 16) return null;
  for (let index = 0; index < 16; index += 1) {
    if (!Number.isFinite(m[index]!)) return null;
  }
  if (
    Math.abs(m[3]!) > 1e-9 ||
    Math.abs(m[7]!) > 1e-9 ||
    Math.abs(m[11]!) > 1e-9 ||
    Math.abs(m[15]! - 1) > 1e-9
  ) {
    return null;
  }
  const columns: Vec3[] = [
    [m[0]!, m[1]!, m[2]!],
    [m[4]!, m[5]!, m[6]!],
    [m[8]!, m[9]!, m[10]!],
  ];
  const lengths = columns.map((column) => Math.hypot(...column));
  const scale = lengths[0]!;
  const tolerance = Math.max(1e-9, scale * 1e-6);
  if (
    !finitePositive(scale) ||
    lengths.some((length) => Math.abs(length - scale) > tolerance)
  ) {
    return null;
  }
  for (let left = 0; left < 3; left += 1) {
    for (let right = left + 1; right < 3; right += 1) {
      const dot = columns[left]!.reduce(
        (sum, value, axis) => sum + value * columns[right]![axis]!,
        0,
      );
      if (Math.abs(dot) > scale * tolerance) return null;
    }
  }
  return scale;
};

/** Determinant of the linear (upper-left 3x3) part. */
export const linearDeterminant = (m: Mat16): number =>
  m[0]! * (m[5]! * m[10]! - m[9]! * m[6]!) -
  m[4]! * (m[1]! * m[10]! - m[9]! * m[2]!) +
  m[8]! * (m[1]! * m[6]! - m[5]! * m[2]!);

const AFFINE_ENTRY_TOLERANCE = 1e-12;
const AFFINE_DETERMINANT_FLOOR = 1e-15;

/**
 * What keeps a matrix from being 16 finite numbers with a (0, 0, 0, 1) bottom
 * row, or null when nothing does. Invertibility is each boundary's own rule,
 * because they differ: a glTF node may scale to zero, a placement may not.
 */
export const affineProblem = (
  m: Mat16 | null | undefined,
): "finite" | "affine" | null => {
  if (!m || m.length !== 16) return "finite";
  for (let index = 0; index < 16; index += 1) {
    if (!Number.isFinite(m[index])) return "finite";
  }
  return Math.abs(m[3]!) > AFFINE_ENTRY_TOLERANCE ||
    Math.abs(m[7]!) > AFFINE_ENTRY_TOLERANCE ||
    Math.abs(m[11]!) > AFFINE_ENTRY_TOLERANCE ||
    Math.abs(m[15]! - 1) > AFFINE_ENTRY_TOLERANCE
    ? "affine"
    : null;
};

/**
 * The affine rule for a matrix a host hands in: 16 finite numbers, a
 * (0, 0, 0, 1) bottom row within 1e-12, and a linear part whose determinant
 * exceeds 1e-15 in magnitude. The trame-vtklocal host and its producer check
 * the same numbers. Throws a TypeError naming `label`; returns a copy.
 */
export const validateAffineMatrix = (
  matrix: Mat16,
  label: string,
): number[] => {
  const problem = affineProblem(matrix);
  if (problem === "finite") {
    throw new TypeError(`${label} must contain 16 finite numbers`);
  }
  if (problem === "affine") {
    throw new TypeError(`${label} must be an affine column-major matrix`);
  }
  const determinant = linearDeterminant(matrix);
  if (
    !Number.isFinite(determinant) ||
    Math.abs(determinant) <= AFFINE_DETERMINANT_FLOOR
  ) {
    throw new TypeError(`${label} must be invertible`);
  }
  return Array.from(matrix);
};
