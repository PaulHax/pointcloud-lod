export type Vec3 = [number, number, number];

/** Column-major affine matrix, matching 3D Tiles and glTF conventions. */
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

export interface RtcPrimitiveInput {
  positions: Float32Array | Float64Array;
  normals?: Float32Array | Float64Array;
  uvs?: Float32Array;
  indices?: Uint16Array | Uint32Array;
}

export interface RtcPrimitiveResult {
  positions: Float32Array;
  normals?: Float32Array;
  uvs?: Float32Array;
  indices?: Uint16Array | Uint32Array;
  origin: Vec3;
}

const AFFINE_EPSILON = 1e-12;
const WGS84_SEMIMAJOR_METERS = 6_378_137;
const WGS84_ECCENTRICITY_SQUARED = 6.69437999014e-3;

const finiteMatrix = (matrix: readonly number[], label: string): Mat4 => {
  if (matrix.length !== 16 || matrix.some((value) => !Number.isFinite(value))) {
    throw new Error(`${label} must contain 16 finite numbers`);
  }
  if (
    Math.abs(matrix[3]!) > AFFINE_EPSILON ||
    Math.abs(matrix[7]!) > AFFINE_EPSILON ||
    Math.abs(matrix[11]!) > AFFINE_EPSILON ||
    Math.abs(matrix[15]! - 1) > AFFINE_EPSILON
  ) {
    throw new Error(`${label} must be an affine column-major matrix`);
  }
  return [...matrix] as Mat4;
};

/** Column-major 4x4 product without validation, shared by the checked paths. */
export const multiplyMat4Values = (
  left: readonly number[],
  right: readonly number[],
): number[] => {
  const result = Array.from({ length: 16 }, () => 0);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      let value = 0;
      for (let inner = 0; inner < 4; inner += 1) {
        value += left[inner * 4 + row]! * right[column * 4 + inner]!;
      }
      result[column * 4 + row] = value;
    }
  }
  return result;
};

export const multiplyMat4 = (
  leftInput: readonly number[],
  rightInput: readonly number[],
): Mat4 =>
  finiteMatrix(
    multiplyMat4Values(
      finiteMatrix(leftInput, "left matrix"),
      finiteMatrix(rightInput, "right matrix"),
    ),
    "composed matrix",
  );

export const composeSceneTransform = (
  ecefToScene: readonly number[],
  accumulatedTilesTransform: readonly number[],
): Mat4 => multiplyMat4(ecefToScene, accumulatedTilesTransform);

/**
 * Scene-local ENU vertical exaggeration, in column-major convention.
 * The pivot is invariant: `z' = pivotZ + exaggeration * (z - pivotZ)`.
 */
export const createVerticalExaggerationTransform = (
  exaggeration = 1,
  pivotZ = 0,
): Mat4 => {
  if (!Number.isFinite(exaggeration) || exaggeration <= 0) {
    throw new RangeError("verticalExaggeration must be finite and > 0");
  }
  if (!Number.isFinite(pivotZ)) {
    throw new RangeError("verticalPivotZ must be finite");
  }
  return [
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    exaggeration,
    0,
    0,
    0,
    pivotZ * (1 - exaggeration),
    1,
  ];
};

/** `T(p) * S(1,1,e) * T(-p) * ECEFToScene * accumulatedTile/local`. */
export const composeVerticalExaggeratedSceneTransform = (
  exaggeration: number,
  pivotZ: number,
  ecefToScene: readonly number[],
  accumulatedTilesTransform: readonly number[],
): Mat4 =>
  multiplyMat4(
    createVerticalExaggerationTransform(exaggeration, pivotZ),
    composeSceneTransform(ecefToScene, accumulatedTilesTransform),
  );

export const transformPoint = (
  matrixInput: readonly number[],
  point: readonly [number, number, number],
): Vec3 => {
  const matrix = finiteMatrix(matrixInput, "transform");
  if (point.some((value) => !Number.isFinite(value))) {
    throw new Error("point must contain three finite numbers");
  }
  const [x, y, z] = point;
  const transformed: Vec3 = [
    matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
    matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
    matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14],
  ];
  if (transformed.some((value) => !Number.isFinite(value))) {
    throw new Error("point transform produced a non-finite coordinate");
  }
  return transformed;
};

export const wgs84ToEcef = (
  longitudeDegrees: number,
  latitudeDegrees: number,
  altitudeMeters: number,
): Vec3 => {
  if (
    !Number.isFinite(longitudeDegrees) ||
    !Number.isFinite(latitudeDegrees) ||
    !Number.isFinite(altitudeMeters) ||
    longitudeDegrees < -180 ||
    longitudeDegrees > 180 ||
    latitudeDegrees < -90 ||
    latitudeDegrees > 90
  ) {
    throw new Error("WGS84 coordinates must be finite and in range");
  }
  const longitude = (longitudeDegrees * Math.PI) / 180;
  const latitude = (latitudeDegrees * Math.PI) / 180;
  const sinLatitude = Math.sin(latitude);
  const primeVertical =
    WGS84_SEMIMAJOR_METERS /
    Math.sqrt(1 - WGS84_ECCENTRICITY_SQUARED * sinLatitude * sinLatitude);
  return [
    (primeVertical + altitudeMeters) * Math.cos(latitude) * Math.cos(longitude),
    (primeVertical + altitudeMeters) * Math.cos(latitude) * Math.sin(longitude),
    (primeVertical * (1 - WGS84_ECCENTRICITY_SQUARED) + altitudeMeters) *
      sinLatitude,
  ];
};

export const createEcefToEnuTransform = (
  longitudeDegrees: number,
  latitudeDegrees: number,
  altitudeMeters: number,
): Mat4 => {
  const origin = wgs84ToEcef(longitudeDegrees, latitudeDegrees, altitudeMeters);
  const longitude = (longitudeDegrees * Math.PI) / 180;
  const latitude = (latitudeDegrees * Math.PI) / 180;
  const sinLongitude = Math.sin(longitude);
  const cosLongitude = Math.cos(longitude);
  const sinLatitude = Math.sin(latitude);
  const cosLatitude = Math.cos(latitude);
  const east: Vec3 = [-sinLongitude, cosLongitude, 0];
  const north: Vec3 = [
    -sinLatitude * cosLongitude,
    -sinLatitude * sinLongitude,
    cosLatitude,
  ];
  const up: Vec3 = [
    cosLatitude * cosLongitude,
    cosLatitude * sinLongitude,
    sinLatitude,
  ];

  return [
    east[0],
    north[0],
    up[0],
    0,
    east[1],
    north[1],
    up[1],
    0,
    east[2],
    north[2],
    up[2],
    0,
    -(east[0] * origin[0] + east[1] * origin[1] + east[2] * origin[2]),
    -(north[0] * origin[0] + north[1] * origin[1] + north[2] * origin[2]),
    -(up[0] * origin[0] + up[1] * origin[1] + up[2] * origin[2]),
    1,
  ];
};

const inverseTransposeLinear = (matrix: Mat4): readonly number[] => {
  const a00 = matrix[0];
  const a01 = matrix[4];
  const a02 = matrix[8];
  const a10 = matrix[1];
  const a11 = matrix[5];
  const a12 = matrix[9];
  const a20 = matrix[2];
  const a21 = matrix[6];
  const a22 = matrix[10];
  const c00 = a11 * a22 - a12 * a21;
  const c01 = a12 * a20 - a10 * a22;
  const c02 = a10 * a21 - a11 * a20;
  const determinant = a00 * c00 + a01 * c01 + a02 * c02;
  if (
    !Number.isFinite(determinant) ||
    Math.abs(determinant) <= AFFINE_EPSILON
  ) {
    throw new Error("transform has a degenerate linear component");
  }
  const inverse = [
    c00 / determinant,
    (a02 * a21 - a01 * a22) / determinant,
    (a01 * a12 - a02 * a11) / determinant,
    c01 / determinant,
    (a00 * a22 - a02 * a20) / determinant,
    (a02 * a10 - a00 * a12) / determinant,
    c02 / determinant,
    (a01 * a20 - a00 * a21) / determinant,
    (a00 * a11 - a01 * a10) / determinant,
  ];
  // Multiplication below uses this row-major inverse transposed.
  return inverse;
};

const transformNormal = (
  inverse: readonly number[],
  x: number,
  y: number,
  z: number,
): Vec3 => {
  const transformed: Vec3 = [
    inverse[0]! * x + inverse[3]! * y + inverse[6]! * z,
    inverse[1]! * x + inverse[4]! * y + inverse[7]! * z,
    inverse[2]! * x + inverse[5]! * y + inverse[8]! * z,
  ];
  const length = Math.hypot(...transformed);
  if (!Number.isFinite(length) || length <= AFFINE_EPSILON) {
    throw new Error("normal transform produced a degenerate normal");
  }
  return transformed.map((value) => value / length) as Vec3;
};

export const flattenPrimitiveToRtc = (
  primitive: RtcPrimitiveInput,
  sceneTransformInput: readonly number[],
  nodeTransformInput: readonly number[],
): RtcPrimitiveResult => {
  const sceneTransform = finiteMatrix(sceneTransformInput, "scene transform");
  const nodeTransform = finiteMatrix(nodeTransformInput, "node transform");
  const combined = multiplyMat4(sceneTransform, nodeTransform);
  const normalTransform = inverseTransposeLinear(combined);
  const { positions, normals } = primitive;
  if (positions.length === 0 || positions.length % 3 !== 0) {
    throw new Error("positions must contain complete XYZ tuples");
  }
  if (
    normals &&
    (normals.length !== positions.length || normals.length % 3 !== 0)
  ) {
    throw new Error("normals must match the positions XYZ tuple count");
  }

  const transformed = new Float64Array(positions.length);
  const minimum: Vec3 = [Infinity, Infinity, Infinity];
  const maximum: Vec3 = [-Infinity, -Infinity, -Infinity];
  // `combined` is already validated, so the per-vertex path stays pure
  // arithmetic: no matrix revalidation, copy, or tuple allocation per vertex.
  const [m0, m1, m2, , m4, m5, m6, , m8, m9, m10, , m12, m13, m14] = combined;
  for (let offset = 0; offset < positions.length; offset += 3) {
    const x = positions[offset]!;
    const y = positions[offset + 1]!;
    const z = positions[offset + 2]!;
    const px = m0 * x + m4 * y + m8 * z + m12;
    const py = m1 * x + m5 * y + m9 * z + m13;
    const pz = m2 * x + m6 * y + m10 * z + m14;
    if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(pz)) {
      throw new Error("point transform produced a non-finite coordinate");
    }
    transformed[offset] = px;
    transformed[offset + 1] = py;
    transformed[offset + 2] = pz;
    minimum[0] = Math.min(minimum[0], px);
    minimum[1] = Math.min(minimum[1], py);
    minimum[2] = Math.min(minimum[2], pz);
    maximum[0] = Math.max(maximum[0], px);
    maximum[1] = Math.max(maximum[1], py);
    maximum[2] = Math.max(maximum[2], pz);
  }
  const origin: Vec3 = [
    minimum[0] + (maximum[0] - minimum[0]) / 2,
    minimum[1] + (maximum[1] - minimum[1]) / 2,
    minimum[2] + (maximum[2] - minimum[2]) / 2,
  ];
  const localPositions = new Float32Array(positions.length);
  for (let offset = 0; offset < transformed.length; offset += 3) {
    localPositions[offset] = transformed[offset]! - origin[0];
    localPositions[offset + 1] = transformed[offset + 1]! - origin[1];
    localPositions[offset + 2] = transformed[offset + 2]! - origin[2];
  }

  let localNormals: Float32Array | undefined;
  if (normals) {
    localNormals = new Float32Array(normals.length);
    for (let offset = 0; offset < normals.length; offset += 3) {
      const normal = transformNormal(
        normalTransform,
        normals[offset]!,
        normals[offset + 1]!,
        normals[offset + 2]!,
      );
      localNormals.set(normal, offset);
    }
  }

  return {
    positions: localPositions,
    ...(localNormals ? { normals: localNormals } : {}),
    ...(primitive.uvs ? { uvs: primitive.uvs.slice() } : {}),
    ...(primitive.indices ? { indices: primitive.indices.slice() } : {}),
    origin,
  };
};
