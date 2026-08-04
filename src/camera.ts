/**
 * Pure camera math for LOD selection: frustum extraction/culling from a
 * column-major view-projection matrix and screen-space error estimation.
 * No vtk.js imports — callers hand in plain arrays, so the module works in
 * any renderer, worker, or test without a GL context.
 */

import { finitePositive } from "./numeric";
import type { Bounds, Vec3 } from "./octree";

/** Column-major 4x4 matrix, OpenGL layout (translation in indices 12..14). */
export type Mat16 = ArrayLike<number>;

type CameraViewCommon = {
  /** Column-major view-projection matrix. */
  readonly viewProj: Mat16;
  /** Camera position in world coordinates. */
  readonly position: Vec3;
  /** Viewport width in CSS pixels. */
  readonly viewportWidthCssPx: number;
  /** Viewport height in CSS pixels. */
  readonly viewportHeightCssPx: number;
};

export type PerspectiveCameraView = CameraViewCommon & {
  readonly projection: "perspective";
  /** Vertical field of view, radians. */
  readonly fovY: number;
};

export type OrthographicCameraView = CameraViewCommon & {
  readonly projection: "orthographic";
  /** World-space half-height of the viewport (vtk.js `parallelScale`). */
  readonly parallelScale: number;
};

/**
 * The two projections turn a world spacing into pixels by different laws, and
 * no numeric value in the view distinguishes them, so the mode is carried
 * explicitly rather than inferred.
 */
export type CameraView = PerspectiveCameraView | OrthographicCameraView;

/** Half-space `dot(normal, p) + d >= 0` containing the frustum interior. */
export type Plane = {
  readonly normal: Vec3;
  readonly d: number;
};

const plane = (a: number, b: number, c: number, d: number): Plane | null => {
  const length = Math.hypot(a, b, c);
  if (length === 0) return null;
  return { normal: [a / length, b / length, c / length], d: d / length };
};

/**
 * Gribb–Hartmann frustum extraction. Degenerate planes (zero normal) are
 * dropped, so an identity matrix yields the NDC cube's six half-spaces.
 */
export const frustumPlanes = (m: Mat16): Plane[] => {
  const row = (i: number): [number, number, number, number] => [
    m[i]!,
    m[i + 4]!,
    m[i + 8]!,
    m[i + 12]!,
  ];
  const [r0, r1, r2, r3] = [row(0), row(1), row(2), row(3)];
  const candidates = [
    plane(r3[0] + r0[0], r3[1] + r0[1], r3[2] + r0[2], r3[3] + r0[3]), // left
    plane(r3[0] - r0[0], r3[1] - r0[1], r3[2] - r0[2], r3[3] - r0[3]), // right
    plane(r3[0] + r1[0], r3[1] + r1[1], r3[2] + r1[2], r3[3] + r1[3]), // bottom
    plane(r3[0] - r1[0], r3[1] - r1[1], r3[2] - r1[2], r3[3] - r1[3]), // top
    plane(r3[0] + r2[0], r3[1] + r2[1], r3[2] + r2[2], r3[3] + r2[3]), // near
    plane(r3[0] - r2[0], r3[1] - r2[1], r3[2] - r2[2], r3[3] - r2[3]), // far
  ];
  return candidates.filter((p): p is Plane => p !== null);
};

/**
 * Conservative AABB-vs-frustum test using the positive-vertex distance: true
 * when the bounds may intersect the frustum, false only when they are fully
 * outside at least one plane.
 */
export const boundsIntersectsFrustum = (
  planes: readonly Plane[],
  bounds: Bounds,
): boolean => {
  for (const { normal, d } of planes) {
    const x = normal[0] >= 0 ? bounds.max[0] : bounds.min[0];
    const y = normal[1] >= 0 ? bounds.max[1] : bounds.min[1];
    const z = normal[2] >= 0 ? bounds.max[2] : bounds.min[2];
    const distance = normal[0] * x + normal[1] * y + normal[2] * z + d;
    if (distance < 0) return false;
  }
  return true;
};

/** Distance from a point to the surface of an AABB; 0 inside. */
export const distanceToBounds = (point: Vec3, bounds: Bounds): number => {
  const dx = Math.max(bounds.min[0] - point[0], 0, point[0] - bounds.max[0]);
  const dy = Math.max(bounds.min[1] - point[1], 0, point[1] - bounds.max[1]);
  const dz = Math.max(bounds.min[2] - point[2], 0, point[2] - bounds.max[2]);
  return Math.hypot(dx, dy, dz);
};

/**
 * Projected size of a world-space spacing, in pixels, under perspective
 * projection: how far apart this node's points land on screen. Distance is
 * clamped so a camera inside the node reports a very large (never infinite)
 * error.
 */
export const perspectiveScreenSpaceError = (
  spacing: number,
  distance: number,
  viewportHeightCssPx: number,
  fovY: number,
): number =>
  (spacing * viewportHeightCssPx) /
  (2 * Math.max(distance, 1e-9) * Math.tan(fovY / 2));

/**
 * Projected size of a world-space spacing, in pixels, under parallel
 * projection. Distance does not appear: every point projects at the same
 * scale, fixed only by how much world height the viewport spans. Scale is
 * clamped for the same reason distance is above.
 */
export const orthographicScreenSpaceError = (
  spacing: number,
  viewportHeightCssPx: number,
  parallelScale: number,
): number =>
  (spacing * viewportHeightCssPx) / (2 * Math.max(parallelScale, 1e-9));

/**
 * Screen-space error of one octree node, under whichever projection the view
 * declares: its level's point spacing projected at the node's distance from
 * the camera, or — under a parallel camera, where distance does not enter the
 * law at all — at the world height the viewport spans.
 */
export const nodeScreenSpaceError = (
  bounds: Bounds,
  spacing: number,
  view: CameraView,
): number =>
  view.projection === "orthographic"
    ? orthographicScreenSpaceError(
        spacing,
        view.viewportHeightCssPx,
        view.parallelScale,
      )
    : perspectiveScreenSpaceError(
        spacing,
        distanceToBounds(view.position, bounds),
        view.viewportHeightCssPx,
        view.fovY,
      );

/**
 * A homogeneous w at or below this is unusable: the point sits at or behind
 * the projective horizon and dividing by w yields nothing meaningful.
 */
export const CLIP_W_EPSILON = 1e-9;

export type ProjectedPointCssPx = {
  /** Horizontal css-pixel position, origin at the viewport's left edge. */
  readonly xCssPx: number;
  /** Vertical css-pixel position, origin at the viewport's top edge. */
  readonly yCssPx: number;
  /** Normalized device z (`clip.z / clip.w`), for near/far interval tests. */
  readonly ndcZ: number;
};

/**
 * Project a world point through a column-major view-projection matrix into
 * css-pixel viewport coordinates (y down). Returns null when the viewport
 * dimensions are unusable, any clip coordinate is non-finite, or the
 * homogeneous w is at or below `CLIP_W_EPSILON` (at/behind the camera plane).
 */
export const projectPointToCssPx = (
  viewProj: Mat16,
  point: Vec3,
  viewportWidthCssPx: number,
  viewportHeightCssPx: number,
): ProjectedPointCssPx | null => {
  if (
    !finitePositive(viewportWidthCssPx) ||
    !finitePositive(viewportHeightCssPx)
  ) {
    return null;
  }
  const m = viewProj;
  const [x, y, z] = point;
  const clipX = m[0]! * x + m[4]! * y + m[8]! * z + m[12]!;
  const clipY = m[1]! * x + m[5]! * y + m[9]! * z + m[13]!;
  const clipZ = m[2]! * x + m[6]! * y + m[10]! * z + m[14]!;
  const clipW = m[3]! * x + m[7]! * y + m[11]! * z + m[15]!;
  if (
    !Number.isFinite(clipX) ||
    !Number.isFinite(clipY) ||
    !Number.isFinite(clipZ) ||
    !Number.isFinite(clipW) ||
    clipW <= CLIP_W_EPSILON
  ) {
    return null;
  }
  return {
    xCssPx: ((clipX / clipW + 1) / 2) * viewportWidthCssPx,
    yCssPx: ((1 - clipY / clipW) / 2) * viewportHeightCssPx,
    ndcZ: clipZ / clipW,
  };
};

export type CursorRay = {
  /** The cursor unprojected onto the near plane. */
  readonly origin: Vec3;
  /** Unit direction from the near-plane point towards the far plane. */
  readonly direction: Vec3;
};

/** Cofactor inverse of a column-major 4x4; null when singular or non-finite. */
const invert4 = (m: Mat16): number[] | null => {
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

/** `inverse * [ndc, 1]`, homogenized; null when w collapses or goes wild. */
const unprojectNdc = (
  inverse: readonly number[],
  ndcX: number,
  ndcY: number,
  ndcZ: number,
): Vec3 | null => {
  const x =
    inverse[0]! * ndcX + inverse[4]! * ndcY + inverse[8]! * ndcZ + inverse[12]!;
  const y =
    inverse[1]! * ndcX + inverse[5]! * ndcY + inverse[9]! * ndcZ + inverse[13]!;
  const z =
    inverse[2]! * ndcX +
    inverse[6]! * ndcY +
    inverse[10]! * ndcZ +
    inverse[14]!;
  const w =
    inverse[3]! * ndcX +
    inverse[7]! * ndcY +
    inverse[11]! * ndcZ +
    inverse[15]!;
  if (!Number.isFinite(w) || Math.abs(w) <= CLIP_W_EPSILON) return null;
  const point: Vec3 = [x / w, y / w, z / w];
  for (const coordinate of point) {
    if (!Number.isFinite(coordinate)) return null;
  }
  return point;
};

/**
 * The world-space ray under a css-pixel cursor: invert the view-projection,
 * unproject the cursor at NDC z = -1 and z = +1, and normalize `far - near`.
 * One definition serves perspective and orthographic views alike. Returns
 * null when the matrix is singular or non-finite, the viewport dimensions are
 * unusable, or the cursor is non-finite.
 */
export const cursorRay = (
  viewProj: Mat16,
  cursorXCssPx: number,
  cursorYCssPx: number,
  viewportWidthCssPx: number,
  viewportHeightCssPx: number,
): CursorRay | null => {
  if (
    !Number.isFinite(cursorXCssPx) ||
    !Number.isFinite(cursorYCssPx) ||
    !finitePositive(viewportWidthCssPx) ||
    !finitePositive(viewportHeightCssPx)
  ) {
    return null;
  }
  const inverse = invert4(viewProj);
  if (inverse === null) return null;
  const ndcX = (2 * cursorXCssPx) / viewportWidthCssPx - 1;
  const ndcY = 1 - (2 * cursorYCssPx) / viewportHeightCssPx;
  const near = unprojectNdc(inverse, ndcX, ndcY, -1);
  const far = unprojectNdc(inverse, ndcX, ndcY, 1);
  if (near === null || far === null) return null;
  const dx = far[0] - near[0];
  const dy = far[1] - near[1];
  const dz = far[2] - near[2];
  const length = Math.hypot(dx, dy, dz);
  if (!Number.isFinite(length) || length <= 0) return null;
  return {
    origin: near,
    direction: [dx / length, dy / length, dz / length],
  };
};

/**
 * Camera-motion comparison for the inferred-motion classifier: did this view
 * move, beyond recomputation jitter, relative to a previously rendered one?
 *
 * The epsilon is relative with an absolute floor of the same size, because
 * the compared numbers span map-projection units (~1) and metric frames
 * (~1e6) in the same matrix. Recomputing a double-precision camera product
 * every frame moves the last bits — a few 1e-16 of the terms behind each
 * entry — while the smallest camera change that moves a pixel sits orders
 * above 1e-9, so recomputation jitter never enters the moving regime and no
 * real motion is missed. Entries that are small differences of huge terms
 * would eat that margin; the one place a rendered matrix does that, a
 * scene-derived clip depth range, is excluded from the comparison entirely
 * (see MOTION_MATRIX_INDICES).
 */
const MOTION_RELATIVE_EPSILON = 1e-9;

const movedBeyondJitter = (previous: number, next: number): boolean =>
  Math.abs(previous - next) >
  MOTION_RELATIVE_EPSILON * Math.max(1, Math.abs(previous), Math.abs(next));

/**
 * The `viewProj` entries that describe where the camera is looking, in the
 * column-major layout (`index = column * 4 + row`). The clip-z row
 * (2, 6, 10, 14) is deliberately left out: hosts fold a depth remap derived
 * from the scene's visible bounds into it, so a tile arriving or being
 * evicted rewrites those four numbers while the camera stands perfectly
 * still. Everything a camera move does to the rendered image shows up in the
 * x, y and w rows; the one motion that lives only in clip z — dollying an
 * orthographic camera along its view axis — shows up in the eye point, which
 * is compared alongside the matrix.
 */
const MOTION_MATRIX_INDICES = [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15];

/**
 * Everything about the camera that changes what LOD selects: where it looks
 * from and at, the eye point, the viewport height screen-space error is
 * measured in, and the projection's sizing scalar.
 */
const cameraMotionScalars = (view: CameraView): number[] => [
  ...MOTION_MATRIX_INDICES.map((index) => Number(view.viewProj[index])),
  ...view.position,
  view.viewportHeightCssPx,
  view.projection === "orthographic" ? view.parallelScale : view.fovY,
];

/**
 * Whether `next` renders a genuinely different camera than `previous`.
 * A missing `previous` is a baseline being established, not a movement.
 */
export const cameraMoved = (
  previous: CameraView | null | undefined,
  next: CameraView,
): boolean => {
  if (!previous) return false;
  if (previous.projection !== next.projection) return true;
  const before = cameraMotionScalars(previous);
  const after = cameraMotionScalars(next);
  return before.some((value, index) => movedBeyondJitter(value, after[index]!));
};
