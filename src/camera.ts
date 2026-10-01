/**
 * Pure camera math for LOD selection: frustum extraction and culling from a
 * column-major view-projection matrix, the screen-space error law, cursor
 * rays, and camera comparisons. No vtk.js imports: callers hand in plain
 * arrays, so the module works in any renderer, worker, or test without a GL
 * context.
 */

import {
  invert,
  multiply,
  similarityScale,
  transformPoint,
  type Mat16,
} from "./mat4";
import { finitePositive } from "./numeric";
import type { Bounds, Vec3 } from "./octree";

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

/** Own the camera history even when a host reuses its matrix and position. */
export const copyCameraView = (view: CameraView): CameraView => ({
  ...view,
  position: [...view.position],
  viewProj: Array.from(view.viewProj) as Mat16,
});

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

/** The one screen-space error law; see `PreparedView.screenSpaceError`. */
const screenSpaceErrorLaw = (
  view: CameraView,
): ((length: number, distance: number) => number) => {
  const heightCssPx = view.viewportHeightCssPx;
  if (view.projection === "orthographic") {
    const parallelScale = Math.max(view.parallelScale, 1e-9);
    return (length) => (length * heightCssPx) / (2 * parallelScale);
  }
  const tanHalfFov = Math.tan(view.fovY / 2);
  return (length, distance) =>
    (length * heightCssPx) / (2 * Math.max(distance, 1e-9) * tanHalfFov);
};

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
  const inverse = invert(viewProj);
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
 * The ray through the viewport centre. A valid perspective eye lies on the
 * centre line recovered from the matrix, so the explicit eye is its origin
 * (the cone's true apex), unless an untyped host supplied an eye and a matrix
 * that disagree: the matrix's own point is kept then, rather than inventing a
 * skewed centre ray.
 */
const centerRay = (view: CameraView): CursorRay | null => {
  const ray = cursorRay(
    view.viewProj,
    view.viewportWidthCssPx / 2,
    view.viewportHeightCssPx / 2,
    view.viewportWidthCssPx,
    view.viewportHeightCssPx,
  );
  if (ray === null || view.projection !== "perspective") return ray;
  const eyeDx = view.position[0] - ray.origin[0];
  const eyeDy = view.position[1] - ray.origin[1];
  const eyeDz = view.position[2] - ray.origin[2];
  const eyeAlong =
    eyeDx * ray.direction[0] +
    eyeDy * ray.direction[1] +
    eyeDz * ray.direction[2];
  const eyeDistanceSquared = eyeDx * eyeDx + eyeDy * eyeDy + eyeDz * eyeDz;
  const eyeOffAxis = Math.sqrt(
    Math.max(0, eyeDistanceSquared - eyeAlong * eyeAlong),
  );
  return eyeOffAxis <= 1e-9 * Math.max(1, Math.sqrt(eyeDistanceSquared))
    ? { origin: view.position, direction: ray.direction }
    : ray;
};

const offsetFromRay = (
  ray: CursorRay,
  view: CameraView,
  point: Vec3,
): number => {
  const dx = point[0] - ray.origin[0];
  const dy = point[1] - ray.origin[1];
  const dz = point[2] - ray.origin[2];
  const along =
    dx * ray.direction[0] + dy * ray.direction[1] + dz * ray.direction[2];
  const distanceSquared = dx * dx + dy * dy + dz * dz;
  const perpendicular = Math.sqrt(Math.max(0, distanceSquared - along * along));

  if (view.projection === "orthographic") {
    return perpendicular / view.parallelScale;
  }

  const distance = Math.sqrt(distanceSquared);
  if (distance === 0) return 0;
  return Math.atan2(perpendicular, along);
};

/** A node the view measures: its bounds, and the spacing of its points. */
export type MeasuredNode = {
  readonly bounds: Bounds;
  readonly spacing: number;
};

/**
 * One camera view with everything derived from it computed once: the
 * frustum planes, the centre ray, and the screen-space error law both
 * formats measure detail with.
 */
export type PreparedView = {
  readonly view: CameraView;
  readonly planes: readonly Plane[];
  /**
   * Projected size, in css pixels, of a world length whose nearest point lies
   * `distance` from the eye: how far apart points that far apart land on
   * screen. A parallel view has no distance in its law, because every point
   * projects at the scale its viewport height spans. Both clamp their
   * denominator, so a camera inside a node or a degenerate parallel scale
   * reads as a very large, never infinite, error.
   */
  readonly screenSpaceError: (length: number, distance: number) => number;
  /** A node's point spacing projected at its distance from the eye. */
  readonly nodeScreenSpaceError: (node: MeasuredNode) => number;
  /**
   * How far a point lies off the centre ray. Zero is the innermost cone;
   * larger values form concentric cones moving away from the view centre.
   * Both formats order requests by a node's centre: point selection only
   * compares nodes at the same octree level, where their bounds have equal
   * size, so centres give the unblurred spatial order without large coarse
   * bounding spheres masking one another.
   *
   * Perspective views return radians. Parallel views have no angular spread,
   * so they return perpendicular world-space clearance normalized by the
   * parallel scale. Both are dimensionless and ordered centre-out; their
   * magnitudes are never compared across different views. Infinite when the
   * view-projection has no centre ray.
   */
  readonly centerRayOffset: (point: Vec3) => number;
};

export const prepareView = (view: CameraView): PreparedView => {
  const screenSpaceError = screenSpaceErrorLaw(view);
  const ray = centerRay(view);
  return {
    view,
    planes: frustumPlanes(view.viewProj),
    screenSpaceError,
    nodeScreenSpaceError: (node) =>
      screenSpaceError(
        node.spacing,
        distanceToBounds(view.position, node.bounds),
      ),
    centerRayOffset: (point) =>
      ray === null ? Number.POSITIVE_INFINITY : offsetFromRay(ray, view, point),
  };
};

/**
 * Whether every number a view carries is usable. One that is not would poison
 * the frustum planes, every screen-space error, and the selection comparisons
 * that read them. A field of view at or past a half-turn has no usable
 * tangent, and a non-positive parallel scale inverts the projected spacing.
 */
export const usableView = (view: CameraView): boolean => {
  if (
    !finitePositive(projectionScalar(view)) ||
    (view.projection === "perspective" && view.fovY >= Math.PI) ||
    !finitePositive(view.viewportWidthCssPx) ||
    !finitePositive(view.viewportHeightCssPx)
  ) {
    return false;
  }
  for (const coordinate of view.position) {
    if (!Number.isFinite(coordinate)) return false;
  }
  for (let index = 0; index < view.viewProj.length; index += 1) {
    if (!Number.isFinite(view.viewProj[index])) return false;
  }
  return true;
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

/** The projection's own sizing scalar, whichever kind of projection it is. */
const projectionSizingScalar = (view: CameraView): number =>
  view.projection === "orthographic" ? view.parallelScale : view.fovY;

/**
 * Whether `next` renders a genuinely different camera than `previous`.
 * A missing `previous` is a baseline being established, not a movement.
 *
 * Compared entry by entry and answered at the first difference: this runs on
 * every rendered frame of every view, and the overwhelmingly common answer —
 * a camera that has not moved — is the one case that must read all of them.
 * What is compared is everything about the camera that changes what LOD
 * selects: where it looks from and at, the eye point, the viewport height
 * screen-space error is measured in, and the projection's sizing scalar.
 */
export const cameraMoved = (
  previous: CameraView | null | undefined,
  next: CameraView,
): boolean => {
  if (!previous) return false;
  if (previous.projection !== next.projection) return true;
  for (const index of MOTION_MATRIX_INDICES) {
    if (
      movedBeyondJitter(
        Number(previous.viewProj[index]),
        Number(next.viewProj[index]),
      )
    ) {
      return true;
    }
  }
  for (let axis = 0; axis < 3; axis += 1) {
    if (movedBeyondJitter(previous.position[axis]!, next.position[axis]!)) {
      return true;
    }
  }
  return (
    movedBeyondJitter(previous.viewportHeightCssPx, next.viewportHeightCssPx) ||
    movedBeyondJitter(
      projectionSizingScalar(previous),
      projectionSizingScalar(next),
    )
  );
};

/**
 * A model transform tiles draw under, resolved once: the matrix, its inverse,
 * and the uniform scale of the similarity. The frustum and SSE math in this
 * module assumes a uniform-scale transform, so a matrix that is not a
 * similarity has no usable frame and resolves to null.
 */
export type ModelFrame = {
  readonly matrix: readonly number[];
  readonly inverse: readonly number[];
  readonly scale: number;
};

/** Resolve a model matrix into a frame, or null when it is not a similarity. */
export const modelFrameOf = (m: Mat16): ModelFrame | null => {
  const scale = similarityScale(m);
  if (scale === null) return null;
  const inverse = invert(m);
  if (inverse === null) return null;
  return { matrix: Array.from(m), inverse, scale };
};

/**
 * Restate a world camera in the model's local frame, where the octree's
 * bounds and spacings live. Perspective screen-space error is a ratio of two
 * lengths, so the uniform model scale cancels out of it. Parallel projection
 * has no such ratio: parallelScale is an absolute world height and must be
 * restated in model units alongside the spacings it is compared against.
 */
export const viewInModelFrame = (
  view: CameraView,
  frame: ModelFrame,
): CameraView => {
  const local = {
    ...view,
    viewProj: multiply(view.viewProj, frame.matrix),
    position: transformPoint(frame.inverse, view.position),
  };
  return local.projection === "orthographic"
    ? { ...local, parallelScale: local.parallelScale / frame.scale }
    : local;
};

/**
 * The projection's own scalar: field of view, or parallel scale.
 *
 * Returned as `undefined` for an unrecognized discriminant so a bad value
 * reads as absent rather than being silently treated as perspective.
 */
export const projectionScalar = (view: CameraView): number | undefined =>
  view.projection === "perspective"
    ? view.fovY
    : view.projection === "orthographic"
      ? view.parallelScale
      : undefined;

/**
 * Whether two camera views would produce identical selection work.
 *
 * Hosts restate the camera every frame whether or not it moved, so a member
 * that acts on every restatement re-traverses its whole hierarchy while the
 * user is doing nothing — and, because that happens inside the span the host
 * times, inflates the governor's own frame sample and drives quality down.
 */
export const sameCameraView = (a: CameraView, b: CameraView): boolean => {
  if (
    a.projection !== b.projection ||
    projectionScalar(a) !== projectionScalar(b) ||
    a.viewportWidthCssPx !== b.viewportWidthCssPx ||
    a.viewportHeightCssPx !== b.viewportHeightCssPx ||
    a.position[0] !== b.position[0] ||
    a.position[1] !== b.position[1] ||
    a.position[2] !== b.position[2] ||
    a.viewProj.length !== b.viewProj.length
  ) {
    return false;
  }
  for (let index = 0; index < a.viewProj.length; index += 1) {
    if (a.viewProj[index] !== b.viewProj[index]) return false;
  }
  return true;
};

/** Screen-space extent of a projected box, css pixels, y down. */
export type ScreenAabbCssPx = {
  readonly minXCssPx: number;
  readonly minYCssPx: number;
  readonly maxXCssPx: number;
  readonly maxYCssPx: number;
};

/**
 * Screen AABB of the 8 corners of `bounds`, placed by `matrix` first when one
 * is given. Null when any corner is unprojectable (behind the camera plane or
 * non-finite): a pick prefilter must read that as "may be under the cursor",
 * because a wrong skip is an unpickable visible tile.
 */
export const projectedBoundsAabbCssPx = (
  viewProj: Mat16,
  bounds: Bounds,
  viewportWidthCssPx: number,
  viewportHeightCssPx: number,
  matrix: Mat16 | null = null,
): ScreenAabbCssPx | null => {
  let minXCssPx = Number.POSITIVE_INFINITY;
  let minYCssPx = Number.POSITIVE_INFINITY;
  let maxXCssPx = Number.NEGATIVE_INFINITY;
  let maxYCssPx = Number.NEGATIVE_INFINITY;
  for (const x of [bounds.min[0], bounds.max[0]]) {
    for (const y of [bounds.min[1], bounds.max[1]]) {
      for (const z of [bounds.min[2], bounds.max[2]]) {
        const corner: Vec3 = [x, y, z];
        const projected = projectPointToCssPx(
          viewProj,
          matrix === null ? corner : transformPoint(matrix, corner),
          viewportWidthCssPx,
          viewportHeightCssPx,
        );
        if (projected === null) return null;
        minXCssPx = Math.min(minXCssPx, projected.xCssPx);
        minYCssPx = Math.min(minYCssPx, projected.yCssPx);
        maxXCssPx = Math.max(maxXCssPx, projected.xCssPx);
        maxYCssPx = Math.max(maxYCssPx, projected.yCssPx);
      }
    }
  }
  return { minXCssPx, minYCssPx, maxXCssPx, maxYCssPx };
};
