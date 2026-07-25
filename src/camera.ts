/**
 * Pure camera math for LOD selection: frustum extraction/culling from a
 * column-major view-projection matrix and screen-space error estimation.
 * No vtk.js imports — callers hand in plain arrays, so the module works in
 * any renderer, worker, or test without a GL context.
 */

import type { Bounds, Vec3 } from "./octree";

/** Column-major 4x4 matrix, OpenGL layout (translation in indices 12..14). */
export type Mat16 = ArrayLike<number>;

export interface CameraView {
  /** Column-major view-projection matrix. */
  readonly viewProj: Mat16;
  /** Camera position in world coordinates. */
  readonly position: Vec3;
  /** Vertical field of view, radians. */
  readonly fovY: number;
  /** Viewport height in CSS pixels. */
  readonly viewportHeightCssPx: number;
}

/** Half-space `dot(normal, p) + d >= 0` containing the frustum interior. */
export interface Plane {
  readonly normal: Vec3;
  readonly d: number;
}

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
 * Projected size of a world-space spacing, in pixels, for a perspective
 * camera: how far apart this node's points land on screen. Distance is
 * clamped so a camera inside the node reports a very large (never infinite)
 * error.
 */
export const screenSpaceError = (
  spacing: number,
  distance: number,
  viewportHeightCssPx: number,
  fovY: number,
): number =>
  (spacing * viewportHeightCssPx) /
  (2 * Math.max(distance, 1e-9) * Math.tan(fovY / 2));

/**
 * Screen-space error of one octree node: its level's point spacing projected
 * at the node's distance from the camera.
 */
export const nodeScreenSpaceError = (
  bounds: Bounds,
  spacing: number,
  view: CameraView,
): number =>
  screenSpaceError(
    spacing,
    distanceToBounds(view.position, bounds),
    view.viewportHeightCssPx,
    view.fovY,
  );
