/**
 * Pure, renderer-independent point picking over a set of submitted tiles.
 *
 * Semantics mirror `scene/ray_depth.py` in telesculptor-web: a cloud vertex
 * only chooses a *support depth* — the returned point lies on the cursor ray
 * at that depth, never snapped to the vertex itself. Candidate points are
 * gathered into escalating css-pixel buckets around the cursor; the smallest
 * non-empty bucket wins, and within it the minimum-depth (frontmost) point.
 *
 * No vtk.js imports — callers hand in plain camera views and tile payloads,
 * so the module works in any renderer, worker, or test without a GL context.
 */

import {
  CLIP_W_EPSILON,
  cursorRay,
  projectPointToCssPx,
  type CameraView,
  type CursorRay,
  type Mat16,
} from "./camera";
import { finitePositive } from "./numeric";
import type { Bounds, Vec3 } from "./octree";

/** Mirrors `DEFAULT_PICK_PIXEL_RADIUS` in telesculptor-web `ray_depth.py`. */
export const DEFAULT_PICK_PIXEL_RADIUS = 10;

/**
 * Mirrors `DEFAULT_PICK_PIXEL_RADIUS_MULTIPLIERS` in telesculptor-web
 * `ray_depth.py`. Keep the two in lockstep: a pick the server-side provider
 * would accept must not read as a client-side miss, or vice versa.
 */
export const DEFAULT_PICK_PIXEL_RADIUS_MULTIPLIERS: readonly number[] = [
  1, 2, 10,
];

/** Escalating pick-bucket radii in css pixels, smallest first. */
export const PICK_RADII_CSS_PX: readonly number[] =
  DEFAULT_PICK_PIXEL_RADIUS_MULTIPLIERS.map(
    (multiplier) => DEFAULT_PICK_PIXEL_RADIUS * multiplier,
  );

/** The same radii squared, as `_PICK_RADII_SQ` in `ray_depth.py`. */
const PICK_RADII_SQ: readonly number[] = PICK_RADII_CSS_PX.map(
  (radius) => radius * radius,
);

/** The outermost bucket: nothing further out is a candidate or a hit. */
const WIDEST_PICK_RADIUS_CSS_PX =
  PICK_RADII_CSS_PX[PICK_RADII_CSS_PX.length - 1]!;
const WIDEST_PICK_RADIUS_SQ = PICK_RADII_SQ[PICK_RADII_SQ.length - 1]!;

/** One pickable tile: a submitted payload plus its hierarchy bounds. */
export type PickTile = {
  /** World-space origin the tile-local positions are relative to. */
  readonly origin: Vec3;
  /** Tile-local xyz triplets, `3 * pointCount` floats. */
  readonly positions: Float32Array;
  readonly pointCount: number;
  /**
   * Conservative world-space bounds for the prefilter. Absent bounds keep the
   * tile unconditionally pickable — "candidate", never "skip".
   */
  readonly bounds?: Bounds;
};

export type PointPickResult =
  | {
      readonly status: "hit";
      /** Positive distance from the ray origin along its normalized direction. */
      readonly rayDepth: number;
      /** The cursor ray evaluated at the support depth — not the vertex. */
      readonly pointOnRay: Vec3;
      /** Css-pixel distance from the cursor to the supporting vertex. */
      readonly distancePx: number;
    }
  | { readonly status: "miss" };

const MISS: PointPickResult = { status: "miss" };

/** Everything one pick evaluates points against, derived once per query. */
export type PickQuery = {
  readonly viewProj: Mat16;
  readonly ray: CursorRay;
  readonly cursorXCssPx: number;
  readonly cursorYCssPx: number;
  readonly viewportWidthCssPx: number;
  readonly viewportHeightCssPx: number;
};

/**
 * Bounds prefilter: project the 8 corners, form a 2D screen AABB, expand it
 * by the largest pick radius, and test the cursor against it. Any corner with
 * unusable/behind-camera homogeneous coordinates makes the tile a candidate
 * unconditionally, as do missing bounds — conservative bounds only cost extra
 * sweep work, while a wrong skip is an unpickable visible tile.
 */
const tileIsPickCandidate = (
  query: PickQuery,
  bounds: Bounds | undefined,
): boolean => {
  if (bounds === undefined) return true;
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (const x of [bounds.min[0], bounds.max[0]]) {
    for (const y of [bounds.min[1], bounds.max[1]]) {
      for (const z of [bounds.min[2], bounds.max[2]]) {
        const corner = projectPointToCssPx(
          query.viewProj,
          [x, y, z],
          query.viewportWidthCssPx,
          query.viewportHeightCssPx,
        );
        if (corner === null) return true;
        minX = Math.min(minX, corner.xCssPx);
        minY = Math.min(minY, corner.yCssPx);
        maxX = Math.max(maxX, corner.xCssPx);
        maxY = Math.max(maxY, corner.yCssPx);
      }
    }
  }
  return (
    query.cursorXCssPx >= minX - WIDEST_PICK_RADIUS_CSS_PX &&
    query.cursorXCssPx <= maxX + WIDEST_PICK_RADIUS_CSS_PX &&
    query.cursorYCssPx >= minY - WIDEST_PICK_RADIUS_CSS_PX &&
    query.cursorYCssPx <= maxY + WIDEST_PICK_RADIUS_CSS_PX
  );
};

/**
 * Sweep every point of the candidate tiles. A point is rejected when its
 * projection is unusable (non-finite, or `clip.w <= CLIP_W_EPSILON`), it
 * falls outside the rendered near/far interval (`|ndc z| > 1`), or its ray
 * depth is negative. Survivors land in every bucket whose radius covers their
 * css distance from the cursor; the smallest non-empty bucket then answers
 * with its minimum-depth point, mirroring `ray_depth.py`.
 *
 * The projection is spelled out here instead of calling
 * {@link projectPointToCssPx}: this loop runs once per drawn point — the whole
 * draw budget, on every move of a drag — where the shared helper would allocate
 * an argument array and a result object per point and re-test a viewport that
 * is fixed for the query. Distances stay squared until a winner is known, as
 * `_PICK_RADII_SQ` does.
 */
export const sweepPickPoints = (
  query: PickQuery,
  tiles: readonly PickTile[],
): PointPickResult => {
  const {
    ray,
    viewProj: m,
    cursorXCssPx,
    cursorYCssPx,
    viewportWidthCssPx: width,
    viewportHeightCssPx: height,
  } = query;
  if (!finitePositive(width) || !finitePositive(height)) return MISS;
  const m0 = m[0]!,
    m1 = m[1]!,
    m2 = m[2]!,
    m3 = m[3]!;
  const m4 = m[4]!,
    m5 = m[5]!,
    m6 = m[6]!,
    m7 = m[7]!;
  const m8 = m[8]!,
    m9 = m[9]!,
    m10 = m[10]!,
    m11 = m[11]!;
  const m12 = m[12]!,
    m13 = m[13]!,
    m14 = m[14]!,
    m15 = m[15]!;
  const [rayX, rayY, rayZ] = ray.origin;
  const [dirX, dirY, dirZ] = ray.direction;
  const bucketCount = PICK_RADII_SQ.length;
  /**
   * NaN marks an empty bucket: `!(depth >= NaN)` holds, so the first
   * qualifying point fills a bucket and every later one has to beat it.
   */
  const bestDepth = new Float64Array(bucketCount).fill(Number.NaN);
  const bestDistanceSq = new Float64Array(bucketCount);
  for (const tile of tiles) {
    const [originX, originY, originZ] = tile.origin;
    const { positions } = tile;
    for (let index = 0; index < tile.pointCount; index += 1) {
      const x = originX + positions[index * 3]!;
      const y = originY + positions[index * 3 + 1]!;
      const z = originZ + positions[index * 3 + 2]!;
      const clipW = m3 * x + m7 * y + m11 * z + m15;
      if (!Number.isFinite(clipW) || clipW <= CLIP_W_EPSILON) continue;
      const invW = 1 / clipW;
      // Every remaining test is written as a guard, so a non-finite clip
      // coordinate rejects its point rather than passing through.
      const ndcZ = (m2 * x + m6 * y + m10 * z + m14) * invW;
      if (!(ndcZ >= -1 && ndcZ <= 1)) continue;
      const depth = (x - rayX) * dirX + (y - rayY) * dirY + (z - rayZ) * dirZ;
      if (!(depth > 0)) continue;
      const offsetX =
        (((m0 * x + m4 * y + m8 * z + m12) * invW + 1) / 2) * width -
        cursorXCssPx;
      const offsetY =
        ((1 - (m1 * x + m5 * y + m9 * z + m13) * invW) / 2) * height -
        cursorYCssPx;
      const distanceSq = offsetX * offsetX + offsetY * offsetY;
      if (!(distanceSq <= WIDEST_PICK_RADIUS_SQ)) continue;
      for (let bucket = bucketCount - 1; bucket >= 0; bucket -= 1) {
        if (distanceSq > PICK_RADII_SQ[bucket]!) break;
        if (!(depth >= bestDepth[bucket]!)) {
          bestDepth[bucket] = depth;
          bestDistanceSq[bucket] = distanceSq;
        }
      }
    }
  }
  for (let bucket = 0; bucket < bucketCount; bucket += 1) {
    const depth = bestDepth[bucket]!;
    if (Number.isNaN(depth)) continue;
    return {
      status: "hit",
      rayDepth: depth,
      pointOnRay: [
        rayX + dirX * depth,
        rayY + dirY * depth,
        rayZ + dirZ * depth,
      ],
      distancePx: Math.sqrt(bestDistanceSq[bucket]!),
    };
  }
  return MISS;
};

/**
 * Pick against a tile set: cursor ray, bounds prefilter, point sweep, bucket
 * selection. Returns null when the query cannot be evaluated at all — a
 * singular or non-finite view-projection, unusable viewport dimensions, or a
 * non-finite cursor. A query that evaluates but supports nothing is a valid
 * `miss`, never null.
 */
export const pickPointInTiles = (
  view: CameraView,
  cursorXCssPx: number,
  cursorYCssPx: number,
  tiles: readonly PickTile[],
): PointPickResult | null => {
  const ray = cursorRay(
    view.viewProj,
    cursorXCssPx,
    cursorYCssPx,
    view.viewportWidthCssPx,
    view.viewportHeightCssPx,
  );
  if (ray === null) return null;
  const query: PickQuery = {
    viewProj: view.viewProj,
    ray,
    cursorXCssPx,
    cursorYCssPx,
    viewportWidthCssPx: view.viewportWidthCssPx,
    viewportHeightCssPx: view.viewportHeightCssPx,
  };
  return sweepPickPoints(
    query,
    tiles.filter((tile) => tileIsPickCandidate(query, tile.bounds)),
  );
};
