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
  cursorRay,
  projectPointToCssPx,
  type CameraView,
  type CursorRay,
  type Mat16,
} from "./camera";
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
export const tileIsPickCandidate = (
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
  const radius = PICK_RADII_CSS_PX[PICK_RADII_CSS_PX.length - 1]!;
  return (
    query.cursorXCssPx >= minX - radius &&
    query.cursorXCssPx <= maxX + radius &&
    query.cursorYCssPx >= minY - radius &&
    query.cursorYCssPx <= maxY + radius
  );
};

type BucketBest = {
  readonly depth: number;
  readonly distancePx: number;
};

/**
 * Sweep every point of the candidate tiles. A point is rejected when its
 * projection is unusable (non-finite, or `clip.w <= CLIP_W_EPSILON`), it
 * falls outside the rendered near/far interval (`|ndc z| > 1`), or its ray
 * depth is negative. Survivors land in every bucket whose radius covers their
 * css distance from the cursor; the smallest non-empty bucket then answers
 * with its minimum-depth point, mirroring `ray_depth.py`.
 */
export const sweepPickPoints = (
  query: PickQuery,
  tiles: readonly PickTile[],
): PointPickResult => {
  const { ray } = query;
  const bests: (BucketBest | null)[] = PICK_RADII_CSS_PX.map(() => null);
  for (const tile of tiles) {
    const [originX, originY, originZ] = tile.origin;
    const { positions } = tile;
    for (let index = 0; index < tile.pointCount; index += 1) {
      const x = originX + positions[index * 3]!;
      const y = originY + positions[index * 3 + 1]!;
      const z = originZ + positions[index * 3 + 2]!;
      const projected = projectPointToCssPx(
        query.viewProj,
        [x, y, z],
        query.viewportWidthCssPx,
        query.viewportHeightCssPx,
      );
      if (projected === null || Math.abs(projected.ndcZ) > 1) continue;
      const depth =
        (x - ray.origin[0]) * ray.direction[0] +
        (y - ray.origin[1]) * ray.direction[1] +
        (z - ray.origin[2]) * ray.direction[2];
      // `>=` written as a guard so a NaN depth rejects rather than passes.
      if (!(depth >= 0)) continue;
      const distancePx = Math.hypot(
        projected.xCssPx - query.cursorXCssPx,
        projected.yCssPx - query.cursorYCssPx,
      );
      for (
        let bucket = PICK_RADII_CSS_PX.length - 1;
        bucket >= 0;
        bucket -= 1
      ) {
        if (distancePx > PICK_RADII_CSS_PX[bucket]!) break;
        const best = bests[bucket]!;
        if (best === null || depth < best.depth) {
          bests[bucket] = { depth, distancePx };
        }
      }
    }
  }
  const best = bests.find((entry) => entry !== null);
  if (best === undefined || best === null) return MISS;
  return {
    status: "hit",
    pointOnRay: [
      ray.origin[0] + ray.direction[0] * best.depth,
      ray.origin[1] + ray.direction[1] * best.depth,
      ray.origin[2] + ray.direction[2] * best.depth,
    ],
    distancePx: best.distancePx,
  };
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
