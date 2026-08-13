import { describe, expect, it } from "vitest";

import { cursorRay, type PerspectiveCameraView } from "./camera";
import type { Vec3 } from "./octree";
import {
  DEFAULT_PICK_PIXEL_RADIUS,
  DEFAULT_PICK_PIXEL_RADIUS_MULTIPLIERS,
  PICK_RADII_CSS_PX,
  pickPointInTiles,
  sweepPickPoints,
  type PickQuery,
  type PickTile,
  type PointPickResult,
} from "./picking";

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/**
 * Identity view-projection over a 200x100 viewport: world coordinates are NDC
 * directly, one NDC x unit is 100 css px, one NDC y unit is 50 css px, the
 * cursor ray under the center runs from (x, y, -1) along +z, and a point's
 * ray depth is `z + 1`.
 */
const VIEW: PerspectiveCameraView = {
  projection: "perspective",
  viewProj: IDENTITY,
  position: [0, 0, 0],
  fovY: Math.PI / 2,
  viewportWidthCssPx: 200,
  viewportHeightCssPx: 100,
};
const CENTER: [number, number] = [100, 50];

/** Column-major perspective matrix (symmetric frustum, looking down -Z). */
const perspective = (
  fovY: number,
  aspect: number,
  near: number,
  far: number,
): number[] => {
  const f = 1 / Math.tan(fovY / 2);
  return [
    f / aspect,
    0,
    0,
    0,
    0,
    f,
    0,
    0,
    0,
    0,
    (far + near) / (near - far),
    -1,
    0,
    0,
    (2 * far * near) / (near - far),
    0,
  ];
};

const tile = (points: Vec3[], overrides?: Partial<PickTile>): PickTile => ({
  origin: [0, 0, 0],
  positions: new Float32Array(points.flat()),
  pointCount: points.length,
  ...overrides,
});

const hit = (
  result: PointPickResult | null,
): Extract<PointPickResult, { status: "hit" }> => {
  if (result?.status !== "hit") {
    throw new Error(`expected a hit, got ${JSON.stringify(result)}`);
  }
  return result;
};

describe("pick radii", () => {
  it("mirror scene/ray_depth.py exactly", () => {
    expect(DEFAULT_PICK_PIXEL_RADIUS).toBe(10);
    expect(DEFAULT_PICK_PIXEL_RADIUS_MULTIPLIERS).toEqual([1, 2, 10]);
    expect(PICK_RADII_CSS_PX).toEqual([10, 20, 100]);
  });
});

describe("pickPointInTiles", () => {
  it("answers on the cursor ray at the support depth, not at the vertex", () => {
    // 5 px right of the cursor, half a unit deep.
    const result = hit(
      pickPointInTiles(VIEW, ...CENTER, [tile([[0.05, 0, 0.5]])]),
    );
    expect(result.scenePoint[0]).toBeCloseTo(0);
    expect(result.scenePoint[1]).toBeCloseTo(0);
    expect(result.scenePoint[2]).toBeCloseTo(0.5);
    expect(result.rayDepth).toBeCloseTo(1.5);
    expect(result.distancePx).toBeCloseTo(5, 4);
  });

  it("measures css distance with both viewport dimensions", () => {
    // The same NDC offset is 16 px along x but only 8 px along y.
    const alongY = hit(
      pickPointInTiles(VIEW, ...CENTER, [tile([[0, 0.16, 0]])]),
    );
    expect(alongY.distancePx).toBeCloseTo(8, 4);
    const alongX = hit(
      pickPointInTiles(VIEW, ...CENTER, [tile([[0.16, 0, 0]])]),
    );
    expect(alongX.distancePx).toBeCloseTo(16, 4);
  });

  it("prefers the frontmost point of the smallest non-empty bucket", () => {
    // The winner is neither the closest to the cursor nor the last swept:
    // only minimum depth within the 10 px bucket selects it.
    const tiles = [
      tile([
        [0.05, 0, -0.5], // 5 px, depth 0.5 — frontmost in the 10 px bucket
        [0.02, 0, 0.8], // 2 px, depth 1.8 — closer to the cursor, but behind
        [0.15, 0, -0.9], // 15 px, depth 0.1 — nearer, but outside the bucket
      ]),
    ];
    const result = hit(pickPointInTiles(VIEW, ...CENTER, tiles));
    expect(result.scenePoint[2]).toBeCloseTo(-0.5);
    expect(result.distancePx).toBeCloseTo(5, 4);
  });

  it("escalates through the buckets and misses past the largest", () => {
    const at = (ndcX: number): PointPickResult | null =>
      pickPointInTiles(VIEW, ...CENTER, [tile([[ndcX, 0, 0]])]);
    expect(hit(at(0.15)).distancePx).toBeCloseTo(15, 4); // 20 px bucket
    expect(hit(at(0.5)).distancePx).toBeCloseTo(50, 4); // 100 px bucket
    expect(at(1.5)).toEqual({ status: "miss" }); // past every bucket
  });

  it("composes tile origin with tile-local positions", () => {
    const composed = tile([[-0.45, 0, 0]], { origin: [0.5, 0, 0] });
    // World position is origin + local = (0.05, 0, 0): 5 px from the cursor.
    // Ignoring the origin would land 45 px out instead.
    const result = hit(pickPointInTiles(VIEW, ...CENTER, [composed]));
    expect(result.distancePx).toBeCloseTo(5, 4);
  });

  it("rejects points outside the rendered near/far interval", () => {
    expect(pickPointInTiles(VIEW, ...CENTER, [tile([[0, 0, -1.5]])])).toEqual({
      status: "miss",
    });
    expect(pickPointInTiles(VIEW, ...CENTER, [tile([[0, 0, 1.5]])])).toEqual({
      status: "miss",
    });
  });

  it("rejects near/far/behind-camera points under a perspective view", () => {
    const view: PerspectiveCameraView = {
      ...VIEW,
      viewProj: perspective(Math.PI / 2, 2, 0.1, 100),
      viewportWidthCssPx: 800,
      viewportHeightCssPx: 400,
    };
    const at = (point: Vec3): PointPickResult | null =>
      pickPointInTiles(view, 400, 200, [tile([point])]);
    expect(at([0, 0, -0.05])).toEqual({ status: "miss" }); // before near
    expect(at([0, 0, -200])).toEqual({ status: "miss" }); // beyond far
    expect(at([0, 0, 5])).toEqual({ status: "miss" }); // behind the camera
    const inFront = hit(at([0, 0, -10]));
    expect(inFront.scenePoint[2]).toBeCloseTo(-10);
    expect(inFront.distancePx).toBeCloseTo(0);
  });

  it("skips non-finite points without poisoning the sweep", () => {
    const result = hit(
      pickPointInTiles(VIEW, ...CENTER, [
        tile([
          [Number.NaN, 0, 0],
          [0.05, 0, 0],
        ]),
      ]),
    );
    expect(result.distancePx).toBeCloseTo(5, 4);
  });

  it("prefilters a tile whose bounds sit outside the cursor's reach", () => {
    // The point is dead under the cursor, but the tile's declared bounds
    // project 300+ px away even after the largest-radius expansion: the
    // prefilter must skip the tile without sweeping it.
    const mislocated = tile([[0, 0, 0]], {
      bounds: { min: [3, -0.5, -0.5], max: [4, 0.5, 0.5] },
    });
    expect(pickPointInTiles(VIEW, ...CENTER, [mislocated])).toEqual({
      status: "miss",
    });
  });

  it("expands the prefilter by the largest pick radius, not the smallest", () => {
    // The bounds' screen AABB starts 50 px right of the cursor: past the 10
    // and 20 px buckets, but the 100 px bucket can still catch the point, so
    // the prefilter must keep the tile.
    const offset = tile([[0.6, 0, 0]], {
      bounds: { min: [0.5, -0.1, -0.1], max: [0.7, 0.1, 0.1] },
    });
    const result = hit(pickPointInTiles(VIEW, ...CENTER, [offset]));
    expect(result.distancePx).toBeCloseTo(60, 4);
  });

  it("keeps a tile with missing bounds conservatively pickable", () => {
    const unbounded = tile([[0, 0, 0]]);
    expect(
      hit(pickPointInTiles(VIEW, ...CENTER, [unbounded])).distancePx,
    ).toBeCloseTo(0);
  });

  it("keeps a tile with a behind-camera bounds corner unconditionally", () => {
    const view: PerspectiveCameraView = {
      ...VIEW,
      viewProj: perspective(Math.PI / 2, 2, 0.1, 100),
      viewportWidthCssPx: 800,
      viewportHeightCssPx: 400,
    };
    // The z = 0.5 corners sit behind the camera, so no screen AABB is
    // usable and the tile must stay a candidate outright. Dropping just the
    // bad corners would keep the survivors' AABB, whose far z = -6 corners
    // project 100+ px left of the cursor — perspective magnification puts
    // this in-bounds point far outside their hull — and skip the tile.
    const straddling = tile([[2.8, 0, -2]], {
      bounds: { min: [2, -1, -6], max: [4, 1, 0.5] },
    });
    const result = hit(pickPointInTiles(view, 680, 200, [straddling]));
    expect(result.scenePoint[2]).toBeCloseTo(-2);
    expect(result.distancePx).toBeCloseTo(0);
  });

  it("returns miss, not null, for a valid sweep over nothing", () => {
    expect(pickPointInTiles(VIEW, ...CENTER, [])).toEqual({ status: "miss" });
    expect(pickPointInTiles(VIEW, ...CENTER, [tile([])])).toEqual({
      status: "miss",
    });
  });

  it("returns null, not miss, for an unusable query", () => {
    const singular = { ...VIEW, viewProj: IDENTITY.map(() => 0) };
    expect(pickPointInTiles(singular, ...CENTER, [tile([[0, 0, 0]])])).toBe(
      null,
    );
    const broken = {
      ...VIEW,
      viewProj: [...IDENTITY.slice(0, 15), Number.NaN],
    };
    expect(pickPointInTiles(broken, ...CENTER, [tile([[0, 0, 0]])])).toBe(null);
    const flatViewport = { ...VIEW, viewportWidthCssPx: 0 };
    expect(pickPointInTiles(flatViewport, ...CENTER, [tile([[0, 0, 0]])])).toBe(
      null,
    );
    expect(pickPointInTiles(VIEW, Number.NaN, 50, [tile([[0, 0, 0]])])).toBe(
      null,
    );
  });
});

describe("sweepPickPoints", () => {
  const query = (rayOriginZ: number): PickQuery => ({
    viewProj: IDENTITY,
    ray: { origin: [0, 0, rayOriginZ], direction: [0, 0, 1] },
    cursorXCssPx: CENTER[0],
    cursorYCssPx: CENTER[1],
    viewportWidthCssPx: 200,
    viewportHeightCssPx: 100,
  });

  it("rejects points at non-positive ray depth even when they project fine", () => {
    // The crafted ray starts past the point: clip-space tests all pass, but
    // the support depth is negative and the point must not answer.
    const behindRay = sweepPickPoints(query(5), [tile([[0, 0, 0.5]])]);
    expect(behindRay).toEqual({ status: "miss" });
    expect(sweepPickPoints(query(0.5), [tile([[0, 0, 0.5]])])).toEqual({
      status: "miss",
    });
    const inFrontOfRay = sweepPickPoints(query(-5), [tile([[0, 0, 0.5]])]);
    expect(hit(inFrontOfRay).scenePoint[2]).toBeCloseTo(0.5);
  });
});

describe("cursorRay and the sweep agree on the pick model", () => {
  it("keeps the hit on the ray the same helpers built", () => {
    // Independently known under the identity view: the center ray runs from
    // (0, 0, -1) along +z, and the vertex at z = 0.5 supports depth 1.5.
    const ray = cursorRay(VIEW.viewProj, ...CENTER, 200, 100)!;
    expect(ray.origin[0]).toBeCloseTo(0);
    expect(ray.origin[1]).toBeCloseTo(0);
    expect(ray.origin[2]).toBeCloseTo(-1);
    expect(ray.direction[0]).toBeCloseTo(0);
    expect(ray.direction[1]).toBeCloseTo(0);
    expect(ray.direction[2]).toBeCloseTo(1);
    const result = hit(
      pickPointInTiles(VIEW, ...CENTER, [tile([[0.05, 0, 0.5]])]),
    );
    const depth = 1.5;
    for (const axis of [0, 1, 2] as const) {
      expect(result.scenePoint[axis]).toBeCloseTo(
        ray.origin[axis] + ray.direction[axis] * depth,
      );
    }
  });
});
