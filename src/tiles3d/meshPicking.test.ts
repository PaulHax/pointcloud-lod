import { describe, expect, it } from "vitest";

import type { CameraView } from "../camera";
import {
  createMeshPickSet,
  pickSubmittedTriangles,
  type SubmittedMeshTile,
} from "./meshPicking";
import { createVerticalExaggerationTransform } from "./rtc";

const view: CameraView = {
  projection: "orthographic",
  viewProj: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 1],
  position: [0, 0, -10],
  viewportWidthCssPx: 100,
  viewportHeightCssPx: 100,
  parallelScale: 1,
};

const tile = (id: string, z: number): SubmittedMeshTile => ({
  id,
  origin: [0, 0, z],
  primitives: [
    {
      positions: new Float32Array([-0.5, -0.5, 0, 0.5, -0.5, 0, 0, 0.5, 0]),
      indices: new Uint16Array([0, 1, 2]),
    },
  ],
});

describe("submitted mesh picking", () => {
  it("returns the frontmost submitted triangle and common positive ray depth", () => {
    const result = pickSubmittedTriangles(view, 50, 50, [
      tile("far", 3),
      tile("near", 0),
    ]);
    expect(result).toEqual({
      status: "hit",
      rayDepth: 5,
      pointOnRay: [0, 0, 0],
      distancePx: 0,
    });
  });

  it("uses indexed and unindexed triangles and applies the live anchor matrix", () => {
    const translated: SubmittedMeshTile = {
      ...tile("translated", 0),
      primitives: [
        {
          positions: new Float32Array([
            -1.5, -0.5, 0, -0.5, -0.5, 0, -1, 0.5, 0,
          ]),
        },
      ],
    };
    const result = pickSubmittedTriangles(
      view,
      50,
      50,
      [translated],
      [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    );
    expect(result?.status).toBe("hit");
  });

  it("returns the same pick and occlusion depth for baked vertical geometry", () => {
    const composed = pickSubmittedTriangles(
      view,
      50,
      50,
      [tile("source", 1)],
      createVerticalExaggerationTransform(3, 0),
    );
    const baked = pickSubmittedTriangles(view, 50, 50, [tile("baked", 3)]);

    expect(composed).toEqual(baked);
    expect(baked).toMatchObject({
      status: "hit",
      rayDepth: 8,
      pointOnRay: [0, 0, 3],
    });
  });

  it("tracks refinement swaps using only the exact submitted draw set", () => {
    const set = createMeshPickSet();
    set.replaceDrawn([tile("parent", 1)]);
    expect(set.pick(view, 50, 50)).toMatchObject({ rayDepth: 6 });
    set.replaceDrawn([tile("child", 0)]);
    expect(set.pick(view, 50, 50)).toMatchObject({ rayDepth: 5 });
    set.replaceDrawn([]);
    expect(set.pick(view, 50, 50)).toEqual({ status: "miss" });
  });

  it("reports unavailable for invalid queries and mirrors hits as occlusion depth", () => {
    const set = createMeshPickSet();
    set.replaceDrawn([tile("mesh", 0)]);
    expect(
      set.pick(
        { ...view, viewProj: Array.from({ length: 16 }, () => 0) },
        50,
        50,
      ),
    ).toBeNull();
    expect(set.occlusionDepth(view, 50, 50)).toEqual({
      status: "hit",
      rayDepth: 5,
    });
    set.replaceDrawn([]);
    expect(set.occlusionDepth(view, 50, 50)).toEqual({ status: "clear" });
    set.dispose();
    expect(set.pick(view, 50, 50)).toBeNull();
  });

  it("uses a conservative projected tile-bounds prefilter", () => {
    const outside: SubmittedMeshTile = {
      ...tile("outside", 0),
      bounds: { min: [0.5, 0.5, 0], max: [0.9, 0.9, 0] },
    };
    expect(pickSubmittedTriangles(view, 10, 90, [outside])).toEqual({
      status: "miss",
    });
  });
});
