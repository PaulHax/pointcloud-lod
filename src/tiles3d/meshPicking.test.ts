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
      scenePoint: [0, 0, 0],
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
    const anchor = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1] as const;
    const result = pickSubmittedTriangles(view, 50, 50, [translated], {
      drawn: Array.from(anchor) as never,
      scene: Array.from(anchor) as never,
    });
    expect(result?.status).toBe("hit");
  });

  it("finds the hit where it is drawn but reports it where it is stored", () => {
    // Terrain drawn at 3x vertical exaggeration. The cursor lands on the
    // exaggerated surface — so the ray must be cast against it — but the point
    // handed back is what the app writes into a control point, and that must
    // be the real elevation, unchanged by a display-only scale.
    const exaggeration = createVerticalExaggerationTransform(3, 0);
    const composed = pickSubmittedTriangles(view, 50, 50, [tile("source", 1)], {
      drawn: Array.from(exaggeration) as never,
      scene: null,
    });
    const baked = pickSubmittedTriangles(view, 50, 50, [tile("baked", 3)]);

    // Same pixel, same depth through the drawn scene as geometry truly at z=3.
    expect(composed).toMatchObject({ status: "hit", rayDepth: 8 });
    expect(baked).toMatchObject({
      status: "hit",
      rayDepth: 8,
      scenePoint: [0, 0, 3],
    });
    // ...but the source geometry really sits at z=1, and that is what is reported.
    expect((composed as unknown as { scenePoint: number[] }).scenePoint[2]).toBeCloseTo(1);
  });

  it("reports the unexaggerated point through the pick set's placement", () => {
    const set = createMeshPickSet();
    set.replaceDrawn([tile("terrain", 1)]);
    set.setPlacement({
      drawn: Array.from(createVerticalExaggerationTransform(4, 0)) as never,
      scene: null,
    });
    const hit = set.pick(view, 50, 50) as unknown as { scenePoint: number[] };
    expect(hit.scenePoint[2]).toBeCloseTo(1);
    // Occlusion still compares in drawn space: 4x lifts the surface toward the
    // camera, so it must occlude at the drawn depth, not the stored one.
    expect(set.occlusionDepth(view, 50, 50)).toEqual({
      status: "hit",
      rayDepth: 9,
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
