import { describe, expect, it, vi } from "vitest";

import type { PerspectiveCameraView } from "../camera";
import type { TilesetTile } from "./tilesetSource";
import { createVerticalExaggerationTransform } from "./rtc";
import { traverseTileset, type TileReadiness } from "./traversal";

const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] as const;
const lookAway = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -10, 0, 0, 1] as const;
const view = (
  position: [number, number, number],
  viewProj: ArrayLike<number> = identity,
): PerspectiveCameraView => ({
  projection: "perspective",
  viewProj,
  position,
  fovY: Math.PI / 2,
  viewportWidthCssPx: 100,
  viewportHeightCssPx: 100,
});

const matrix = (x = 0, y = 0, z = 0) => [
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
  1,
  0,
  x,
  y,
  z,
  1,
];

const makeTile = (
  id: string,
  geometricError: number,
  children: TilesetTile[] = [],
  transform: readonly number[] = identity,
): TilesetTile => ({
  id,
  geometricError,
  boundingVolume: {
    center: [0, 0, 0],
    halfAxes: [1, 0, 0, 0, 1, 0, 0, 0, 1],
  },
  transform,
  worldTransform: transform,
  contentUri: `${id}.glb`,
  contentUrl: `/tiles/${id}.glb`,
  children,
});

const withoutContent = (tile: TilesetTile): TilesetTile => {
  const { contentUri: _uri, contentUrl: _url, ...contentless } = tile;
  return contentless;
};

const tree = () => {
  const leftLeaf = makeTile("root/0/0", 0, [], matrix(-0.5));
  const rightLeaf = makeTile("root/1/0", 0, [], matrix(0.5));
  const left = makeTile("root/0", 0.5, [leftLeaf], matrix(-0.5));
  const right = makeTile("root/1", 0.5, [rightLeaf], matrix(0.5));
  return makeTile("root", 8, [left, right]);
};

const select = (
  root: TilesetTile,
  readiness: Record<string, TileReadiness> = {},
  overrides: Partial<Parameters<typeof traverseTileset>[0]> = {},
) =>
  traverseTileset({
    root,
    camera: view([0, 0, 10]),
    maximumScreenSpaceErrorPx: 4,
    readiness: (id) => readiness[id] ?? "unloaded",
    ...overrides,
  });

describe("traverseTileset", () => {
  it("pins near/far camera selection and stable document order", () => {
    expect(
      select(tree(), {}, { camera: view([0, 0, 120]) }).desiredTileIds,
    ).toEqual(["root"]);
    expect(
      select(tree(), {}, { camera: view([0, 0, 10]) }).desiredTileIds,
    ).toEqual(["root/0", "root/1"]);
    expect(
      select(tree(), {}, { camera: view([0, 0, 2]) }).desiredTileIds,
    ).toEqual(["root/0/0", "root/1/0"]);
  });

  it("relaxes the effective SSE threshold from normalized quality", () => {
    const high = select(tree(), {}, { qualityFraction: 1 });
    const low = select(tree(), {}, { qualityFraction: 0.05 });
    expect(high.effectiveScreenSpaceErrorPx).toBe(4);
    expect(low.effectiveScreenSpaceErrorPx).toBe(80);
    expect(high.desiredTileIds).toEqual(["root/0", "root/1"]);
    expect(low.desiredTileIds).toEqual(["root"]);
  });

  it("holds a REPLACE parent until every required child is submitted", () => {
    const root = tree();
    const parentOnly = select(root, { root: "submitted" });
    expect(parentOnly.requestedTileIds).toEqual(["root", "root/0", "root/1"]);
    expect(parentOnly.drawnTileIds).toEqual(["root"]);

    const oneChild = select(root, {
      root: "submitted",
      "root/0": "submitted",
      "root/1": "decoded",
    });
    expect(oneChild.drawnTileIds).toEqual(["root"]);

    const everyChild = select(root, {
      root: "submitted",
      "root/0": "submitted",
      "root/1": "submitted",
    });
    expect(everyChild.drawnTileIds).toEqual(["root/0", "root/1"]);
  });

  it("traverses a contentless root without selecting blank content", () => {
    const readiness = vi.fn<(id: string) => TileReadiness>(() => "unloaded");
    const root = withoutContent(
      makeTile("root", 0, [makeTile("root/0", 0), makeTile("root/1", 0)]),
    );

    const loading = select(root, {}, { readiness });
    expect(loading.desiredTileIds).toEqual(["root/0", "root/1"]);
    expect(loading.requestedTileIds).toEqual(["root/0", "root/1"]);
    expect(loading.drawnTileIds).toEqual([]);
    expect(readiness).not.toHaveBeenCalledWith("root");

    const ready = select(root, {
      "root/0": "submitted",
      "root/1": "submitted",
    });
    expect(ready.drawnTileIds).toEqual(["root/0", "root/1"]);
  });

  it("holds the nearest submitted REPLACE ancestor across a contentless intermediate", () => {
    const structural = withoutContent(
      makeTile("root/0", 0, [makeTile("root/0/0", 0), makeTile("root/0/1", 0)]),
    );
    const root = makeTile("root", 8, [structural]);
    const readiness = vi.fn<(id: string) => TileReadiness>((id) =>
      id === "root" || id === "root/0/0" ? "submitted" : "unloaded",
    );

    const waiting = select(root, {}, { readiness });
    expect(waiting.desiredTileIds).toEqual(["root/0/0", "root/0/1"]);
    expect(waiting.requestedTileIds).toEqual(["root", "root/0/0", "root/0/1"]);
    expect(waiting.drawnTileIds).toEqual(["root"]);
    expect(readiness).not.toHaveBeenCalledWith("root/0");

    const ready = select(root, {
      root: "submitted",
      "root/0/0": "submitted",
      "root/0/1": "submitted",
    });
    expect(ready.drawnTileIds).toEqual(["root/0/0", "root/0/1"]);
  });

  it("holds submitted descendants while a coarser desired parent is admitted", () => {
    const root = tree();
    const result = select(
      root,
      {
        root: "decoded",
        "root/0": "submitted",
        "root/1": "submitted",
      },
      { camera: view([0, 0, 120]) },
    );
    expect(result.desiredTileIds).toEqual(["root"]);
    expect(result.requestedTileIds).toEqual(["root", "root/0", "root/1"]);
    expect(result.drawnTileIds).toEqual(["root/0", "root/1"]);
  });

  it("keeps the nearest submitted fallback through nested failure and admission", () => {
    const root = tree();
    const waiting = select(
      root,
      {
        root: "submitted",
        "root/0": "submitted",
        "root/1": "submitted",
        "root/0/0": "submitted",
        "root/1/0": "failed",
      },
      { camera: view([0, 0, 2]) },
    );
    expect(waiting.desiredTileIds).toEqual(["root/0/0", "root/1/0"]);
    expect(waiting.drawnTileIds).toEqual(["root/0/0", "root/1"]);
    expect(waiting.requestedTileIds).toEqual([
      "root/0/0",
      "root/1",
      "root/1/0",
    ]);

    const ready = select(
      root,
      {
        root: "submitted",
        "root/0": "submitted",
        "root/1": "submitted",
        "root/0/0": "submitted",
        "root/1/0": "submitted",
      },
      { camera: view([0, 0, 2]) },
    );
    expect(ready.drawnTileIds).toEqual(["root/0/0", "root/1/0"]);
  });

  it("culls outside volumes but keeps frustum-edge intersections", () => {
    const outside = makeTile("outside", 0, [], matrix(4));
    expect(select(outside).desiredTileIds).toEqual([]);
    expect(select(outside).culledTileIds).toEqual(["outside"]);
    expect(select(makeTile("edge", 0, [], matrix(2))).desiredTileIds).toEqual([
      "edge",
    ]);
    expect(
      select(tree(), {}, { camera: view([0, 0, 10], lookAway) }).desiredTileIds,
    ).toEqual([]);
  });

  it("uses accumulated f64 and caller model transforms for culling", () => {
    const child = makeTile("root/0", 0, [], matrix(4));
    const root = {
      ...makeTile("root", 8, [child], matrix(-4)),
      boundingVolume: {
        center: [0, 0, 0] as const,
        halfAxes: [5, 0, 0, 0, 5, 0, 0, 0, 5] as const,
      },
    };
    expect(select(root).desiredTileIds).toEqual(["root/0"]);
    expect(select(root, {}, { modelMatrix: matrix(4) }).desiredTileIds).toEqual(
      ["root"],
    );
  });

  it("applies pivoted vertical exaggeration to traversal bounds and SSE", () => {
    const shallow = makeTile("shallow", 0);
    expect(select(shallow).desiredTileIds).toEqual(["shallow"]);
    expect(
      select(
        shallow,
        {},
        {
          modelMatrix: createVerticalExaggerationTransform(2, 10),
        },
      ),
    ).toMatchObject({ desiredTileIds: [], culledTileIds: ["shallow"] });

    const orthographic = {
      projection: "orthographic" as const,
      viewProj: identity,
      position: [0, 0, 100] as [number, number, number],
      parallelScale: 100,
      viewportWidthCssPx: 100,
      viewportHeightCssPx: 100,
    };
    expect(select(tree(), {}, { camera: orthographic }).desiredTileIds).toEqual(
      ["root"],
    );
    expect(
      select(
        tree(),
        {},
        {
          camera: orthographic,
          modelMatrix: createVerticalExaggerationTransform(3, 0),
        },
      ).desiredTileIds,
    ).toEqual(["root/0", "root/1"]);
  });

  it("refines while the camera is inside a volume", () => {
    expect(
      select(tree(), {}, { camera: view([0, 0, 0]) }).desiredTileIds,
    ).toEqual(["root/0/0", "root/1/0"]);
  });

  it("supports orthographic SSE without a renderer dependency", () => {
    const result = select(
      tree(),
      {},
      {
        camera: {
          projection: "orthographic",
          viewProj: identity,
          position: [0, 0, 100],
          parallelScale: 100,
          viewportWidthCssPx: 100,
          viewportHeightCssPx: 100,
        },
      },
    );
    expect(result.desiredTileIds).toEqual(["root"]);
  });

  it.each([0, -1, Infinity, Number.NaN])(
    "rejects invalid maximum SSE %s",
    (maximumScreenSpaceErrorPx) => {
      expect(() => select(tree(), {}, { maximumScreenSpaceErrorPx })).toThrow(
        RangeError,
      );
    },
  );
});
