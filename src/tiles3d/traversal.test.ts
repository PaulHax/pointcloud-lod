import { describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";

import type { PerspectiveCameraView } from "../camera";
import { parseTileset, type TilesetTile } from "./tilesetSource";
import { createVerticalExaggerationTransform } from "./rtc";
import { traverseTileset, type TileReadiness } from "./traversal";
import { parseSubtree } from "./subtree";
import type { SubtreeStoreSnapshot } from "./subtreeStore";
import {
  makeSubtreeFixture,
  SUBTREE_METADATA_SCHEMA,
} from "../../test/fixtures/subtreeFixture";

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

const implicitRoot = (): TilesetTile => ({
  ...makeTile("root", 16, []),
  contentUri: undefined,
  contentUrl: undefined,
  implicitAddress: { level: 0, x: 0, y: 0 },
  implicitTiling: {
    subdivisionScheme: "QUADTREE",
    subtreeLevels: 2,
    availableLevels: 3,
    subtreeUriTemplate: "subtrees/{level}/{x}/{y}.subtree",
    subtreeUrlTemplate: "/tiles/subtrees/{level}/{x}/{y}.subtree",
    contentUriTemplate: "content/{level}/{x}/{y}.glb",
    contentUrlTemplate: "/tiles/content/{level}/{x}/{y}.glb",
    metadataSchema: SUBTREE_METADATA_SCHEMA,
  },
});

const subtreeSnapshot = (subtree: ReturnType<typeof parseSubtree>) =>
  ({
    revision: "r1",
    configGeneration: 1,
    selected: 1,
    active: 0,
    queued: 0,
    retrying: 0,
    ready: 1,
    failed: 0,
    cached: 1,
    cachedBytes: subtree.byteLength,
    cacheHits: 0,
    cacheMisses: 1,
    cacheEvictions: 0,
    workPending: false,
    disposed: false,
    entries: [
      {
        id: "subtree/0/0/0",
        url: "/tiles/subtrees/0/0/0.subtree",
        status: "ready",
        attempt: 1,
      },
    ],
    subtreeById: new Map([["subtree/0/0/0", subtree]]),
  }) satisfies SubtreeStoreSnapshot;

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
  it("traverses the checked-in producer fixture to its four quadrant contents", () => {
    const directory = new URL(
      "../../test/fixtures/tiles3d-implicit/",
      import.meta.url,
    );
    const source = parseTileset(
      JSON.parse(readFileSync(new URL("tileset.json", directory), "utf8")),
      "/fixture",
    );
    const bytes = readFileSync(new URL("subtrees/0/0/0.subtree", directory));
    const subtree = parseSubtree(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
      4,
      source.root.implicitTiling!.metadataSchema,
    );
    const snapshot = {
      ...subtreeSnapshot(subtree),
      entries: [
        {
          id: "subtree/0/0/0",
          url: "/fixture/subtrees/0/0/0.subtree",
          status: "ready" as const,
          attempt: 1,
        },
      ],
    } satisfies SubtreeStoreSnapshot;
    const result = select(
      source.root,
      {},
      {
        subtrees: snapshot,
        // Degenerate planes deliberately disable culling here; this is a wire
        // contract test, while ordinary traversal tests pin frustum behavior.
        camera: view(
          [0, 0, 0],
          Array.from({ length: 16 }, () => 0),
        ),
      },
    );

    expect(result.neededSubtreeRequests).toEqual([]);
    expect(result.desiredTileIds).toEqual([
      "root/0",
      "root/1",
      "root/2",
      "root/3",
    ]);
    expect(
      result.desiredTileIds.map((id) => result.tileById.get(id)?.contentUri),
    ).toEqual([
      "content/1/0/0.glb",
      "content/1/1/0.glb",
      "content/1/0/1.glb",
      "content/1/1/1.glb",
    ]);
  });
  it("requests a visible unknown implicit root and does not pretend it is empty", () => {
    const result = select(implicitRoot());
    expect(result.desiredTileIds).toEqual([]);
    expect(result.requestedTileIds).toEqual([]);
    expect(result.neededSubtreeRequests).toEqual([
      {
        id: "subtree/0/0/0",
        url: "/tiles/subtrees/0/0/0.subtree",
      },
    ]);
  });

  it("applies the tileset root transform while its subtree is unknown", () => {
    const transform = matrix(-100);
    const root: TilesetTile = {
      ...implicitRoot(),
      boundingVolume: {
        center: [100, 0, 0],
        halfAxes: [1, 0, 0, 0, 1, 0, 0, 0, 1],
      },
      transform,
      worldTransform: transform,
    };

    const result = select(root);

    expect(result.culledTileIds).toEqual([]);
    expect(result.neededSubtreeRequests).toEqual([
      {
        id: "subtree/0/0/0",
        url: "/tiles/subtrees/0/0/0.subtree",
      },
    ]);
  });

  it("does not fetch an implicit subtree outside the view frustum", () => {
    const result = select(implicitRoot(), {}, { modelMatrix: matrix(100) });
    expect(result.culledTileIds).toEqual(["root"]);
    expect(result.neededSubtreeRequests).toEqual([]);
  });

  it("materializes known implicit content, halves error, and uses metadata bounds", () => {
    const boxes = new Map(
      Array.from(
        { length: 5 },
        (_value, index) =>
          [index, [0, 0, 0, 1 + index / 10, 0, 0, 0, 1, 0, 0, 0, 1]] as const,
      ),
    );
    const subtree = parseSubtree(
      makeSubtreeFixture({ boxes }),
      2,
      SUBTREE_METADATA_SCHEMA,
    );
    const result = select(
      implicitRoot(),
      {},
      {
        subtrees: subtreeSnapshot(subtree),
        camera: view([0, 0, 2]),
      },
    );

    expect(result.desiredTileIds).toEqual([
      "root/0",
      "root/1",
      "root/2",
      "root/3",
    ]);
    expect(result.neededSubtreeRequests).toEqual([]);
    expect(result.tileById.get("root/3")).toMatchObject({
      geometricError: 8,
      contentUri: "content/1/1/1.glb",
      boundingVolume: { halfAxes: [1.4, 0, 0, 0, 1, 0, 0, 0, 1] },
    });
  });

  it("holds the nearest submitted parent at an unknown child-subtree boundary", () => {
    const childSubtrees = Array.from({ length: 16 }, () => false);
    childSubtrees[0] = true;
    const subtree = parseSubtree(
      makeSubtreeFixture({ childSubtreeAvailability: childSubtrees }),
      2,
      SUBTREE_METADATA_SCHEMA,
    );
    const result = select(
      implicitRoot(),
      {
        root: "submitted",
        "root/0": "submitted",
        "root/1": "submitted",
        "root/2": "submitted",
        "root/3": "submitted",
      },
      { subtrees: subtreeSnapshot(subtree), camera: view([0, 0, 2]) },
    );

    expect(result.neededSubtreeRequests).toContainEqual({
      id: "subtree/2/0/0",
      url: "/tiles/subtrees/2/0/0.subtree",
    });
    expect(result.drawnTileIds).toContain("root/0");
  });

  it("keeps a submitted fallback across a contentless ancestor and loaded child-subtree boundary", () => {
    const childSubtrees = Array.from({ length: 16 }, () => false);
    childSubtrees[0] = true;
    const rootSubtree = parseSubtree(
      makeSubtreeFixture({
        contentAvailability: [true, false, true, true, true],
        childSubtreeAvailability: childSubtrees,
      }),
      2,
      SUBTREE_METADATA_SCHEMA,
    );
    const childSubtree = parseSubtree(
      makeSubtreeFixture({
        tileAvailability: [true, false, false, false, false],
        contentAvailability: [true, false, false, false, false],
      }),
      2,
      SUBTREE_METADATA_SCHEMA,
    );
    const snapshot = {
      ...subtreeSnapshot(rootSubtree),
      selected: 2,
      ready: 2,
      cached: 2,
      cachedBytes: rootSubtree.byteLength + childSubtree.byteLength,
      entries: [
        {
          id: "subtree/0/0/0",
          url: "/tiles/subtrees/0/0/0.subtree",
          status: "ready" as const,
          attempt: 1,
        },
        {
          id: "subtree/2/0/0",
          url: "/tiles/subtrees/2/0/0.subtree",
          status: "ready" as const,
          attempt: 1,
        },
      ],
      subtreeById: new Map([
        ["subtree/0/0/0", rootSubtree],
        ["subtree/2/0/0", childSubtree],
      ]),
    } satisfies SubtreeStoreSnapshot;
    const result = select(
      implicitRoot(),
      { root: "submitted" },
      {
        subtrees: snapshot,
        camera: view(
          [0, 0, 2],
          Array.from({ length: 16 }, () => 0),
        ),
      },
    );

    expect(result.neededSubtreeRequests).toEqual([]);
    expect(result.desiredTileIds).toContain("root/0/0");
    expect(result.desiredTileIds).not.toContain("root/0");
    expect(result.requestedTileIds).toContain("root/0/0");
    expect(result.drawnTileIds).toEqual(["root"]);
  });
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
