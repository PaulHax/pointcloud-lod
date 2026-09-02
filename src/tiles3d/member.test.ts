import { beforeEach, describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";

import type { CameraView } from "../camera";
import { createMemoryPool } from "../memoryPool";
import { createSubmissionScheduler } from "../submissionScheduler";
import type { StreamedMemberContext } from "../streamedMember";
import {
  TileUnsupportedExtensionError,
  type DecodeTileRequest,
  type DecodedTileContent,
} from "./decode";
import { createTiles3dMember } from "./member";
import type { Tiles3dMemberConfig, Tiles3dMemberStats } from "./memberTypes";
import { actorInstances, resetStubs } from "../../test/stubs/vtkStub";

const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const view: CameraView = {
  projection: "perspective",
  viewProj: identity,
  position: [0, 0, -1],
  fovY: Math.PI / 2,
  viewportWidthCssPx: 100,
  viewportHeightCssPx: 100,
};

const tile = (
  uri: string,
  geometricError: number,
  children: unknown[] = [],
) => ({
  geometricError,
  refine: "REPLACE",
  boundingVolume: { box: [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1] },
  content: { uri },
  children,
});

const contentlessTile = (geometricError: number, children: unknown[]) => ({
  geometricError,
  refine: "REPLACE",
  boundingVolume: { box: [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1] },
  children,
});

const document = (withChildren = false) => ({
  asset: { version: "1.1" },
  geometricError: 8,
  root: tile(
    "root.glb",
    withChildren ? 8 : 0,
    withChildren ? [tile("left.glb", 0), tile("right.glb", 0)] : [],
  ),
});

const decoded = (z = 0): DecodedTileContent => ({
  origin: [0, 0, z],
  primitives: [
    {
      positions: new Float32Array([-0.5, -0.5, 0, 0.5, -0.5, 0, 0, 0.5, 0]),
      indices: new Uint16Array([0, 1, 2]),
      material: {
        baseColorFactor: [1, 1, 1, 1],
        raw: {
          version: 1,
          kind: "gltf-material",
          alphaMode: "OPAQUE",
          alphaCutoff: 0.5,
          doubleSided: false,
          unlit: false,
          metallicFactor: 1,
          roughnessFactor: 1,
          emissiveFactor: [0, 0, 0],
        },
      },
    },
  ],
  byteEstimate: { geometry: 42, textures: 0 },
});

const settle = async (): Promise<void> => {
  for (let index = 0; index < 6; index += 1) await Promise.resolve();
  await new Promise((resolve) => setTimeout(resolve, 0));
};

const harness = (withChildren = false, schedulerBytes = 1024) => {
  const renderer = { addActor: vi.fn(), removeActor: vi.fn() };
  const scheduleRender = vi.fn();
  const decodedRequests: DecodeTileRequest[] = [];
  const contentRequests: string[] = [];
  const memory = createMemoryPool({ totalBytes: 4096 });
  const submissions = createSubmissionScheduler({
    scheduleRender,
    maxBytesPerFrame: schedulerBytes,
    maxTimeMsPerFrame: 100,
    now: () => 0,
  });
  const context: StreamedMemberContext = {
    renderer,
    scheduleRender,
    memory,
    submissions,
    textureCapabilities: {
      capabilityKey: "compressed-texture-v1:astc-4x4",
      compressedFormats: ["astc-4x4"],
    },
    workers: {
      size: 3,
      decode: (request) => {
        decodedRequests.push(request);
        return { promise: Promise.resolve(decoded()), cancel: vi.fn() };
      },
    },
    devicePixelRatio: 1,
    onWorkChange: vi.fn(),
  };
  const config: Tiles3dMemberConfig = {
    endpoint: "/tiles",
    revision: "r1",
    tilesetToScene: identity,
    maximumScreenSpaceErrorPx: 4,
    minConcurrency: 1,
    maxConcurrency: 3,
    fetchTileset: async () => ({
      ok: true,
      status: 200,
      statusText: "OK",
      json: async () => document(withChildren),
    }),
    fetchContent: async (url) => {
      contentRequests.push(url);
      return {
        ok: true,
        status: 200,
        statusText: "OK",
        arrayBuffer: async () => new ArrayBuffer(8),
      };
    },
  };
  return {
    context,
    config,
    renderer,
    memory,
    submissions,
    decodedRequests,
    contentRequests,
  };
};

describe("createTiles3dMember", () => {
  beforeEach(resetStubs);

  it.each([104, 110])(
    "recovers mixed pooled and GPU-evicted children within a %i-byte budget",
    async (budget) => {
      const h = harness(true, 1024);
      const member = createTiles3dMember(h.context, h.config);
      const drain = async () => {
        for (let i = 0; i < 15; i++) {
          await settle();
          while (h.submissions.hasPending()) h.submissions.prepareFrame();
          if (member.governorInputs().work.operations === 0) break;
        }
      };
      member.applyAllocation({
        qualityFraction: 1,
        memoryBudgetBytes: 4096,
        regime: "stationary",
      });
      member.setCamera(view);
      await drain();
      expect(member.stats()).toMatchObject({
        memoryConstrained: false,
        renderer: { drawnTiles: 2 },
      });
      member.setCamera({ ...view, position: [0, 0, -1000] });
      await drain();
      expect(member.stats()).toMatchObject({
        memoryConstrained: false,
        renderer: { drawnTiles: 1 },
      });
      member.applyAllocation({
        qualityFraction: 1,
        memoryBudgetBytes: budget,
        regime: "stationary",
      });
      await drain();
      const away = [...identity];
      away[12] = 10;
      member.setCamera({ ...view, viewProj: away });
      await drain();
      expect(member.stats()).toMatchObject({
        memoryConstrained: false,
        renderer: { drawnTiles: 0 },
      });
      member.setCamera(view);
      await drain();
      expect(member.stats()).toMatchObject({
        memoryConstrained: false,
        renderer: { drawnTiles: 2, drawnTileIds: ["root/0", "root/1"] },
      });
      expect(member.pick(view, 50, 50)?.status).toBe("hit");
      member.dispose();
    },
  );

  it("batches drawn coverage when many pooled tiles are selected again", async () => {
    const h = harness();
    const manifest = {
      ...document(),
      root: tile(
        "root.glb",
        8,
        Array.from({ length: 24 }, (_, i) => tile(`${i}.glb`, 0)),
      ),
    };
    const member = createTiles3dMember(h.context, {
      ...h.config,
      fetchTileset: async () => ({
        ok: true,
        status: 200,
        statusText: "OK",
        json: async () => manifest,
      }),
    });
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 4096,
      regime: "stationary",
    });
    member.setCamera(view);
    for (
      let i = 0;
      i < 30 &&
      (member.stats() as Tiles3dMemberStats).renderer.drawnTiles !== 24;
      i++
    ) {
      await settle();
      while (h.submissions.hasPending()) h.submissions.prepareFrame();
    }
    expect(member.stats()).toMatchObject({ renderer: { drawnTiles: 24 } });
    member.setCamera({ ...view, position: [0, 0, -1000] });
    expect(member.stats()).toMatchObject({ renderer: { drawnTiles: 1 } });
    const before = (member.stats() as Tiles3dMemberStats).selectionPasses;
    const decodeCount = h.decodedRequests.length;
    member.setCamera(view);
    expect(member.stats()).toMatchObject({ renderer: { drawnTiles: 24 } });
    expect(member.pick(view, 50, 50)?.status).toBe("hit");
    expect(h.decodedRequests).toHaveLength(decodeCount);
    expect(h.submissions.hasPending()).toBe(false);
    // One request-selection pass and one coverage pass per refresh, including
    // the follow-up that releases the ancestor after pooled admissions.
    expect(
      (member.stats() as Tiles3dMemberStats).selectionPasses - before,
    ).toBeLessThanOrEqual(4);
    member.dispose();
  });

  it("drives implicit subtree arrival into full quadrant content selection", async () => {
    const h = harness();
    const directory = new URL(
      "../../test/fixtures/tiles3d-implicit/",
      import.meta.url,
    );
    const manifest = JSON.parse(
      readFileSync(new URL("tileset.json", directory), "utf8"),
    );
    // The checked-in wire fixture's ECEF placement is independent of this
    // lifecycle test; keep its producer metadata boxes but render in identity.
    manifest.root.transform = identity;
    const subtreeBytes = readFileSync(
      new URL("subtrees/0/0/0.subtree", directory),
    );
    const subtreeFetch = vi.fn(async () => ({
      ok: true,
      status: 200,
      statusText: "OK",
      arrayBuffer: async () =>
        subtreeBytes.buffer.slice(
          subtreeBytes.byteOffset,
          subtreeBytes.byteOffset + subtreeBytes.byteLength,
        ),
    }));
    const member = createTiles3dMember(h.context, {
      ...h.config,
      fetchTileset: async () => ({
        ok: true,
        status: 200,
        statusText: "OK",
        json: async () => manifest,
      }),
      fetchSubtree: subtreeFetch,
    });
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 4096,
      regime: "stationary",
    });
    member.setCamera({
      ...view,
      viewProj: Array.from({ length: 16 }, () => 0),
      position: [0, 0, 0],
    });
    await settle();

    expect(subtreeFetch).toHaveBeenCalledOnce();
    expect(h.contentRequests).toEqual([
      "/tiles/content/0/0/0.glb",
      "/tiles/content/1/0/0.glb",
      "/tiles/content/1/1/0.glb",
      "/tiles/content/1/0/1.glb",
      "/tiles/content/1/1/1.glb",
    ]);
    expect(member.stats()).toMatchObject({
      selectedTiles: 4,
      requestedTiles: 5,
      subtrees: { cached: 1, failed: 0, workPending: false },
      queue: { ready: 5, workPending: false },
    });
    expect(member.governorInputs()).toMatchObject({
      physicalHierarchyOperations: 0,
      work: { operations: expect.any(Number) },
    });
    member.dispose();
  });

  it("weighs itself by how far its root sits above its own error cutoff", async () => {
    // The governor can only read importance as a relative weight, so it has to
    // arrive on the shared [0, 1] scale. A constant here is what let a mesh be
    // water-filled down to the quality floor beneath a point cloud reporting
    // raw pixels.
    const h = harness(true);
    const member = createTiles3dMember(h.context, {
      ...h.config,
      maximumScreenSpaceErrorPx: 16,
    });
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 4096,
      regime: "stationary",
    });
    member.setCamera(view);
    await settle();

    const importance = member.governorInputs().projectedImportance;
    expect(importance).toBeGreaterThan(0);
    expect(importance).toBeLessThanOrEqual(1);

    member.setActive(false);
    expect(member.governorInputs().projectedImportance).toBe(0);
  });

  it("ignores restatements of an unchanged camera and anchor matrix", async () => {
    const h = harness(true);
    let traversals = 0;
    const member = createTiles3dMember(h.context, {
      ...h.config,
      fetchTileset: async () => {
        return {
          ok: true,
          status: 200,
          statusText: "OK",
          json: async () => document(true),
        };
      },
      fetchContent: async () => {
        traversals += 1;
        return {
          ok: true,
          status: 200,
          statusText: "OK",
          arrayBuffer: async () => new ArrayBuffer(8),
        };
      },
    });
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 4096,
      regime: "stationary",
    });
    member.setCamera(view);
    await settle();
    const afterFirst = traversals;

    // The host restates both every pre-paint pass, whether or not they moved.
    for (let index = 0; index < 5; index += 1) {
      member.setCamera({ ...view });
      member.setModelMatrix(null);
    }
    await settle();
    expect(traversals).toBe(afterFirst);

    const selectionPasses = (member.stats() as Tiles3dMemberStats)
      .selectionPasses;
    for (let serial = 0; serial < 5; serial += 1) member.prepareFrame();
    expect((member.stats() as Tiles3dMemberStats).selectionPasses).toBe(
      selectionPasses,
    );

    member.dispose();
  });

  it("implements the streamed member lifecycle without a second memory owner", async () => {
    const h = harness();
    const member = createTiles3dMember(h.context, h.config);
    expect(h.memory.memberCount()).toBe(0);
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 4096,
      regime: "stationary",
    });
    member.setCamera(view);
    await settle();
    expect(member.governorInputs()).toMatchObject({
      // This fixture's root has geometricError 0 — already exact, so it asks
      // the view for no refinement quality even though it is on screen.
      projectedImportance: 0,
      qualityDemand: 1,
      work: { operations: 1 },
    });
    expect(h.decodedRequests[0]).toMatchObject({
      revision: "r1",
      textureCapabilities: h.context.textureCapabilities,
      contentUrl: "http://localhost/tiles/root.glb",
      dependencyRootUrl: "http://localhost/tiles/",
    });
    h.submissions.prepareFrame();
    expect(h.renderer.addActor).toHaveBeenCalledOnce();
    expect(member.pick(view, 50, 50)).toMatchObject({
      status: "hit",
      rayDepth: 1,
    });
    expect(member.occlusionDepth(view, 50, 50)).toEqual({
      status: "hit",
      rayDepth: 1,
    });

    member.setModelMatrix([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.25, 0, 0, 1]);
    expect(actorInstances[0]?.userMatrix?.[12]).toBe(0.25);
    member.setDevicePixelRatio(2);
    member.beginInteraction();
    member.endInteraction();
    member.setConfig({ ...h.config, maximumScreenSpaceErrorPx: 8 });
    expect(member.stats()).toMatchObject({
      kind: "tiles3d",
      devicePixelRatio: 2,
      interactionDepth: 0,
      configGeneration: 1,
      verticalExaggeration: 1,
      verticalPivotZ: 0,
      maximumScreenSpaceErrorPx: 8,
      renderer: { drawnTiles: 1, drawnTriangles: 1 },
    });

    member.setActive(false);
    expect(actorInstances[0]?.visibility).toBe(false);
    expect(member.governorInputs().work.operations).toBe(0);
    member.setActive(true);
    member.prepareFrame();
    member.dispose();
    expect(actorInstances[0]?.deleted).toBe(true);
    expect((member.stats() as Tiles3dMemberStats).sourceState).toBe("disposed");
  });

  it("keeps exact submitted picking after the decoded CPU cache evicts", async () => {
    const h = harness();
    const member = createTiles3dMember(h.context, h.config);
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 4096,
      regime: "stationary",
    });
    member.setCamera(view);
    await settle();
    h.submissions.prepareFrame();
    expect(member.stats()).toMatchObject({
      queue: { decodedBytes: decoded().byteEstimate.geometry },
      // Submitted chunks own deindexed positions plus vtk Uint32 cell records;
      // this charged representation is separate from the worker cache.
      renderer: { residentGeometryBytes: 52, residentBytes: 52 },
    });

    member.setConfig({ ...h.config, cacheBytes: 0 });
    expect(member.stats()).toMatchObject({
      queue: { decodedBytes: 0 },
      renderer: {
        submittedTiles: 1,
        drawnTiles: 1,
        residentGeometryBytes: 52,
        residentBytes: 52,
      },
    });
    expect(member.pick(view, 50, 50)).toMatchObject({
      status: "hit",
      rayDepth: 1,
    });

    member.dispose();
  });

  it("preserves nested decode causes in public diagnostics", async () => {
    const h = harness(false);
    h.context.workers.decode = () => ({
      promise: Promise.reject(new Error("codec unavailable")),
      cancel: vi.fn(),
    });
    const member = createTiles3dMember(h.context, {
      ...h.config,
      maxAttempts: 1,
    });
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 4096,
      regime: "stationary",
    });
    member.setCamera(view);
    await settle();

    expect((member.stats() as Tiles3dMemberStats).lastError).toMatch(
      /content decode failed: codec unavailable/,
    );
  });

  it("holds a submitted parent until every child crosses paced admission", async () => {
    const h = harness(true, 128);
    const member = createTiles3dMember(h.context, h.config);
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 4096,
      regime: "stationary",
    });
    member.setCamera(view);
    await settle();
    expect(h.submissions.stats().queuedJobs).toBe(3);
    h.submissions.prepareFrame();
    expect((member.stats() as Tiles3dMemberStats).renderer).toMatchObject({
      drawnTiles: 1,
      submittedTiles: 2,
    });
    h.submissions.prepareFrame();
    expect((member.stats() as Tiles3dMemberStats).renderer).toMatchObject({
      drawnTiles: 2,
      submittedTiles: 2,
    });
    expect(h.renderer.removeActor).toHaveBeenCalledOnce();
    member.dispose();
  });

  it("loads and picks descendants of a contentless root without blank work", async () => {
    const h = harness();
    const rootContentless = {
      asset: { version: "1.1" },
      geometricError: 8,
      root: contentlessTile(0, [tile("left.glb", 0), tile("right.glb", 0)]),
    };
    const member = createTiles3dMember(h.context, {
      ...h.config,
      fetchTileset: async () => ({
        ok: true,
        status: 200,
        statusText: "OK",
        json: async () => rootContentless,
      }),
    });
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 4096,
      regime: "stationary",
    });
    member.setCamera(view);
    await settle();

    expect(h.contentRequests).toEqual(["/tiles/left.glb", "/tiles/right.glb"]);
    expect(h.decodedRequests.map((request) => request.contentUrl)).toEqual([
      "http://localhost/tiles/left.glb",
      "http://localhost/tiles/right.glb",
    ]);
    while (h.submissions.hasPending()) h.submissions.prepareFrame();
    expect(member.stats()).toMatchObject({
      selectedTiles: 2,
      requestedTiles: 2,
      renderer: { submittedTiles: 2, drawnTiles: 2 },
    });
    expect(member.pick(view, 50, 50)?.status).toBe("hit");
    member.dispose();
  });

  it("keeps a real parent fallback across a contentless intermediate", async () => {
    const h = harness(false, 128);
    const intermediateContentless = {
      asset: { version: "1.1" },
      geometricError: 8,
      root: tile("root.glb", 8, [
        contentlessTile(0, [tile("left.glb", 0), tile("right.glb", 0)]),
      ]),
    };
    const member = createTiles3dMember(h.context, {
      ...h.config,
      fetchTileset: async () => ({
        ok: true,
        status: 200,
        statusText: "OK",
        json: async () => intermediateContentless,
      }),
    });
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 4096,
      regime: "stationary",
    });
    member.setCamera(view);
    await settle();

    expect(h.contentRequests).toEqual([
      "/tiles/root.glb",
      "/tiles/left.glb",
      "/tiles/right.glb",
    ]);
    h.submissions.prepareFrame();
    expect(member.stats()).toMatchObject({
      selectedTiles: 2,
      requestedTiles: 3,
      renderer: { submittedTiles: 2, drawnTiles: 1 },
    });
    h.submissions.prepareFrame();
    expect(member.stats()).toMatchObject({
      renderer: { submittedTiles: 2, drawnTiles: 2 },
    });
    expect(member.pick(view, 50, 50)?.status).toBe("hit");
    member.dispose();
  });

  it("cancels selection work and partial submissions without stale callbacks", async () => {
    const h = harness(false, 16);
    const member = createTiles3dMember(h.context, h.config);
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 4096,
      regime: "stationary",
    });
    member.setCamera(view);
    await settle();
    // Shrink before the queued geometry admission crosses the boundary.
    expect(h.renderer.addActor).not.toHaveBeenCalled();
    member.setActive(false);
    while (h.submissions.hasPending()) h.submissions.prepareFrame();
    expect(h.renderer.addActor).not.toHaveBeenCalled();
    expect(actorInstances.every((actor) => actor.deleted)).toBe(true);
    member.dispose();
  });

  it("maps quality to SSE and bounded concurrency and contains source errors", async () => {
    const h = harness();
    const onError = vi.fn();
    const member = createTiles3dMember(h.context, {
      ...h.config,
      fetchTileset: async () => {
        throw new Error("offline");
      },
      onError,
    });
    member.applyAllocation({
      qualityFraction: 0.25,
      memoryBudgetBytes: 2048,
      regime: "moving",
    });
    member.setCamera(view);
    await settle();
    expect(onError).toHaveBeenCalledOnce();
    expect(member.governorInputs().physicalHierarchyOperations).toBe(0);
    expect(member.stats()).toMatchObject({
      sourceState: "failed",
      sseMultiplier: 4,
      effectiveScreenSpaceErrorPx: 16,
      errorCount: 1,
    });
    member.dispose();
  });

  it("surfaces typed decode failures through queue and member telemetry", async () => {
    const h = harness();
    const onError = vi.fn();
    h.context.workers.decode = () => ({
      promise: Promise.reject(
        new TileUnsupportedExtensionError(
          "/tiles/root.glb",
          "EXT_mesh_gpu_instancing",
        ),
      ),
      cancel: vi.fn(),
    });
    const member = createTiles3dMember(h.context, {
      ...h.config,
      maxAttempts: 1,
      onError,
    });
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 4096,
      regime: "stationary",
    });
    member.setCamera(view);
    await settle();

    const error = onError.mock.calls[0]?.[0];
    expect(error).toMatchObject({ name: "ContentQueueDecodeError" });
    expect(error.cause).toMatchObject({
      name: "TileUnsupportedExtensionError",
      stage: "profile",
      extension: "EXT_mesh_gpu_instancing",
    });
    expect(member.stats()).toMatchObject({
      errorCount: 1,
      lastError: expect.stringContaining("unsupported required glTF extension"),
      queue: { failed: 1, workPending: false },
    });
    member.dispose();
  });

  it("retries ready decoded content when its GPU allocation grows", async () => {
    const h = harness();
    const member = createTiles3dMember(h.context, h.config);
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 1,
      regime: "stationary",
    });
    member.setCamera(view);
    await settle();
    expect(h.submissions.stats().queuedJobs).toBe(0);
    expect((member.stats() as Tiles3dMemberStats).errorCount).toBe(0);
    expect((member.stats() as Tiles3dMemberStats).lastError).toBeNull();
    expect(
      (member.stats() as Tiles3dMemberStats).queue?.decodedBytes,
    ).toBeGreaterThan(1);
    // Blocked on a budget increase, not on work: nothing will progress until
    // something outside the member acts. Claiming pending work here stops the
    // view governor sampling capacity for every member in the view, so the
    // whole view's adaptive quality freezes behind one over-budget tileset.
    expect(member.governorInputs().work.operations).toBe(0);
    expect(member.stats()).toMatchObject({
      memoryConstrained: true,
      irreducibleBudget: true,
      renderer: { residentBytes: 0 },
    });
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 4096,
      regime: "stationary",
    });
    // A bigger allowance clears the latch and the work resumes on its own.
    expect((member.stats() as Tiles3dMemberStats).irreducibleBudget).toBe(
      false,
    );
    expect(h.submissions.stats().queuedJobs).toBeGreaterThan(0);
    expect(member.governorInputs().work.operations).toBeGreaterThan(0);
    h.submissions.prepareFrame();
    expect(h.renderer.addActor).toHaveBeenCalledOnce();
    member.dispose();
  });

  it("swaps submitted descendants for a fitting parent before enforcing a smaller byte share", async () => {
    const h = harness(true, 128);
    const member = createTiles3dMember(h.context, h.config);
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 4096,
      regime: "stationary",
    });
    member.setCamera(view);
    await settle();
    while (h.submissions.hasPending()) h.submissions.prepareFrame();
    expect(member.stats()).toMatchObject({
      memoryConstrained: false,
      renderer: { drawnTiles: 2, submittedTiles: 2 },
    });

    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 60,
      regime: "stationary",
    });
    await settle();
    expect(member.stats()).toMatchObject({
      memoryConstrained: true,
      renderer: { drawnTiles: 2, submittedTiles: 2, pendingTiles: 1 },
    });
    expect(member.governorInputs().work.operations).toBeGreaterThan(0);

    h.submissions.prepareFrame();
    const converged = member.stats() as Tiles3dMemberStats;
    expect(converged.renderer).toMatchObject({
      drawnTiles: 1,
      submittedTiles: 1,
      pendingTiles: 0,
    });
    expect(converged.renderer.residentBytes).toBeLessThanOrEqual(60);
    expect(h.renderer.removeActor).toHaveBeenCalledTimes(3);
    member.dispose();
  });

  it("credits a submitted parent when the complete child frontier fits without the transient peak", async () => {
    const h = harness(true, 88);
    (h.context.workers as any).decode = (request: DecodeTileRequest) => {
      h.decodedRequests.push(request);
      const value = decoded();
      if (request.contentUrl.endsWith("/root.glb")) {
        value.primitives[0]!.normals = new Float32Array([
          0, 0, 1, 0, 0, 1, 0, 0, 1,
        ]);
      }
      return { promise: Promise.resolve(value), cancel: vi.fn() };
    };
    const member = createTiles3dMember(h.context, h.config);
    member.applyAllocation({
      qualityFraction: 1,
      // The parent retains 88 bytes and each child 52. Neither child can
      // coexist with the parent under 110, but the final 104-byte frontier fits.
      memoryBudgetBytes: 110,
      regime: "stationary",
    });
    member.setCamera({
      ...view,
      projection: "orthographic",
      parallelScale: 1e9,
    });
    await settle();
    while (h.submissions.hasPending()) {
      h.submissions.prepareFrame();
      await settle();
    }
    expect(member.stats()).toMatchObject({
      memoryConstrained: false,
      renderer: { drawnTiles: 1, submittedTiles: 1, residentBytes: 88 },
    });

    member.setCamera(view);
    await settle();
    expect(member.stats()).toMatchObject({
      renderer: { drawnTiles: 1, submittedTiles: 1, pendingTiles: 2 },
    });
    h.submissions.prepareFrame();
    // One child is realized but held behind the group barrier. The parent is
    // still the only resident/drawn frontier, so there is no visible gap or
    // over-budget submitted peak.
    expect(member.stats()).toMatchObject({
      renderer: {
        drawnTiles: 1,
        submittedTiles: 1,
        pendingTiles: 2,
        residentBytes: 88,
      },
    });
    h.submissions.prepareFrame();
    await settle();
    expect(member.stats()).toMatchObject({
      memoryConstrained: false,
      renderer: { drawnTiles: 2, submittedTiles: 2, residentBytes: 104 },
    });
    member.dispose();
  });

  it("retries a memory-constrained member when the camera changes its desired frontier", async () => {
    const h = harness(true, 128);
    const member = createTiles3dMember(h.context, h.config);
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 60,
      regime: "stationary",
    });
    member.setCamera(view);
    await settle();
    while (h.submissions.hasPending()) {
      h.submissions.prepareFrame();
      await settle();
    }
    expect(member.stats()).toMatchObject({
      memoryConstrained: true,
      renderer: { drawnTiles: 1, residentBytes: 52 },
    });

    member.setCamera({
      ...view,
      projection: "orthographic",
      parallelScale: 1e9,
    });
    await settle();
    expect(member.stats()).toMatchObject({
      memoryConstrained: false,
      renderer: { drawnTiles: 1, residentBytes: 52 },
    });
    member.dispose();
  });

  it("surfaces permanent adapter rejection once without retrying forever", async () => {
    const h = harness();
    (h.context.workers as any).decode = (request: DecodeTileRequest) => {
      h.decodedRequests.push(request);
      const invalid = decoded();
      invalid.primitives[0]!.material.baseColorTexture = {
        kind: "rgba",
        rgba: new Uint8Array(2048),
        width: 32,
        height: 16,
        colorSpace: "srgb",
        sampler: {
          magFilter: 9729,
          minFilter: 9987,
          wrapS: 10497,
          wrapT: 10497,
        },
      };
      return { promise: Promise.resolve(invalid), cancel: vi.fn() };
    };
    const member = createTiles3dMember(h.context, h.config);
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 4096,
      regime: "stationary",
    });
    member.setCamera(view);
    await settle();
    expect((member.stats() as Tiles3dMemberStats).errorCount).toBe(1);
    expect((member.stats() as Tiles3dMemberStats).lastError).toMatch(
      /texture is 2048 bytes.*submission cap/,
    );
    expect(member.governorInputs()).toMatchObject({
      work: { operations: 0 },
      physicalTileOperations: 0,
    });
    member.applyAllocation({
      qualityFraction: 0.5,
      memoryBudgetBytes: 4096,
      regime: "moving",
    });
    await settle();
    expect(h.decodedRequests).toHaveLength(1);
    expect((member.stats() as Tiles3dMemberStats).errorCount).toBe(1);
    member.dispose();
  });

  it("drops old submitted/pick currency before revision, endpoint, or placement reload", async () => {
    const h = harness();
    const member = createTiles3dMember(h.context, h.config);
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 4096,
      regime: "stationary",
    });
    member.setCamera(view);
    await settle();
    h.submissions.prepareFrame();
    expect(member.pick(view, 50, 50)?.status).toBe("hit");
    member.setConfig({
      ...h.config,
      endpoint: "/tiles-next",
      revision: "r2",
      tilesetToScene: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0, 1],
    });
    expect(h.renderer.removeActor).toHaveBeenCalledOnce();
    expect(member.pick(view, 50, 50)).toEqual({ status: "miss" });
    expect(member.stats()).toMatchObject({
      sourceState: "loading",
      revision: "r2",
      queue: null,
      renderer: { submittedTiles: 0, drawnTiles: 0 },
    });
    member.dispose();
  });

  it("places vertical exaggeration after the tile origin without changing the live anchor", async () => {
    const h = harness();
    const tilesetToScene = [
      0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0.25, 0.5, 0.25, 1,
    ];
    const member = createTiles3dMember(h.context, {
      ...h.config,
      tilesetToScene,
      verticalExaggeration: 2,
      verticalPivotZ: 0.5,
    });
    const anchor = [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] as const;
    member.setModelMatrix(anchor);
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 4096,
      regime: "stationary",
    });
    member.setCamera(view);
    await settle();

    // Decoded vertices stay in unexaggerated scene coordinates: exaggeration is a
    // render-time placement, so changing it never re-fetches or re-decodes.
    expect(h.decodedRequests[0]?.tilesetToScene).toEqual(tilesetToScene);
    h.submissions.prepareFrame();
    expect(actorInstances[0]?.userMatrix).toEqual([
      0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 2, 0, 0, 0, -0.5, 1,
    ]);
    expect(anchor).toEqual([0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    expect(member.stats()).toMatchObject({
      configGeneration: 1,
      verticalExaggeration: 2,
      verticalPivotZ: 0.5,
    });

    const decodedRequestCount = h.decodedRequests.length;
    member.setConfig({
      ...h.config,
      tilesetToScene,
      verticalExaggeration: 4,
      verticalPivotZ: -0.25,
    });
    expect(h.renderer.removeActor).not.toHaveBeenCalled();
    expect(member.stats()).toMatchObject({
      configGeneration: 1,
      verticalExaggeration: 4,
      verticalPivotZ: -0.25,
      renderer: { submittedTiles: 1 },
    });
    expect(actorInstances[0]?.userMatrix).toEqual([
      0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0.75, 1,
    ]);
    await settle();
    expect(h.decodedRequests).toHaveLength(decodedRequestCount);
    member.dispose();
  });

  it("re-places rather than reloads when a host rebuilds equal wasm URLs each push", async () => {
    // Hosts resolve wasm URLs against the page every time they push config, so
    // the object is fresh on each call. Decode inputs are its URLs, not its
    // identity: an equal set must not look like a decode change.
    const wasmUrls = () => ({
      draco: { wrapperUrl: "/draco.js", wasmUrl: "/draco.wasm" },
      basis: { encoderUrl: "/basis.js", wasmUrl: "/basis.wasm" },
    });
    const h = harness();
    const member = createTiles3dMember(h.context, {
      ...h.config,
      wasm: wasmUrls(),
    });
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 4096,
      regime: "stationary",
    });
    member.setCamera(view);
    await settle();
    h.submissions.prepareFrame();
    const decodedRequestCount = h.decodedRequests.length;
    expect(decodedRequestCount).toBeGreaterThan(0);

    member.setConfig({
      ...h.config,
      wasm: wasmUrls(),
      verticalExaggeration: 3,
    });

    expect(h.renderer.removeActor).not.toHaveBeenCalled();
    expect(member.stats()).toMatchObject({
      configGeneration: 1,
      sourceState: "ready",
      verticalExaggeration: 3,
      renderer: { submittedTiles: 1 },
    });
    await settle();
    expect(h.decodedRequests).toHaveLength(decodedRequestCount);
    member.dispose();
  });

  it("reloads when the wasm URLs a tile decodes with actually change", async () => {
    const h = harness();
    const member = createTiles3dMember(h.context, {
      ...h.config,
      wasm: { draco: { wrapperUrl: "/draco.js", wasmUrl: "/draco.wasm" } },
    });
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 4096,
      regime: "stationary",
    });
    member.setCamera(view);
    await settle();
    h.submissions.prepareFrame();

    member.setConfig({
      ...h.config,
      wasm: { draco: { wrapperUrl: "/draco.js", wasmUrl: "/draco-v2.wasm" } },
    });

    expect(h.renderer.removeActor).toHaveBeenCalledOnce();
    expect(member.stats()).toMatchObject({
      configGeneration: 2,
      sourceState: "loading",
      renderer: { submittedTiles: 0 },
    });
    member.dispose();
  });

  it("retries a requested decoded tile after allocation shrink cancels partial admission", async () => {
    const h = harness(false, 64);
    const member = createTiles3dMember(h.context, h.config);
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 4096,
      regime: "stationary",
    });
    member.setCamera(view);
    await settle();
    // Keep the tile partially queued; the shrink itself performs cancellation.
    expect((member.stats() as Tiles3dMemberStats).renderer).toMatchObject({
      pendingTiles: 1,
      submittedTiles: 0,
    });
    member.setConfig({ ...h.config, cacheBytes: 0 });
    expect((member.stats() as Tiles3dMemberStats).queue?.decodedBytes).toBe(0);
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 1,
      regime: "stationary",
    });
    await settle();
    expect((member.stats() as Tiles3dMemberStats).renderer.pendingTiles).toBe(
      0,
    );
    expect(h.decodedRequests.length).toBeGreaterThan(1);
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 4096,
      regime: "stationary",
    });
    await settle();
    while (h.submissions.hasPending()) h.submissions.prepareFrame();
    expect(h.renderer.addActor).toHaveBeenCalledOnce();
    expect((member.stats() as Tiles3dMemberStats).errorCount).toBe(0);
    member.dispose();
  });

  it("rejects non-affine and singular ECEF transforms at the factory boundary", () => {
    const h = harness();
    expect(() =>
      createTiles3dMember(h.context, {
        ...h.config,
        tilesetToScene: [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
      }),
    ).toThrow(/affine/);
    expect(() =>
      createTiles3dMember(h.context, {
        ...h.config,
        tilesetToScene: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
      }),
    ).toThrow(/invertible/);
  });

  it.each([
    ["verticalExaggeration", 0],
    ["verticalExaggeration", -1],
    ["verticalExaggeration", Number.NaN],
    ["verticalExaggeration", Number.POSITIVE_INFINITY],
    ["verticalPivotZ", Number.NaN],
    ["verticalPivotZ", Number.NEGATIVE_INFINITY],
  ] as const)("rejects invalid %s values", (field, value) => {
    const h = harness();
    expect(() =>
      createTiles3dMember(h.context, { ...h.config, [field]: value }),
    ).toThrow(new RegExp(field));
  });

  it("rejects non-number vertical configuration at the runtime boundary", () => {
    const h = harness();
    expect(() =>
      createTiles3dMember(h.context, {
        ...h.config,
        verticalExaggeration: true as unknown as number,
      }),
    ).toThrow(/verticalExaggeration/);
    expect(() =>
      createTiles3dMember(h.context, {
        ...h.config,
        verticalPivotZ: "0" as unknown as number,
      }),
    ).toThrow(/verticalPivotZ/);
    expect(() =>
      createTiles3dMember(h.context, {
        ...h.config,
        verticalExaggeration: null as unknown as number,
      }),
    ).toThrow(/verticalExaggeration/);
  });

  it("composes anchor after tilesetToScene for traversal while decode keeps anchor live", async () => {
    const h = harness();
    const member = createTiles3dMember(h.context, {
      ...h.config,
      tilesetToScene: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 0, 0, 1],
    });
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 4096,
      regime: "stationary",
    });
    member.setCamera(view);
    await settle();
    expect(h.decodedRequests).toHaveLength(0);
    member.setModelMatrix([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -10, 0, 0, 1]);
    await settle();
    expect(h.decodedRequests).toHaveLength(1);
    expect(h.decodedRequests[0]?.tilesetToScene[12]).toBe(10);
    member.dispose();
  });
});
