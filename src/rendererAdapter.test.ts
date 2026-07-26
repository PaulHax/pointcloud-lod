import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { PerspectiveCameraView } from "./camera";
import { createLodController } from "./controller";
import { keyToString, type VoxelKey } from "./octree";
import { createRendererAdapter } from "./rendererAdapter";
import type { NodeInfo, TileData, TileSource } from "./tileSource";
import {
  actorInstances,
  mapperInstances,
  polyDataInstances,
  resetStubs,
  type StubMapper,
  type StubPolyData,
} from "../test/stubs/vtkStub";

const tile = (
  origin: [number, number, number],
  pointCount = 2,
  withRgb = true,
): TileData => ({
  origin,
  positions: new Float32Array(pointCount * 3),
  rgb: withRgb ? new Uint8Array(pointCount * 3).fill(200) : undefined,
  pointCount,
});

const KEY_A = { level: 1, x: 0, y: 0, z: 0 };
const KEY_B = { level: 1, x: 1, y: 0, z: 0 };

const makeAdapter = (options?: {
  diameterCssPx?: number;
  devicePixelRatio?: number;
  visible?: boolean;
}) => {
  const renderer = { addActor: vi.fn(), removeActor: vi.fn() };
  const scheduleRender = vi.fn();
  const adapter = createRendererAdapter({
    renderer,
    scheduleRender,
    ...options,
  });
  return { adapter, renderer, scheduleRender };
};

/** Live actors, keyed by the payload they draw — the adapter's visible truth. */
const drawnPositions = (): Set<Float32Array> => {
  const positions = new Set<Float32Array>();
  for (const actor of actorInstances) {
    if (actor.deleted || !actor.visibility) continue;
    const mapper = actor.mapper as StubMapper;
    const polyData = mapper.inputData as StubPolyData;
    positions.add(polyData.points as Float32Array);
  }
  return positions;
};

beforeEach(resetStubs);
afterEach(() => vi.useRealTimers());

describe("createRendererAdapter", () => {
  it("creates one polydata/mapper/actor per added tile", () => {
    const { adapter, renderer, scheduleRender } = makeAdapter({
      diameterCssPx: 3,
      devicePixelRatio: 2,
    });
    const data = tile([10, 20, 30]);
    adapter.applyBatch({ added: [{ key: KEY_A, tile: data }], removed: [] });

    expect(adapter.stats()).toEqual({
      submittedTiles: 1,
      submittedPoints: 2,
      submittedBytes: 94,
      pooledTiles: 0,
      pooledPoints: 0,
      pooledBytes: 0,
      gpuResidentTiles: 1,
      gpuResidentPoints: 2,
      gpuResidentBytes: 94,
      resourceCeilingBytes: 256 * 1024 * 1024,
      drawnTiles: 1,
      drawnPoints: 2,
      visible: true,
      diameterCssPx: 3,
      devicePixelRatio: 2,
    });
    expect(renderer.addActor).toHaveBeenCalledTimes(1);
    expect(scheduleRender).toHaveBeenCalledTimes(1);

    expect(polyDataInstances[0]!.points).toBe(data.positions);
    expect((polyDataInstances[0]!.scalars as { values: unknown }).values).toBe(
      data.rgb,
    );
    expect(mapperInstances[0]!.inputData).toBe(polyDataInstances[0]);
    expect(mapperInstances[0]!.static).toBe(true);
    expect(mapperInstances[0]!.scaleFactor).toBe(2);
    expect(actorInstances[0]!.pointSize).toBe(3);
    // Identity base: the tile matrix is a plain translation to the origin.
    expect(actorInstances[0]!.userMatrix!.slice(12, 15)).toEqual([10, 20, 30]);
  });

  it("skips scalars for tiles without RGB", () => {
    const { adapter } = makeAdapter();
    adapter.applyBatch({
      added: [{ key: KEY_A, tile: tile([0, 0, 0], 2, false) }],
      removed: [],
    });
    expect(polyDataInstances[0]!.scalars).toBeNull();
  });

  it("composes the base matrix with each tile origin", () => {
    const { adapter } = makeAdapter();
    adapter.applyBatch({
      added: [{ key: KEY_A, tile: tile([1, 2, 3]) }],
      removed: [],
    });

    // base = translation by (10, 20, 30): composed translation adds up.
    adapter.setBaseMatrix([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 20, 30, 1]);
    expect(actorInstances[0]!.userMatrix!.slice(12, 15)).toEqual([11, 22, 33]);

    // base with a scale of 2: rotation/scale part multiplies the origin.
    adapter.setBaseMatrix([2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1]);
    expect(actorInstances[0]!.userMatrix!.slice(12, 15)).toEqual([2, 4, 6]);
    expect(actorInstances[0]!.userMatrix![0]).toBe(2);
  });

  it("does not repaint for an unchanged base matrix", () => {
    const { adapter, scheduleRender } = makeAdapter();
    adapter.applyBatch({
      added: [{ key: KEY_A, tile: tile([1, 2, 3]) }],
      removed: [],
    });
    const matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 20, 30, 1];
    adapter.setBaseMatrix(matrix);

    scheduleRender.mockClear();
    adapter.setBaseMatrix([...matrix]);
    expect(scheduleRender).not.toHaveBeenCalled();

    matrix[12] = 20;
    adapter.setBaseMatrix(matrix);
    expect(scheduleRender).toHaveBeenCalledTimes(1);
    expect(actorInstances[0]!.userMatrix!.slice(12, 15)).toEqual([21, 22, 33]);
  });

  it("fans out CSS diameter and DPR to every tile actor", () => {
    const { adapter, scheduleRender } = makeAdapter();
    adapter.applyBatch({
      added: [
        { key: KEY_A, tile: tile([0, 0, 0]) },
        { key: KEY_B, tile: tile([1, 0, 0]) },
      ],
      removed: [],
    });

    scheduleRender.mockClear();
    adapter.setPointDiameterCssPx(7);
    adapter.setDevicePixelRatio(2);
    expect(actorInstances.map((a) => a.pointSize)).toEqual([7, 7]);
    expect(mapperInstances.map((m) => m.scaleFactor)).toEqual([2, 2]);
    expect(scheduleRender).toHaveBeenCalledTimes(2);

    // No-op updates do not schedule renders.
    scheduleRender.mockClear();
    adapter.setPointDiameterCssPx(7);
    adapter.setDevicePixelRatio(2);
    expect(scheduleRender).not.toHaveBeenCalled();
  });

  it("keeps apparent CSS diameter invariant across DPR", () => {
    const { adapter, scheduleRender } = makeAdapter({
      diameterCssPx: 2.5,
      devicePixelRatio: 1,
    });
    adapter.applyBatch({
      added: [{ key: KEY_A, tile: tile([0, 0, 0]) }],
      removed: [],
    });
    expect(actorInstances[0]!.pointSize).toBe(2.5);
    expect(mapperInstances[0]!.scaleFactor).toBe(1);

    scheduleRender.mockClear();
    adapter.setDevicePixelRatio(2);
    expect(actorInstances[0]!.pointSize).toBe(2.5);
    expect(mapperInstances[0]!.scaleFactor).toBe(2);
    expect(scheduleRender).toHaveBeenCalledTimes(1);
  });

  it("dispose releases everything and is idempotent", () => {
    const { adapter, renderer } = makeAdapter();
    adapter.applyBatch({
      added: [
        { key: KEY_A, tile: tile([0, 0, 0]) },
        { key: KEY_B, tile: tile([1, 0, 0]) },
      ],
      removed: [],
    });
    adapter.dispose();
    adapter.dispose();

    expect(adapter.stats().gpuResidentTiles).toBe(0);
    expect(renderer.removeActor).toHaveBeenCalledTimes(2);
    expect(actorInstances.every((a) => a.deleted)).toBe(true);

    // Batches after dispose are ignored.
    adapter.applyBatch({
      added: [{ key: KEY_A, tile: tile([0, 0, 0]) }],
      removed: [],
    });
    expect(adapter.stats().gpuResidentTiles).toBe(0);
  });
});

describe("adapter visibility", () => {
  it("hides and shows with nothing submitted yet", () => {
    const { adapter, renderer, scheduleRender } = makeAdapter();
    adapter.setVisible(false);
    adapter.setVisible(true);

    expect(actorInstances).toHaveLength(0);
    expect(renderer.removeActor).not.toHaveBeenCalled();
    expect(scheduleRender).toHaveBeenCalledTimes(2);

    // Loading afterwards behaves exactly as it would have before the toggle.
    adapter.applyBatch({
      added: [{ key: KEY_A, tile: tile([0, 0, 0]) }],
      removed: [],
    });
    expect(adapter.stats()).toMatchObject({ submittedTiles: 1, drawnTiles: 1 });
    expect(actorInstances[0]!.visibility).toBe(true);
  });

  it("keeps submitted actors and their resources across a hide/show", () => {
    const { adapter, renderer, scheduleRender } = makeAdapter();
    adapter.applyBatch({
      added: [
        { key: KEY_A, tile: tile([0, 0, 0]) },
        { key: KEY_B, tile: tile([1, 0, 0]) },
      ],
      removed: [],
    });

    scheduleRender.mockClear();
    adapter.setVisible(false);
    expect(actorInstances.map((a) => a.visibility)).toEqual([false, false]);
    expect(actorInstances.some((a) => a.deleted)).toBe(false);
    expect(renderer.removeActor).not.toHaveBeenCalled();
    expect(adapter.stats()).toMatchObject({
      submittedTiles: 2,
      submittedPoints: 4,
      gpuResidentTiles: 2,
      drawnTiles: 0,
      drawnPoints: 0,
      visible: false,
    });

    adapter.setVisible(true);
    expect(actorInstances.map((a) => a.visibility)).toEqual([true, true]);
    expect(adapter.stats()).toMatchObject({
      submittedTiles: 2,
      drawnTiles: 2,
      drawnPoints: 4,
      visible: true,
    });
    expect(scheduleRender).toHaveBeenCalledTimes(2);
    expect(renderer.addActor).toHaveBeenCalledTimes(2);

    // A hide and a show inside one task, with no batch in between, is a
    // no-op the consumer never has to compensate for.
    scheduleRender.mockClear();
    adapter.setVisible(false);
    adapter.setVisible(true);
    expect(actorInstances.map((a) => a.visibility)).toEqual([true, true]);
    expect(adapter.stats().drawnTiles).toBe(2);
  });

  it("accepts tiles that arrive while hidden and draws them on show", () => {
    const { adapter, renderer, scheduleRender } = makeAdapter();
    adapter.setVisible(false);

    // Loading continues while hidden: batches must not be dropped.
    adapter.applyBatch({
      added: [{ key: KEY_A, tile: tile([0, 0, 0]) }],
      removed: [],
    });
    expect(renderer.addActor).toHaveBeenCalledTimes(1);
    expect(actorInstances[0]!.visibility).toBe(false);
    expect(adapter.stats()).toMatchObject({
      submittedTiles: 1,
      gpuResidentTiles: 1,
      drawnTiles: 0,
    });

    scheduleRender.mockClear();
    adapter.setVisible(true);
    expect(actorInstances[0]!.visibility).toBe(true);
    expect(adapter.stats()).toMatchObject({ drawnTiles: 1, drawnPoints: 2 });
    expect(scheduleRender).toHaveBeenCalledTimes(1);
  });

  it("restores tiles resubmitted after a controller reactivation", () => {
    const { adapter, renderer } = makeAdapter();
    const data = tile([0, 0, 0]);
    adapter.applyBatch({ added: [{ key: KEY_A, tile: data }], removed: [] });
    adapter.setVisible(false);

    // Deactivation drops residency: the removal is what frees the cloud.
    adapter.applyBatch({ added: [], removed: [KEY_A] });
    expect(adapter.stats()).toMatchObject({
      submittedTiles: 0,
      pooledTiles: 1,
      drawnTiles: 0,
    });

    // Reactivation resubmits the cached payload while still hidden.
    adapter.applyBatch({ added: [{ key: KEY_A, tile: data }], removed: [] });
    expect(actorInstances).toHaveLength(1);
    expect(renderer.addActor).toHaveBeenCalledTimes(1);
    expect(actorInstances[0]!.visibility).toBe(false);

    adapter.setVisible(true);
    expect(actorInstances[0]!.visibility).toBe(true);
    expect(adapter.stats()).toMatchObject({
      submittedTiles: 1,
      pooledTiles: 0,
      drawnTiles: 1,
    });
  });

  it("starts hidden when constructed hidden", () => {
    const { adapter } = makeAdapter({ visible: false });
    adapter.applyBatch({
      added: [{ key: KEY_A, tile: tile([0, 0, 0]) }],
      removed: [],
    });
    expect(actorInstances[0]!.visibility).toBe(false);
    expect(adapter.stats()).toMatchObject({
      submittedTiles: 1,
      drawnTiles: 0,
      visible: false,
    });
  });
});

describe("adapter resource pool", () => {
  it("hides a removed actor synchronously and pools it", () => {
    const { adapter, renderer } = makeAdapter();
    adapter.applyBatch({
      added: [{ key: KEY_A, tile: tile([0, 0, 0]) }],
      removed: [],
    });
    adapter.applyBatch({ added: [], removed: [KEY_A] });

    expect(actorInstances[0]!.visibility).toBe(false);
    expect(adapter.stats()).toMatchObject({
      submittedTiles: 0,
      submittedPoints: 0,
      submittedBytes: 0,
      pooledTiles: 1,
      pooledPoints: 2,
      pooledBytes: 94,
      gpuResidentTiles: 1,
      gpuResidentBytes: 94,
      drawnTiles: 0,
    });
    expect(renderer.removeActor).not.toHaveBeenCalled();

    adapter.dispose();
    expect(renderer.removeActor).toHaveBeenCalledTimes(1);
    expect(actorInstances[0]!.deleted).toBe(true);
    expect(mapperInstances[0]!.deleted).toBe(true);
    expect(polyDataInstances[0]!.deleted).toBe(true);
    expect(adapter.stats().gpuResidentTiles).toBe(0);
  });

  it("brings the pool back under its ceiling on a batch of pure removals", () => {
    const { adapter, renderer } = makeAdapter();
    adapter.applyBatch({
      added: [
        { key: KEY_A, tile: tile([0, 0, 0]) },
        { key: KEY_B, tile: tile([1, 0, 0]) },
      ],
      removed: [],
    });
    // Below one tile, so nothing may stay pooled once it is off screen.
    adapter.setResourceCeilingBytes(50);
    expect(adapter.stats()).toMatchObject({ submittedTiles: 2, pooledTiles: 0 });

    // Deactivating a cloud sends exactly this: removals and nothing else. The
    // pool used to be trimmed only while adding, so these actors stayed on the
    // GPU until some other cloud happened to add a tile.
    adapter.applyBatch({ added: [], removed: [KEY_A, KEY_B] });

    expect(adapter.stats()).toMatchObject({
      submittedTiles: 0,
      pooledTiles: 0,
      gpuResidentTiles: 0,
      gpuResidentBytes: 0,
    });
    expect(renderer.removeActor).toHaveBeenCalledTimes(2);
  });

  it("reuses a pooled actor when the same payload returns", () => {
    const { adapter, renderer } = makeAdapter();
    const data = tile([0, 0, 0]);
    adapter.applyBatch({ added: [{ key: KEY_A, tile: data }], removed: [] });
    adapter.applyBatch({ added: [], removed: [KEY_A] });
    adapter.applyBatch({ added: [{ key: KEY_A, tile: data }], removed: [] });

    expect(renderer.addActor).toHaveBeenCalledTimes(1);
    expect(renderer.removeActor).not.toHaveBeenCalled();
    expect(actorInstances).toHaveLength(1);
    expect(adapter.stats()).toMatchObject({
      submittedTiles: 1,
      pooledTiles: 0,
      drawnTiles: 1,
    });
  });

  it("releases a pooled actor rather than resurrect an obsolete payload", () => {
    const { adapter, renderer } = makeAdapter();
    adapter.applyBatch({
      added: [{ key: KEY_A, tile: tile([0, 0, 0]) }],
      removed: [],
    });
    adapter.applyBatch({ added: [], removed: [KEY_A] });

    const refreshed = tile([5, 0, 0], 3);
    adapter.applyBatch({
      added: [{ key: KEY_A, tile: refreshed }],
      removed: [],
    });

    expect(actorInstances).toHaveLength(2);
    expect(actorInstances[0]!.deleted).toBe(true);
    expect(renderer.removeActor).toHaveBeenCalledTimes(1);
    expect(polyDataInstances[1]!.points).toBe(refreshed.positions);
    expect(actorInstances[1]!.userMatrix!.slice(12, 15)).toEqual([5, 0, 0]);
    expect(adapter.stats()).toMatchObject({
      submittedTiles: 1,
      submittedPoints: 3,
      pooledTiles: 0,
      gpuResidentTiles: 1,
    });
  });

  it("replaces a submitted tile whose payload changed", () => {
    const { adapter, renderer, scheduleRender } = makeAdapter();
    const data = tile([0, 0, 0]);
    adapter.applyBatch({ added: [{ key: KEY_A, tile: data }], removed: [] });

    // A re-addition without a paired removal is the controller reporting a
    // new payload for a key already on screen.
    const refreshed = tile([0, 0, 0], 4);
    scheduleRender.mockClear();
    adapter.applyBatch({
      added: [{ key: KEY_A, tile: refreshed }],
      removed: [],
    });

    expect(actorInstances).toHaveLength(2);
    expect(actorInstances[0]!.deleted).toBe(true);
    expect(actorInstances[1]!.visibility).toBe(true);
    expect(renderer.removeActor).toHaveBeenCalledTimes(1);
    expect(renderer.addActor).toHaveBeenCalledTimes(2);
    expect(polyDataInstances[1]!.points).toBe(refreshed.positions);
    expect(scheduleRender).toHaveBeenCalledTimes(1);
    expect(adapter.stats()).toMatchObject({
      submittedTiles: 1,
      submittedPoints: 4,
      pooledTiles: 0,
      gpuResidentTiles: 1,
    });
  });

  it("ignores a repeated addition of the payload already on screen", () => {
    const { adapter, renderer, scheduleRender } = makeAdapter();
    const data = tile([0, 0, 0]);
    adapter.applyBatch({ added: [{ key: KEY_A, tile: data }], removed: [] });

    scheduleRender.mockClear();
    adapter.applyBatch({ added: [{ key: KEY_A, tile: data }], removed: [] });
    expect(actorInstances).toHaveLength(1);
    expect(renderer.addActor).toHaveBeenCalledTimes(1);
    expect(renderer.removeActor).not.toHaveBeenCalled();
    expect(scheduleRender).not.toHaveBeenCalled();
  });

  it("evicts pooled resources at the shared-memory ceiling", () => {
    const { adapter, renderer } = makeAdapter();
    adapter.setResourceCeilingBytes(94);
    adapter.applyBatch({
      added: [{ key: KEY_A, tile: tile([0, 0, 0]) }],
      removed: [],
    });
    adapter.applyBatch({ added: [], removed: [KEY_A] });
    adapter.applyBatch({
      added: [{ key: KEY_B, tile: tile([1, 0, 0]) }],
      removed: [],
    });

    expect(renderer.removeActor).toHaveBeenCalledTimes(1);
    expect(actorInstances[0]!.deleted).toBe(true);
    expect(adapter.stats()).toMatchObject({
      submittedTiles: 1,
      submittedBytes: 94,
      pooledTiles: 0,
      pooledBytes: 0,
      gpuResidentTiles: 1,
      gpuResidentBytes: 94,
    });
  });

  it("never trims a submitted actor, however tight the ceiling", () => {
    const { adapter, renderer } = makeAdapter();
    adapter.applyBatch({
      added: [
        { key: KEY_A, tile: tile([0, 0, 0]) },
        { key: KEY_B, tile: tile([1, 0, 0]) },
      ],
      removed: [],
    });
    adapter.applyBatch({ added: [], removed: [KEY_B] });

    adapter.setResourceCeilingBytes(0);
    expect(renderer.removeActor).toHaveBeenCalledTimes(1);
    expect(actorInstances[0]!.deleted).toBe(false);
    expect(actorInstances[1]!.deleted).toBe(true);
    expect(adapter.stats()).toMatchObject({
      submittedTiles: 1,
      submittedBytes: 94,
      pooledTiles: 0,
      pooledBytes: 0,
    });

    // Still tight: the next addition trims nothing it is not allowed to.
    adapter.applyBatch({
      added: [{ key: KEY_B, tile: tile([2, 0, 0]) }],
      removed: [],
    });
    expect(adapter.stats()).toMatchObject({
      submittedTiles: 2,
      pooledTiles: 0,
    });
    expect(actorInstances[0]!.deleted).toBe(false);
  });

  it("dispose releases submitted and pooled actors exactly once", () => {
    const { adapter, renderer } = makeAdapter();
    adapter.applyBatch({
      added: [
        { key: KEY_A, tile: tile([0, 0, 0]) },
        { key: KEY_B, tile: tile([1, 0, 0]) },
      ],
      removed: [],
    });
    adapter.applyBatch({ added: [], removed: [KEY_B] });

    adapter.dispose();
    const released = renderer.removeActor.mock.calls.map(([actor]) => actor);
    expect(released).toHaveLength(2);
    expect(new Set(released).size).toBe(2);
    expect(adapter.stats()).toMatchObject({
      submittedTiles: 0,
      pooledTiles: 0,
      gpuResidentTiles: 0,
    });
  });
});

const IDENTITY_VIEW_PROJ = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
/** Pushes everything 10 units off in clip x: nothing is visible. */
const LOOK_AWAY_VIEW_PROJ = [
  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -10, 0, 0, 1,
];

const cameraView = (viewProj: number[]): PerspectiveCameraView => ({
  projection: "perspective",
  viewProj,
  position: [0, 0, 0],
  fovY: Math.PI / 2,
  viewportHeightCssPx: 100,
});

/** Root plus four level-1 leaves, all inside the identity frustum. */
const STRESS_TREE: Record<string, number> = {
  "0-0-0-0": 100,
  "1-0-0-0": 100,
  "1-1-0-0": 100,
  "1-0-1-0": 100,
  "1-1-1-0": 100,
};

const makeStressSource = () => {
  const payloads = new Map<string, TileData>();
  const payloadFor = (keyString: string): TileData => {
    const existing = payloads.get(keyString);
    if (existing !== undefined) return existing;
    const pointCount = STRESS_TREE[keyString]!;
    const created: TileData = {
      origin: [0, 0, 0],
      positions: new Float32Array(pointCount * 3),
      rgb: new Uint8Array(pointCount * 3),
      pointCount,
    };
    payloads.set(keyString, created);
    return created;
  };
  const source: TileSource = {
    metadata: () => ({ pointCount: 500 }),
    async nodes(key: VoxelKey): Promise<NodeInfo[]> {
      if (keyToString(key) !== "0-0-0-0") throw new Error("no such page");
      return Object.entries(STRESS_TREE).map(([keyString, pointCount]) => {
        const [level, x, y, z] = keyString.split("-").map(Number);
        return {
          key: { level: level!, x: x!, y: y!, z: z! },
          pointCount,
          bounds: { min: [-0.5, -0.5, -0.5], max: [0.5, 0.5, 0.5] },
          spacing: 0.1 / 2 ** level!,
        };
      });
    },
    async loadTile(key: VoxelKey): Promise<TileData> {
      return payloadFor(keyToString(key));
    },
  };
  return { source, payloadFor };
};

const settle = async (): Promise<void> => {
  for (let turn = 0; turn < 30; turn += 1) await Promise.resolve();
};

describe("adapter driven by a live controller", () => {
  it("draws exactly the controller's submitted set through rapid churn", async () => {
    const { adapter, renderer } = makeAdapter();
    const { source, payloadFor } = makeStressSource();
    // Mirrors the submitted state the controller's deltas describe.
    const submitted = new Map<string, TileData>();
    const controller = createLodController({
      source,
      onTiles: (batch) => {
        for (const key of batch.removed) submitted.delete(keyToString(key));
        for (const { key, tile: payload } of batch.added) {
          submitted.set(keyToString(key), payload);
        }
        adapter.applyBatch(batch);
      },
      scheduleRender: () => {},
      pointBudget: 500,
      selectionDelayMs: 0,
      memory: 64 * 1024 * 1024,
    });
    const submittedPositions = (): Set<Float32Array> =>
      new Set([...submitted.values()].map((payload) => payload.positions));

    controller.setCamera(cameraView(IDENTITY_VIEW_PROJ));
    await settle();
    expect(submitted.size).toBe(5);
    expect(drawnPositions()).toEqual(submittedPositions());

    // Budget churn, culling churn, a hide across both, then a full
    // deactivation/reactivation cycle — all coalesced into disjoint deltas.
    controller.setPointBudget(200);
    await settle();
    controller.setPointBudget(500);
    await settle();
    adapter.setVisible(false);
    controller.setCamera(cameraView(LOOK_AWAY_VIEW_PROJ));
    await settle();
    expect(submitted.size).toBe(0);
    controller.setCamera(cameraView(IDENTITY_VIEW_PROJ));
    await settle();
    controller.setActive(false);
    await settle();
    controller.setActive(true);
    await settle();
    adapter.setVisible(true);

    expect(submitted.size).toBe(5);
    expect(drawnPositions()).toEqual(submittedPositions());
    const controllerStats = controller.stats();
    expect(adapter.stats()).toMatchObject({
      submittedTiles: controllerStats.residentTiles,
      submittedPoints: controllerStats.residentPoints,
      submittedBytes: controllerStats.residentBytes,
      drawnTiles: controllerStats.residentTiles,
    });
    // Payload identity survived the churn: nothing was rebuilt from scratch.
    expect(submitted.get("0-0-0-0")).toBe(payloadFor("0-0-0-0"));

    // A hide/show with the controller quiet emits no batch at all: only the
    // adapter's own state can bring the cloud back.
    adapter.setVisible(false);
    adapter.setVisible(true);
    await settle();
    expect(drawnPositions()).toEqual(submittedPositions());

    // Controller teardown hands back everything the renderer still holds.
    controller.dispose();
    expect(submitted.size).toBe(0);
    expect(adapter.stats()).toMatchObject({
      submittedTiles: 0,
      drawnTiles: 0,
      pooledTiles: 5,
    });

    adapter.dispose();
    const released = renderer.removeActor.mock.calls.map(([actor]) => actor);
    expect(new Set(released).size).toBe(released.length);
    expect(adapter.stats().gpuResidentTiles).toBe(0);
  });
});
