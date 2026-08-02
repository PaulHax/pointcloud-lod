import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  ROOT_CUBE,
  createPageGraphSource,
  levelOf,
  type PageGraphSource,
} from "../test/fixtures/pageGraph";
import {
  distanceToBounds,
  type CameraView,
  type OrthographicCameraView,
  type PerspectiveCameraView,
} from "./camera";
import {
  createLodController,
  type LodController,
  type LodControllerOptions,
  type TileBatch,
} from "./controller";
import { createMemoryPool, type MemoryPool } from "./memoryPool";
import {
  ROOT_KEY,
  childKeys,
  keyFromString,
  keyToString,
  nodeBounds,
  type VoxelKey,
} from "./octree";
import type {
  NodeInfo,
  TileData,
  TileSource,
  TileSourceMetadata,
} from "./tileSource";

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
/** Pushes everything 10 units off in clip x: nothing is visible. */
const LOOK_AWAY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -10, 0, 0, 1];

const VIEW: PerspectiveCameraView = {
  projection: "perspective",
  viewProj: IDENTITY,
  position: [0, 0, 0],
  fovY: Math.PI / 2,
  viewportWidthCssPx: 100,
  viewportHeightCssPx: 100,
};

/** Same framing, parallel projection: half-height 1 over a 100 px viewport. */
const ORTHOGRAPHIC_VIEW: OrthographicCameraView = {
  projection: "orthographic",
  viewProj: IDENTITY,
  position: [0, 0, 0],
  parallelScale: 1,
  viewportWidthCssPx: 100,
  viewportHeightCssPx: 100,
};

const METADATA: TileSourceMetadata = {
  pointCount: 1000,
};

type FakeNode = {
  pointCount: number;
  bounds?: {
    min: [number, number, number];
    max: [number, number, number];
  };
  spacing?: number;
  children?: string[];
  pageRef?: boolean;
};

type FakeEntry = FakeNode & {
  /** Entries revealed by loading the page rooted at this key. */
  pageNodes?: Record<string, FakeNode>;
};

type Deferred = {
  resolve: (tile?: Partial<TileData>) => void;
  reject: (error: Error) => void;
  aborted: boolean;
};

const makeTile = (
  pointCount: number,
  overrides?: Partial<TileData>,
): TileData => ({
  origin: [0, 0, 0],
  positions: new Float32Array(pointCount * 3),
  pointCount,
  ...overrides,
});

/** Fake source: single root page + per-key deferred tile loads. */
const makeFakeSource = (tree: Record<string, FakeEntry>) => {
  const deferred = new Map<string, Deferred>();
  const loadCalls: string[] = [];
  const toInfos = (entries: Record<string, FakeNode>): NodeInfo[] =>
    Object.entries(entries).map(([keyString, entry]) => {
      const key = keyFromString(keyString);
      return {
        key,
        pointCount: entry.pointCount,
        bounds: entry.bounds ?? {
          min: [-0.5, -0.5, -0.5],
          max: [0.5, 0.5, 0.5],
        },
        spacing: entry.spacing ?? 0.1 / 2 ** key.level,
        // Array.map would hand keyFromString an index as a second argument.
        children: entry.children?.map((c) => keyFromString(c)),
        pageRef: entry.pageRef,
      };
    });

  const source: TileSource = {
    metadata: () => METADATA,
    async nodes(key: VoxelKey) {
      const keyString = keyToString(key);
      if (keyString === "0-0-0-0") return toInfos(tree);
      const entry = tree[keyString];
      if (entry?.pageNodes) return toInfos(entry.pageNodes);
      throw new Error(`no page for ${keyString}`);
    },
    loadTile(key: VoxelKey, opts) {
      const keyString = keyToString(key);
      loadCalls.push(keyString);
      return new Promise<TileData>((resolve, reject) => {
        const d: Deferred = {
          resolve: (overrides) => {
            const entry = tree[keyString];
            const pointCount =
              entry?.pageNodes?.[keyString]?.pointCount ??
              entry?.pointCount ??
              1;
            resolve(makeTile(pointCount, overrides));
          },
          reject,
          aborted: false,
        };
        deferred.set(keyString, d);
        opts?.signal?.addEventListener("abort", () => {
          d.aborted = true;
          const error = new Error("aborted");
          error.name = "AbortError";
          reject(error);
        });
      });
    },
  };
  return { source, deferred, loadCalls };
};

/** Source that ignores the abort signal: cancellation is advisory only. */
const uncancellableSource = (resolvers: Array<(tile: TileData) => void>) => ({
  metadata: () => METADATA,
  async nodes() {
    return [
      {
        key: { level: 0, x: 0, y: 0, z: 0 },
        pointCount: 100,
        bounds: {
          min: [-0.5, -0.5, -0.5] as [number, number, number],
          max: [0.5, 0.5, 0.5] as [number, number, number],
        },
        spacing: 0.1,
        children: [],
      },
    ];
  },
  // Deliberately ignores opts.signal: models an uncancellable getter.
  loadTile: () => new Promise<TileData>((r) => resolvers.push(r)),
});

const collectBatches = () => {
  const batches: TileBatch[] = [];
  const scheduleRender = vi.fn();
  const onTiles = (batch: TileBatch) => {
    batches.push({
      added: [...batch.added],
      removed: [...batch.removed],
    });
  };
  return { batches, onTiles, scheduleRender };
};

/**
 * Models the consumer the way the renderer adapter behaves, and refuses
 * everything the controller must never emit: a key in both halves of one
 * batch, a re-addition of a payload already on screen, or a removal of
 * something the renderer was never handed.
 */
const mirrorRenderer = () => {
  const visible = new Map<string, TileData>();
  const batches: TileBatch[] = [];
  // Batches arrive in a microtask, where a thrown assertion would never reach
  // the test — violations are recorded and asserted from the test body.
  const violations: string[] = [];
  const scheduleRender = vi.fn();
  const onTiles = (batch: TileBatch) => {
    batches.push({ added: [...batch.added], removed: [...batch.removed] });
    const removed = batch.removed.map(keyToString);
    for (const entry of batch.added) {
      const keyString = keyToString(entry.key);
      if (removed.includes(keyString)) {
        violations.push(`added and removed ${keyString}`);
      }
    }
    for (const keyString of removed) {
      if (!visible.delete(keyString)) {
        violations.push(`removed unsubmitted ${keyString}`);
      }
    }
    for (const { key, tile } of batch.added) {
      const keyString = keyToString(key);
      // Re-adding a key is only ever legal to swap in a different payload.
      if (visible.get(keyString) === tile) {
        violations.push(`re-added ${keyString}`);
      }
      visible.set(keyString, tile);
    }
  };
  const visiblePoints = (): number =>
    [...visible.values()].reduce((sum, tile) => sum + tile.pointCount, 0);
  const touchedKeys = (): string[] =>
    batches.flatMap((batch) => [
      ...batch.added.map((entry) => keyToString(entry.key)),
      ...batch.removed.map(keyToString),
    ]);
  return {
    visible,
    batches,
    violations,
    onTiles,
    scheduleRender,
    visiblePoints,
    touchedKeys,
  };
};

const settle = async (): Promise<void> => {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
};

/**
 * Bootstrap the root page, look at the cloud, and land every outstanding
 * fetch. Tests that resolve selectively or count turns drive this by hand.
 */
const bootAndLand = async (
  controller: LodController,
  deferred: Map<string, Deferred>,
): Promise<void> => {
  await settle(); // root hierarchy page
  controller.setCamera(VIEW);
  await settle();
  for (const d of deferred.values()) d.resolve();
  await settle();
};

const makeHarness = <
  S extends { onTiles: (batch: TileBatch) => void; scheduleRender: () => void },
>(
  tree: Record<string, FakeEntry>,
  sink: S,
  overrides?: Partial<LodControllerOptions>,
) => {
  const fake = makeFakeSource(tree);
  const controller = createLodController({
    source: fake.source,
    onTiles: sink.onTiles,
    scheduleRender: sink.scheduleRender,
    pointBudget: 1000,
    selectionDelayMs: 0,
    ...overrides,
  });
  return { controller, ...fake, ...sink };
};

const makeController = (
  tree: Record<string, FakeEntry>,
  overrides?: Partial<LodControllerOptions>,
) => makeHarness(tree, collectBatches(), overrides);

const makeMirrored = (
  tree: Record<string, FakeEntry>,
  overrides?: Partial<LodControllerOptions>,
) => makeHarness(tree, mirrorRenderer(), overrides);

const SMALL_TREE: Record<string, FakeEntry> = {
  "0-0-0-0": { pointCount: 100, children: ["1-0-0-0", "1-1-0-0"] },
  "1-0-0-0": { pointCount: 60 },
  "1-1-0-0": { pointCount: 60 },
};

/**
 * One leaf, sized so the density-aware spacing lands at 5 css px: 0.9 world
 * spacing over +/-1 bounds is what makes the Auto diameter come out at 4.
 */
const AUTO_LEAF: Record<string, FakeEntry> = {
  "0-0-0-0": {
    pointCount: 100,
    bounds: { min: [-1, -1, -1], max: [1, 1, 1] },
    spacing: 0.9,
  },
};

describe("createLodController", () => {
  it("reports required work and revisions independently of telemetry", async () => {
    const { controller, deferred } = makeController(SMALL_TREE);
    const duringHierarchy = controller.stats();
    expect(duringHierarchy.workPending).toBe(true);
    expect(duringHierarchy.workRevision).toBeGreaterThan(0);

    await bootAndLand(controller, deferred);
    const settled = controller.stats();
    expect(settled.workPending).toBe(false);
    expect(settled.workRevision).toBeGreaterThan(duringHierarchy.workRevision);
    controller.dispose();
  });

  it("selects within the budget with a parent-closed set", async () => {
    const { controller, loadCalls, deferred, batches } = makeController(
      SMALL_TREE,
      { pointBudget: 200 },
    );
    await settle(); // root hierarchy page
    controller.setCamera(VIEW);
    await settle();

    // 100 + 60 + 60 exceeds 200: root plus exactly one child fetches.
    expect(loadCalls).toHaveLength(2);
    expect(loadCalls[0]).toBe("0-0-0-0"); // coarse level first

    deferred.get(loadCalls[0]!)!.resolve();
    deferred.get(loadCalls[1]!)!.resolve();
    await settle();

    const added = batches.flatMap((b) =>
      b.added.map((a) => keyToString(a.key)),
    );
    expect(added.sort()).toEqual([loadCalls[0], loadCalls[1]].sort());
    expect(controller.stats().residentPoints).toBe(160);
    controller.dispose();
  });

  it("raising the budget refines further", async () => {
    const { controller, loadCalls, deferred } = makeController(SMALL_TREE, {
      pointBudget: 200,
    });
    await bootAndLand(controller, deferred);

    controller.setPointBudget(1000);
    await settle();
    expect(loadCalls).toHaveLength(3);
    controller.dispose();
  });

  it("changes progressive density without selection, I/O, or tile batches", async () => {
    const onDensityFraction = vi.fn();
    const { controller, deferred, loadCalls, batches } = makeController(
      SMALL_TREE,
      { onDensityFraction },
    );
    await bootAndLand(controller, deferred);
    const before = controller.stats();
    const beforeLoads = [...loadCalls];
    const beforeBatches = batches.length;
    const beforeKeys = controller.activeKeys();

    controller.setDensityFraction(0.25);

    const after = controller.stats();
    expect(after.densityFraction).toBe(0.25);
    expect(after.drawnPoints).toBe(55);
    expect(after.selection.generation).toBe(before.selection.generation);
    expect(after.selection.targetRevision).toBe(
      before.selection.targetRevision,
    );
    expect(after.selection.targetPoints).toBe(before.selection.targetPoints);
    expect(controller.activeKeys()).toEqual(beforeKeys);
    expect(loadCalls).toEqual(beforeLoads);
    expect(batches).toHaveLength(beforeBatches);
    expect(onDensityFraction).toHaveBeenLastCalledWith(0.25);
    controller.dispose();
  });

  it("never replaces selected tiles when only the point budget grows", async () => {
    const tree: Record<string, FakeEntry> = {
      "0-0-0-0": {
        pointCount: 10,
        spacing: 10,
        bounds: { min: [-1, -1, -0.5], max: [1, 1, 0.5] },
        children: ["1-0-0-0", "1-1-0-0"],
      },
      "1-0-0-0": {
        pointCount: 80,
        spacing: 1,
        bounds: { min: [-0.5, -0.2, -0.5], max: [-0.1, 0.2, 0.5] },
      },
      "1-1-0-0": {
        pointCount: 30,
        spacing: 1,
        bounds: { min: [0.1, -0.2, -0.5], max: [0.5, 0.2, 0.5] },
      },
    };
    const { controller, loadCalls } = makeController(tree, {
      pointBudget: 50,
    });
    await settle();
    controller.setCamera({ ...VIEW, position: [0.5, 0, 2] });
    await settle();
    expect(loadCalls).toEqual(["0-0-0-0", "1-1-0-0"]);
    const initialRevision = controller.stats().selection.targetRevision;

    // The farther 80-point child now fits by itself, but replacing the
    // selected 30-point child would make a larger budget look less refined.
    controller.setPointBudget(90);
    await settle();
    expect(loadCalls).toEqual(["0-0-0-0", "1-1-0-0"]);
    expect(controller.stats().selection.targetRevision).toBe(initialRevision);

    // Once the extra budget truly covers both, the near child is additive.
    controller.setPointBudget(120);
    await settle();
    expect(loadCalls).toEqual(["0-0-0-0", "1-1-0-0", "1-0-0-0"]);
    controller.dispose();
  });

  it("keeps a near-tied selected tile until a challenger is materially more important", async () => {
    const tree: Record<string, FakeEntry> = {
      "0-0-0-0": {
        pointCount: 100,
        spacing: 1,
        bounds: { min: [-1, -1, -0.5], max: [1, 1, 0.5] },
        children: ["1-0-0-0", "1-1-0-0"],
      },
      "1-0-0-0": {
        pointCount: 60,
        spacing: 0.1,
        bounds: { min: [-0.5, -0.2, -0.5], max: [-0.1, 0.2, 0.5] },
      },
      "1-1-0-0": {
        pointCount: 60,
        spacing: 0.1,
        bounds: { min: [0.1, -0.2, -0.5], max: [0.5, 0.2, 0.5] },
      },
    };
    const { controller, loadCalls } = makeController(tree, {
      pointBudget: 160,
    });
    await settle();

    controller.setCamera({ ...VIEW, position: [-0.5, 0, 2] });
    await settle();
    expect(loadCalls).toEqual(["0-0-0-0", "1-0-0-0"]);

    // The right child becomes about 8% more important. That is too small a
    // difference to replace a large actor at the budget boundary.
    controller.setCamera({ ...VIEW, position: [0.5, 0, 2] });
    await settle();
    expect(loadCalls).toEqual(["0-0-0-0", "1-0-0-0"]);

    // Once the difference clears the hysteresis band, the selection follows.
    controller.setCamera({ ...VIEW, position: [1, 0, 2] });
    await settle();
    expect(loadCalls).toEqual(["0-0-0-0", "1-0-0-0", "1-1-0-0"]);
    controller.dispose();
  });

  it("cancels in-flight fetches when the camera looks away", async () => {
    const { controller, deferred, batches } = makeController(SMALL_TREE);
    await settle();
    controller.setCamera(VIEW);
    await settle();

    deferred.get("0-0-0-0")!.resolve();
    await settle();
    expect(controller.stats().residentTiles).toBe(1);

    controller.setCamera({ ...VIEW, viewProj: LOOK_AWAY });
    await settle();

    // The unresolved child fetches were aborted, the resident root removed.
    expect(deferred.get("1-0-0-0")!.aborted).toBe(true);
    expect(deferred.get("1-1-0-0")!.aborted).toBe(true);
    const removed = batches.flatMap((b) =>
      b.removed.map((k) => keyToString(k)),
    );
    expect(removed).toContain("0-0-0-0");
    expect(controller.stats().residentTiles).toBe(0);
    expect(controller.stats().inFlight).toBe(0);
    controller.dispose();
  });

  it("adopts a canceled fetch when the same key is reselected", async () => {
    // Aborting is advisory: the COPC getter takes no signal, so a canceled
    // request still resolves. Looking back must adopt that live read rather
    // than start a rival one, and its payload must claim residency.
    const resolvers: Array<(tile: TileData) => void> = [];
    const sink = collectBatches();
    const controller = createLodController({
      source: uncancellableSource(resolvers),
      onTiles: sink.onTiles,
      scheduleRender: sink.scheduleRender,
      pointBudget: 1000,
      selectionDelayMs: 0,
    });

    await settle();
    controller.setCamera(VIEW);
    await settle();
    expect(resolvers).toHaveLength(1);

    // Look away (aborts, but the read stays physically alive), then look back.
    controller.setCamera({ ...VIEW, viewProj: LOOK_AWAY });
    await settle();
    expect(controller.stats().inFlight).toBe(0);
    controller.setCamera(VIEW);
    await settle();
    // No second read of a key already being read: the running one is adopted.
    expect(resolvers).toHaveLength(1);
    expect(controller.stats()).toMatchObject({
      inFlight: 1,
      physicalTileOperations: 1,
    });

    // Its payload is what the selection is waiting for, so it goes resident
    // instead of being thrown away for a redundant round trip.
    resolvers[0]!(makeTile(100));
    await settle();
    expect(resolvers).toHaveLength(1);
    expect(controller.stats()).toMatchObject({
      inFlight: 0,
      physicalTileOperations: 0,
      residentTiles: 1,
      residentPoints: 100,
      cachedTiles: 0,
    });
    controller.dispose();
  });

  it("reuses cached tiles without refetching", async () => {
    const { controller, deferred, loadCalls } = makeController(SMALL_TREE);
    await bootAndLand(controller, deferred);
    const fetchesBefore = loadCalls.length;

    controller.setCamera({ ...VIEW, viewProj: LOOK_AWAY });
    await settle();
    controller.setCamera(VIEW);
    await settle();

    expect(loadCalls.length).toBe(fetchesBefore); // all served from cache
    expect(controller.stats().residentTiles).toBe(3);
    controller.dispose();
  });

  it("deactivation releases submitted tiles and reuses bounded decoded payloads", async () => {
    const { controller, deferred, loadCalls, batches } =
      makeController(SMALL_TREE);
    await bootAndLand(controller, deferred);

    expect(controller.stats()).toMatchObject({
      active: true,
      decodedTiles: 3,
      residentTiles: 3,
      cachedTiles: 0,
    });
    const fetchesBefore = loadCalls.length;
    batches.length = 0;

    controller.setActive(false);
    await settle();
    expect(controller.stats()).toMatchObject({
      active: false,
      decodedTiles: 3,
      residentTiles: 0,
      cachedTiles: 3,
      memoryBudgetBytes: 0,
    });
    // Deactivation is when the cache ceiling matters: the GPU-pool share just
    // fell to zero while the payloads stayed. The stats must report the LRU's
    // own bound, and the contents must honour it — this is the invariant the
    // browser stress scenarios watch, locked here so `cacheBytes` cannot be
    // dropped from `stats()` without a unit failure.
    const deactivated = controller.stats();
    expect(deactivated.cacheBytes).toBe(256 * 1024 * 1024);
    expect(deactivated.cachedBytes).toBeGreaterThan(0);
    expect(deactivated.cachedBytes).toBeLessThanOrEqual(deactivated.cacheBytes);
    expect(batches.flatMap((batch) => batch.removed)).toHaveLength(3);

    // Camera changes while hidden update the stored view but cannot submit or
    // request tiles. Showing selects against that latest view from the LRU.
    controller.setCamera({ ...VIEW, viewProj: LOOK_AWAY });
    controller.setCamera(VIEW);
    controller.setActive(true);
    await settle();
    expect(loadCalls).toHaveLength(fetchesBefore);
    expect(controller.stats()).toMatchObject({
      active: true,
      decodedTiles: 3,
      residentTiles: 3,
      cachedTiles: 0,
    });
    controller.dispose();
  });

  it("an initially inactive controller defers hierarchy I/O until activated", async () => {
    const graph = createPageGraphSource({ depth: 0 });
    const controller = createLodController({
      source: graph.source,
      onTiles: () => {},
      scheduleRender: () => {},
      selectionDelayMs: 0,
      active: false,
    });

    controller.setCamera(VIEW);
    await settle();
    expect(graph.pageCalls).toHaveLength(0);
    expect(controller.stats()).toMatchObject({
      active: false,
      hierarchyInFlight: 0,
      queuedPages: 0,
      memoryBudgetBytes: 0,
    });

    controller.setActive(true);
    await settle();
    expect(graph.pageCalls).toEqual(["0-0-0-0"]);
    expect(controller.stats().active).toBe(true);
    controller.dispose();
  });

  it("evicts deselected tiles beyond the cache byte bound", async () => {
    const { controller, deferred, loadCalls } = makeController(SMALL_TREE, {
      cacheBytes: 1, // nothing survives deselection
    });
    await bootAndLand(controller, deferred);

    controller.setCamera({ ...VIEW, viewProj: LOOK_AWAY });
    await settle();
    expect(controller.stats().cachedBytes).toBe(0);

    const fetchesBefore = loadCalls.length;
    controller.setCamera(VIEW);
    await settle();
    expect(loadCalls.length).toBeGreaterThan(fetchesBefore); // refetched
    controller.dispose();
  });

  it("never applies a tile that resolves after setSource", async () => {
    const { controller, deferred, batches, source } =
      makeController(SMALL_TREE);
    await settle();
    controller.setCamera(VIEW);
    await settle();

    const stale = deferred.get("0-0-0-0")!;
    controller.setSource(source); // same source object; state still drops
    await settle();
    stale.resolve();
    await settle();

    const added = batches.flatMap((b) =>
      b.added.map((a) => keyToString(a.key)),
    );
    expect(added).toHaveLength(0);
    controller.dispose();
  });

  it("loads hierarchy pages lazily behind pageRef entries", async () => {
    const tree: Record<string, FakeEntry> = {
      "0-0-0-0": { pointCount: 10, children: ["1-0-0-0"] },
      "1-0-0-0": {
        pointCount: 0,
        pageRef: true,
        pageNodes: { "1-0-0-0": { pointCount: 40 } },
      },
    };
    const { controller, loadCalls, deferred } = makeController(tree);
    await settle();
    controller.setCamera(VIEW);
    await settle(); // triggers page load, reselects on arrival
    await settle();

    expect(loadCalls).toContain("1-0-0-0");
    for (const d of deferred.values()) d.resolve();
    await settle();
    expect(controller.stats().residentPoints).toBe(50);
    controller.dispose();
  });

  it("re-measures a node whose page replaced its hierarchy entry", async () => {
    // A page reference is a stand-in for the subtree it names, and the entry
    // that replaces it is a different node with its own spacing. Screen-space
    // error is cached per key against the current view, and the reference's
    // value is cached the moment its page is queued — so the replacement has
    // to drop it, or the node is measured for the rest of the view by the
    // placeholder it superseded.
    const tree: Record<string, FakeEntry> = {
      "0-0-0-0": { pointCount: 10, spacing: 0.1, children: ["1-0-0-0"] },
      "1-0-0-0": {
        pointCount: 0,
        pageRef: true,
        // Deliberately unlike the node behind it: both shipped sources happen
        // to agree here, and nothing in the contract says they must.
        spacing: 4,
        pageNodes: { "1-0-0-0": { pointCount: 40, spacing: 0.03 } },
      },
    };
    const { controller, deferred } = makeController(tree);
    await settle();
    // Parallel projection, so a projected spacing is the world spacing times
    // viewportHeight / (2 * parallelScale) = 50, with no distance in it.
    controller.setCamera(ORTHOGRAPHIC_VIEW);
    await settle();
    await settle();
    for (const d of deferred.values()) d.resolve();
    await settle();

    const frontier = controller.stats().selection.readyTerminalFrontier;
    expect(frontier.leafNodes).toBe(1);
    expect(frontier.projectedSpacingCssPx.p50).toBeCloseTo(0.03 * 50, 6);
    controller.dispose();
  });

  it("coalesces same-tick arrivals into one batch and one render", async () => {
    const { controller, deferred, batches, scheduleRender } =
      makeController(SMALL_TREE);
    await settle();
    controller.setCamera(VIEW);
    await settle();

    scheduleRender.mockClear();
    batches.length = 0;
    for (const d of deferred.values()) d.resolve();
    await settle();

    expect(batches).toHaveLength(1);
    expect(batches[0]!.added).toHaveLength(3);
    expect(scheduleRender).toHaveBeenCalledTimes(1);
    controller.dispose();
  });

  it("dispose removes residents, cancels fetches, and is idempotent", async () => {
    const { controller, deferred, batches } = makeController(SMALL_TREE);
    await settle();
    controller.setCamera(VIEW);
    await settle();
    deferred.get("0-0-0-0")!.resolve();
    await settle();

    batches.length = 0;
    controller.dispose();
    controller.dispose();

    expect(batches).toHaveLength(1);
    expect(batches[0]!.removed.map(keyToString)).toEqual(["0-0-0-0"]);
    expect(deferred.get("1-0-0-0")!.aborted).toBe(true);

    // A late resolve after dispose must do nothing.
    deferred.get("1-1-0-0")!.resolve();
    await settle();
    expect(batches).toHaveLength(1);
    expect(controller.stats().residentTiles).toBe(0);
  });

  it("dispose reports removals queued before the flush that never ran", async () => {
    const { controller, deferred, batches } = makeController(SMALL_TREE);
    await settle();
    controller.setCamera(VIEW);
    await settle();
    deferred.get("0-0-0-0")!.resolve();
    await settle();
    expect(controller.stats().residentTiles).toBe(1);

    batches.length = 0;
    // Deselect and tear down within one task: the removal is queued for a
    // microtask flush that dispose cancels.
    controller.setCamera({ ...VIEW, viewProj: LOOK_AWAY });
    expect(controller.stats().residentTiles).toBe(0);
    controller.dispose();
    await settle();

    const removed = batches.flatMap((b) => b.removed.map(keyToString));
    expect(removed).toContain("0-0-0-0");
  });

  it("does not fetch culled subtrees", async () => {
    const tree: Record<string, FakeEntry> = {
      "0-0-0-0": {
        pointCount: 10,
        bounds: { min: [-8, -8, -8], max: [8, 8, 8] },
        children: ["1-0-0-0"],
      },
      "1-0-0-0": {
        pointCount: 40,
        bounds: { min: [-8, -8, -8], max: [0, 0, 0] },
      },
    };
    // The whole tree is 50 points against the harness's 1000-point budget, so
    // the frustum is the only thing that can hold a node back.
    const { controller, loadCalls } = makeController(tree);
    await settle();
    // Clip space shifted by -5: only world coords in [4,6]^3 are visible.
    // The root cube [-8,8]^3 straddles that region, but the negative octant
    // '1-0-0-0' ([-8,0]^3) is fully outside and must never be fetched.
    controller.setCamera({
      ...VIEW,
      viewProj: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -5, -5, -5, 1],
    });
    await settle();
    expect(loadCalls).toContain("0-0-0-0");
    expect(loadCalls).not.toContain("1-0-0-0");
    controller.dispose();
  });
});

describe("createLodController — ready frontier and presentation", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  const makePresented = (
    tree: Record<string, FakeEntry>,
    overrides: Partial<LodControllerOptions> &
      Required<Pick<LodControllerOptions, "presentation">>,
  ) => {
    // Declared before construction: the constructor reports the first diameter
    // synchronously, and that first value is what the Auto tests read.
    const diameters: number[] = [];
    return {
      diameters,
      ...makeController(tree, {
        onPointDiameterCssPx: (value) => diameters.push(value),
        ...overrides,
      }),
    };
  };

  it("reports leaf, tile-readiness, and budget terminal coverage", async () => {
    const tree: Record<string, FakeEntry> = {
      "0-0-0-0": {
        pointCount: 100,
        spacing: 4,
        children: ["1-0-0-0", "1-1-0-0"],
      },
      "1-0-0-0": { pointCount: 10, spacing: 1 },
      "1-1-0-0": { pointCount: 10, spacing: 1 },
    };
    const { controller, deferred } = makePresented(tree, {
      pointBudget: 110,
      presentation: { mode: "fixed", diameterCssPx: 3 },
    });
    await settle();
    controller.setCamera({ ...VIEW, position: [0, 0, 10] });
    await settle();

    deferred.get("0-0-0-0")!.resolve();
    await settle();
    expect(controller.stats().selection.readyTerminalFrontier).toMatchObject({
      count: 1,
      leafNodes: 0,
      tileBlockedNodes: 1,
      budgetBlockedNodes: 1,
    });

    const selectedChild = [...deferred.keys()].find(
      (key) => key !== "0-0-0-0",
    )!;
    deferred.get(selectedChild)!.resolve();
    await settle();
    expect(controller.stats().selection.readyTerminalFrontier).toMatchObject({
      count: 2,
      leafNodes: 1,
      tileBlockedNodes: 0,
      budgetBlockedNodes: 1,
    });
    controller.dispose();
  });

  it("reports cutoff terminals instead of storage-level descendants", async () => {
    const { controller, deferred } = makePresented(SMALL_TREE, {
      presentation: { mode: "fixed", diameterCssPx: 2 },
    });
    await settle();
    controller.setRefinementCutoffPx(1_000_000);
    controller.setCamera({ ...VIEW, position: [0, 0, 10] });
    await settle();
    deferred.get("0-0-0-0")!.resolve();
    await settle();

    expect(controller.stats().selection.readyTerminalFrontier).toMatchObject({
      count: 1,
      cutoffNodes: 1,
      leafNodes: 0,
    });
    controller.dispose();
  });

  it("projects the spacing supplied by each hierarchy node", async () => {
    const tree: Record<string, FakeEntry> = {
      "0-0-0-0": {
        pointCount: 100,
        spacing: 100,
        children: ["1-0-0-0"],
      },
      "1-0-0-0": { pointCount: 10, spacing: 0.001 },
    };
    const { controller, deferred } = makePresented(tree, {
      presentation: { mode: "fixed", diameterCssPx: 2 },
    });
    await settle();
    controller.setCamera({ ...VIEW, position: [0, 0, 10] });
    await settle();
    for (const pending of deferred.values()) pending.resolve();
    await settle();

    const p75 =
      controller.stats().selection.readyTerminalFrontier.projectedSpacingCssPx
        .p75;
    expect(p75).not.toBeNull();
    expect(p75!).toBeLessThan(0.01);
    controller.dispose();
  });

  /** Projected spacing of the whole ready frontier, the diagnostic hosts read. */
  const projectedSpacingP75 = (controller: LodController): number | null =>
    controller.stats().selection.readyTerminalFrontier.projectedSpacingCssPx
      .p75;

  it("projects spacing from the parallel scale, not the distance, under an orthographic camera", async () => {
    const tree: Record<string, FakeEntry> = {
      "0-0-0-0": { pointCount: 100, spacing: 2 },
    };
    const near = makePresented(tree, {
      presentation: { mode: "fixed", diameterCssPx: 2 },
    });
    const far = makePresented(tree, {
      presentation: { mode: "fixed", diameterCssPx: 2 },
    });
    await settle();
    // Half-height 1 over 100 px = 50 px/m, so a 2 m spacing projects to 100 px
    // whether the camera sits on the node or a kilometre away.
    near.controller.setCamera({ ...ORTHOGRAPHIC_VIEW, position: [0, 0, 10] });
    far.controller.setCamera({ ...ORTHOGRAPHIC_VIEW, position: [0, 0, 1000] });
    await settle();
    for (const pending of near.deferred.values()) pending.resolve();
    for (const pending of far.deferred.values()) pending.resolve();
    await settle();

    expect(projectedSpacingP75(near.controller)).toBeCloseTo(100);
    expect(projectedSpacingP75(far.controller)).toBeCloseTo(100);

    // Zooming a parallel camera in halves the scale and doubles the spacing.
    near.controller.setCamera({
      ...ORTHOGRAPHIC_VIEW,
      position: [0, 0, 10],
      parallelScale: 0.5,
    });
    await settle();
    expect(projectedSpacingP75(near.controller)).toBeCloseTo(200);

    near.controller.dispose();
    far.controller.dispose();
  });

  it("treats a projection-mode swap as a camera change", async () => {
    const tree: Record<string, FakeEntry> = {
      "0-0-0-0": { pointCount: 100, spacing: 2 },
    };
    const { controller, deferred } = makePresented(tree, {
      presentation: { mode: "fixed", diameterCssPx: 2 },
    });
    await settle();
    controller.setCamera(VIEW);
    await settle();
    for (const pending of deferred.values()) pending.resolve();
    await settle();
    const perspectiveSpacing = projectedSpacingP75(controller);
    const generation = controller.stats().selection.generation;

    // Same matrix, same position, same viewport: only the projection differs,
    // and it must not be mistaken for the unchanged view a host re-feeds every
    // render.
    controller.setCamera(ORTHOGRAPHIC_VIEW);
    await settle();
    expect(controller.stats().selection.generation).toBeGreaterThan(generation);
    expect(projectedSpacingP75(controller)).not.toBeCloseTo(
      perspectiveSpacing!,
    );
    controller.dispose();
  });

  it("treats a viewport resize as a camera change, an identical view as none", async () => {
    const { controller, deferred } = makeController(SMALL_TREE);
    await bootAndLand(controller, deferred);
    const generation = controller.stats().selection.generation;

    // Hosts re-feed the camera every render: an identical view is a no-op.
    controller.setCamera({ ...VIEW });
    await settle();
    expect(controller.stats().selection.generation).toBe(generation);

    // A width-only resize changes the aspect and must re-run selection.
    controller.setCamera({ ...VIEW, viewportWidthCssPx: 150 });
    await settle();
    const widened = controller.stats().selection.generation;
    expect(widened).toBeGreaterThan(generation);

    // So must a height-only resize on top of it.
    controller.setCamera({
      ...VIEW,
      viewportWidthCssPx: 150,
      viewportHeightCssPx: 150,
    });
    await settle();
    expect(controller.stats().selection.generation).toBeGreaterThan(widened);
    controller.dispose();
  });

  it("keeps the ready parent terminal while a hierarchy page is unavailable", async () => {
    let resolveChildPage!: (infos: NodeInfo[]) => void;
    let resolveRootTile!: (tile: TileData) => void;
    const source: TileSource = {
      metadata: () => ({ pointCount: 110 }),
      async nodes(key) {
        if (key.level === 0) {
          return [
            {
              key: { level: 0, x: 0, y: 0, z: 0 },
              pointCount: 100,
              bounds: { min: [-1, -1, -1], max: [1, 1, 1] },
              spacing: 1,
              children: [{ level: 1, x: 0, y: 0, z: 0 }],
            },
            {
              key: { level: 1, x: 0, y: 0, z: 0 },
              pointCount: 0,
              bounds: { min: [-1, -1, -1], max: [0, 0, 0] },
              spacing: 0.5,
              pageRef: true,
            },
          ];
        }
        return new Promise<NodeInfo[]>((resolve) => {
          resolveChildPage = resolve;
        });
      },
      loadTile(key) {
        if (key.level === 0) {
          return new Promise<TileData>((resolve) => {
            resolveRootTile = resolve;
          });
        }
        return new Promise<TileData>(() => {});
      },
    };
    const controller = createLodController({
      source,
      onTiles: () => {},
      scheduleRender: () => {},
      selectionDelayMs: 0,
    });
    await settle();
    controller.setCamera({ ...VIEW, position: [0, 0, 10] });
    await settle();
    resolveRootTile(makeTile(100));
    await settle();
    expect(controller.stats().selection.readyTerminalFrontier).toMatchObject({
      count: 1,
      hierarchyBlockedNodes: 1,
    });

    resolveChildPage([
      {
        key: { level: 1, x: 0, y: 0, z: 0 },
        pointCount: 10,
        bounds: { min: [-1, -1, -1], max: [0, 0, 0] },
        spacing: 0.5,
      },
    ]);
    await settle();
    expect(controller.stats().selection.readyTerminalFrontier).toMatchObject({
      count: 1,
      hierarchyBlockedNodes: 0,
      tileBlockedNodes: 1,
    });
    controller.dispose();
  });

  it("keeps a mixed frontier covered by its coarsest terminal spacing", async () => {
    const tree: Record<string, FakeEntry> = {
      "0-0-0-0": {
        pointCount: 100,
        spacing: 0.08,
        children: ["1-0-0-0", "1-1-0-0", "1-0-1-0", "1-1-1-0"],
      },
      "1-0-0-0": { pointCount: 10, spacing: 0.02 },
      "1-1-0-0": { pointCount: 10, spacing: 0.02 },
      "1-0-1-0": { pointCount: 10, spacing: 0.02 },
      "1-1-1-0": { pointCount: 10, spacing: 0.02 },
    };
    const { controller, deferred } = makePresented(tree, {
      pointBudget: 130,
      presentation: {
        mode: "auto",
        userScale: 1,
        minDiameterCssPx: 0.5,
        maxDiameterCssPx: 10,
      },
    });
    await settle();
    controller.setCamera(ORTHOGRAPHIC_VIEW);
    await settle();
    for (const pending of deferred.values()) pending.resolve();
    await settle();

    const stats = controller.stats();
    expect(stats.selection.readyTerminalFrontier).toMatchObject({
      count: 4,
      budgetBlockedNodes: 1,
      projectedSpacingCssPx: {
        p75: 1,
        max: 4,
      },
    });
    expect(stats.presentation.diameterCssPx).toBe(4);
    controller.dispose();
  });

  it("holds the selected Auto diameter while detail tiles arrive", async () => {
    const tree: Record<string, FakeEntry> = {
      "0-0-0-0": {
        pointCount: 100,
        spacing: 0.08,
        children: ["1-0-0-0", "1-1-0-0", "1-0-1-0", "1-1-1-0"],
      },
      "1-0-0-0": { pointCount: 10, spacing: 0.02 },
      "1-1-0-0": { pointCount: 10, spacing: 0.02 },
      "1-0-1-0": { pointCount: 10, spacing: 0.02 },
      "1-1-1-0": { pointCount: 10, spacing: 0.02 },
    };
    const { controller, deferred, diameters } = makePresented(tree, {
      pointBudget: 1_000,
      presentation: {
        mode: "auto",
        userScale: 1,
        minDiameterCssPx: 0.5,
        maxDiameterCssPx: 10,
      },
    });
    await settle();
    controller.setCamera(ORTHOGRAPHIC_VIEW);
    await settle();

    // The complete selection's level-1 spacing is known before its payloads
    // arrive. Auto adopts it once, instead of drawing early detail at the
    // root's 4 px spacing and shrinking every actor when the last child lands.
    expect(controller.stats().presentation.diameterCssPx).toBeCloseTo(1);
    expect(diameters).toEqual([2, 1]);

    deferred.get("0-0-0-0")!.resolve();
    await settle();
    expect(controller.stats().presentation.diameterCssPx).toBeCloseTo(1);

    for (const keyString of tree["0-0-0-0"]!.children!) {
      deferred.get(keyString)!.resolve();
      await settle();
      expect(controller.stats().presentation.diameterCssPx).toBeCloseTo(1);
    }
    expect(diameters).toEqual([2, 1]);
    expect(
      controller.stats().selection.readyTerminalFrontier.projectedSpacingCssPx
        .max,
    ).toBeCloseTo(1);
    controller.dispose();
  });

  it("sizes Auto points for the effective progressively thinned spacing", async () => {
    const tree: Record<string, FakeEntry> = {
      "0-0-0-0": { pointCount: 100, spacing: 0.08 },
    };
    const { controller, deferred } = makePresented(tree, {
      presentation: {
        mode: "auto",
        userScale: 1,
        minDiameterCssPx: 0.5,
        maxDiameterCssPx: 10,
      },
    });
    await settle();
    controller.setCamera(ORTHOGRAPHIC_VIEW);
    await settle();
    deferred.get("0-0-0-0")!.resolve();
    await settle();
    expect(controller.stats().presentation.diameterCssPx).toBeCloseTo(4);
    const generation = controller.stats().selection.generation;

    controller.setDensityFraction(0.25);
    const thinned = controller.stats();
    expect(thinned.presentation.diameterCssPx).toBeCloseTo(8);
    expect(
      thinned.selection.readyTerminalFrontier.projectedSpacingCssPx.max,
    ).toBeCloseTo(8);
    expect(thinned.selection.generation).toBe(generation);

    controller.setDensityFraction(1);
    expect(controller.stats().presentation.diameterCssPx).toBeCloseTo(4);
    controller.dispose();
  });

  it("uses the largest terminal spacing with clamps throughout interaction", async () => {
    const { controller, deferred, diameters } = makePresented(AUTO_LEAF, {
      presentation: { mode: "auto", userScale: 1 },
    });
    expect(diameters).toEqual([2]);
    await settle();
    controller.setCamera({ ...VIEW, position: [0, 0, 10] });
    await settle();
    deferred.get("0-0-0-0")!.resolve();
    await settle();
    expect(controller.stats().presentation.diameterCssPx).toBeCloseTo(4);

    vi.advanceTimersByTime(300);
    await settle();
    expect(
      controller.stats().selection.readyTerminalFrontier.projectedSpacingCssPx
        .p75,
    ).toBeCloseTo(5);
    expect(controller.stats().presentation.diameterCssPx).toBeCloseTo(4);

    controller.beginInteraction();
    expect(controller.stats().presentation.diameterCssPx).toBeCloseTo(4);
    controller.endInteraction();
    expect(controller.stats().presentation.diameterCssPx).toBeCloseTo(4);
    vi.advanceTimersByTime(300);
    await settle();
    expect(controller.stats().presentation.diameterCssPx).toBeCloseTo(4);

    controller.setPresentation({
      mode: "auto",
      userScale: 0.5,
      minDiameterCssPx: 1.5,
      maxDiameterCssPx: 5,
    });
    expect(controller.stats().presentation.diameterCssPx).toBe(2.5);
    controller.dispose();
  });

  it("applies Auto size changes during interaction without a release phase", async () => {
    const { controller, deferred } = makePresented(AUTO_LEAF, {
      presentation: { mode: "auto", userScale: 1 },
    });
    await settle();
    controller.setCamera({ ...VIEW, position: [0, 0, 10] });
    await settle();
    deferred.get("0-0-0-0")!.resolve();
    await settle();
    vi.advanceTimersByTime(300);
    await settle();
    controller.beginInteraction();
    controller.setPresentation({
      mode: "auto",
      userScale: 0.75,
    });
    expect(controller.stats().presentation.diameterCssPx).toBeCloseTo(3);

    controller.endInteraction();
    expect(controller.stats().presentation.diameterCssPx).toBeCloseTo(3);

    vi.advanceTimersByTime(300);
    await settle();
    expect(controller.stats().presentation.diameterCssPx).toBeCloseTo(3);
    controller.dispose();
  });

  it("applies repeated settled Auto scale changes immediately", async () => {
    const { controller, deferred, diameters } = makePresented(AUTO_LEAF, {
      presentation: { mode: "auto", userScale: 1 },
    });
    await settle();
    controller.setCamera({ ...VIEW, position: [0, 0, 10] });
    await settle();
    deferred.get("0-0-0-0")!.resolve();
    await settle();
    vi.advanceTimersByTime(2000);
    await settle();
    expect(controller.stats().presentation.diameterCssPx).toBe(4);

    controller.setPresentation({ mode: "auto", userScale: 1.2 });
    let stats = controller.stats().presentation;
    expect(stats.diameterCssPx).toBeCloseTo(4.8);

    controller.setPresentation({ mode: "auto", userScale: 1.4 });
    stats = controller.stats().presentation;
    expect(stats.diameterCssPx).toBeCloseTo(5.6);
    expect(diameters.at(-2)).toBeCloseTo(4.8);
    expect(diameters.at(-1)).toBeCloseTo(5.6);
    controller.dispose();
  });
});

describe("createLodController — failing sources", () => {
  it("stops re-requesting a tile whose fetch keeps failing", async () => {
    const errors: unknown[] = [];
    const { controller, loadCalls, deferred } = makeController(SMALL_TREE, {
      onError: (error) => errors.push(error),
    });
    await settle();
    controller.setCamera(VIEW);
    await settle();
    expect(loadCalls).toHaveLength(3);

    const failEveryFetch = async (): Promise<void> => {
      for (const d of deferred.values()) d.reject(new Error("500"));
      await settle();
    };
    await failEveryFetch();

    // Retries are bounded: after the allowance runs out the keys are dropped,
    // so any number of further selections issues nothing.
    for (let round = 0; round < 10; round += 1) {
      controller.refresh();
      await settle();
      await failEveryFetch();
    }
    expect(loadCalls).toHaveLength(9); // 3 tiles x 3 attempts
    expect(errors).toHaveLength(9);
    controller.dispose();
  });

  it("refetches a rested tile once the outage ends", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    const { controller, loadCalls, deferred } = makeController(SMALL_TREE, {
      onError: () => {},
    });
    await settle();
    controller.setCamera(VIEW);
    await settle();

    // A brief outage burns the whole allowance for every tile.
    for (let round = 0; round < 4; round += 1) {
      for (const d of deferred.values()) d.reject(new Error("500"));
      await settle();
      controller.refresh();
      await settle();
    }
    expect(loadCalls).toHaveLength(9);

    // The server recovers. Resting is a backoff, not an eviction: once the
    // keys have been quiet long enough they are fetchable again, and this
    // time they load.
    vi.setSystemTime(31_000);
    controller.refresh();
    await settle();
    expect(loadCalls).toHaveLength(12);

    for (const d of deferred.values()) d.resolve();
    await settle();
    expect(controller.stats().residentTiles).toBe(3);

    controller.dispose();
    vi.useRealTimers();
  });

  it("stops re-requesting a hierarchy page that keeps failing", async () => {
    const errors: unknown[] = [];
    const bounds = {
      min: [-0.5, -0.5, -0.5] as [number, number, number],
      max: [0.5, 0.5, 0.5] as [number, number, number],
    };
    let pageCalls = 0;
    const source: TileSource = {
      metadata: () => METADATA,
      async nodes(key: VoxelKey) {
        pageCalls += 1;
        if (keyToString(key) !== "0-0-0-0") throw new Error("500");
        return [
          {
            key: { level: 0, x: 0, y: 0, z: 0 },
            pointCount: 0, // structural: no tile of its own
            bounds,
            spacing: 0.1,
            children: [{ level: 1, x: 0, y: 0, z: 0 }],
          },
          {
            key: { level: 1, x: 0, y: 0, z: 0 },
            pointCount: 60,
            bounds,
            spacing: 0.05,
            pageRef: true, // its page is the one that always fails
          },
        ];
      },
      loadTile: () => new Promise<TileData>(() => {}),
    };
    const sink = collectBatches();
    const controller = createLodController({
      source,
      onTiles: sink.onTiles,
      scheduleRender: sink.scheduleRender,
      selectionDelayMs: 0,
      onError: (error) => errors.push(error),
    });
    await settle();
    controller.setCamera(VIEW);
    await settle();

    for (let round = 0; round < 10; round += 1) {
      controller.refresh();
      await settle();
    }
    expect(pageCalls).toBe(4); // the root page, then 3 attempts at the child
    expect(errors).toHaveLength(3);
    controller.dispose();
  });

  it("recovers when the very first hierarchy request fails", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    let offline = true;
    let pageCalls = 0;
    const fake = makeFakeSource(SMALL_TREE);
    const source: TileSource = {
      metadata: fake.source.metadata,
      async nodes(key: VoxelKey) {
        pageCalls += 1;
        if (offline) throw new Error("503");
        return fake.source.nodes(key);
      },
      loadTile: fake.source.loadTile,
    };
    const sink = collectBatches();
    const controller = createLodController({
      source,
      onTiles: sink.onTiles,
      scheduleRender: sink.scheduleRender,
      pointBudget: 1000,
      selectionDelayMs: 0,
      onError: () => {},
    });
    await settle();
    controller.setCamera(VIEW);
    await settle();

    // The bootstrap request burns its allowance while the server is down.
    for (let round = 0; round < 6; round += 1) {
      controller.refresh();
      await settle();
    }
    expect(pageCalls).toBe(3);
    expect(controller.stats().residentTiles).toBe(0);

    // Nothing else can ask for the root page, so selection has to. Once the
    // backoff lapses and the server is back, the controller bootstraps.
    offline = false;
    vi.setSystemTime(31_000);
    controller.refresh();
    await settle();
    for (const d of fake.deferred.values()) d.resolve();
    await settle();
    expect(controller.stats().residentTiles).toBe(3);

    controller.dispose();
    vi.useRealTimers();
  });

  it("refetches a transiently failed tile with the camera never touched", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    const { controller, loadCalls, deferred } = makeController(SMALL_TREE, {
      onError: () => {},
    });
    await settle();
    controller.setCamera(VIEW);
    await settle();
    expect(loadCalls).toHaveLength(3);

    // One tile 500s while its siblings land. The scene is converged — still
    // camera, settled budget, nothing else in flight — so no selection pass
    // will ever run to ask again.
    deferred.get("0-0-0-0")!.resolve();
    deferred.get("1-1-0-0")!.resolve();
    deferred.get("1-0-0-0")!.reject(new Error("500"));
    await settle();
    expect(controller.stats().residentTiles).toBe(2);
    const generation = controller.stats().selection.generation;

    await vi.advanceTimersByTimeAsync(1000);
    await settle();
    expect(loadCalls).toEqual([
      "0-0-0-0",
      "1-0-0-0",
      "1-1-0-0",
      "1-0-0-0", // the retry the controller issued by itself
    ]);
    deferred.get("1-0-0-0")!.resolve();
    await settle();
    expect(controller.stats().residentTiles).toBe(3);
    // No reselection was needed to get there.
    expect(controller.stats().selection.generation).toBe(generation);

    controller.dispose();
    vi.useRealTimers();
  });

  it("waits out the backoff before refetching a tile that spent its attempts", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    const { controller, loadCalls, deferred } = makeController(SMALL_TREE, {
      onError: () => {},
    });
    await settle();
    controller.setCamera(VIEW);
    await settle();
    expect(loadCalls).toHaveLength(3);

    // The retry timers burn the whole allowance on their own: no refresh, no
    // camera move, three attempts a second apart.
    for (let attempt = 0; attempt < 3; attempt += 1) {
      for (const d of deferred.values()) d.reject(new Error("500"));
      await settle();
      await vi.advanceTimersByTimeAsync(1000);
      await settle();
    }
    expect(loadCalls).toHaveLength(9); // 3 tiles x 3 attempts

    // Now the keys rest: the prompt retry must not fire again until the whole
    // backoff has elapsed since the last failure (at 2000 ms).
    await vi.advanceTimersByTimeAsync(28_999);
    await settle();
    expect(loadCalls).toHaveLength(9);

    await vi.advanceTimersByTimeAsync(2);
    await settle();
    expect(loadCalls).toHaveLength(12);
    for (const d of deferred.values()) d.resolve();
    await settle();
    expect(controller.stats().residentTiles).toBe(3);

    controller.dispose();
    vi.useRealTimers();
  });

  it("dispose cancels armed retries", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    const { controller, loadCalls, deferred } = makeController(SMALL_TREE, {
      onError: () => {},
    });
    await settle();
    controller.setCamera(VIEW);
    await settle();
    for (const d of deferred.values()) d.reject(new Error("500"));
    await settle();
    expect(loadCalls).toHaveLength(3);

    controller.dispose();
    expect(vi.getTimerCount()).toBe(0);
    await vi.advanceTimersByTimeAsync(60_000);
    await settle();
    expect(loadCalls).toHaveLength(3);

    vi.useRealTimers();
  });

  it("a new source cancels retries armed against the old one", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    const { controller, deferred } = makeController(SMALL_TREE, {
      onError: () => {},
    });
    await settle();
    controller.setCamera(VIEW);
    await settle();
    for (const d of deferred.values()) d.reject(new Error("500"));
    await settle();

    const replacement = makeFakeSource(SMALL_TREE);
    controller.setSource(replacement.source);
    expect(vi.getTimerCount()).toBe(0);
    await settle();
    const afterSwap = replacement.loadCalls.length;
    await vi.advanceTimersByTimeAsync(60_000);
    await settle();
    expect(replacement.loadCalls).toHaveLength(afterSwap);

    controller.dispose();
    vi.useRealTimers();
  });

  it("refetches a transiently failed hierarchy page with the camera never touched", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    let offline = true;
    let pageCalls = 0;
    const bounds = {
      min: [-0.5, -0.5, -0.5] as [number, number, number],
      max: [0.5, 0.5, 0.5] as [number, number, number],
    };
    const source: TileSource = {
      metadata: () => METADATA,
      async nodes(key: VoxelKey) {
        pageCalls += 1;
        if (keyToString(key) === "0-0-0-0") {
          return [
            {
              key: { level: 0, x: 0, y: 0, z: 0 },
              pointCount: 0, // structural: no tile of its own
              bounds,
              spacing: 0.1,
              children: [{ level: 1, x: 0, y: 0, z: 0 }],
            },
            {
              key: { level: 1, x: 0, y: 0, z: 0 },
              pointCount: 60,
              bounds,
              spacing: 0.05,
              pageRef: true,
            },
          ];
        }
        if (offline) throw new Error("500");
        return [
          {
            key: { level: 1, x: 0, y: 0, z: 0 },
            pointCount: 60,
            bounds,
            spacing: 0.05,
          },
        ];
      },
      loadTile: async () => makeTile(60),
    };
    const sink = collectBatches();
    const controller = createLodController({
      source,
      onTiles: sink.onTiles,
      scheduleRender: sink.scheduleRender,
      selectionDelayMs: 0,
      onError: () => {},
    });
    await settle();
    controller.setCamera(VIEW);
    await settle();

    // The page that unblocks the only refinable subtree failed, so nothing is
    // drawn and nothing is left to trigger another selection pass.
    expect(pageCalls).toBe(2);
    expect(controller.stats().residentTiles).toBe(0);

    offline = false;
    await vi.advanceTimersByTimeAsync(1000);
    await settle();
    expect(pageCalls).toBe(3);
    expect(controller.stats().residentTiles).toBe(1);

    controller.dispose();
    vi.useRealTimers();
  });

  it("keeps aborted fetches retryable", async () => {
    // Deselection, setSource, and dispose abort normally — they must never
    // count against the failure allowance, however often they happen.
    const { controller, loadCalls } = makeController(SMALL_TREE);
    await settle();

    for (let round = 1; round <= 4; round += 1) {
      controller.setCamera(VIEW);
      await settle();
      expect(loadCalls).toHaveLength(3 * round);
      controller.setCamera({ ...VIEW, viewProj: LOOK_AWAY });
      await settle();
    }
    controller.dispose();
  });

  it("a new source clears the failure memory", async () => {
    const errors: unknown[] = [];
    const { controller, loadCalls, deferred } = makeController(SMALL_TREE, {
      onError: (error) => errors.push(error),
    });
    await settle();
    controller.setCamera(VIEW);
    await settle();
    for (let round = 0; round < 4; round += 1) {
      for (const d of deferred.values()) d.reject(new Error("500"));
      await settle();
      controller.refresh();
      await settle();
    }
    expect(loadCalls).toHaveLength(9);

    const replacement = makeFakeSource(SMALL_TREE);
    controller.setSource(replacement.source);
    await settle();
    controller.refresh();
    await settle();
    expect(replacement.loadCalls).toHaveLength(3);
    controller.dispose();
  });
});

describe("createLodController — budget and memory ceiling", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  const makeBudgeted = (
    tree: Record<string, FakeEntry>,
    pointBudget: number,
    memory?: number,
  ) => makeController(tree, { pointBudget, memory });

  it("applies a budget drop immediately during interaction", async () => {
    const { controller, deferred } = makeBudgeted(SMALL_TREE, 1000);
    // t=0: interacting
    await bootAndLand(controller, deferred);
    expect(controller.stats().residentTiles).toBe(3);

    // Lower the budget mid-gesture: both children (60 points each) fall out of
    // the 150-point selection and leave submitted work immediately.
    controller.setPointBudget(150);
    await settle();
    expect(controller.stats().residentTiles).toBe(1);
    expect(controller.stats().cachedTiles).toBe(2);

    // Settling does not change the budget or resurrect dormant actors.
    vi.advanceTimersByTime(300);
    await settle();
    expect(controller.stats().residentTiles).toBe(1);
    controller.dispose();
  });

  it("counts nested begin/end pairs down to zero", async () => {
    const { controller } = makeBudgeted(SMALL_TREE, 2_000_000);
    await settle();
    controller.setCamera(VIEW);
    await settle();

    controller.beginInteraction();
    controller.beginInteraction();
    expect(controller.stats().interactionDepth).toBe(2);
    controller.endInteraction();
    expect(controller.stats().interactionDepth).toBe(1);
    // An unbalanced extra end must not drive the count negative.
    controller.endInteraction();
    controller.endInteraction();
    expect(controller.stats().interactionDepth).toBe(0);
    controller.dispose();
  });

  // Below the 100k-resident-point measurement threshold the controller
  // converts bytes to points with the 16 bytes/point fallback, so a byte
  // budget of 16 * N caps selection at N points.
  const BYTES_PER_POINT = 16;

  it("the memory budget caps the point budget", async () => {
    const { controller, deferred } = makeBudgeted(
      SMALL_TREE,
      1000,
      150 * BYTES_PER_POINT,
    );
    await bootAndLand(controller, deferred);
    // Root (100 points) fits the 150-point ceiling; the 60-point children
    // would overshoot and stay out.
    expect(controller.stats().pointBudget).toBe(150);
    expect(controller.stats().residentTiles).toBe(1);
    expect(controller.stats().residentPoints).toBe(100);
    controller.dispose();
  });

  it("controllers on a shared pool split the byte budget and rebalance", async () => {
    const pool = createMemoryPool({ totalBytes: 300 * BYTES_PER_POINT });
    const first = makeFakeSource(SMALL_TREE);
    const sink = collectBatches();
    const controller = createLodController({
      source: first.source,
      onTiles: sink.onTiles,
      scheduleRender: sink.scheduleRender,
      selectionDelayMs: 0,
      memory: pool,
    });
    await settle();
    controller.setCamera(VIEW);
    await settle();
    for (const d of first.deferred.values()) d.resolve();
    await settle();
    // Alone in the pool: the full 300-point ceiling fits all three tiles.
    expect(controller.stats().memoryBudgetBytes).toBe(300 * BYTES_PER_POINT);
    expect(controller.stats().residentTiles).toBe(3);

    // A second cloud joins: shares drop to 150 points each and the first
    // controller reselects down — N clouds must not multiply GPU memory by N.
    const second = makeFakeSource(SMALL_TREE);
    const other = createLodController({
      source: second.source,
      onTiles: () => {},
      scheduleRender: () => {},
      selectionDelayMs: 0,
      memory: pool,
    });
    await settle();
    expect(controller.stats().memoryBudgetBytes).toBe(150 * BYTES_PER_POINT);
    expect(controller.stats().residentTiles).toBe(1);

    // A hidden cloud leaves GPU-budget membership without being disposed: the
    // visible survivor's share grows and its children are reselected.
    other.setActive(false);
    await settle();
    expect(pool.memberCount()).toBe(1);
    expect(other.stats().memoryBudgetBytes).toBe(0);
    for (const d of first.deferred.values()) d.resolve();
    await settle();
    expect(controller.stats().memoryBudgetBytes).toBe(300 * BYTES_PER_POINT);
    expect(controller.stats().residentTiles).toBe(3);
    other.dispose();
    controller.dispose();
  });
});

describe("createLodController — selection stats", () => {
  it("reports why the latest selection stopped refining", async () => {
    const { controller, deferred } = makeController(SMALL_TREE, {
      pointBudget: 150,
    });
    await settle();
    controller.setCamera(VIEW);
    await settle();
    deferred.get("0-0-0-0")!.resolve();
    await settle();

    expect(controller.stats().selection).toMatchObject({
      targetTiles: 1,
      targetPoints: 100,
      selectedNodes: 1,
      budgetSkippedNodes: 2,
      budgetSkippedPoints: 120,
    });
    controller.dispose();
  });

  it("tracks frontier identity and refinement-cutoff changes", async () => {
    const { controller, deferred, source } = makeController(SMALL_TREE);
    await settle();
    controller.setCamera({ ...VIEW, position: [0, 0, 5] });
    await settle();
    deferred.get("0-0-0-0")!.resolve();
    await settle();
    const before = controller.stats();

    controller.setRefinementCutoffPx(1_000_000_000);
    const cutoff = controller.stats();
    expect(cutoff.refinementCutoffPx).toBe(1_000_000_000);
    expect(cutoff.selection.generation).toBeGreaterThan(
      before.selection.generation,
    );
    expect(cutoff.selection.sseStoppedNodes).toBeGreaterThan(0);
    expect(cutoff.selection.readyTerminalFrontier.cutoffNodes).toBeGreaterThan(
      0,
    );

    for (const d of deferred.values()) d.resolve();
    await settle();
    const targetRevision = controller.stats().selection.targetRevision;
    controller.refresh();
    expect(controller.stats().selection.targetRevision).toBe(targetRevision);

    controller.setSource(source);
    expect(controller.stats().selection.targetTiles).toBe(0);
    controller.dispose();
  });
});

describe("createLodController — batch coalescing", () => {
  const AWAY = { ...VIEW, viewProj: LOOK_AWAY };

  it("emits no actor for a cached tile selected and dropped in one task", async () => {
    const { controller, deferred, batches, visible, violations } =
      makeMirrored(SMALL_TREE);
    await bootAndLand(controller, deferred);
    controller.setCamera(AWAY);
    await settle();
    expect(visible.size).toBe(0);
    expect(controller.stats().cachedTiles).toBe(3);

    batches.length = 0;
    // One task, two selections: the cache hits make all three resident and
    // the next selection drops them again before the flush microtask runs.
    controller.setCamera(VIEW);
    expect(controller.stats().residentTiles).toBe(3);
    controller.setCamera(AWAY);
    expect(controller.stats().residentTiles).toBe(0);
    await settle();

    expect(violations).toEqual([]);
    expect(batches).toEqual([]);
    expect(visible.size).toBe(0);
    controller.dispose();
  });

  it("emits no actor for a decoded tile dropped before its flush", async () => {
    const { controller, deferred, batches, visible, violations } =
      makeMirrored(SMALL_TREE);
    await settle();
    controller.setCamera(VIEW);
    await settle();
    batches.length = 0;

    deferred.get("0-0-0-0")!.resolve();
    // Exactly one turn: the fetch continuation claims residency and queues a
    // flush that has not run yet.
    await Promise.resolve();
    expect(controller.stats().residentTiles).toBe(1);
    controller.setCamera(AWAY);
    expect(controller.stats().residentTiles).toBe(0);
    await settle();

    expect(violations).toEqual([]);
    expect(batches).toEqual([]);
    expect(visible.size).toBe(0);
    controller.dispose();
  });

  it("leaves the renderer untouched when a tile is dropped and reselected", async () => {
    const { controller, deferred, batches, visible, violations } =
      makeMirrored(SMALL_TREE);
    await bootAndLand(controller, deferred);
    expect(visible.size).toBe(3);

    batches.length = 0;
    controller.setCamera(AWAY);
    controller.setCamera(VIEW);
    await settle();

    expect(violations).toEqual([]);
    expect(batches).toEqual([]);
    expect(visible.size).toBe(3);
    expect(controller.stats().residentTiles).toBe(3);
    expect(controller.stats().residentPoints).toBe(220);
    controller.dispose();
  });

  it("keeps renderer state equal to residency across a long sequence", async () => {
    const { controller, deferred, visible, visiblePoints, violations } =
      makeMirrored(SMALL_TREE);
    const agrees = (): void => {
      const stats = controller.stats();
      expect(violations).toEqual([]);
      expect(visible.size).toBe(stats.residentTiles);
      expect(visiblePoints()).toBe(stats.residentPoints);
      expect(stats.residentPoints).toBeGreaterThanOrEqual(0);
      expect(stats.residentBytes).toBeGreaterThanOrEqual(0);
      expect(stats.decodedBytes).toBe(stats.residentBytes + stats.cachedBytes);
    };

    await settle();
    agrees();
    controller.setCamera(VIEW);
    await settle();
    agrees();

    deferred.get("0-0-0-0")!.resolve();
    await settle();
    agrees();

    controller.setPointBudget(150); // drops both children
    await settle();
    agrees();

    for (const d of deferred.values()) d.resolve();
    await settle();
    agrees();

    controller.setPointBudget(1000);
    await settle();
    agrees();

    controller.setCamera(AWAY);
    await settle();
    agrees();

    controller.setCamera(VIEW);
    await settle();
    agrees();

    controller.setActive(false);
    await settle();
    expect(controller.stats().residentPoints).toBe(0);
    expect(controller.stats().residentBytes).toBe(0);
    agrees();

    controller.setActive(true);
    await settle();
    agrees();

    controller.dispose();
    await settle();
    expect(visible.size).toBe(0);
  });

  it("never batches a structural node that carries no tile", async () => {
    const structural: Record<string, FakeEntry> = {
      "0-0-0-0": { pointCount: 0, children: ["1-0-0-0"] },
      "1-0-0-0": { pointCount: 40 },
    };
    const { controller, deferred, visible, touchedKeys, violations } =
      makeMirrored(structural);
    await bootAndLand(controller, deferred);
    expect([...visible.keys()]).toEqual(["1-0-0-0"]);

    controller.setCamera(AWAY);
    await settle();
    controller.dispose();
    await settle();

    expect(violations).toEqual([]);
    expect(touchedKeys()).not.toContain("0-0-0-0");
    expect(visible.size).toBe(0);
  });

  it("dispose takes back exactly what the renderer was handed", async () => {
    const { controller, deferred, batches, visible, violations } =
      makeMirrored(SMALL_TREE);
    await settle();
    controller.setCamera(VIEW);
    await settle();
    batches.length = 0;

    // Residency the consumer has not been told about yet: tearing down must
    // not ask it to remove an actor it never created.
    deferred.get("0-0-0-0")!.resolve();
    await Promise.resolve();
    expect(controller.stats().residentTiles).toBe(1);
    controller.dispose();
    await settle();

    expect(violations).toEqual([]);
    expect(batches).toEqual([]);
    expect(visible.size).toBe(0);
  });
});

describe("createLodController — decoded single ownership", () => {
  /** What `tileBytes` charges for a position-only tile of this size. */
  const decodedBytes = (pointCount: number): number => pointCount * 12 + 64;

  it("reuses a canceled request's payload instead of reading the key again", async () => {
    const resolvers: Array<(tile: TileData) => void> = [];
    const sink = mirrorRenderer();
    const controller = createLodController({
      source: uncancellableSource(resolvers),
      onTiles: sink.onTiles,
      scheduleRender: sink.scheduleRender,
      pointBudget: 1000,
      selectionDelayMs: 0,
    });
    await settle();
    controller.setCamera(VIEW);
    await settle();
    expect(resolvers).toHaveLength(1);

    // Cancel (advisory: the getter keeps running) and let it land unwanted:
    // the single decoded copy rests in the cache.
    controller.setCamera({ ...VIEW, viewProj: LOOK_AWAY });
    await settle();
    resolvers[0]!(makeTile(100));
    await settle();
    expect(controller.stats()).toMatchObject({
      decodedTiles: 1,
      cachedTiles: 1,
      residentTiles: 0,
      inFlight: 0,
      physicalTileOperations: 0,
    });

    // Looking back must spend that copy, not read the same bytes again.
    controller.setCamera(VIEW);
    await settle();
    expect(resolvers).toHaveLength(1);
    const stats = controller.stats();
    expect(stats).toMatchObject({
      residentTiles: 1,
      residentPoints: 100,
      cachedTiles: 0,
      cachedBytes: 0,
      decodedTiles: 1,
      inFlight: 0,
    });
    expect(stats.decodedBytes).toBe(decodedBytes(100));
    expect(stats.residentBytes).toBe(stats.decodedBytes);
    expect(sink.violations).toEqual([]);
    expect(sink.visible.size).toBe(1);
    controller.dispose();
  });

  it("keeps one owner when a hide/show adopts the running read", async () => {
    // The trame bridge hides and shows a cloud on ordinary visibility
    // changes. That must not start a second read of a key already being read,
    // and the payload the first read delivers has to reach the renderer.
    const resolvers: Array<(tile: TileData) => void> = [];
    const sink = mirrorRenderer();
    const controller = createLodController({
      source: uncancellableSource(resolvers),
      onTiles: sink.onTiles,
      scheduleRender: sink.scheduleRender,
      pointBudget: 1000,
      selectionDelayMs: 0,
    });
    await settle();
    controller.setCamera(VIEW);
    await settle();
    expect(resolvers).toHaveLength(1);

    controller.setActive(false);
    await settle();
    controller.setActive(true);
    await settle();
    expect(resolvers).toHaveLength(1);
    expect(controller.stats()).toMatchObject({
      inFlight: 1,
      physicalTileOperations: 1,
    });

    resolvers[0]!(makeTile(100));
    await settle();
    const stats = controller.stats();
    expect(stats.residentTiles).toBe(1);
    expect(stats.decodedTiles).toBe(1);
    expect(stats.decodedBytes).toBe(decodedBytes(100));
    expect(stats.cachedTiles).toBe(0);
    expect(sink.violations).toEqual([]);
    expect(sink.visible.size).toBe(1);
    controller.dispose();
  });

  it("hands the cached payload to residency instead of copying it", async () => {
    const { controller, deferred } = makeController(SMALL_TREE);
    await bootAndLand(controller, deferred);
    const residentBytes = controller.stats().residentBytes;
    expect(controller.stats().decodedTiles).toBe(3);

    controller.setCamera({ ...VIEW, viewProj: LOOK_AWAY });
    await settle();
    expect(controller.stats()).toMatchObject({
      residentTiles: 0,
      residentBytes: 0,
      cachedTiles: 3,
      decodedTiles: 3,
    });
    expect(controller.stats().decodedBytes).toBe(residentBytes);

    controller.setCamera(VIEW);
    await settle();
    expect(controller.stats()).toMatchObject({
      residentTiles: 3,
      cachedTiles: 0,
      cachedBytes: 0,
      decodedTiles: 3,
    });
    expect(controller.stats().decodedBytes).toBe(residentBytes);
    controller.dispose();
  });
});

describe("createLodController — bounded physical work", () => {
  const TILE_CEILING = 3;
  const PAGE_CEILING = 2;

  /**
   * The page graph is driven by hand, so `refinementCutoffPx: 0` keeps
   * screen-space error out of the picture entirely: what gets requested is
   * then a pure function of the frustum and the point budget.
   */
  const makeScheduled = (
    graph: PageGraphSource,
    overrides?: Partial<LodControllerOptions>,
  ) => {
    const sink = mirrorRenderer();
    const controller = createLodController({
      source: graph.source,
      onTiles: sink.onTiles,
      scheduleRender: sink.scheduleRender,
      pointBudget: 1_000_000,
      selectionDelayMs: 0,
      refinementCutoffPx: 0,
      fetchConcurrency: TILE_CEILING,
      hierarchyConcurrency: PAGE_CEILING,
      ...overrides,
    });
    return { controller, ...sink };
  };

  /** Land everything outstanding until the whole page graph is resolved. */
  const drainPages = async (
    graph: PageGraphSource,
    onRound?: () => void,
  ): Promise<void> => {
    let round = 0;
    while (round < 200 && graph.activePages().length > 0) {
      round += 1;
      onRound?.();
      graph.landPages();
      await settle();
    }
  };

  it("never runs more physical tile reads than fetchConcurrency", async () => {
    // Cancellation is advisory here: the abandoned reads stay unresolved, so
    // the only thing that can bound real I/O is counting them.
    const graph = createPageGraphSource({
      depth: 1,
      branching: 8,
      pointsPerNode: 10,
    });
    const { controller } = makeScheduled(graph);
    await settle();
    controller.setCamera(VIEW);
    await drainPages(graph);
    expect(graph.tileCalls.length).toBe(TILE_CEILING);

    // Look away and back repeatedly. Every flip cancels the outstanding
    // fetches and re-selects the same tiles; nothing may start until the
    // abandoned reads actually finish.
    for (let flip = 0; flip < 12; flip += 1) {
      controller.setCamera({ ...VIEW, viewProj: LOOK_AWAY });
      await settle();
      controller.setCamera(VIEW);
      await settle();
      expect(graph.activeTiles().length).toBeLessThanOrEqual(TILE_CEILING);
      expect(controller.stats().physicalTileOperations).toBeLessThanOrEqual(
        TILE_CEILING,
      );
    }
    expect(graph.tileCalls.length).toBe(TILE_CEILING);

    // The ceiling is a queue, not a deadlock: landing the abandoned reads
    // releases the slots and the next tiles go out.
    graph.landTiles();
    await settle();
    expect(graph.tileCalls.length).toBeGreaterThan(TILE_CEILING);
    expect(controller.stats().physicalTileOperations).toBeLessThanOrEqual(
      TILE_CEILING,
    );
    controller.dispose();
  });

  it("holds a tile slot across an epoch change until the read settles", async () => {
    const stale = createPageGraphSource({ depth: 0, pointsPerNode: 10 });
    const { controller } = makeScheduled(stale, { fetchConcurrency: 1 });
    await settle();
    controller.setCamera(VIEW);
    await drainPages(stale);
    expect(stale.tileCalls).toEqual(["0-0-0-0"]);

    const fresh = createPageGraphSource({ depth: 0, pointsPerNode: 10 });
    controller.setSource(fresh.source);
    await settle();
    fresh.landPages();
    await settle();
    // The replacement wants the same node, but the abandoned read still owns
    // the only slot: a source swap must not double real I/O.
    expect(controller.stats().physicalTileOperations).toBe(1);
    expect(fresh.tileCalls).toEqual([]);

    stale.landTiles();
    await settle();
    expect(fresh.tileCalls).toEqual(["0-0-0-0"]);
    // The stale payload was never wanted, so nothing holds it.
    expect(controller.stats()).toMatchObject({
      decodedTiles: 0,
      residentTiles: 0,
      cachedTiles: 0,
    });
    controller.dispose();
  });

  it("spends a landed payload instead of re-reading a queued key", async () => {
    const graph = createPageGraphSource({
      depth: 1,
      branching: 1,
      pointsPerNode: 10,
    });
    const { controller, visible } = makeScheduled(graph, {
      fetchConcurrency: 1,
    });
    await settle();
    controller.setCamera(VIEW);
    await drainPages(graph);
    expect(graph.tileCalls).toEqual(["0-0-0-0"]);

    // Deselect and reselect while the read still holds the only slot: the key
    // goes back on the queue with its payload still physically in flight.
    controller.setPointBudget(1);
    await settle();
    controller.setPointBudget(1000);
    await settle();
    expect(controller.stats()).toMatchObject({
      physicalTileOperations: 1,
      residentTiles: 0,
    });

    // The read lands. Serving the queued key from what just arrived is the
    // difference between one read and two — and, when the second read fails,
    // between a drawn tile and a hole over a payload the controller holds.
    graph.landTiles();
    await settle();
    graph.landTiles();
    await settle();
    expect(graph.tileCalls).toEqual(["0-0-0-0", "1-0-0-0"]);
    const stats = controller.stats();
    expect(stats.residentTiles).toBe(stats.selection.targetTiles);
    expect(stats).toMatchObject({ queuedTiles: 0, cachedTiles: 0 });
    expect(visible.size).toBe(stats.residentTiles);
    controller.dispose();
  });

  it("reads each tile key once however often the camera churns", async () => {
    // Cancellation never destroys a decoded payload: the read either stays
    // physically alive or lands in the cache. A reselect must therefore adopt
    // one or spend the other — re-reading the same bytes was ~40% of all tile
    // I/O under a panning camera.
    const graph = createPageGraphSource({
      depth: 2,
      branching: 4,
      pointsPerNode: 5,
    });
    const { controller, violations, visible } = makeScheduled(graph, {
      fetchConcurrency: 1,
    });
    await settle();

    // The camera moves faster than the I/O: every round cancels the
    // outstanding reads and reselects the same keys before any of them lands.
    for (let round = 0; round < 40; round += 1) {
      controller.setCamera({ ...VIEW, viewProj: LOOK_AWAY });
      await settle();
      controller.setCamera(VIEW);
      await settle();
      graph.landPages();
      await settle();
      graph.landTiles();
      await settle();
      expect(new Set(graph.tileCalls).size).toBe(graph.tileCalls.length);
    }

    // Settle looking at the cloud and drain everything outstanding.
    controller.setCamera(VIEW);
    await drainPages(graph);
    let round = 0;
    while (
      round < 200 &&
      (controller.stats().queuedTiles > 0 || graph.activeTiles().length > 0)
    ) {
      round += 1;
      graph.landTiles();
      await settle();
    }

    // Fully settled means no hole: every selected tile is on screen, and no
    // key was ever read twice. The lower bounds are what keep this from
    // passing vacuously: a broken page path that selects nothing satisfies
    // every dedup assertion with an empty call list.
    const stats = controller.stats();
    expect(stats.selection.targetTiles).toBeGreaterThan(0);
    expect(stats.residentTiles).toBe(stats.selection.targetTiles);
    expect(visible.size).toBe(stats.residentTiles);
    expect(graph.tileCalls.length).toBeGreaterThanOrEqual(stats.residentTiles);
    expect(new Set(graph.tileCalls).size).toBe(graph.tileCalls.length);
    expect(violations).toEqual([]);
    controller.dispose();
  });

  it("never runs more physical page reads than hierarchyConcurrency", async () => {
    const graph = createPageGraphSource({
      depth: 3,
      branching: 4,
      pointsPerNode: 1,
    });
    const { controller } = makeScheduled(graph);
    await settle();
    controller.setCamera(VIEW);
    await settle();

    await drainPages(graph, () => {
      expect(graph.activePages().length).toBeLessThanOrEqual(PAGE_CEILING);
      expect(
        controller.stats().physicalHierarchyOperations,
      ).toBeLessThanOrEqual(PAGE_CEILING);
    });

    // A page graph far wider and deeper than the ceiling still resolves whole,
    // and every page is read exactly once.
    expect(graph.pageCalls.length).toBe(graph.pageKeys().length);
    expect(new Set(graph.pageCalls).size).toBe(graph.pageCalls.length);
    expect(graph.pageKeys().length).toBeGreaterThan(PAGE_CEILING * 10);
    controller.dispose();
  });

  it("requests each page once however often selection reruns", async () => {
    const graph = createPageGraphSource({ depth: 2, branching: 2 });
    const { controller } = makeScheduled(graph);
    await settle();
    controller.setCamera(VIEW);
    for (let round = 0; round < 5; round += 1) {
      controller.refresh();
      await settle();
    }
    expect(graph.pageCalls).toEqual(["0-0-0-0"]);

    graph.landPages();
    await settle();
    for (let round = 0; round < 5; round += 1) {
      controller.refresh();
      await settle();
    }
    expect(new Set(graph.pageCalls).size).toBe(graph.pageCalls.length);
    controller.dispose();
  });

  it("reads the coarsest blocked page before any deeper one", async () => {
    const graph = createPageGraphSource({ depth: 2, branching: 2 });
    const { controller } = makeScheduled(graph, { hierarchyConcurrency: 1 });
    await settle();
    // Camera inside the first octant: its level-2 pages score a far larger
    // screen-space error than the level-1 page of the octant behind the
    // camera, so only the level tiebreak keeps the coarse page ahead.
    controller.setCamera({ ...VIEW, position: [-0.25, -0.25, -0.25] });
    await drainPages(graph);

    const levels = graph.pageCalls.map(levelOf);
    expect(levels.length).toBe(graph.pageKeys().length);
    expect(levels).toEqual([...levels].sort((a, b) => a - b));
    controller.dispose();
  });

  it("orders same-level pages by screen-space error", async () => {
    const position: [number, number, number] = [0.4, 0.3, 0.2];
    const graph = createPageGraphSource({ depth: 1, branching: 8 });
    const { controller } = makeScheduled(graph, { hierarchyConcurrency: 1 });
    await settle();
    controller.setCamera({ ...VIEW, position });
    await drainPages(graph);

    // Same level and same spacing, so descending error is ascending distance.
    const expected = childKeys(ROOT_KEY)
      .map((key) => ({
        keyString: keyToString(key),
        distance: distanceToBounds(position, nodeBounds(ROOT_CUBE, key)),
      }))
      .sort((a, b) => a.distance - b.distance)
      .map((node) => node.keyString);
    expect(graph.pageCalls).toEqual(["0-0-0-0", ...expected]);
    controller.dispose();
  });

  it("holds a page slot across an epoch change until the read settles", async () => {
    const stale = createPageGraphSource({ depth: 1, branching: 2 });
    const { controller } = makeScheduled(stale, { hierarchyConcurrency: 1 });
    await settle();
    controller.setCamera(VIEW);
    await settle();
    expect(stale.pageCalls).toEqual(["0-0-0-0"]);

    const fresh = createPageGraphSource({ depth: 1, branching: 2 });
    controller.setSource(fresh.source);
    await settle();
    expect(controller.stats().physicalHierarchyOperations).toBe(1);
    expect(fresh.pageCalls).toEqual([]);

    // Landing the abandoned read frees the slot; its entries are ignored.
    stale.landPages();
    await settle();
    expect(fresh.pageCalls).toEqual(["0-0-0-0"]);
    expect(controller.stats().selection.targetTiles).toBe(0);
    expect(stale.tileCalls).toEqual([]);
    controller.dispose();
  });

  it("skips pages for subtrees the frustum rejects", async () => {
    const graph = createPageGraphSource({ depth: 1, branching: 8 });
    const { controller } = makeScheduled(graph);
    await settle();
    // Clip space shifted by -1.2: only world x >= 0.2 is visible, so the four
    // octants left of the root centre are wholly outside.
    controller.setCamera({
      ...VIEW,
      viewProj: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -1.2, 0, 0, 1],
    });
    await drainPages(graph);

    expect(graph.pageCalls).toHaveLength(5); // the root page plus four octants
    for (const keyString of graph.pageCalls.slice(1)) {
      expect(keyFromString(keyString).x).toBe(1);
    }
    controller.dispose();
  });

  it("keeps tile and page work inside their own ceilings under churn", async () => {
    const graph = createPageGraphSource({
      depth: 2,
      branching: 4,
      pointsPerNode: 5,
    });
    const { controller, violations } = makeScheduled(graph);
    await settle();

    for (let round = 0; round < 30; round += 1) {
      controller.setCamera(
        round % 2 === 0 ? VIEW : { ...VIEW, viewProj: LOOK_AWAY },
      );
      await settle();
      const stats = controller.stats();
      expect(stats.physicalTileOperations).toBeLessThanOrEqual(TILE_CEILING);
      expect(stats.physicalHierarchyOperations).toBeLessThanOrEqual(
        PAGE_CEILING,
      );
      expect(graph.activeTiles().length).toBeLessThanOrEqual(TILE_CEILING);
      expect(graph.activePages().length).toBeLessThanOrEqual(PAGE_CEILING);
      // Total I/O is the documented sum of the two explicit ceilings.
      expect(
        stats.physicalTileOperations + stats.physicalHierarchyOperations,
      ).toBeLessThanOrEqual(TILE_CEILING + PAGE_CEILING);
      if (round % 3 === 0) {
        graph.landPages();
        graph.landTiles();
        await settle();
      }
    }
    // The ceilings only mean something if work actually ran up against them:
    // an implementation that issued nothing at all satisfies every
    // upper-bound assertion in the loop.
    expect(graph.pageCalls.length).toBeGreaterThan(0);
    expect(graph.tileCalls.length).toBeGreaterThan(0);
    expect(violations).toEqual([]);
    controller.dispose();
  });

  it("accepts a zero point budget as draw nothing", async () => {
    const graph = createPageGraphSource({
      depth: 1,
      branching: 2,
      pointsPerNode: 5,
    });
    const { controller, visible } = makeScheduled(graph);
    await settle();
    controller.setCamera(VIEW);
    await drainPages(graph);
    let round = 0;
    while (
      round < 50 &&
      (controller.stats().queuedTiles > 0 || graph.activeTiles().length > 0)
    ) {
      round += 1;
      graph.landTiles();
      await settle();
    }
    expect(controller.stats().residentTiles).toBeGreaterThan(0);

    // Zero is the share a view governor hands a deactivated member. Rejecting
    // it would keep the previous budget silently in force while the
    // governor's diagnostics report the member draws nothing.
    controller.setPointBudget(0);
    await settle();
    expect(controller.stats().selection.targetTiles).toBe(0);
    expect(controller.stats().residentTiles).toBe(0);
    expect(visible.size).toBe(0);

    // And back: the payloads waited in the cache, so restoring the budget
    // restores the picture without a single new read.
    const readsBefore = graph.tileCalls.length;
    controller.setPointBudget(1_000_000);
    await settle();
    expect(controller.stats().residentTiles).toBeGreaterThan(0);
    expect(visible.size).toBeGreaterThan(0);
    expect(graph.tileCalls.length).toBe(readsBefore);
    controller.dispose();
  });

  it("rejects an invalid hierarchy concurrency at construction", () => {
    for (const value of [Number.NaN, Number.POSITIVE_INFINITY, 0, -1]) {
      expect(() =>
        createLodController({
          source: createPageGraphSource().source,
          onTiles: () => {},
          scheduleRender: () => {},
          hierarchyConcurrency: value,
        }),
      ).toThrow(/hierarchyConcurrency/);
    }
  });
});

describe("createLodController — numeric configuration", () => {
  const NON_FINITE = [
    Number.NaN,
    Number.POSITIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
  ];

  const make = (overrides: Partial<LodControllerOptions>) =>
    createLodController({
      source: makeFakeSource(SMALL_TREE).source,
      onTiles: () => {},
      scheduleRender: () => {},
      selectionDelayMs: 0,
      ...overrides,
    });

  it("throws for a construction option outside its range", () => {
    for (const value of [...NON_FINITE, 0, -1, -1e9]) {
      expect(() => make({ pointBudget: value })).toThrow(/pointBudget/);
      expect(() => make({ cacheBytes: value })).toThrow(/cacheBytes/);
      expect(() => make({ fetchConcurrency: value })).toThrow(
        /fetchConcurrency/,
      );
      expect(() => make({ memory: value })).toThrow(/memory/);
    }
    for (const value of [...NON_FINITE, -1, -1e9]) {
      expect(() => make({ selectionDelayMs: value })).toThrow(
        /selectionDelayMs/,
      );
      expect(() => make({ refinementCutoffPx: value })).toThrow(
        /refinementCutoffPx/,
      );
    }
    for (const value of [...NON_FINITE, -0.01, 1.01]) {
      expect(() => make({ densityFraction: value })).toThrow(/densityFraction/);
    }
  });

  it("throws for an unusable or inverted presentation contract", () => {
    const broken = [
      { mode: "fixed", diameterCssPx: Number.NaN },
      { mode: "fixed", diameterCssPx: 0 },
      { mode: "fixed", diameterCssPx: Number.POSITIVE_INFINITY },
      { mode: "auto", userScale: Number.NaN },
      { mode: "auto", userScale: -1 },
      { mode: "auto", userScale: 1, minDiameterCssPx: 5, maxDiameterCssPx: 2 },
      {
        mode: "auto",
        userScale: 1,
        minDiameterCssPx: Number.NEGATIVE_INFINITY,
      },
    ] as const;
    for (const presentation of broken) {
      expect(() => make({ presentation })).toThrow();
    }
  });

  it("truncates fractional construction counts", () => {
    const controller = make({ pointBudget: 1000.9, fetchConcurrency: 2.7 });
    expect(controller.stats().pointBudget).toBe(1000);
    controller.dispose();
  });

  it("ignores live setter values outside their range", async () => {
    const { controller, deferred } = makeController(SMALL_TREE);
    await bootAndLand(controller, deferred);
    const before = controller.stats();
    expect(before.residentTiles).toBe(3);

    // Zero is not in this list: it is a valid budget meaning "draw nothing",
    // the share a governor hands a deactivated member.
    for (const value of [...NON_FINITE, -1, -1e9]) {
      controller.setPointBudget(value);
      const after = controller.stats();
      expect(after.pointBudget).toBe(before.pointBudget);
      expect(after.memoryCeilingPoints).toBe(before.memoryCeilingPoints);
      expect(after.residentTiles).toBe(before.residentTiles);
      expect(after.residentPoints).toBe(before.residentPoints);
    }
    for (const value of [...NON_FINITE, -1]) {
      controller.setRefinementCutoffPx(value);
      expect(controller.stats().refinementCutoffPx).toBe(
        before.refinementCutoffPx,
      );
    }
    for (const value of [...NON_FINITE, -0.01, 1.01]) {
      controller.setDensityFraction(value);
      expect(controller.stats().densityFraction).toBe(before.densityFraction);
    }
    for (const presentation of [
      { mode: "fixed", diameterCssPx: Number.NaN },
      { mode: "fixed", diameterCssPx: -2 },
      { mode: "auto", userScale: Number.POSITIVE_INFINITY },
      { mode: "auto", userScale: 1, minDiameterCssPx: 9, maxDiameterCssPx: 1 },
    ] as const) {
      controller.setPresentation(presentation);
      expect(controller.stats().presentation).toEqual(before.presentation);
    }

    // A usable value still applies, truncated to whole points.
    controller.setPointBudget(150.9);
    expect(controller.stats().pointBudget).toBe(150);
    expect(controller.stats().residentTiles).toBe(1);
    controller.dispose();
  });

  it("ignores a camera view carrying any non-finite number", async () => {
    const { controller, deferred } = makeController(SMALL_TREE);
    await bootAndLand(controller, deferred);
    const before = controller.stats();

    const broken: CameraView[] = [
      { ...VIEW, position: [Number.NaN, 0, 0] },
      { ...VIEW, fovY: Number.POSITIVE_INFINITY },
      { ...VIEW, fovY: 0 },
      // A field of view of a half-turn or more has no usable tangent.
      { ...VIEW, fovY: Math.PI },
      { ...VIEW, viewportWidthCssPx: 0 },
      { ...VIEW, viewportWidthCssPx: Number.NaN },
      { ...VIEW, viewportHeightCssPx: 0 },
      { ...VIEW, viewportHeightCssPx: Number.NaN },
      { ...VIEW, viewProj: [...IDENTITY.slice(0, 15), Number.NaN] },
      { ...ORTHOGRAPHIC_VIEW, parallelScale: Number.NaN },
      { ...ORTHOGRAPHIC_VIEW, parallelScale: 0 },
      { ...ORTHOGRAPHIC_VIEW, parallelScale: -1 },
      // Untyped hosts can hand over anything; an unknown mode is not a camera.
      { ...VIEW, projection: "isometric" } as unknown as CameraView,
    ];
    for (const view of broken) {
      controller.setCamera(view);
      await settle();
      const after = controller.stats();
      expect(after.selection.generation).toBe(before.selection.generation);
      expect(after.residentTiles).toBe(before.residentTiles);
      expect(Number.isFinite(after.selection.projectedImportance)).toBe(true);
    }
    controller.dispose();
  });

  it("treats a memory pool that answers nonsense as no memory", async () => {
    const brokenPool: MemoryPool = {
      register: () => ({ budgetBytes: () => Number.NaN, release: () => {} }),
      memberCount: () => 1,
    };
    const { controller, loadCalls } = makeController(SMALL_TREE, {
      memory: brokenPool,
    });
    await settle();
    controller.setCamera(VIEW);
    await settle();

    const stats = controller.stats();
    expect(Number.isFinite(stats.pointBudget)).toBe(true);
    expect(stats.pointBudget).toBe(0);
    expect(stats.memoryCeilingPoints).toBe(0);
    expect(stats.memoryBudgetBytes).toBe(0);
    expect(loadCalls).toEqual([]);
    controller.dispose();
  });
});

describe("createLodController — pickPoint", () => {
  // Every fake tile carries zeroed positions at the world origin, which the
  // identity view-projection puts at the viewport center: css (50, 50). The
  // cursor ray there runs from (0, 0, -1) along +z, so the support depth of
  // the origin is 1 and the picked point is the origin itself.
  const pickCenter = (controller: LodController, view: CameraView = VIEW) =>
    controller.pickPoint(view, 50, 50);

  it("hits over the submitted tile set, on the cursor ray", async () => {
    const { controller, deferred } = makeController(SMALL_TREE);
    await bootAndLand(controller, deferred);

    const result = pickCenter(controller);
    expect(result?.status).toBe("hit");
    if (result?.status !== "hit") throw new Error("unreachable");
    expect(result.pointOnRay[0]).toBeCloseTo(0);
    expect(result.pointOnRay[1]).toBeCloseTo(0);
    expect(result.pointOnRay[2]).toBeCloseTo(0);
    expect(result.distancePx).toBeCloseTo(0);
    controller.dispose();
  });

  it("never picks points outside the prefix currently being drawn", async () => {
    const { controller, deferred } = makeController(SMALL_TREE);
    await bootAndLand(controller, deferred);
    expect(pickCenter(controller)?.status).toBe("hit");

    controller.setDensityFraction(0);
    expect(controller.stats().drawnPoints).toBe(0);
    expect(pickCenter(controller)).toEqual({ status: "miss" });

    controller.setDensityFraction(1);
    expect(pickCenter(controller)?.status).toBe("hit");
    controller.dispose();
  });

  it("answers miss, not null, when a valid sweep supports nothing", async () => {
    const { controller, deferred } = makeController(SMALL_TREE);
    // Nothing submitted yet: a valid query over an empty set is a miss.
    await settle();
    expect(pickCenter(controller)).toEqual({ status: "miss" });

    await bootAndLand(controller, deferred);
    // A cursor far outside every bucket misses too.
    expect(controller.pickPoint(VIEW, 550, 50)).toEqual({ status: "miss" });
    controller.dispose();
  });

  it("does not pick resident tiles ahead of the renderer flush", async () => {
    const { controller, deferred } = makeController(SMALL_TREE);
    await settle();
    controller.setCamera(VIEW);
    await settle();

    deferred.get("0-0-0-0")!.resolve();
    // One turn: the payload claims residency, but the batched flush that
    // hands it to the renderer is still queued behind this continuation.
    await Promise.resolve();
    expect(controller.activeKeys().resident).toContain("0-0-0-0");
    expect(controller.activeKeys().submitted).toEqual([]);
    // What the renderer is not yet drawing must not answer a pick.
    expect(pickCenter(controller)).toEqual({ status: "miss" });

    await settle();
    expect(controller.activeKeys().submitted).toContain("0-0-0-0");
    expect(pickCenter(controller)?.status).toBe("hit");
    controller.dispose();
  });

  it("does not pick tiles that fell back to the decoded cache", async () => {
    const { controller, deferred } = makeController(SMALL_TREE);
    await bootAndLand(controller, deferred);
    expect(pickCenter(controller)?.status).toBe("hit");

    controller.setCamera({ ...VIEW, viewProj: LOOK_AWAY });
    await settle();
    expect(controller.stats().cachedTiles).toBeGreaterThan(0);
    expect(controller.stats().residentTiles).toBe(0);
    // The payloads are still decoded, but nothing is rendered: the query
    // view still looks straight at them, and must miss anyway.
    expect(pickCenter(controller)).toEqual({ status: "miss" });
    controller.dispose();
  });

  it("is unavailable while inactive and again after dispose", async () => {
    const { controller, deferred } = makeController(SMALL_TREE);
    await bootAndLand(controller, deferred);

    controller.setActive(false);
    await settle();
    expect(pickCenter(controller)).toBeNull();

    controller.setActive(true);
    await settle();
    expect(pickCenter(controller)?.status).toBe("hit");

    controller.dispose();
    expect(pickCenter(controller)).toBeNull();
  });

  it("is unavailable for an invalid view or cursor, never a miss", async () => {
    const { controller, deferred } = makeController(SMALL_TREE);
    await bootAndLand(controller, deferred);

    const unusable: CameraView[] = [
      { ...VIEW, viewportWidthCssPx: 0 },
      { ...VIEW, viewportHeightCssPx: Number.NaN },
      { ...VIEW, viewProj: [...IDENTITY.slice(0, 15), Number.NaN] },
      // All-finite but singular: the cursor ray cannot be built.
      { ...VIEW, viewProj: IDENTITY.map(() => 0) },
      { ...VIEW, position: [Number.NaN, 0, 0] },
    ];
    for (const view of unusable) {
      expect(pickCenter(controller, view)).toBeNull();
    }
    expect(controller.pickPoint(VIEW, Number.NaN, 50)).toBeNull();
    expect(controller.pickPoint(VIEW, 50, Number.NEGATIVE_INFINITY)).toBeNull();
    controller.dispose();
  });
});
