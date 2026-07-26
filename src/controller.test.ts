import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  createLodController,
  type LodControllerOptions,
  type TileBatch,
} from "./controller";
import { createMemoryPool, type MemoryPool } from "./memoryPool";
import { keyToString, type VoxelKey } from "./octree";
import type {
  NodeInfo,
  TileData,
  TileSource,
  TileSourceMetadata,
} from "./tileSource";

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
/** Pushes everything 10 units off in clip x: nothing is visible. */
const LOOK_AWAY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -10, 0, 0, 1];

const VIEW = {
  viewProj: IDENTITY,
  position: [0, 0, 0] as [number, number, number],
  fovY: Math.PI / 2,
  viewportHeightCssPx: 100,
};

const METADATA: TileSourceMetadata = {
  pointCount: 1000,
};

interface FakeEntry {
  pointCount: number;
  bounds?: {
    min: [number, number, number];
    max: [number, number, number];
  };
  spacing?: number;
  children?: string[];
  pageRef?: boolean;
  /** Entries revealed by loading the page rooted at this key. */
  pageNodes?: Record<
    string,
    {
      pointCount: number;
      bounds?: {
        min: [number, number, number];
        max: [number, number, number];
      };
      spacing?: number;
      children?: string[];
    }
  >;
}

interface Deferred {
  resolve: (tile?: Partial<TileData>) => void;
  reject: (error: Error) => void;
  aborted: boolean;
}

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
  const toInfos = (
    entries: Record<
      string,
      {
        pointCount: number;
        bounds?: {
          min: [number, number, number];
          max: [number, number, number];
        };
        spacing?: number;
        children?: string[];
        pageRef?: boolean;
      }
    >,
  ): NodeInfo[] =>
    Object.entries(entries).map(([keyString, entry]) => {
      const [level, x, y, z] = keyString.split("-").map(Number);
      return {
        key: { level: level!, x: x!, y: y!, z: z! },
        pointCount: entry.pointCount,
        bounds: entry.bounds ?? {
          min: [-0.5, -0.5, -0.5],
          max: [0.5, 0.5, 0.5],
        },
        spacing: entry.spacing ?? 0.1 / 2 ** level!,
        children: entry.children?.map((c) => {
          const [cl, cx, cy, cz] = c.split("-").map(Number);
          return { level: cl!, x: cx!, y: cy!, z: cz! };
        }),
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

const makeController = (
  tree: Record<string, FakeEntry>,
  options?: { pointBudget?: number; cacheBytes?: number },
) => {
  const fake = makeFakeSource(tree);
  const sink = collectBatches();
  const controller = createLodController({
    source: fake.source,
    onTiles: sink.onTiles,
    scheduleRender: sink.scheduleRender,
    pointBudget: options?.pointBudget ?? 1000,
    selectionDelayMs: 0,
    cacheBytes: options?.cacheBytes,
  });
  return { controller, ...fake, ...sink };
};

const makeMirrored = (
  tree: Record<string, FakeEntry>,
  options?: { pointBudget?: number },
) => {
  const fake = makeFakeSource(tree);
  const sink = mirrorRenderer();
  const controller = createLodController({
    source: fake.source,
    onTiles: sink.onTiles,
    scheduleRender: sink.scheduleRender,
    pointBudget: options?.pointBudget ?? 1000,
    selectionDelayMs: 0,
  });
  return { controller, ...fake, ...sink };
};

const SMALL_TREE: Record<string, FakeEntry> = {
  "0-0-0-0": { pointCount: 100, children: ["1-0-0-0", "1-1-0-0"] },
  "1-0-0-0": { pointCount: 60 },
  "1-1-0-0": { pointCount: 60 },
};

describe("createLodController", () => {
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
    await settle();
    controller.setCamera(VIEW);
    await settle();
    for (const d of deferred.values()) d.resolve();
    await settle();

    controller.setPointBudget(1000);
    await settle();
    expect(loadCalls).toHaveLength(3);
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

  it("ignores a superseded fetch that resolves after its replacement", async () => {
    // Aborting is advisory: the COPC getter takes no signal, so a superseded
    // request still resolves. The stale continuation must not retire the live
    // in-flight slot (which would uncap fetchConcurrency) or double-count
    // resident points.
    const resolvers: Array<(tile: TileData) => void> = [];
    const source: TileSource = {
      metadata: () => METADATA,
      async nodes() {
        return [
          {
            key: { level: 0, x: 0, y: 0, z: 0 },
            pointCount: 100,
            bounds: { min: [-0.5, -0.5, -0.5], max: [0.5, 0.5, 0.5] },
            spacing: 0.1,
            children: [],
          },
        ];
      },
      // Deliberately ignores opts.signal: models an uncancellable getter.
      loadTile: () => new Promise<TileData>((r) => resolvers.push(r)),
    };
    const sink = collectBatches();
    const controller = createLodController({
      source,
      onTiles: sink.onTiles,
      scheduleRender: sink.scheduleRender,
      pointBudget: 1000,
      selectionDelayMs: 0,
    });

    await settle();
    controller.setCamera(VIEW);
    await settle();
    expect(resolvers).toHaveLength(1);

    // Look away (aborts, but the fetch stays live), then look back: the same
    // key is refetched into a fresh in-flight slot.
    controller.setCamera({ ...VIEW, viewProj: LOOK_AWAY });
    await settle();
    controller.setCamera(VIEW);
    await settle();
    expect(resolvers).toHaveLength(2);
    expect(controller.stats().inFlight).toBe(1);

    // The stale first fetch lands. It must not free the live slot.
    resolvers[0]!(makeTile(100));
    await settle();
    expect(controller.stats().inFlight).toBe(1);
    expect(controller.stats().residentTiles).toBe(0);

    // The live fetch still completes normally, exactly once.
    resolvers[1]!(makeTile(100));
    await settle();
    expect(controller.stats().inFlight).toBe(0);
    expect(controller.stats().residentTiles).toBe(1);
    expect(controller.stats().residentPoints).toBe(100);
    controller.dispose();
  });

  it("reuses cached tiles without refetching", async () => {
    const { controller, deferred, loadCalls } = makeController(SMALL_TREE);
    await settle();
    controller.setCamera(VIEW);
    await settle();
    for (const d of deferred.values()) d.resolve();
    await settle();
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
    await settle();
    controller.setCamera(VIEW);
    await settle();
    for (const d of deferred.values()) d.resolve();
    await settle();

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

  it("evicts deselected tiles beyond the cache byte bound", async () => {
    const { controller, deferred, loadCalls } = makeController(SMALL_TREE, {
      cacheBytes: 1, // nothing survives deselection
    });
    await settle();
    controller.setCamera(VIEW);
    await settle();
    for (const d of deferred.values()) d.resolve();
    await settle();

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
    const fake = makeFakeSource(tree);
    const sink = collectBatches();
    const controller = createLodController({
      source: fake.source,
      onTiles: sink.onTiles,
      scheduleRender: sink.scheduleRender,
      selectionDelayMs: 0,
    });
    await settle();
    // Clip space shifted by -5: only world coords in [4,6]^3 are visible.
    // The root cube [-8,8]^3 straddles that region, but the negative octant
    // '1-0-0-0' ([-8,0]^3) is fully outside and must never be fetched.
    controller.setCamera({
      ...VIEW,
      viewProj: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -5, -5, -5, 1],
    });
    await settle();
    expect(fake.loadCalls).toContain("0-0-0-0");
    expect(fake.loadCalls).not.toContain("1-0-0-0");
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
    options: {
      pointBudget?: number;
      presentation:
        | { mode: "fixed"; diameterCssPx: number }
        | {
            mode: "auto";
            userScale: number;
            minDiameterCssPx?: number;
            maxDiameterCssPx?: number;
          };
    },
  ) => {
    const fake = makeFakeSource(tree);
    const sink = collectBatches();
    const diameters: number[] = [];
    const controller = createLodController({
      source: fake.source,
      onTiles: sink.onTiles,
      scheduleRender: sink.scheduleRender,
      pointBudget: options.pointBudget ?? 1000,
      selectionDelayMs: 0,
      interactionSettleMs: 300,
      presentation: options.presentation,
      onPointDiameterCssPx: (value) => diameters.push(value),
    });
    return { controller, diameters, ...fake, ...sink };
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

  it("uses the density-aware p75 with clamps throughout interaction", async () => {
    const leaf: Record<string, FakeEntry> = {
      "0-0-0-0": {
        pointCount: 100,
        bounds: { min: [-1, -1, -1], max: [1, 1, 1] },
        spacing: 0.9,
      },
    };
    const { controller, deferred, diameters } = makePresented(leaf, {
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
    const leaf: Record<string, FakeEntry> = {
      "0-0-0-0": {
        pointCount: 100,
        bounds: { min: [-1, -1, -1], max: [1, 1, 1] },
        spacing: 0.9,
      },
    };
    const { controller, deferred } = makePresented(leaf, {
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
    const leaf: Record<string, FakeEntry> = {
      "0-0-0-0": {
        pointCount: 100,
        bounds: { min: [-1, -1, -1], max: [1, 1, 1] },
        spacing: 0.9,
      },
    };
    const { controller, deferred, diameters } = makePresented(leaf, {
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
    const fake = makeFakeSource(SMALL_TREE);
    const sink = collectBatches();
    const controller = createLodController({
      source: fake.source,
      onTiles: sink.onTiles,
      scheduleRender: sink.scheduleRender,
      pointBudget: 1000,
      selectionDelayMs: 0,
      onError: (error) => errors.push(error),
    });
    await settle();
    controller.setCamera(VIEW);
    await settle();
    expect(fake.loadCalls).toHaveLength(3);

    const failEveryFetch = async (): Promise<void> => {
      for (const d of fake.deferred.values()) d.reject(new Error("500"));
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
    expect(fake.loadCalls).toHaveLength(9); // 3 tiles x 3 attempts
    expect(errors).toHaveLength(9);
    controller.dispose();
  });

  it("refetches a rested tile once the outage ends", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    const fake = makeFakeSource(SMALL_TREE);
    const sink = collectBatches();
    const controller = createLodController({
      source: fake.source,
      onTiles: sink.onTiles,
      scheduleRender: sink.scheduleRender,
      pointBudget: 1000,
      selectionDelayMs: 0,
      onError: () => {},
    });
    await settle();
    controller.setCamera(VIEW);
    await settle();

    // A brief outage burns the whole allowance for every tile.
    for (let round = 0; round < 4; round += 1) {
      for (const d of fake.deferred.values()) d.reject(new Error("500"));
      await settle();
      controller.refresh();
      await settle();
    }
    expect(fake.loadCalls).toHaveLength(9);

    // The server recovers. Resting is a backoff, not an eviction: once the
    // keys have been quiet long enough they are fetchable again, and this
    // time they load.
    vi.setSystemTime(31_000);
    controller.refresh();
    await settle();
    expect(fake.loadCalls).toHaveLength(12);

    for (const d of fake.deferred.values()) d.resolve();
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
    const fake = makeFakeSource(SMALL_TREE);
    const sink = collectBatches();
    const controller = createLodController({
      source: fake.source,
      onTiles: sink.onTiles,
      scheduleRender: sink.scheduleRender,
      pointBudget: 1000,
      selectionDelayMs: 0,
      onError: (error) => errors.push(error),
    });
    await settle();
    controller.setCamera(VIEW);
    await settle();
    for (let round = 0; round < 4; round += 1) {
      for (const d of fake.deferred.values()) d.reject(new Error("500"));
      await settle();
      controller.refresh();
      await settle();
    }
    expect(fake.loadCalls).toHaveLength(9);

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
  ) => {
    const fake = makeFakeSource(tree);
    const sink = collectBatches();
    const controller = createLodController({
      source: fake.source,
      onTiles: sink.onTiles,
      scheduleRender: sink.scheduleRender,
      pointBudget,
      selectionDelayMs: 0,
      interactionSettleMs: 300,
      ...(memory !== undefined ? { memory } : {}),
    });
    return { controller, ...fake, ...sink };
  };

  it("applies a budget drop immediately during interaction", async () => {
    const { controller, deferred } = makeBudgeted(SMALL_TREE, 1000);
    await settle();
    controller.setCamera(VIEW); // t=0: interacting
    await settle();
    for (const d of deferred.values()) d.resolve();
    await settle();
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
    await settle();
    controller.setCamera(VIEW);
    await settle();
    for (const d of deferred.values()) d.resolve();
    await settle();
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
    await settle();
    controller.setCamera(VIEW);
    await settle();
    for (const d of deferred.values()) d.resolve();
    await settle();
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
    await settle();
    controller.setCamera(VIEW);
    await settle();
    for (const d of deferred.values()) d.resolve();
    await settle();
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
    await settle();
    controller.setCamera(VIEW);
    await settle();
    for (const d of deferred.values()) d.resolve();
    await settle();
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
    loadTile: () => new Promise<TileData>((r) => resolvers.push(r)),
  });

  it("drops a canceled request's payload while its replacement is live", async () => {
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

    // Cancel (advisory: the getter keeps running) and re-request the key.
    controller.setCamera({ ...VIEW, viewProj: LOOK_AWAY });
    await settle();
    controller.setCamera(VIEW);
    await settle();
    expect(resolvers).toHaveLength(2);
    expect(controller.stats().inFlight).toBe(1);

    // The canceled request lands first. The live request owns the key, so
    // this payload is a duplicate and nothing may hold it.
    resolvers[0]!(makeTile(100));
    await settle();
    expect(controller.stats()).toMatchObject({
      decodedTiles: 0,
      decodedBytes: 0,
      cachedTiles: 0,
      residentTiles: 0,
      inFlight: 1,
    });

    resolvers[1]!(makeTile(100));
    await settle();
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

  it("keeps one owner when a canceled request lands after residency", async () => {
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
    controller.setCamera({ ...VIEW, viewProj: LOOK_AWAY });
    await settle();
    controller.setCamera(VIEW);
    await settle();

    resolvers[1]!(makeTile(100));
    await settle();
    expect(controller.stats().residentTiles).toBe(1);

    resolvers[0]!(makeTile(100));
    await settle();
    const stats = controller.stats();
    expect(stats.decodedTiles).toBe(1);
    expect(stats.decodedBytes).toBe(decodedBytes(100));
    expect(stats.cachedTiles).toBe(0);
    controller.dispose();
  });

  it("hands the cached payload to residency instead of copying it", async () => {
    const { controller, deferred } = makeController(SMALL_TREE);
    await settle();
    controller.setCamera(VIEW);
    await settle();
    for (const d of deferred.values()) d.resolve();
    await settle();
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
      expect(() => make({ interactionSettleMs: value })).toThrow(
        /interactionSettleMs/,
      );
      expect(() => make({ refinementCutoffPx: value })).toThrow(
        /refinementCutoffPx/,
      );
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
    await settle();
    controller.setCamera(VIEW);
    await settle();
    for (const d of deferred.values()) d.resolve();
    await settle();
    const before = controller.stats();
    expect(before.residentTiles).toBe(3);

    for (const value of [...NON_FINITE, 0, -1, -1e9]) {
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
    await settle();
    controller.setCamera(VIEW);
    await settle();
    for (const d of deferred.values()) d.resolve();
    await settle();
    const before = controller.stats();

    const broken = [
      { ...VIEW, position: [Number.NaN, 0, 0] as [number, number, number] },
      { ...VIEW, fovY: Number.POSITIVE_INFINITY },
      { ...VIEW, viewportHeightCssPx: 0 },
      { ...VIEW, viewportHeightCssPx: Number.NaN },
      { ...VIEW, viewProj: [...IDENTITY.slice(0, 15), Number.NaN] },
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
      totalBytes: () => Number.NaN,
      setTotalBytes: () => {},
      memberCount: () => 1,
    };
    const fake = makeFakeSource(SMALL_TREE);
    const sink = collectBatches();
    const controller = createLodController({
      source: fake.source,
      onTiles: sink.onTiles,
      scheduleRender: sink.scheduleRender,
      selectionDelayMs: 0,
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
    expect(fake.loadCalls).toEqual([]);
    controller.dispose();
  });
});
