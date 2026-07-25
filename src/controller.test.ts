import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createLodController, type TileBatch } from "./controller";
import type { AdaptiveBudgetOptions } from "./adaptiveBudget";
import { createMemoryPool } from "./memoryPool";
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
  viewportHeight: 100,
};

const METADATA: TileSourceMetadata = {
  pointCount: 1000,
  cube: { center: [0, 0, 0], halfSize: 0.5 },
  spacing: 0.1,
};

interface FakeEntry {
  pointCount: number;
  children?: string[];
  pageRef?: boolean;
  /** Entries revealed by loading the page rooted at this key. */
  pageNodes?: Record<string, { pointCount: number; children?: string[] }>;
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
      { pointCount: number; children?: string[]; pageRef?: boolean }
    >,
  ): NodeInfo[] =>
    Object.entries(entries).map(([keyString, entry]) => {
      const [level, x, y, z] = keyString.split("-").map(Number);
      return {
        key: { level: level!, x: x!, y: y!, z: z! },
        pointCount: entry.pointCount,
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
      activeTiles: 3,
      activePoints: 220,
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
      activeTiles: 0,
      activePoints: 0,
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
      activeTiles: 3,
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

  it("does not fetch culled subtrees", async () => {
    const tree: Record<string, FakeEntry> = {
      "0-0-0-0": { pointCount: 10, children: ["1-0-0-0"] },
      "1-0-0-0": { pointCount: 40 },
    };
    const meta: TileSourceMetadata = {
      ...METADATA,
      cube: { center: [0, 0, 0], halfSize: 8 },
    };
    const fake = makeFakeSource(tree);
    fake.source.metadata = () => meta;
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

describe("createLodController — adaptive budget", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  const makeAdaptive = (
    tree: Record<string, FakeEntry>,
    adaptive: boolean | AdaptiveBudgetOptions,
    initialBudget: number,
    memory?: number,
  ) => {
    const fake = makeFakeSource(tree);
    const sink = collectBatches();
    const controller = createLodController({
      source: fake.source,
      onTiles: sink.onTiles,
      scheduleRender: sink.scheduleRender,
      // Adaptive mode has no configured point ceiling: the initial budget
      // rides the adaptive options; pointBudget is the fixed-mode budget.
      ...(adaptive === false
        ? { pointBudget: initialBudget }
        : {
            adaptive: {
              initialBudget,
              interactionInitialBudget: initialBudget,
              maxStep: 0.25,
              ...(typeof adaptive === "object" ? adaptive : {}),
            },
          }),
      selectionDelayMs: 0,
      interactionSettleMs: 300,
      ...(memory !== undefined ? { memory } : {}),
    });
    return { controller, ...fake, ...sink };
  };

  it("starts at the initial budget and reports the interaction regime", async () => {
    const { controller } = makeAdaptive(SMALL_TREE, true, 2_000_000);
    await settle();
    expect(controller.stats().pointBudget).toBe(2_000_000);
    expect(controller.stats().interacting).toBe(false);
    controller.dispose();
  });

  it("drops immediately and settles only after the outermost interaction ends", async () => {
    const { controller } = makeAdaptive(
      SMALL_TREE,
      { interactionInitialBudget: 500_000 },
      2_000_000,
    );
    await settle();
    controller.setCamera(VIEW);
    await settle();
    vi.advanceTimersByTime(300);
    await settle();
    expect(controller.stats().pointBudget).toBe(2_000_000);

    controller.beginInteraction();
    controller.beginInteraction();
    expect(controller.stats()).toMatchObject({
      interacting: true,
      interactionDepth: 2,
      pointBudget: 500_000,
    });
    controller.endInteraction();
    vi.advanceTimersByTime(400);
    expect(controller.stats().interacting).toBe(true);
    controller.endInteraction();
    vi.advanceTimersByTime(299);
    expect(controller.stats().interacting).toBe(true);
    vi.advanceTimersByTime(1);
    await settle();
    expect(controller.stats()).toMatchObject({
      interacting: false,
      interactionDepth: 0,
      pointBudget: 2_000_000,
    });
    controller.dispose();
  });

  it("keeps interaction and stationary budgets independent", async () => {
    const { controller } = makeAdaptive(
      SMALL_TREE,
      { minSamples: 4 },
      2_000_000,
    );
    await settle();
    controller.setCamera(VIEW); // t=0: interacting
    await settle();
    // Four slow frames (80ms) against the 33ms interaction target → one shrink.
    for (let i = 0; i < 4; i += 1) {
      vi.setSystemTime(i);
      controller.recordFrame(80);
    }
    vi.setSystemTime(10);
    expect(controller.stats().interacting).toBe(true);
    expect(controller.stats().pointBudget).toBe(1_500_000); // interaction shrank
    vi.setSystemTime(400); // past interactionSettleMs → settled
    expect(controller.stats().interacting).toBe(false);
    expect(controller.stats().pointBudget).toBe(2_000_000); // stationary untouched
    controller.dispose();
  });

  it("a shrink during interaction deactivates deselected tiles immediately", async () => {
    // Point counts scaled to millions so a budget change crosses a selection
    // boundary: at 3M all three tiles fit; at 2.25M only the root and one child.
    const bigTree: Record<string, FakeEntry> = {
      "0-0-0-0": { pointCount: 1_000_000, children: ["1-0-0-0", "1-1-0-0"] },
      "1-0-0-0": { pointCount: 1_000_000 },
      "1-1-0-0": { pointCount: 1_000_000 },
    };
    const { controller, deferred, batches } = makeAdaptive(
      bigTree,
      { minSamples: 4 },
      3_000_000,
    );
    await settle();
    controller.setCamera(VIEW); // t=0: interacting, budget 3M → all fit
    await settle();
    for (const d of deferred.values()) d.resolve();
    await settle();
    expect(controller.stats().residentTiles).toBe(3);
    batches.length = 0;

    // Slow interaction frames shrink the budget to 2.25M. The deselected
    // child immediately leaves submitted/GPU-resident work; its payload stays
    // eligible for the decoded CPU cache.
    for (let i = 0; i < 4; i += 1) {
      vi.setSystemTime(i);
      controller.recordFrame(80);
    }
    await settle();
    vi.setSystemTime(10);
    expect(controller.stats().pointBudget).toBe(2_250_000);
    expect(controller.stats().residentTiles).toBe(2);
    expect(controller.stats().cachedTiles).toBe(1);
    expect(batches.flatMap((b) => b.removed)).toHaveLength(1);
    controller.dispose();
  });

  it("applies a fixed-budget drop immediately during interaction", async () => {
    const { controller, deferred } = makeAdaptive(SMALL_TREE, false, 1000);
    await settle();
    controller.setCamera(VIEW); // t=0: interacting
    await settle();
    for (const d of deferred.values()) d.resolve();
    await settle();
    expect(controller.stats().residentTiles).toBe(3);

    // Lower the fixed budget mid-gesture: both children (60 points each) fall
    // out of the 150-point selection and leave submitted work immediately.
    controller.setPointBudget(150);
    await settle();
    expect(controller.stats().residentTiles).toBe(1);
    expect(controller.stats().cachedTiles).toBe(2);

    // Settling does not change a fixed budget or resurrect dormant actors.
    vi.advanceTimersByTime(300);
    await settle();
    expect(controller.stats().residentTiles).toBe(1);
    controller.dispose();
  });

  it("applies the stationary budget after the camera settles, with no further frames", async () => {
    // Same millions-scaled tree: 3M fits all three tiles, 2.25M only two.
    const bigTree: Record<string, FakeEntry> = {
      "0-0-0-0": { pointCount: 1_000_000, children: ["1-0-0-0", "1-1-0-0"] },
      "1-0-0-0": { pointCount: 1_000_000 },
      "1-1-0-0": { pointCount: 1_000_000 },
    };
    const { controller, deferred } = makeAdaptive(
      bigTree,
      { minSamples: 4 },
      3_000_000,
    );
    await settle();
    controller.setCamera(VIEW); // t=0: arms the settle timer (fires at 300ms)
    await settle();
    for (const d of deferred.values()) d.resolve();
    await settle();
    expect(controller.stats().residentTiles).toBe(3);

    // Slow interaction frames shrink the interaction budget; the deselected
    // tile leaves submitted work immediately.
    for (let i = 0; i < 4; i += 1) {
      vi.setSystemTime(i);
      controller.recordFrame(80);
    }
    await settle();
    expect(controller.stats().pointBudget).toBe(2_250_000);
    expect(controller.stats().residentTiles).toBe(2);
    expect(controller.stats().cachedTiles).toBe(1);

    // No further frames — the host renders on demand and goes quiet. Advancing
    // past the settle window fires the settle timer, which re-applies the
    // (untouched) stationary 3M budget; the decoded cached tile is selected
    // and submitted again without a network request.
    vi.advanceTimersByTime(400);
    await settle();
    expect(controller.stats().interacting).toBe(false);
    expect(controller.stats().pointBudget).toBe(3_000_000);
    expect(controller.stats().residentTiles).toBe(3);
    controller.dispose();
  });

  it("setPointBudget is a no-op with adaptive enabled — there is no point ceiling", async () => {
    const { controller } = makeAdaptive(SMALL_TREE, true, 2_000_000);
    await settle();
    expect(controller.stats().pointBudget).toBe(2_000_000);
    controller.setPointBudget(1_000_000);
    expect(controller.stats().pointBudget).toBe(2_000_000);
    controller.dispose();
  });

  it("grows past any former fixed ceiling while frames are fast", async () => {
    // The old design pinned the budget at a configured point count; now only
    // frame time and memory govern. With fast frames and ample memory the
    // budget keeps climbing 25% per adjustment, sailing past 3M.
    const { controller } = makeAdaptive(
      SMALL_TREE,
      { minSamples: 4, cooldownMs: 0 },
      3_000_000,
    );
    await settle();
    controller.setCamera(VIEW);
    await settle();
    vi.setSystemTime(400); // settled regime
    for (let i = 0; i < 20; i += 1) {
      vi.setSystemTime(400 + i);
      controller.recordFrame(1); // far under the 16ms stationary target
    }
    expect(controller.stats().pointBudget).toBeGreaterThan(3_000_000);
    controller.dispose();
  });

  it("an unchanged camera view does not re-enter the interaction regime", async () => {
    // Hosts call setCamera before every paint with a freshly built (but often
    // identical) view — including paints triggered by the settle reselect
    // itself. An identical view must not stamp a camera change, or the regime
    // would flip back to interaction and oscillate between the two budgets.
    const { controller } = makeAdaptive(
      SMALL_TREE,
      { minSamples: 4 },
      2_000_000,
    );
    await settle();
    controller.setCamera(VIEW); // t=0: a real camera change → interacting
    await settle();
    expect(controller.stats().interacting).toBe(true);

    vi.setSystemTime(400); // past interactionSettleMs → settled
    vi.advanceTimersByTime(400); // settle timer fires
    expect(controller.stats().interacting).toBe(false);

    // A per-render feed of an equal (fresh object) view stays stationary.
    controller.setCamera({ ...VIEW, position: [...VIEW.position] });
    expect(controller.stats().interacting).toBe(false);

    // A genuinely different view re-enters interaction.
    controller.setCamera({ ...VIEW, viewportHeight: 200 });
    expect(controller.stats().interacting).toBe(true);
    controller.dispose();
  });

  it("recordFrame is a no-op when adaptive is disabled", async () => {
    const { controller } = makeController(SMALL_TREE, { pointBudget: 500 });
    await settle();
    controller.setCamera(VIEW);
    await settle();
    for (let i = 0; i < 20; i += 1) controller.recordFrame(5000);
    expect(controller.stats().pointBudget).toBe(500);
    controller.dispose();
  });

  // Below the 100k-resident-point measurement threshold the controller
  // converts bytes to points with the 16 bytes/point fallback, so a byte
  // budget of 16 * N caps selection at N points.
  const BYTES_PER_POINT = 16;

  it("the memory budget caps the adaptive budget", async () => {
    const { controller, deferred } = makeAdaptive(
      SMALL_TREE,
      true,
      2_000_000,
      150 * BYTES_PER_POINT,
    );
    await settle();
    expect(controller.stats().pointBudget).toBe(150);
    controller.setCamera(VIEW);
    await settle();
    for (const d of deferred.values()) d.resolve();
    await settle();
    // Root (100 points) fits the 150-point ceiling; the 60-point children
    // would overshoot and stay out.
    expect(controller.stats().residentTiles).toBe(1);
    expect(controller.stats().residentPoints).toBe(100);
    controller.dispose();
  });

  it("the memory budget caps a fixed budget too", async () => {
    const fake = makeFakeSource(SMALL_TREE);
    const sink = collectBatches();
    const controller = createLodController({
      source: fake.source,
      onTiles: sink.onTiles,
      scheduleRender: sink.scheduleRender,
      pointBudget: 1000,
      selectionDelayMs: 0,
      memory: 150 * BYTES_PER_POINT,
    });
    await settle();
    controller.setCamera(VIEW);
    await settle();
    for (const d of fake.deferred.values()) d.resolve();
    await settle();
    expect(controller.stats().pointBudget).toBe(150);
    expect(controller.stats().residentTiles).toBe(1);
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
      adaptive: true,
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
    // Advance past the settle window before changing the shared ceiling.
    vi.setSystemTime(1000);
    const second = makeFakeSource(SMALL_TREE);
    const other = createLodController({
      source: second.source,
      onTiles: () => {},
      scheduleRender: () => {},
      adaptive: true,
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

// --- phase0-bench (removable; see app/telesculptor_web/app/bench/README.md) ---
describe("createLodController — diagnostic stats", () => {
  // makeTile allocates Float32Array(pointCount * 3); tileBytes adds 64.
  const bytesOf = (pointCount: number) => pointCount * 12 + 64;

  const makeAdaptiveController = (adaptive: AdaptiveBudgetOptions) => {
    const fake = makeFakeSource(SMALL_TREE);
    const sink = collectBatches();
    const controller = createLodController({
      source: fake.source,
      onTiles: sink.onTiles,
      scheduleRender: sink.scheduleRender,
      adaptive,
      selectionDelayMs: 0,
    });
    return { controller, ...fake, ...sink };
  };

  it("counts fetches and bytes, and cache misses before hits", async () => {
    const { controller, deferred, loadCalls } = makeController(SMALL_TREE);
    await settle();
    controller.setCamera(VIEW);
    await settle();
    expect(controller.stats().cacheMisses).toBe(3);
    expect(controller.stats().cacheHits).toBe(0);
    expect(controller.stats().fetchedTiles).toBe(0);

    for (const d of deferred.values()) d.resolve();
    await settle();
    expect(controller.stats().fetchedTiles).toBe(3);
    expect(controller.stats().fetchedBytes).toBe(
      bytesOf(100) + 2 * bytesOf(60),
    );
    expect(controller.stats().cachedTiles).toBe(0);

    // Deselect everything: the three tiles move to the LRU.
    controller.setCamera({ ...VIEW, viewProj: LOOK_AWAY });
    await settle();
    expect(controller.stats().cachedTiles).toBe(3);

    // Looking back is served entirely from the LRU — hits, no new fetches.
    const fetchesBefore = loadCalls.length;
    controller.setCamera(VIEW);
    await settle();
    expect(loadCalls.length).toBe(fetchesBefore);
    expect(controller.stats().cacheHits).toBe(3);
    expect(controller.stats().cacheMisses).toBe(3);
    expect(controller.stats().cachedTiles).toBe(0);
    expect(controller.stats().fetchedTiles).toBe(3);
    controller.dispose();
  });

  it("counts fetches cancelled by deselection, not completed ones", async () => {
    const { controller, deferred } = makeController(SMALL_TREE);
    await settle();
    controller.setCamera(VIEW);
    await settle();
    deferred.get("0-0-0-0")!.resolve();
    await settle();
    expect(controller.stats().cancelledFetches).toBe(0);

    controller.setCamera({ ...VIEW, viewProj: LOOK_AWAY });
    await settle();
    expect(controller.stats().cancelledFetches).toBe(2); // both children
    expect(controller.stats().fetchedTiles).toBe(1); // the root completed
    controller.dispose();
  });

  it("counts fetches cancelled by dispose", async () => {
    const { controller } = makeController(SMALL_TREE);
    await settle();
    controller.setCamera(VIEW);
    await settle();
    expect(controller.stats().inFlight).toBe(3);

    controller.dispose();
    expect(controller.stats().cancelledFetches).toBe(3);
    expect(controller.stats().fetchedTiles).toBe(0);
  });

  it("counts a superseded fetch as fetched, never as a second cancellation", async () => {
    // Aborting is advisory, so a superseded request still resolves: its bytes
    // crossed the network and count once, and the abort that superseded it was
    // already counted where it happened.
    const resolvers: Array<(tile: TileData) => void> = [];
    const source: TileSource = {
      metadata: () => METADATA,
      async nodes() {
        return [
          {
            key: { level: 0, x: 0, y: 0, z: 0 },
            pointCount: 100,
            children: [],
          },
        ];
      },
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
    controller.setCamera({ ...VIEW, viewProj: LOOK_AWAY }); // aborts (advisory)
    await settle();
    controller.setCamera(VIEW); // refetch into a fresh slot
    await settle();
    expect(controller.stats().cancelledFetches).toBe(1);

    resolvers[0]!(makeTile(100)); // the superseded fetch lands
    await settle();
    expect(controller.stats().fetchedTiles).toBe(1);
    expect(controller.stats().fetchedBytes).toBe(bytesOf(100));
    expect(controller.stats().cancelledFetches).toBe(1);
    expect(controller.stats().residentTiles).toBe(0);
    controller.dispose();
  });

  it("reports a resident tile/point histogram by octree level", async () => {
    const { controller, deferred } = makeController(SMALL_TREE);
    await settle();
    expect(controller.stats().residentLevels).toEqual([]);
    controller.setCamera(VIEW);
    await settle();
    for (const d of deferred.values()) d.resolve();
    await settle();

    expect(controller.stats().residentLevels).toEqual([
      { level: 0, tiles: 1, points: 100 },
      { level: 1, tiles: 2, points: 120 },
    ]);
    controller.dispose();
    expect(controller.stats().residentLevels).toEqual([]);
  });

  it("keeps counters across setSource so consumers can take deltas", async () => {
    const { controller, deferred, source } = makeController(SMALL_TREE);
    await settle();
    controller.setCamera(VIEW);
    await settle();
    deferred.get("0-0-0-0")!.resolve();
    await settle();

    controller.setSource(source); // drops residents, caches, in-flight fetches
    const after = controller.stats();
    expect(after.fetchedTiles).toBe(1);
    expect(after.fetchedBytes).toBe(bytesOf(100));
    expect(after.cacheMisses).toBe(3);
    expect(after.cancelledFetches).toBe(2); // the two in-flight children
    expect(after.cachedTiles).toBe(0); // the LRU itself is cleared
    expect(after.residentTiles).toBe(0);
    controller.dispose();
  });

  it("reports adaptive track stats only when adaptive is enabled", async () => {
    const fixed = makeController(SMALL_TREE);
    await settle();
    expect(fixed.controller.stats().adaptive).toBeNull();
    fixed.controller.dispose();

    const { controller } = makeAdaptiveController({ initialBudget: 2_000_000 });
    await settle();
    expect(controller.stats().adaptive?.stationary).toEqual({
      budget: 2_000_000,
      samples: 0,
      estimateMs: null,
    });
    expect(controller.stats().adaptive?.interaction.budget).toBe(1_000_000);

    controller.setCamera(VIEW); // interacting: the frame lands on that track
    controller.recordFrame(20);
    expect(controller.stats().adaptive?.interaction).toEqual({
      budget: 1_000_000,
      samples: 1,
      estimateMs: 20,
    });
    expect(controller.stats().adaptive?.stationary.samples).toBe(0);
    controller.dispose();
  });

  it("reports why the latest selection stopped refining", async () => {
    const { controller, deferred } = makeController(SMALL_TREE, {
      pointBudget: 150,
    });
    await settle();
    controller.setCamera(VIEW);
    await settle();
    deferred.get("0-0-0-0")!.resolve();
    await settle();

    const stats = controller.stats();
    expect(stats.controllerInstanceId).toBeGreaterThan(0);
    expect(stats.sourceEpoch).toBe(0);
    expect(stats.selection).toMatchObject({
      targetTiles: 1,
      targetPoints: 100,
      selectedNodes: 1,
      budgetSkippedNodes: 2,
      budgetSkippedPoints: 120,
    });
    expect(stats.queuedTiles).toBe(0);
    expect(stats.hierarchyPagesLoading).toBe(0);
    expect(stats.hierarchyPagesLoaded).toBe(1);
    controller.dispose();
  });

  it("tracks source/frontier identity and refinement-cutoff changes", async () => {
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
    expect(cutoff.selection.frontierSse.count).toBeGreaterThan(0);

    for (const d of deferred.values()) d.resolve();
    await settle();
    const targetRevision = controller.stats().selection.targetRevision;
    controller.refresh();
    expect(controller.stats().selection.targetRevision).toBe(targetRevision);

    controller.setSource(source);
    expect(controller.stats().sourceEpoch).toBe(1);
    expect(controller.stats().selection.targetTiles).toBe(0);
    controller.dispose();
  });
});
// --- end phase0-bench ---
