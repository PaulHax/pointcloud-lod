import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { createLodController, type TileBatch } from './controller';
import type { AdaptiveBudgetOptions } from './adaptiveBudget';
import { keyToString, type VoxelKey } from './octree';
import type {
  NodeInfo,
  TileData,
  TileSource,
  TileSourceMetadata,
} from './tileSource';

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
      const [level, x, y, z] = keyString.split('-').map(Number);
      return {
        key: { level: level!, x: x!, y: y!, z: z! },
        pointCount: entry.pointCount,
        children: entry.children?.map((c) => {
          const [cl, cx, cy, cz] = c.split('-').map(Number);
          return { level: cl!, x: cx!, y: cy!, z: cz! };
        }),
        pageRef: entry.pageRef,
      };
    });

  const source: TileSource = {
    metadata: () => METADATA,
    async nodes(key: VoxelKey) {
      const keyString = keyToString(key);
      if (keyString === '0-0-0-0') return toInfos(tree);
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
        opts?.signal?.addEventListener('abort', () => {
          d.aborted = true;
          const error = new Error('aborted');
          error.name = 'AbortError';
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
  '0-0-0-0': { pointCount: 100, children: ['1-0-0-0', '1-1-0-0'] },
  '1-0-0-0': { pointCount: 60 },
  '1-1-0-0': { pointCount: 60 },
};

describe('createLodController', () => {
  it('selects within the budget with a parent-closed set', async () => {
    const { controller, loadCalls, deferred, batches } = makeController(
      SMALL_TREE,
      { pointBudget: 200 },
    );
    await settle(); // root hierarchy page
    controller.setCamera(VIEW);
    await settle();

    // 100 + 60 + 60 exceeds 200: root plus exactly one child fetches.
    expect(loadCalls).toHaveLength(2);
    expect(loadCalls[0]).toBe('0-0-0-0'); // coarse level first

    deferred.get(loadCalls[0]!)!.resolve();
    deferred.get(loadCalls[1]!)!.resolve();
    await settle();

    const added = batches.flatMap((b) => b.added.map((a) => keyToString(a.key)));
    expect(added.sort()).toEqual([loadCalls[0], loadCalls[1]].sort());
    expect(controller.stats().residentPoints).toBe(160);
    controller.dispose();
  });

  it('raising the budget refines further', async () => {
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

  it('cancels in-flight fetches when the camera looks away', async () => {
    const { controller, deferred, batches } = makeController(SMALL_TREE);
    await settle();
    controller.setCamera(VIEW);
    await settle();

    deferred.get('0-0-0-0')!.resolve();
    await settle();
    expect(controller.stats().residentTiles).toBe(1);

    controller.setCamera({ ...VIEW, viewProj: LOOK_AWAY });
    await settle();

    // The unresolved child fetches were aborted, the resident root removed.
    expect(deferred.get('1-0-0-0')!.aborted).toBe(true);
    expect(deferred.get('1-1-0-0')!.aborted).toBe(true);
    const removed = batches.flatMap((b) =>
      b.removed.map((k) => keyToString(k)),
    );
    expect(removed).toContain('0-0-0-0');
    expect(controller.stats().residentTiles).toBe(0);
    expect(controller.stats().inFlight).toBe(0);
    controller.dispose();
  });

  it('reuses cached tiles without refetching', async () => {
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

  it('evicts deselected tiles beyond the cache byte bound', async () => {
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

  it('never applies a tile that resolves after setSource', async () => {
    const { controller, deferred, batches, source } =
      makeController(SMALL_TREE);
    await settle();
    controller.setCamera(VIEW);
    await settle();

    const stale = deferred.get('0-0-0-0')!;
    controller.setSource(source); // same source object; state still drops
    await settle();
    stale.resolve();
    await settle();

    const added = batches.flatMap((b) => b.added.map((a) => keyToString(a.key)));
    expect(added).toHaveLength(0);
    controller.dispose();
  });

  it('loads hierarchy pages lazily behind pageRef entries', async () => {
    const tree: Record<string, FakeEntry> = {
      '0-0-0-0': { pointCount: 10, children: ['1-0-0-0'] },
      '1-0-0-0': {
        pointCount: 0,
        pageRef: true,
        pageNodes: { '1-0-0-0': { pointCount: 40 } },
      },
    };
    const { controller, loadCalls, deferred } = makeController(tree);
    await settle();
    controller.setCamera(VIEW);
    await settle(); // triggers page load, reselects on arrival
    await settle();

    expect(loadCalls).toContain('1-0-0-0');
    for (const d of deferred.values()) d.resolve();
    await settle();
    expect(controller.stats().residentPoints).toBe(50);
    controller.dispose();
  });

  it('coalesces same-tick arrivals into one batch and one render', async () => {
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

  it('dispose removes residents, cancels fetches, and is idempotent', async () => {
    const { controller, deferred, batches } = makeController(SMALL_TREE);
    await settle();
    controller.setCamera(VIEW);
    await settle();
    deferred.get('0-0-0-0')!.resolve();
    await settle();

    batches.length = 0;
    controller.dispose();
    controller.dispose();

    expect(batches).toHaveLength(1);
    expect(batches[0]!.removed.map(keyToString)).toEqual(['0-0-0-0']);
    expect(deferred.get('1-0-0-0')!.aborted).toBe(true);

    // A late resolve after dispose must do nothing.
    deferred.get('1-1-0-0')!.resolve();
    await settle();
    expect(batches).toHaveLength(1);
    expect(controller.stats().residentTiles).toBe(0);
  });

  it('does not fetch culled subtrees', async () => {
    const tree: Record<string, FakeEntry> = {
      '0-0-0-0': { pointCount: 10, children: ['1-0-0-0'] },
      '1-0-0-0': { pointCount: 40 },
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
    expect(fake.loadCalls).toContain('0-0-0-0');
    expect(fake.loadCalls).not.toContain('1-0-0-0');
    controller.dispose();
  });
});

describe('createLodController — adaptive budget', () => {
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
    pointBudget: number,
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
      adaptive,
    });
    return { controller, ...fake, ...sink };
  };

  it('starts at the ceiling and reports the interaction regime', async () => {
    const { controller } = makeAdaptive(SMALL_TREE, true, 2_000_000);
    await settle();
    expect(controller.stats().pointBudget).toBe(2_000_000);
    expect(controller.stats().interacting).toBe(false);
    controller.dispose();
  });

  it('keeps interaction and stationary budgets independent', async () => {
    const { controller } = makeAdaptive(SMALL_TREE, { minSamples: 4 }, 2_000_000);
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

  it('a shrink during interaction retains visible resident tiles', async () => {
    // Point counts scaled to millions so a budget change crosses a selection
    // boundary: at 3M all three tiles fit; at 2.25M only the root and one child.
    const bigTree: Record<string, FakeEntry> = {
      '0-0-0-0': { pointCount: 1_000_000, children: ['1-0-0-0', '1-1-0-0'] },
      '1-0-0-0': { pointCount: 1_000_000 },
      '1-1-0-0': { pointCount: 1_000_000 },
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
    // child is still on screen mid-gesture, so it stays resident — removing
    // and re-adding it moments later would read as per-tile flashing.
    for (let i = 0; i < 4; i += 1) {
      vi.setSystemTime(i);
      controller.recordFrame(80);
    }
    await settle();
    vi.setSystemTime(10);
    expect(controller.stats().pointBudget).toBe(2_250_000);
    expect(controller.stats().residentTiles).toBe(3);
    expect(batches.flatMap((b) => b.removed)).toHaveLength(0);
    controller.dispose();
  });

  it('consolidates retained tiles once the camera settles (adaptive off)', async () => {
    // Retention + settle consolidation are regime behaviors, not adaptive
    // ones: a fixed-budget drop mid-gesture must not flash either.
    const { controller, deferred } = makeAdaptive(SMALL_TREE, false, 1000);
    await settle();
    controller.setCamera(VIEW); // t=0: interacting
    await settle();
    for (const d of deferred.values()) d.resolve();
    await settle();
    expect(controller.stats().residentTiles).toBe(3);

    // Lower the fixed budget mid-gesture: both children (60 points each) fall
    // out of the 150-point selection but stay resident — still on screen.
    controller.setPointBudget(150);
    await settle();
    expect(controller.stats().residentTiles).toBe(3);

    // Once the camera has been still for the settle window, the retained
    // children are dropped in one consolidation pass.
    vi.advanceTimersByTime(300);
    await settle();
    expect(controller.stats().residentTiles).toBe(1);
    controller.dispose();
  });

  it('applies the stationary budget after the camera settles, with no further frames', async () => {
    // Same millions-scaled tree: 3M fits all three tiles, 2.25M only two.
    const bigTree: Record<string, FakeEntry> = {
      '0-0-0-0': { pointCount: 1_000_000, children: ['1-0-0-0', '1-1-0-0'] },
      '1-0-0-0': { pointCount: 1_000_000 },
      '1-1-0-0': { pointCount: 1_000_000 },
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
    // tile is retained (anti-flashing) but the budget has moved.
    for (let i = 0; i < 4; i += 1) {
      vi.setSystemTime(i);
      controller.recordFrame(80);
    }
    await settle();
    expect(controller.stats().pointBudget).toBe(2_250_000);
    expect(controller.stats().residentTiles).toBe(3);

    // No further frames — the host renders on demand and goes quiet. Advancing
    // past the settle window fires the settle timer, which re-applies the
    // (untouched) stationary 3M budget; the retained tile is selected again.
    vi.advanceTimersByTime(400);
    await settle();
    expect(controller.stats().interacting).toBe(false);
    expect(controller.stats().pointBudget).toBe(3_000_000);
    expect(controller.stats().residentTiles).toBe(3);
    controller.dispose();
  });

  it('setPointBudget lowers the ceiling and clamps the effective budget', async () => {
    const { controller } = makeAdaptive(SMALL_TREE, true, 2_000_000);
    await settle();
    expect(controller.stats().pointBudget).toBe(2_000_000);
    controller.setPointBudget(1_000_000); // new ceiling
    expect(controller.stats().pointBudget).toBe(1_000_000);
    controller.dispose();
  });

  it('an unchanged camera view does not re-enter the interaction regime', async () => {
    // Hosts call setCamera before every paint with a freshly built (but often
    // identical) view — including paints triggered by the settle reselect
    // itself. An identical view must not stamp a camera change, or the regime
    // would flip back to interaction and oscillate between the two budgets.
    const { controller } = makeAdaptive(SMALL_TREE, { minSamples: 4 }, 2_000_000);
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

  it('recordFrame is a no-op when adaptive is disabled', async () => {
    const { controller } = makeController(SMALL_TREE, { pointBudget: 500 });
    await settle();
    controller.setCamera(VIEW);
    await settle();
    for (let i = 0; i < 20; i += 1) controller.recordFrame(5000);
    expect(controller.stats().pointBudget).toBe(500);
    controller.dispose();
  });
});
