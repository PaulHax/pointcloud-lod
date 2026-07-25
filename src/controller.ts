/**
 * LOD controller: turns camera movement into a bounded set of submitted tiles.
 *
 * Selection is a pure pass (frustum cull → screen-space-error priority →
 * parent-closed point-budget selection); the controller owns the impure rest:
 * lazy hierarchy pages, a bounded fetch queue with cancellation, an LRU for
 * deselected tiles, and batched delivery to the consumer.
 *
 * COPC hierarchies are additive — children add detail while parents keep
 * rendering — so there is no parent/child swap and no hole risk: the
 * parent-closed selection invariant is the entire hole-free story.
 */

import {
  frustumPlanes,
  nodeScreenSpaceError,
  cubeIntersectsFrustum,
  type CameraView,
} from "./camera";
import {
  createAdaptiveBudget,
  type AdaptiveBudget,
  type AdaptiveBudgetOptions,
  // --- phase0-bench (removable; see app/telesculptor_web/app/bench/README.md) ---
  type AdaptiveBudgetStats,
  // --- end phase0-bench ---
} from "./adaptiveBudget";
import { selectNodes } from "./budget";
import { createLruCache } from "./lru";
import {
  createMemoryPool,
  type MemoryPool,
  type MemoryPoolMember,
} from "./memoryPool";
import {
  ROOT_KEY,
  childKeys,
  keyFromString,
  keyToString,
  nodeCube,
  pointSpacing,
  type VoxelKey,
} from "./octree";
import type { TileData, TileSource } from "./tileSource";

export interface TileBatch {
  readonly added: readonly { key: VoxelKey; tile: TileData }[];
  readonly removed: readonly VoxelKey[];
}

export interface LodControllerOptions {
  source: TileSource;
  /** Receives batched tile arrivals/removals (typically a renderer adapter). */
  onTiles: (batch: TileBatch) => void;
  /**
   * Coalescing render request — called once per applied batch, never once per
   * tile. Must not render synchronously more than once per event loop turn.
   */
  scheduleRender: () => void;
  /**
   * Fixed visible-point budget when `adaptive` is off. Default 2,000,000.
   * Ignored with `adaptive` enabled — there is no configured point ceiling;
   * frame time and the memory budget are the governors. The memory-derived
   * point cap applies in both modes.
   */
  pointBudget?: number;
  /**
   * Adapt the visible-point budget to measured render duration (Phase 5). Pass
   * `true` for defaults, an options object to tune, or omit/false for a fixed
   * `pointBudget`. When enabled, feed each render's wall-time to `recordFrame`.
   */
  adaptive?: AdaptiveBudgetOptions | boolean;
  /**
   * GPU-memory budget for resident tile bytes. Pass a `MemoryPool` to share
   * one byte budget across controllers on the same GPU (each gets an even
   * share), a number of bytes for a private budget, or omit for a private
   * budget sized by `defaultMemoryBudgetBytes()`. The controller converts its
   * byte share into a point ceiling using the measured bytes-per-point of
   * resident tiles, so the frame-time loop can never climb into an
   * out-of-memory failure it has no way to sense.
   */
  memory?: MemoryPool | number;
  /**
   * How long after the last camera change the controller still treats itself
   * as "interacting" for adaptive budgeting, ms. Default 300.
   */
  interactionSettleMs?: number;
  /** Parallel tile fetches. Default 6. */
  fetchConcurrency?: number;
  /** CPU cache for deselected tiles, bytes. Default 256 MiB. */
  cacheBytes?: number;
  /**
   * Whether this controller may request and submit tiles. Default true.
   * Inactive controllers retain only byte-bounded decoded cache entries; no
   * renderer resources remain live.
   */
  active?: boolean;
  /** Trailing debounce for camera-driven reselection, ms. Default 150. */
  selectionDelayMs?: number;
  /**
   * Nodes whose projected point spacing is below this many pixels are not
   * refined further. Default 1.
   */
  refinementCutoffPx?: number;
  /** Non-abort fetch/hierarchy failures land here. Default: console.warn. */
  onError?: (error: unknown) => void;
}

export interface LodControllerStats {
  /** Whether selection, requests, and renderer submission are enabled. */
  readonly active: boolean;
  readonly residentTiles: number;
  readonly residentPoints: number;
  /** Decoded bytes backing currently submitted tiles. */
  readonly residentBytes: number;
  /** Submitted tiles; explicit alias for `residentTiles`. */
  readonly activeTiles: number;
  /** Submitted points; explicit alias for `residentPoints`. */
  readonly activePoints: number;
  /** Decoded payloads across submitted tiles and the dormant CPU cache. */
  readonly decodedTiles: number;
  /** Decoded bytes across submitted tiles and the dormant CPU cache. */
  readonly decodedBytes: number;
  readonly cachedBytes: number;
  readonly inFlight: number;
  /** Effective visible-point budget currently driving selection. */
  readonly pointBudget: number;
  /** This controller's byte share of its memory pool. */
  readonly memoryBudgetBytes: number;
  /** Memory-derived point ceiling the budget can never exceed. */
  readonly memoryCeilingPoints: number;
  /** Whether the controller currently treats itself as interacting. */
  readonly interacting: boolean;
  /** Explicit interaction nesting depth reported by the host. */
  readonly interactionDepth: number;
  // --- phase0-bench (removable; see app/telesculptor_web/app/bench/README.md) ---
  /**
   * Diagnostic counters below are cumulative for the controller's lifetime and
   * are deliberately NOT reset by `setSource()` — consumers take deltas.
   */
  /** LRU entry count (deselected tiles kept for cheap reselection). */
  readonly cachedTiles: number;
  /** Completed tile fetches, including ones whose result was discarded. */
  readonly fetchedTiles: number;
  /** Bytes of completed tile fetches (same population as `fetchedTiles`). */
  readonly fetchedBytes: number;
  /** Fetches this controller aborted (deselection, setSource, dispose). */
  readonly cancelledFetches: number;
  /** Selection entries served from the LRU without a fetch. */
  readonly cacheHits: number;
  /** Selection entries that had to be queued for a fetch. */
  readonly cacheMisses: number;
  /** Resident tile/point histogram by octree level, ascending. */
  readonly residentLevels: readonly {
    level: number;
    tiles: number;
    points: number;
  }[];
  /** Both adaptive tracks' budgets and estimates; null when adaptive is off. */
  readonly adaptive: AdaptiveBudgetStats | null;
  /** Stable identity for this controller closure. */
  readonly controllerInstanceId: number;
  /** Increments whenever `setSource()` invalidates the controller's source. */
  readonly sourceEpoch: number;
  /** Tile fetches selected but not yet started because concurrency is full. */
  readonly queuedTiles: number;
  /** Hierarchy pages currently being requested. */
  readonly hierarchyPagesLoading: number;
  /** Hierarchy pages available to traversal. */
  readonly hierarchyPagesLoaded: number;
  /** Whether a camera-driven selection pass is waiting on its debounce timer. */
  readonly selectionPending: boolean;
  /** Whether the interaction-to-stationary consolidation timer is armed. */
  readonly settlePending: boolean;
  /** Current screen-space refinement cutoff. */
  readonly refinementCutoffPx: number;
  /** Latest selection pass and the reasons traversal stopped. */
  readonly selection: LodSelectionStats;
  // --- end phase0-bench ---
}

export interface LodSelectionStats {
  readonly generation: number;
  /** Increments only when the selected key set changes. */
  readonly targetRevision: number;
  readonly targetTiles: number;
  readonly targetPoints: number;
  readonly consideredNodes: number;
  readonly availableNodes: number;
  readonly selectedNodes: number;
  readonly hierarchyUnavailableNodes: number;
  readonly hierarchyPageBlockedNodes: number;
  readonly frustumCulledNodes: number;
  readonly frustumCulledPoints: number;
  readonly leafNodes: number;
  readonly sseStoppedNodes: number;
  readonly budgetSkippedNodes: number;
  readonly budgetSkippedPoints: number;
  /** Root projected screen-space error, used for cross-cloud allocation. */
  readonly projectedImportance: number;
  /** SSE distribution for selected nodes stopped by the cutoff. */
  readonly frontierSse: {
    readonly count: number;
    readonly p50: number | null;
    readonly p95: number | null;
    readonly max: number | null;
  };
}

export interface LodController {
  /** Update the camera; selection reruns debounced (leading edge immediate). */
  setCamera(view: CameraView): void;
  /** Enter a camera interaction. Nested calls are reference counted. */
  beginInteraction(): void;
  /** Leave a camera interaction; the outermost end starts the settle window. */
  endInteraction(): void;
  /**
   * Set the fixed visible-point budget (non-adaptive mode only; with
   * `adaptive` enabled the budget is governed by frame time and memory and
   * this is a no-op).
   */
  setPointBudget(points: number): void;
  /**
   * Report one rendered frame's duration (ms) to the adaptive budget loop.
   * No-op unless `adaptive` is enabled. The host measures the render wall-time
   * and calls this once per painted frame.
   */
  recordFrame(durationMs: number): void;
  /**
   * Swap the tile source (e.g. a new asset revision behind a new endpoint).
   * All resident tiles, caches, hierarchy state, and in-flight requests are
   * dropped; nothing stale can apply afterwards.
   */
  setSource(source: TileSource): void;
  /** Force an immediate reselection with the current camera. */
  refresh(): void;
  /** Change the screen-space refinement cutoff and reselect immediately. */
  setRefinementCutoffPx(pixels: number): void;
  /**
   * Enable or disable renderer submission. Disabling immediately removes all
   * submitted tiles, cancels tile requests, and moves decoded payloads into
   * the byte-bounded CPU cache. Re-enabling selects against the latest camera.
   */
  setActive(active: boolean): void;
  stats(): LodControllerStats;
  /** Cancel everything and release all tiles. Idempotent. */
  dispose(): void;
}

interface HierarchyEntry {
  pointCount: number;
  children: readonly VoxelKey[] | null;
  pageRef: boolean;
}

const isAbortError = (error: unknown): boolean =>
  error instanceof Error && error.name === "AbortError";

const tileBytes = (tile: TileData): number =>
  tile.positions.byteLength + (tile.rgb?.byteLength ?? 0) + 64;

let nextControllerInstanceId = 1;

const percentile = (values: readonly number[], p: number): number | null => {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const rank = Math.ceil(p * sorted.length);
  return sorted[Math.min(sorted.length, Math.max(1, rank)) - 1] ?? null;
};

// --- phase0-bench (removable; see app/telesculptor_web/app/bench/README.md) ---
/** Resident tiles/points grouped by octree level, ascending. Diagnostics only. */
const levelHistogram = (
  tiles: ReadonlyMap<string, TileData>,
): { level: number; tiles: number; points: number }[] => {
  const byLevel = new Map<
    number,
    { level: number; tiles: number; points: number }
  >();
  for (const [keyString, tile] of tiles) {
    const { level } = keyFromString(keyString);
    const entry = byLevel.get(level);
    if (entry === undefined) {
      byLevel.set(level, { level, tiles: 1, points: tile.pointCount });
    } else {
      entry.tiles += 1;
      entry.points += tile.pointCount;
    }
  }
  return [...byLevel.values()].sort((a, b) => a.level - b.level);
};
// --- end phase0-bench ---

export const createLodController = (
  options: LodControllerOptions,
): LodController => {
  const {
    onTiles,
    scheduleRender,
    fetchConcurrency = 6,
    cacheBytes = 256 * 1024 * 1024,
    selectionDelayMs = 150,
    interactionSettleMs = 300,
    refinementCutoffPx: initialRefinementCutoffPx = 1,
    onError = (error) => console.warn("pointcloud-lod:", error),
  } = options;

  let source = options.source;
  let pointBudget = options.pointBudget ?? 2_000_000;
  let refinementCutoffPx = initialRefinementCutoffPx;
  let active = options.active ?? true;
  let view: CameraView | null = null;
  let disposed = false;
  const controllerInstanceId = nextControllerInstanceId++;

  // Adaptive budget (Phase 5): when enabled, the loop moves the effective
  // budget between a floor and the memory-derived ceiling, tracking a target
  // frame time. When disabled, `pointBudget` is fixed (memory still caps it).
  const adaptiveBudget: AdaptiveBudget | null = options.adaptive
    ? createAdaptiveBudget(
        typeof options.adaptive === "object" ? options.adaptive : {},
      )
    : null;

  let lastCameraChange = Number.NEGATIVE_INFINITY;
  let explicitInteractionSeen = false;
  let interactionDepth = 0;
  let interactionSettling = false;
  // Effective budget applied by the most recent selection; lets `recordFrame`
  // reselect only when the adaptive budget (or interaction regime) has moved.
  let lastSelectionBudget = pointBudget;
  const isInteracting = (now: number): boolean =>
    active &&
    (explicitInteractionSeen
      ? interactionDepth > 0 || interactionSettling
      : now - lastCameraChange < interactionSettleMs);

  const currentBudget = (now: number): number => {
    if (!active) return 0;
    return adaptiveBudget === null
      ? Math.min(pointBudget, memoryCeilingPoints())
      : adaptiveBudget.budget(isInteracting(now));
  };

  // Bumped on setSource/dispose; every async continuation checks it.
  let epoch = 0;

  const hierarchy = new Map<string, HierarchyEntry>();
  const pagesLoaded = new Set<string>();
  const pagesLoading = new Set<string>();

  /** Tiles currently delivered to the consumer. */
  const resident = new Map<string, TileData>();
  let residentPoints = 0;
  let residentBytes = 0;
  /** Deselected tiles kept for cheap reselection. */
  const cache = createLruCache<string, TileData>({ maxBytes: cacheBytes });

  // Memory governor: the byte share converts to a point ceiling via the
  // measured bytes-per-point of resident tiles (falling back to an estimate
  // until enough points are resident to measure). The frame-time loop cannot
  // sense GPU memory — frames stay fast right up until an allocation fails —
  // so this ceiling is what keeps "adaptive" from meaning "climb until the
  // context is lost".
  const memoryPool: MemoryPool =
    typeof options.memory === "object"
      ? options.memory
      : createMemoryPool(
          typeof options.memory === "number"
            ? { totalBytes: options.memory }
            : {},
        );
  let poolMember: MemoryPoolMember | null = null;
  const joinMemoryPool = (): void => {
    if (poolMember !== null) return;
    poolMember = memoryPool.register(() => {
      if (disposed || !active) return;
      applyMemoryCeiling();
      requestSelection();
    });
  };
  if (active) joinMemoryPool();

  const FALLBACK_BYTES_PER_POINT = 16;
  const MEASURE_MIN_POINTS = 100_000;

  const bytesPerPoint = (): number =>
    residentPoints >= MEASURE_MIN_POINTS
      ? residentBytes / residentPoints
      : FALLBACK_BYTES_PER_POINT;

  const memoryCeilingPoints = (): number =>
    poolMember === null
      ? 0
      : Math.max(1, Math.floor(poolMember.budgetBytes() / bytesPerPoint()));

  // Last ceiling handed to the adaptive loop. The bytes-per-point estimate
  // drifts as tiles arrive, and setMaxBudget discards a track's samples when
  // it clamps a budget — a small dead-band keeps estimate jitter from
  // re-clamping every selection.
  let appliedCeiling = 0;
  const applyMemoryCeiling = (): void => {
    if (adaptiveBudget === null || poolMember === null) return;
    const ceiling = memoryCeilingPoints();
    if (
      appliedCeiling > 0 &&
      Math.abs(ceiling - appliedCeiling) / appliedCeiling <= 0.02
    ) {
      return;
    }
    appliedCeiling = ceiling;
    adaptiveBudget.setMaxBudget(ceiling);
  };
  applyMemoryCeiling();

  let target: ReadonlySet<string> = new Set<string>();
  let targetRevision = 0;
  let selectionGeneration = 0;
  let selectionStats: LodSelectionStats = {
    generation: 0,
    targetRevision: 0,
    targetTiles: 0,
    targetPoints: 0,
    consideredNodes: 0,
    availableNodes: 0,
    selectedNodes: 0,
    hierarchyUnavailableNodes: 0,
    hierarchyPageBlockedNodes: 0,
    frustumCulledNodes: 0,
    frustumCulledPoints: 0,
    leafNodes: 0,
    sseStoppedNodes: 0,
    budgetSkippedNodes: 0,
    budgetSkippedPoints: 0,
    projectedImportance: 0,
    frontierSse: { count: 0, p50: null, p95: null, max: null },
  };
  const inFlight = new Map<string, AbortController>();
  let queue: string[] = [];

  // --- phase0-bench (removable; see app/telesculptor_web/app/bench/README.md) ---
  // Lifetime-cumulative diagnostic counters. Never reset (not even by
  // setSource/dropEverything) so consumers can take deltas across a run.
  let fetchedTiles = 0;
  let fetchedBytes = 0;
  let cancelledFetches = 0;
  let cacheHits = 0;
  let cacheMisses = 0;
  // --- end phase0-bench ---

  let pendingAdded: { key: VoxelKey; tile: TileData }[] = [];
  let pendingRemoved: VoxelKey[] = [];
  let flushScheduled = false;

  const scheduleFlush = (): void => {
    if (flushScheduled) return;
    flushScheduled = true;
    queueMicrotask(() => {
      flushScheduled = false;
      if (disposed) return;
      if (pendingAdded.length === 0 && pendingRemoved.length === 0) return;
      const batch: TileBatch = {
        added: pendingAdded,
        removed: pendingRemoved,
      };
      pendingAdded = [];
      pendingRemoved = [];
      onTiles(batch);
      scheduleRender();
    });
  };

  const loadPage = (key: VoxelKey): void => {
    const keyString = keyToString(key);
    if (pagesLoaded.has(keyString) || pagesLoading.has(keyString)) return;
    pagesLoading.add(keyString);
    const requestEpoch = epoch;
    source
      .nodes(key)
      .then((infos) => {
        if (disposed || requestEpoch !== epoch) return;
        pagesLoading.delete(keyString);
        pagesLoaded.add(keyString);
        for (const info of infos) {
          hierarchy.set(keyToString(info.key), {
            pointCount: info.pointCount,
            children: info.children ?? null,
            pageRef: info.pageRef === true,
          });
        }
        runSelection();
      })
      .catch((error) => {
        if (disposed || requestEpoch !== epoch) return;
        pagesLoading.delete(keyString);
        onError(error);
      });
  };

  const childrenOf = (
    key: VoxelKey,
    entry: HierarchyEntry,
  ): readonly VoxelKey[] =>
    entry.children ??
    childKeys(key).filter((child) => hierarchy.has(keyToString(child)));

  const pump = (): void => {
    while (inFlight.size < fetchConcurrency && queue.length > 0) {
      const keyString = queue.shift()!;
      if (
        !target.has(keyString) ||
        resident.has(keyString) ||
        inFlight.has(keyString)
      ) {
        continue;
      }
      const abort = new AbortController();
      inFlight.set(keyString, abort);
      const requestEpoch = epoch;
      const key = keyFromString(keyString);
      source
        .loadTile(key, { signal: abort.signal })
        .then((tile) => {
          // --- phase0-bench (removable; see app/telesculptor_web/app/bench/README.md) ---
          // A fetch that resolves counts as fetched even when its payload is
          // discarded (superseded slot, stale epoch, no longer targeted) — the
          // network cost was paid. It is not also counted as a cancellation;
          // cancellations are counted where `abort()` is called. Arrivals after
          // dispose are ignored: disposed stats are frozen.
          if (!disposed) {
            fetchedTiles += 1;
            fetchedBytes += tileBytes(tile);
          }
          // --- end phase0-bench ---
          if (disposed || requestEpoch !== epoch) return;
          // Aborting is advisory: the COPC getter takes no signal, so a
          // superseded request still resolves. Only the continuation that owns
          // the current in-flight slot may retire it or claim residency —
          // otherwise a stale arrival frees a live slot (uncapping
          // fetchConcurrency) and double-counts resident points and bytes.
          if (inFlight.get(keyString) !== abort) {
            cache.set(keyString, tile, tileBytes(tile));
            pump();
            return;
          }
          inFlight.delete(keyString);
          if (target.has(keyString) && !resident.has(keyString)) {
            resident.set(keyString, tile);
            residentPoints += tile.pointCount;
            residentBytes += tileBytes(tile);
            pendingAdded.push({ key, tile });
            scheduleFlush();
          } else {
            cache.set(keyString, tile, tileBytes(tile));
          }
          pump();
        })
        .catch((error) => {
          if (disposed || requestEpoch !== epoch) return;
          if (inFlight.get(keyString) === abort) inFlight.delete(keyString);
          if (!isAbortError(error)) onError(error);
          pump();
        });
    }
  };

  const runSelection = (): void => {
    if (disposed || !active || view === null) return;
    const now = Date.now();
    // Re-derive the memory ceiling first: resident bytes (and with them the
    // bytes-per-point estimate) changed since the last selection.
    applyMemoryCeiling();
    const budget = currentBudget(now);
    lastSelectionBudget = budget;
    const currentView = view;
    const meta = source.metadata();
    const planes = frustumPlanes(currentView.viewProj);
    const sseByKey = new Map<string, number>();
    const sse = (key: VoxelKey): number => {
      const keyString = keyToString(key);
      let value = sseByKey.get(keyString);
      if (value === undefined) {
        value = nodeScreenSpaceError(
          nodeCube(meta.cube, key),
          pointSpacing(meta.spacing, key.level),
          currentView,
        );
        sseByKey.set(keyString, value);
      }
      return value;
    };

    const neededPages: VoxelKey[] = [];
    let hierarchyUnavailableNodes = 0;
    let hierarchyPageBlockedNodes = 0;
    let frustumCulledNodes = 0;
    let frustumCulledPoints = 0;
    let leafNodes = 0;
    let sseStoppedNodes = 0;
    const selection = selectNodes({
      root: ROOT_KEY,
      pointBudget: budget,
      priority: sse,
      getNode: (key) => {
        const keyString = keyToString(key);
        const entry = hierarchy.get(keyString);
        if (entry === undefined) {
          hierarchyUnavailableNodes += 1;
          return undefined;
        }
        if (entry.pageRef && !pagesLoaded.has(keyString)) {
          neededPages.push(key);
          hierarchyPageBlockedNodes += 1;
          return undefined;
        }
        if (!cubeIntersectsFrustum(planes, nodeCube(meta.cube, key))) {
          frustumCulledNodes += 1;
          frustumCulledPoints += entry.pointCount;
          return undefined;
        }
        const availableChildren = childrenOf(key, entry);
        let children = availableChildren;
        if (availableChildren.length === 0) {
          leafNodes += 1;
        } else if (sse(key) < refinementCutoffPx) {
          sseStoppedNodes += 1;
          children = [];
        }
        return { pointCount: entry.pointCount, children };
      },
    });

    const previousTarget = target;
    target = selection.selected;
    if (
      previousTarget.size !== target.size ||
      [...target].some((keyString) => !previousTarget.has(keyString))
    ) {
      targetRevision += 1;
    }
    selectionGeneration += 1;
    const frontierSse = [...target]
      .map(keyFromString)
      .filter((key) => {
        const entry = hierarchy.get(keyToString(key));
        return (
          !!entry &&
          childrenOf(key, entry).length > 0 &&
          sse(key) < refinementCutoffPx
        );
      })
      .map(sse);
    selectionStats = {
      generation: selectionGeneration,
      targetRevision,
      targetTiles: target.size,
      targetPoints: selection.totalPoints,
      consideredNodes: selection.consideredNodes,
      availableNodes: selection.availableNodes,
      selectedNodes: selection.selectedNodes,
      hierarchyUnavailableNodes,
      hierarchyPageBlockedNodes,
      frustumCulledNodes,
      frustumCulledPoints,
      leafNodes,
      sseStoppedNodes,
      budgetSkippedNodes: selection.budgetSkippedNodes,
      budgetSkippedPoints: selection.budgetSkippedPoints,
      projectedImportance: target.size > 0 ? sse(ROOT_KEY) : 0,
      frontierSse: {
        count: frontierSse.length,
        p50: percentile(frontierSse, 0.5),
        p95: percentile(frontierSse, 0.95),
        max: frontierSse.length > 0 ? Math.max(...frontierSse) : null,
      },
    };
    for (const key of neededPages) loadPage(key);

    // Deselected submitted tiles leave renderer/GPU residency immediately.
    // Their decoded payload may remain in the byte-bounded CPU LRU for cheap
    // reactivation; no dormant actor is retained.
    for (const [keyString, tile] of [...resident]) {
      if (target.has(keyString)) continue;
      resident.delete(keyString);
      residentPoints -= tile.pointCount;
      residentBytes -= tileBytes(tile);
      cache.set(keyString, tile, tileBytes(tile));
      pendingRemoved.push(keyFromString(keyString));
    }

    // Cancel fetches that no longer matter.
    for (const [keyString, abort] of [...inFlight]) {
      if (target.has(keyString)) continue;
      inFlight.delete(keyString);
      // --- phase0-bench (removable; see app/telesculptor_web/app/bench/README.md) ---
      cancelledFetches += 1;
      // --- end phase0-bench ---
      abort.abort();
    }

    // Reuse cached tiles immediately; queue the rest, coarse levels first.
    const toFetch: string[] = [];
    for (const keyString of target) {
      if (resident.has(keyString) || inFlight.has(keyString)) continue;
      const cached = cache.get(keyString);
      if (cached !== undefined) {
        // --- phase0-bench (removable; see app/telesculptor_web/app/bench/README.md) ---
        cacheHits += 1;
        // --- end phase0-bench ---
        cache.delete(keyString);
        resident.set(keyString, cached);
        residentPoints += cached.pointCount;
        residentBytes += tileBytes(cached);
        pendingAdded.push({ key: keyFromString(keyString), tile: cached });
      } else {
        // --- phase0-bench (removable; see app/telesculptor_web/app/bench/README.md) ---
        cacheMisses += 1;
        // --- end phase0-bench ---
        toFetch.push(keyString);
      }
    }
    queue = toFetch.sort((a, b) => {
      const ka = keyFromString(a);
      const kb = keyFromString(b);
      if (ka.level !== kb.level) return ka.level - kb.level;
      return sse(kb) - sse(ka);
    });

    scheduleFlush();
    pump();
  };

  let lastSelection = Number.NEGATIVE_INFINITY;
  let selectionTimer: ReturnType<typeof setTimeout> | null = null;
  // Fires once the camera has been still for interactionSettleMs so the
  // interaction→stationary budget flip is applied even when the host renders
  // on demand and stops calling recordFrame after the last interaction frame.
  let settleTimer: ReturnType<typeof setTimeout> | null = null;

  const clearSettleTimer = (): void => {
    if (settleTimer !== null) {
      clearTimeout(settleTimer);
      settleTimer = null;
    }
  };

  const armSettleTimer = (): void => {
    clearSettleTimer();
    settleTimer = setTimeout(() => {
      settleTimer = null;
      if (disposed) return;
      if (explicitInteractionSeen) interactionSettling = false;
      // Explicit completion is also the stationary refinement trigger. In
      // inference mode a budget change is the only missing work.
      if (
        explicitInteractionSeen ||
        currentBudget(Date.now()) !== lastSelectionBudget
      ) {
        runSelection();
      }
    }, interactionSettleMs);
  };

  const requestSelection = (): void => {
    if (!active) return;
    const now = Date.now();
    if (now - lastSelection >= selectionDelayMs) {
      lastSelection = now;
      runSelection();
      return;
    }
    if (selectionTimer === null) {
      selectionTimer = setTimeout(
        () => {
          selectionTimer = null;
          lastSelection = Date.now();
          runSelection();
        },
        selectionDelayMs - (now - lastSelection),
      );
    }
  };

  const dropEverything = (): void => {
    epoch += 1;
    // --- phase0-bench (removable; see app/telesculptor_web/app/bench/README.md) ---
    // setSource and dispose abort every live fetch; both count. A cancellation
    // is "a fetch this controller gave up on", regardless of the reason.
    cancelledFetches += inFlight.size;
    // --- end phase0-bench ---
    for (const abort of inFlight.values()) abort.abort();
    inFlight.clear();
    queue = [];
    hierarchy.clear();
    pagesLoaded.clear();
    pagesLoading.clear();
    cache.clear();
    if (target.size > 0) targetRevision += 1;
    target = new Set();
    selectionStats = {
      ...selectionStats,
      targetRevision,
      targetTiles: 0,
      targetPoints: 0,
      consideredNodes: 0,
      availableNodes: 0,
      selectedNodes: 0,
      hierarchyUnavailableNodes: 0,
      hierarchyPageBlockedNodes: 0,
      frustumCulledNodes: 0,
      frustumCulledPoints: 0,
      leafNodes: 0,
      sseStoppedNodes: 0,
      budgetSkippedNodes: 0,
      budgetSkippedPoints: 0,
      projectedImportance: 0,
      frontierSse: { count: 0, p50: null, p95: null, max: null },
    };
    for (const keyString of resident.keys()) {
      pendingRemoved.push(keyFromString(keyString));
    }
    resident.clear();
    residentPoints = 0;
    residentBytes = 0;
    pendingAdded = [];
  };

  // Hosts feed the camera on every render, unconditionally; an unchanged view
  // must be a no-op. Stamping it as a camera change would mark any render as
  // "interacting" — including renders the settle-timer reselect itself
  // triggers — flipping the regime back and oscillating between the two
  // budgets forever.
  const sameView = (a: CameraView, b: CameraView): boolean => {
    if (
      a.fovY !== b.fovY ||
      a.viewportHeight !== b.viewportHeight ||
      a.position[0] !== b.position[0] ||
      a.position[1] !== b.position[1] ||
      a.position[2] !== b.position[2] ||
      a.viewProj.length !== b.viewProj.length
    ) {
      return false;
    }
    for (let i = 0; i < a.viewProj.length; i += 1) {
      if (a.viewProj[i] !== b.viewProj[i]) return false;
    }
    return true;
  };

  // Bootstrap: hierarchy root page loads eagerly; selection waits for camera.
  if (active) loadPage(ROOT_KEY);

  return {
    setCamera(nextView) {
      if (disposed) return;
      if (view !== null && sameView(view, nextView)) return;
      view = nextView;
      if (!active) return;
      if (!explicitInteractionSeen) {
        lastCameraChange = Date.now();
        // Camera timestamps are a fallback for hosts without lifecycle events.
        if (adaptiveBudget !== null) armSettleTimer();
      }
      requestSelection();
    },

    beginInteraction() {
      if (disposed) return;
      explicitInteractionSeen = true;
      interactionDepth += 1;
      if (interactionDepth !== 1) return;
      clearSettleTimer();
      interactionSettling = false;
      // Bypass camera debounce before the first expensive moving frame.
      runSelection();
    },

    endInteraction() {
      if (disposed || interactionDepth === 0) return;
      interactionDepth -= 1;
      if (interactionDepth !== 0) return;
      interactionSettling = true;
      armSettleTimer();
    },

    setPointBudget(points) {
      if (disposed) return;
      // Adaptive mode has no configured point ceiling — frame time and the
      // memory budget govern; a fixed point count has nothing to say.
      if (adaptiveBudget !== null) return;
      pointBudget = points;
      runSelection();
    },

    recordFrame(durationMs) {
      if (disposed || !active || adaptiveBudget === null) return;
      const now = Date.now();
      adaptiveBudget.recordFrame(durationMs, {
        interacting: isInteracting(now),
        now,
      });
      // Reselect only when the effective budget (value or regime) actually
      // moved — recordFrame fires every frame; adjustments are rare.
      if (currentBudget(now) !== lastSelectionBudget) requestSelection();
    },

    setSource(nextSource) {
      if (disposed) return;
      clearSettleTimer();
      interactionSettling = false;
      dropEverything();
      adaptiveBudget?.reset();
      source = nextSource;
      scheduleFlush();
      if (active) loadPage(ROOT_KEY);
    },

    refresh() {
      if (disposed) return;
      runSelection();
    },

    setRefinementCutoffPx(pixels) {
      if (disposed || !Number.isFinite(pixels) || pixels < 0) return;
      if (pixels === refinementCutoffPx) return;
      refinementCutoffPx = pixels;
      runSelection();
    },

    setActive(nextActive) {
      if (disposed || nextActive === active) return;
      active = nextActive;
      if (active) {
        joinMemoryPool();
        appliedCeiling = 0;
        applyMemoryCeiling();
        if (!pagesLoaded.has(keyToString(ROOT_KEY))) loadPage(ROOT_KEY);
        runSelection();
        return;
      }

      if (selectionTimer !== null) {
        clearTimeout(selectionTimer);
        selectionTimer = null;
      }
      clearSettleTimer();
      poolMember?.release();
      poolMember = null;
      appliedCeiling = 0;
      queue = [];
      cancelledFetches += inFlight.size;
      for (const abort of inFlight.values()) abort.abort();
      inFlight.clear();

      if (target.size > 0) targetRevision += 1;
      target = new Set();
      selectionGeneration += 1;
      selectionStats = {
        ...selectionStats,
        generation: selectionGeneration,
        targetRevision,
        targetTiles: 0,
        targetPoints: 0,
        consideredNodes: 0,
        availableNodes: 0,
        selectedNodes: 0,
        hierarchyUnavailableNodes: 0,
        hierarchyPageBlockedNodes: 0,
        frustumCulledNodes: 0,
        frustumCulledPoints: 0,
        leafNodes: 0,
        sseStoppedNodes: 0,
        budgetSkippedNodes: 0,
        budgetSkippedPoints: 0,
        projectedImportance: 0,
        frontierSse: { count: 0, p50: null, p95: null, max: null },
      };

      // Suppress any not-yet-delivered additions, then remove every actor the
      // consumer may already own. Adapter removal is idempotent for the former.
      pendingAdded = [];
      for (const [keyString, tile] of resident) {
        cache.set(keyString, tile, tileBytes(tile));
        pendingRemoved.push(keyFromString(keyString));
      }
      resident.clear();
      residentPoints = 0;
      residentBytes = 0;
      scheduleFlush();
    },

    stats() {
      const now = Date.now();
      const cachedBytes = cache.totalBytes();
      return {
        active,
        residentTiles: resident.size,
        residentPoints,
        residentBytes,
        activeTiles: resident.size,
        activePoints: residentPoints,
        decodedTiles: resident.size + cache.count(),
        decodedBytes: residentBytes + cachedBytes,
        cachedBytes,
        inFlight: inFlight.size,
        pointBudget: currentBudget(now),
        memoryBudgetBytes: poolMember?.budgetBytes() ?? 0,
        memoryCeilingPoints: memoryCeilingPoints(),
        interacting: isInteracting(now),
        interactionDepth,
        // --- phase0-bench (removable; see app/telesculptor_web/app/bench/README.md) ---
        cachedTiles: cache.count(),
        fetchedTiles,
        fetchedBytes,
        cancelledFetches,
        cacheHits,
        cacheMisses,
        // Computed on demand; resident tiles number in the tens.
        residentLevels: levelHistogram(resident),
        adaptive: adaptiveBudget?.stats() ?? null,
        controllerInstanceId,
        sourceEpoch: epoch,
        queuedTiles: queue.length,
        hierarchyPagesLoading: pagesLoading.size,
        hierarchyPagesLoaded: pagesLoaded.size,
        selectionPending: selectionTimer !== null,
        settlePending: settleTimer !== null,
        refinementCutoffPx,
        selection: selectionStats,
        // --- end phase0-bench ---
      };
    },

    dispose() {
      if (disposed) return;
      if (selectionTimer !== null) {
        clearTimeout(selectionTimer);
        selectionTimer = null;
      }
      clearSettleTimer();
      interactionDepth = 0;
      interactionSettling = false;
      const removed = [...resident.keys()].map(keyFromString);
      dropEverything();
      poolMember?.release();
      poolMember = null;
      disposed = true;
      pendingRemoved = [];
      if (removed.length > 0) {
        onTiles({ added: [], removed });
        scheduleRender();
      }
    },
  };
};
