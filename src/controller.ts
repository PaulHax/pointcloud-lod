/**
 * LOD controller: turns camera movement into a bounded set of resident tiles.
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
} from './camera';
import {
  createAdaptiveBudget,
  type AdaptiveBudget,
  type AdaptiveBudgetOptions,
} from './adaptiveBudget';
import { selectNodes } from './budget';
import { createLruCache } from './lru';
import {
  createMemoryPool,
  type MemoryPool,
  type MemoryPoolMember,
} from './memoryPool';
import {
  ROOT_KEY,
  childKeys,
  keyFromString,
  keyToString,
  nodeCube,
  pointSpacing,
  type VoxelKey,
} from './octree';
import type { TileData, TileSource } from './tileSource';

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
  readonly residentTiles: number;
  readonly residentPoints: number;
  /** Bytes of resident tile data (the GPU-memory proxy). */
  readonly residentBytes: number;
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
}

export interface LodController {
  /** Update the camera; selection reruns debounced (leading edge immediate). */
  setCamera(view: CameraView): void;
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
  error instanceof Error && error.name === 'AbortError';

const tileBytes = (tile: TileData): number =>
  tile.positions.byteLength + (tile.rgb?.byteLength ?? 0) + 64;

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
    refinementCutoffPx = 1,
    onError = (error) => console.warn('pointcloud-lod:', error),
  } = options;

  let source = options.source;
  let pointBudget = options.pointBudget ?? 2_000_000;
  let view: CameraView | null = null;
  let disposed = false;

  // Adaptive budget (Phase 5): when enabled, the loop moves the effective
  // budget between a floor and the memory-derived ceiling, tracking a target
  // frame time. When disabled, `pointBudget` is fixed (memory still caps it).
  const adaptiveBudget: AdaptiveBudget | null = options.adaptive
    ? createAdaptiveBudget(
        typeof options.adaptive === 'object' ? options.adaptive : {},
      )
    : null;

  let lastCameraChange = Number.NEGATIVE_INFINITY;
  // Effective budget applied by the most recent selection; lets `recordFrame`
  // reselect only when the adaptive budget (or interaction regime) has moved.
  let lastSelectionBudget = pointBudget;
  // True when the last selection kept a deselected-but-visible resident tile
  // alive to avoid mid-gesture flashing; the settle reselect consolidates.
  let retainedResident = false;

  const isInteracting = (now: number): boolean =>
    now - lastCameraChange < interactionSettleMs;

  const currentBudget = (now: number): number =>
    adaptiveBudget === null
      ? Math.min(pointBudget, memoryCeilingPoints())
      : adaptiveBudget.budget(isInteracting(now));

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
    typeof options.memory === 'object'
      ? options.memory
      : createMemoryPool(
          typeof options.memory === 'number'
            ? { totalBytes: options.memory }
            : {},
        );
  const poolMember: MemoryPoolMember = memoryPool.register(() => {
    if (disposed) return;
    applyMemoryCeiling();
    requestSelection();
  });

  const FALLBACK_BYTES_PER_POINT = 16;
  const MEASURE_MIN_POINTS = 100_000;

  const bytesPerPoint = (): number =>
    residentPoints >= MEASURE_MIN_POINTS
      ? residentBytes / residentPoints
      : FALLBACK_BYTES_PER_POINT;

  const memoryCeilingPoints = (): number =>
    Math.max(1, Math.floor(poolMember.budgetBytes() / bytesPerPoint()));

  // Last ceiling handed to the adaptive loop. The bytes-per-point estimate
  // drifts as tiles arrive, and setMaxBudget discards a track's samples when
  // it clamps a budget — a small dead-band keeps estimate jitter from
  // re-clamping every selection.
  let appliedCeiling = 0;
  const applyMemoryCeiling = (): void => {
    if (adaptiveBudget === null) return;
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
  const inFlight = new Map<string, AbortController>();
  let queue: string[] = [];

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
    if (disposed || view === null) return;
    const now = Date.now();
    // Re-derive the memory ceiling first: resident bytes (and with them the
    // bytes-per-point estimate) changed since the last selection.
    applyMemoryCeiling();
    const budget = currentBudget(now);
    const interacting = isInteracting(now);
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
    const selection = selectNodes({
      root: ROOT_KEY,
      pointBudget: budget,
      priority: sse,
      getNode: (key) => {
        const keyString = keyToString(key);
        const entry = hierarchy.get(keyString);
        if (entry === undefined) return undefined;
        if (entry.pageRef && !pagesLoaded.has(keyString)) {
          neededPages.push(key);
          return undefined;
        }
        if (!cubeIntersectsFrustum(planes, nodeCube(meta.cube, key))) {
          return undefined;
        }
        const children =
          sse(key) < refinementCutoffPx ? [] : childrenOf(key, entry);
        return { pointCount: entry.pointCount, children };
      },
    });

    target = selection.selected;
    for (const key of neededPages) loadPage(key);

    // Deselected residents move to the LRU (evicting only unselected tiles) —
    // except mid-interaction: dropping a still-visible tile (a budget shrink
    // or SSE change during a gesture) only to re-add it moments later reads as
    // per-tile flashing. Retain in-frustum residents while interacting; the
    // settle reselect consolidates once the camera stops. Off-frustum
    // residents are invisible and drop immediately in either regime.
    retainedResident = false;
    for (const [keyString, tile] of [...resident]) {
      if (target.has(keyString)) continue;
      if (
        interacting &&
        cubeIntersectsFrustum(
          planes,
          nodeCube(meta.cube, keyFromString(keyString)),
        )
      ) {
        retainedResident = true;
        continue;
      }
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
      abort.abort();
    }

    // Reuse cached tiles immediately; queue the rest, coarse levels first.
    const toFetch: string[] = [];
    for (const keyString of target) {
      if (resident.has(keyString) || inFlight.has(keyString)) continue;
      const cached = cache.get(keyString);
      if (cached !== undefined) {
        cache.delete(keyString);
        resident.set(keyString, cached);
        residentPoints += cached.pointCount;
        residentBytes += tileBytes(cached);
        pendingAdded.push({ key: keyFromString(keyString), tile: cached });
      } else {
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
  // Fires once the camera has been still for interactionSettleMs, so that
  // (a) the interaction→stationary budget flip is applied even when the host
  // renders on demand and stops calling recordFrame after the last
  // interaction frame, and (b) residents retained mid-gesture to avoid
  // flashing are consolidated against the settled selection.
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
      // The regime has flipped to stationary; reselect if that changed the
      // effective budget from the last selection, or if the last selection
      // retained tiles it would otherwise have dropped (mid-gesture
      // anti-flashing) — the settled pass applies the real selection.
      if (currentBudget(Date.now()) !== lastSelectionBudget || retainedResident) {
        runSelection();
      }
    }, interactionSettleMs);
  };

  const requestSelection = (): void => {
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
    for (const abort of inFlight.values()) abort.abort();
    inFlight.clear();
    queue = [];
    hierarchy.clear();
    pagesLoaded.clear();
    pagesLoading.clear();
    cache.clear();
    target = new Set();
    for (const keyString of resident.keys()) {
      pendingRemoved.push(keyFromString(keyString));
    }
    resident.clear();
    residentPoints = 0;
    residentBytes = 0;
    pendingAdded = [];
    retainedResident = false;
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
  loadPage(ROOT_KEY);

  return {
    setCamera(nextView) {
      if (disposed) return;
      if (view !== null && sameView(view, nextView)) return;
      view = nextView;
      lastCameraChange = Date.now();
      // Re-arm on every camera change so the timer only fires once the camera
      // has been still for the full settle window. Armed even without an
      // adaptive budget: retained-resident consolidation needs it too.
      armSettleTimer();
      requestSelection();
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
      if (disposed || adaptiveBudget === null) return;
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
      dropEverything();
      adaptiveBudget?.reset();
      source = nextSource;
      scheduleFlush();
      loadPage(ROOT_KEY);
    },

    refresh() {
      if (disposed) return;
      runSelection();
    },

    stats() {
      const now = Date.now();
      return {
        residentTiles: resident.size,
        residentPoints,
        residentBytes,
        cachedBytes: cache.totalBytes(),
        inFlight: inFlight.size,
        pointBudget: currentBudget(now),
        memoryBudgetBytes: poolMember.budgetBytes(),
        memoryCeilingPoints: memoryCeilingPoints(),
        interacting: isInteracting(now),
      };
    },

    dispose() {
      if (disposed) return;
      if (selectionTimer !== null) {
        clearTimeout(selectionTimer);
        selectionTimer = null;
      }
      clearSettleTimer();
      const removed = [...resident.keys()].map(keyFromString);
      dropEverything();
      poolMember.release();
      disposed = true;
      pendingRemoved = [];
      if (removed.length > 0) {
        onTiles({ added: [], removed });
        scheduleRender();
      }
    },
  };
};
