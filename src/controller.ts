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
   * Maximum points resident at once. Default 2,000,000. With `adaptive`
   * enabled this is the ceiling (the user quality control); the loop keeps the
   * effective budget at or below it.
   */
  pointBudget?: number;
  /**
   * Adapt the visible-point budget to measured render duration (Phase 5). Pass
   * `true` for defaults, an options object to tune, or omit/false for a fixed
   * `pointBudget`. When enabled, feed each render's wall-time to `recordFrame`.
   */
  adaptive?: AdaptiveBudgetOptions | boolean;
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
  readonly cachedBytes: number;
  readonly inFlight: number;
  /** Effective visible-point budget currently driving selection. */
  readonly pointBudget: number;
  /** Whether the controller currently treats itself as interacting. */
  readonly interacting: boolean;
}

export interface LodController {
  /** Update the camera; selection reruns debounced (leading edge immediate). */
  setCamera(view: CameraView): void;
  /**
   * Set the visible-point budget. With `adaptive` enabled this sets the
   * ceiling (the user quality control); otherwise it is the fixed budget.
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

  // Adaptive budget (Phase 5): when enabled, `pointBudget` above is the
  // ceiling and the loop moves the effective budget between a floor and it,
  // tracking a target frame time. When disabled, `pointBudget` is fixed.
  const adaptiveBudget: AdaptiveBudget | null = options.adaptive
    ? createAdaptiveBudget({
        ...(typeof options.adaptive === 'object' ? options.adaptive : {}),
        maxBudget: pointBudget,
        initialBudget:
          (typeof options.adaptive === 'object'
            ? options.adaptive.initialBudget
            : undefined) ?? pointBudget,
        minBudget: Math.min(
          (typeof options.adaptive === 'object'
            ? options.adaptive.minBudget
            : undefined) ?? 200_000,
          pointBudget,
        ),
      })
    : null;

  let lastCameraChange = Number.NEGATIVE_INFINITY;
  // Effective budget applied by the most recent selection; lets `recordFrame`
  // reselect only when the adaptive budget (or interaction regime) has moved.
  let lastSelectionBudget = pointBudget;

  const isInteracting = (now: number): boolean =>
    now - lastCameraChange < interactionSettleMs;

  const currentBudget = (now: number): number =>
    adaptiveBudget === null
      ? pointBudget
      : adaptiveBudget.budget(isInteracting(now));

  // Bumped on setSource/dispose; every async continuation checks it.
  let epoch = 0;

  const hierarchy = new Map<string, HierarchyEntry>();
  const pagesLoaded = new Set<string>();
  const pagesLoading = new Set<string>();

  /** Tiles currently delivered to the consumer. */
  const resident = new Map<string, TileData>();
  let residentPoints = 0;
  /** Deselected tiles kept for cheap reselection. */
  const cache = createLruCache<string, TileData>({ maxBytes: cacheBytes });

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
          inFlight.delete(keyString);
          if (target.has(keyString)) {
            resident.set(keyString, tile);
            residentPoints += tile.pointCount;
            pendingAdded.push({ key, tile });
            scheduleFlush();
          } else {
            cache.set(keyString, tile, tileBytes(tile));
          }
          pump();
        })
        .catch((error) => {
          if (disposed || requestEpoch !== epoch) return;
          inFlight.delete(keyString);
          if (!isAbortError(error)) onError(error);
          pump();
        });
    }
  };

  const runSelection = (): void => {
    if (disposed || view === null) return;
    const budget = currentBudget(Date.now());
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

    // Deselected residents move to the LRU (evicting only unselected tiles).
    for (const [keyString, tile] of [...resident]) {
      if (target.has(keyString)) continue;
      resident.delete(keyString);
      residentPoints -= tile.pointCount;
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
    pendingAdded = [];
  };

  // Bootstrap: hierarchy root page loads eagerly; selection waits for camera.
  loadPage(ROOT_KEY);

  return {
    setCamera(nextView) {
      if (disposed) return;
      view = nextView;
      lastCameraChange = Date.now();
      requestSelection();
    },

    setPointBudget(points) {
      if (disposed) return;
      pointBudget = points;
      if (adaptiveBudget !== null) {
        // The config budget is the ceiling; the loop owns the effective value.
        adaptiveBudget.setMaxBudget(points);
      }
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
        cachedBytes: cache.totalBytes(),
        inFlight: inFlight.size,
        pointBudget: currentBudget(now),
        interacting: isInteracting(now),
      };
    },

    dispose() {
      if (disposed) return;
      if (selectionTimer !== null) {
        clearTimeout(selectionTimer);
        selectionTimer = null;
      }
      const removed = [...resident.keys()].map(keyFromString);
      dropEverything();
      disposed = true;
      pendingRemoved = [];
      if (removed.length > 0) {
        onTiles({ added: [], removed });
        scheduleRender();
      }
    },
  };
};
