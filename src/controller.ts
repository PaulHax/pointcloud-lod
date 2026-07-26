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
  boundsIntersectsFrustum,
  type CameraView,
} from "./camera";
import { selectNodes } from "./budget";
import { percentile } from "./adaptiveBudget";
import { createLruCache } from "./lru";
import { finiteAtLeast, wholeAtLeast } from "./numeric";
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
  type Bounds,
  type VoxelKey,
} from "./octree";
import type { TileData, TileSource } from "./tileSource";

export interface TileBatch {
  readonly added: readonly { key: VoxelKey; tile: TileData }[];
  readonly removed: readonly VoxelKey[];
}

export interface FixedPointPresentation {
  readonly mode: "fixed";
  readonly diameterCssPx: number;
}

export interface AutoPointPresentation {
  readonly mode: "auto";
  /** Multiplier applied after deriving the density-aware Auto diameter. */
  readonly userScale: number;
  /** Bounds for the unscaled density-aware diameter. */
  readonly minDiameterCssPx?: number;
  readonly maxDiameterCssPx?: number;
}

export type PointPresentation = FixedPointPresentation | AutoPointPresentation;

/**
 * Every numeric option is a programmer error when it is not finite or falls
 * outside its documented range: construction throws naming the option and the
 * value. The live setters take the opposite side of the same policy — see
 * `LodController`.
 */
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
   * Visible-point budget driving selection. Default 2,000,000. The
   * memory-derived point cap applies on top.
   */
  pointBudget?: number;
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
   * How long the camera must be still before the stationary refinement pass
   * runs, ms. Default 750.
   */
  interactionSettleMs?: number;
  /** Parallel tile fetches. Default 6. */
  fetchConcurrency?: number;
  /**
   * Parallel hierarchy-page fetches. Default 4.
   *
   * Deliberately a second, separate ceiling rather than a share of
   * `fetchConcurrency`: a page unblocks selection for a whole subtree, while a
   * tile only adds detail to something already drawn. Behind one shared queue
   * a page would wait for tiles that cannot be chosen correctly until it
   * lands, so refinement stalls exactly when the camera is moving fastest.
   * The two ceilings bound total I/O at `fetchConcurrency +
   * hierarchyConcurrency` operations.
   */
  hierarchyConcurrency?: number;
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
  /** Point presentation. Default `{mode: "fixed", diameterCssPx: 2}`. */
  presentation?: PointPresentation;
  /** Receives the one CSS-pixel diameter applied to every active tile. */
  onPointDiameterCssPx?: (diameterCssPx: number) => void;
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
  /** Decoded payloads across submitted tiles and the dormant CPU cache. */
  readonly decodedTiles: number;
  /** Decoded bytes across submitted tiles and the dormant CPU cache. */
  readonly decodedBytes: number;
  readonly cachedBytes: number;
  /** Tile requests whose result the controller still wants. */
  readonly inFlight: number;
  /** Selected tiles waiting for a fetch slot. */
  readonly queuedTiles: number;
  /**
   * Tile fetch/decode operations physically running, cancelled ones included:
   * what `fetchConcurrency` actually bounds. Cancellation is advisory, so this
   * can exceed `inFlight` until the abandoned promises settle.
   */
  readonly physicalTileOperations: number;
  /** Hierarchy pages requested whose result the controller still wants. */
  readonly hierarchyInFlight: number;
  /** Needed hierarchy pages waiting for a slot. */
  readonly queuedPages: number;
  /** Hierarchy page operations physically running, cancelled ones included. */
  readonly physicalHierarchyOperations: number;
  /** The ceiling `physicalTileOperations` is held under. */
  readonly fetchConcurrency: number;
  /** The ceiling `physicalHierarchyOperations` is held under. */
  readonly hierarchyConcurrency: number;
  /** Effective visible-point budget currently driving selection. */
  readonly pointBudget: number;
  /** This controller's byte share of its memory pool. */
  readonly memoryBudgetBytes: number;
  /** Memory-derived point ceiling the budget can never exceed. */
  readonly memoryCeilingPoints: number;
  /** Explicit interaction nesting depth reported by the host. */
  readonly interactionDepth: number;
  /** Current explicit presentation and emitted uniform diameter. */
  readonly presentation: {
    readonly config: PointPresentation;
    readonly diameterCssPx: number;
  };
  /** LRU entry count (deselected tiles kept for cheap reselection). */
  readonly cachedTiles: number;
  /**
   * The LRU's byte ceiling — the bound `cachedBytes` is actually held under.
   * Distinct from `memoryBudgetBytes`, which is this controller's share of the
   * GPU pool and falls to zero when it is deactivated, while the CPU cache
   * deliberately keeps its payloads.
   */
  readonly cacheBytes: number;
  /** Current screen-space refinement cutoff. */
  readonly refinementCutoffPx: number;
  /** Latest selection pass and the reasons traversal stopped. */
  readonly selection: LodSelectionStats;
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
  /** Projected-spacing distribution for the ready terminal coverage frontier. */
  readonly readyTerminalFrontier: {
    readonly count: number;
    readonly leafNodes: number;
    readonly cutoffNodes: number;
    readonly hierarchyBlockedNodes: number;
    readonly tileBlockedNodes: number;
    readonly budgetBlockedNodes: number;
    readonly projectedSpacingCssPx: {
      readonly p25: number | null;
      readonly p50: number | null;
      readonly p75: number | null;
      readonly p95: number | null;
      readonly max: number | null;
    };
  };
}

/**
 * Setters carry live wire input, so they validate rather than throw: a value
 * that is not finite or is out of range is ignored and changes no state.
 * Construction is where an invalid number is fatal.
 */
export interface LodController {
  /**
   * Update the camera; selection reruns debounced (leading edge immediate).
   * A view with any non-finite number is ignored.
   */
  setCamera(view: CameraView): void;
  /** Enter a camera interaction. Nested calls are reference counted. */
  beginInteraction(): void;
  /** Leave a camera interaction; the outermost end starts the settle window. */
  endInteraction(): void;
  /**
   * Set the visible-point budget, truncated to whole points; the memory
   * ceiling still caps it. Anything below one point is ignored.
   */
  setPointBudget(points: number): void;
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
   * Replace the explicit Fixed/Auto point-presentation contract. Undefined
   * restores the default; a contract with unusable or inverted bounds is
   * ignored.
   */
  setPresentation(presentation: PointPresentation | undefined): void;
  /**
   * Enable or disable renderer submission. Disabling immediately removes all
   * submitted tiles, cancels tile requests, and moves decoded payloads into
   * the byte-bounded CPU cache. Re-enabling selects against the latest camera.
   */
  setActive(active: boolean): void;
  stats(): LodControllerStats;
  /**
   * Diagnostics: the tiles held on screen, and the set last handed to the
   * consumer. A pending flush is the only reason they differ, so once the
   * controller is quiet they agree — and `submitted` is then exactly what the
   * renderer adapter must hold, key for key.
   */
  activeKeys(): { readonly resident: string[]; readonly submitted: string[] };
  /** Cancel everything and release all tiles. Idempotent. */
  dispose(): void;
}

interface HierarchyEntry {
  pointCount: number;
  bounds: Bounds;
  spacing: number;
  children: readonly VoxelKey[] | null;
  pageRef: boolean;
}

const isAbortError = (error: unknown): boolean =>
  error instanceof Error && error.name === "AbortError";

const tileBytes = (tile: TileData): number =>
  tile.positions.byteLength + (tile.rgb?.byteLength ?? 0) + 64;

/** The frontier stats report null for "nothing measured" rather than NaN. */
const percentileOrNull = (
  values: readonly number[],
  p: number,
): number | null => (values.length === 0 ? null : percentile(values, p));

/**
 * Numeric policy, applied at every public boundary:
 *
 * - construction options are programmer errors — an invalid one throws with
 *   the offending name and value;
 * - live setters carry wire input — an invalid value is ignored and leaves
 *   the controller exactly as it was.
 *
 * Either way nothing non-finite reaches selection, the memory ceiling, or the
 * statistics: one NaN in `Math.min` silently empties a cloud and every
 * comparison downstream of it answers false forever.
 */
const outOfRange = (value: number, min: number): boolean =>
  !Number.isFinite(value) || value < min;

/** Finite and strictly positive: what every diameter, scale, and share needs. */
const notPositive = (value: number): boolean =>
  !Number.isFinite(value) || value <= 0;

const DEFAULT_PRESENTATION: FixedPointPresentation = {
  mode: "fixed",
  diameterCssPx: 2,
};
const DEFAULT_AUTO_MIN_DIAMETER_CSS_PX = 1.5;
const DEFAULT_AUTO_MAX_DIAMETER_CSS_PX = 4;
const INITIAL_AUTO_DIAMETER_CSS_PX = 2;

type PresentationCheck =
  | { readonly presentation: PointPresentation }
  | { readonly error: string };

/** One validation body for both the throwing and the ignoring boundary. */
const checkPresentation = (
  value: PointPresentation | undefined,
): PresentationCheck => {
  const presentation = value ?? DEFAULT_PRESENTATION;
  if (presentation.mode === "fixed") {
    if (notPositive(presentation.diameterCssPx)) {
      return {
        error: `Fixed diameterCssPx must be finite and > 0, got ${presentation.diameterCssPx}`,
      };
    }
    return {
      presentation: { mode: "fixed", diameterCssPx: presentation.diameterCssPx },
    };
  }
  const min = presentation.minDiameterCssPx ?? DEFAULT_AUTO_MIN_DIAMETER_CSS_PX;
  const max = presentation.maxDiameterCssPx ?? DEFAULT_AUTO_MAX_DIAMETER_CSS_PX;
  if (
    notPositive(presentation.userScale) ||
    notPositive(min) ||
    outOfRange(max, min)
  ) {
    return {
      error:
        "Auto userScale/minDiameterCssPx/maxDiameterCssPx must be finite, positive, and ordered",
    };
  }
  return {
    presentation: {
      mode: "auto",
      userScale: presentation.userScale,
      minDiameterCssPx: min,
      maxDiameterCssPx: max,
    },
  };
};

const normalizePresentation = (
  value: PointPresentation | undefined,
): PointPresentation => {
  const checked = checkPresentation(value);
  if ("error" in checked) throw new Error(checked.error);
  return checked.presentation;
};

/**
 * The one scalar that sizes each projection. Hosts feed views from untyped
 * JS, so an unrecognized discriminant reads as absent instead of silently
 * being treated as perspective.
 */
const projectionScalar = (view: CameraView): number | undefined =>
  view.projection === "perspective"
    ? view.fovY
    : view.projection === "orthographic"
      ? view.parallelScale
      : undefined;

/**
 * A camera whose numbers are not all finite would poison the frustum planes,
 * every screen-space error, and the selection comparisons that read them.
 */
const isFiniteView = (view: CameraView): boolean => {
  const scalar = projectionScalar(view);
  // A field of view at or past a half-turn has no usable tangent, and a
  // non-positive parallel scale inverts the projected spacing.
  if (
    scalar === undefined ||
    notPositive(scalar) ||
    (view.projection === "perspective" && view.fovY >= Math.PI) ||
    notPositive(view.viewportHeightCssPx)
  ) {
    return false;
  }
  for (const coordinate of view.position) {
    if (!Number.isFinite(coordinate)) return false;
  }
  for (let index = 0; index < view.viewProj.length; index += 1) {
    if (!Number.isFinite(view.viewProj[index])) return false;
  }
  return true;
};

const samePresentation = (
  left: PointPresentation,
  right: PointPresentation,
): boolean =>
  left.mode === right.mode &&
  (left.mode === "fixed"
    ? left.diameterCssPx === (right as FixedPointPresentation).diameterCssPx
    : left.userScale === (right as AutoPointPresentation).userScale &&
      left.minDiameterCssPx ===
        (right as AutoPointPresentation).minDiameterCssPx &&
      left.maxDiameterCssPx ===
        (right as AutoPointPresentation).maxDiameterCssPx);

const emptyReadyTerminalFrontier =
  (): LodSelectionStats["readyTerminalFrontier"] => ({
    count: 0,
    leafNodes: 0,
    cutoffNodes: 0,
    hierarchyBlockedNodes: 0,
    tileBlockedNodes: 0,
    budgetBlockedNodes: 0,
    projectedSpacingCssPx: {
      p25: null,
      p50: null,
      p75: null,
      p95: null,
      max: null,
    },
  });

export const createLodController = (
  options: LodControllerOptions,
): LodController => {
  const {
    onTiles,
    scheduleRender,
    onPointDiameterCssPx = () => {},
    onError = (error) => console.warn("pointcloud-lod:", error),
  } = options;

  const fetchConcurrency = wholeAtLeast(
    "fetchConcurrency",
    options.fetchConcurrency ?? 6,
    1,
  );
  const hierarchyConcurrency = wholeAtLeast(
    "hierarchyConcurrency",
    options.hierarchyConcurrency ?? 4,
    1,
  );
  const cacheBytes = wholeAtLeast(
    "cacheBytes",
    options.cacheBytes ?? 256 * 1024 * 1024,
    1,
  );
  const selectionDelayMs = finiteAtLeast(
    "selectionDelayMs",
    options.selectionDelayMs ?? 150,
    0,
  );
  const interactionSettleMs = finiteAtLeast(
    "interactionSettleMs",
    options.interactionSettleMs ?? 750,
    0,
  );

  let source = options.source;
  let pointBudget = wholeAtLeast(
    "pointBudget",
    options.pointBudget ?? 2_000_000,
    1,
  );
  let refinementCutoffPx = finiteAtLeast(
    "refinementCutoffPx",
    options.refinementCutoffPx ?? 1,
    0,
  );
  let presentation = normalizePresentation(options.presentation);
  let diameterCssPx =
    presentation.mode === "fixed"
      ? presentation.diameterCssPx
      : INITIAL_AUTO_DIAMETER_CSS_PX;
  let active = options.active ?? true;
  let view: CameraView | null = null;
  let disposed = false;
  onPointDiameterCssPx(diameterCssPx);

  const emitDiameter = (next: number): void => {
    if (next === diameterCssPx) return;
    diameterCssPx = next;
    onPointDiameterCssPx(next);
  };



  let explicitInteractionSeen = false;
  let interactionDepth = 0;
  // Effective budget applied by the most recent selection; lets the settle
  // timer reselect only when the effective budget has actually moved.
  let lastSelectionBudget = pointBudget;
  const currentBudget = (): number =>
    active ? Math.min(pointBudget, memoryCeilingPoints()) : 0;

  // Bumped on setSource/dispose; every async continuation checks it.
  let epoch = 0;

  const hierarchy = new Map<string, HierarchyEntry>();

  // Screen-space error depends only on a node and the current view, so values
  // stay valid until the view moves. Selection and the ready-frontier walk
  // both need them, and the frontier walk reruns on every tile arrival.
  // A node whose hierarchy entry has not landed yet is deliberately NOT
  // cached: it scores 0 now, and its page may arrive before the view moves.
  const sseByKey = new Map<string, number>();
  const sseFor = (keyString: string): number => {
    const cached = sseByKey.get(keyString);
    if (cached !== undefined) return cached;
    const entry = hierarchy.get(keyString);
    if (entry === undefined || view === null) return 0;
    const value = nodeScreenSpaceError(entry.bounds, entry.spacing, view);
    sseByKey.set(keyString, value);
    return value;
  };
  /**
   * Request order for both queues: coarse levels first, then the largest
   * screen-space error. That is the selected frontier working outwards, so a
   * hierarchy page is fetched in the order the pages it unblocks would be.
   */
  const byFrontierPriority = (a: string, b: string): number => {
    const levelA = keyFromString(a).level;
    const levelB = keyFromString(b).level;
    if (levelA !== levelB) return levelA - levelB;
    return sseFor(b) - sseFor(a);
  };

  const pagesLoaded = new Set<string>();
  /** Hierarchy pages requested whose result is still wanted. */
  const pagesInFlight = new Map<string, AbortController>();
  let pageQueue: string[] = [];

  // Selection re-requests whatever it still needs, so an endpoint that always
  // fails would be re-issued on every pass forever. Non-abort failures are
  // counted per key and the key rests once it runs out of attempts. The rest
  // is a backoff, not an eviction: a transient outage must not blank a tile
  // for the life of the controller, so the allowance is restored once the key
  // has been quiet for RETRY_BACKOFF_MS and the key becomes fetchable again.
  // A new source (dropEverything) is a fresh start. Aborts never count —
  // deselection, setSource, and dispose cancel normally and stay retryable.
  const MAX_ATTEMPTS = 3;
  const RETRY_BACKOFF_MS = 30_000;
  interface FailureRecord {
    count: number;
    lastMs: number;
  }
  const pageFailures = new Map<string, FailureRecord>();
  const tileFailures = new Map<string, FailureRecord>();

  const recordFailure = (
    failures: Map<string, FailureRecord>,
    keyString: string,
  ): void => {
    failures.set(keyString, {
      count: (failures.get(keyString)?.count ?? 0) + 1,
      lastMs: Date.now(),
    });
  };
  /** Clears the record once the backoff elapses, restoring a full allowance. */
  const resting = (
    failures: Map<string, FailureRecord>,
    keyString: string,
  ): boolean => {
    const record = failures.get(keyString);
    if (record === undefined || record.count < MAX_ATTEMPTS) return false;
    if (Date.now() - record.lastMs < RETRY_BACKOFF_MS) return true;
    failures.delete(keyString);
    return false;
  };

  /** Tiles currently delivered to the consumer. */
  const resident = new Map<string, TileData>();
  let residentPoints = 0;
  let residentBytes = 0;
  /** Deselected tiles kept for cheap reselection. */
  const cache = createLruCache<string, TileData>({ maxBytes: cacheBytes });

  // Memory governor: the byte share converts to a point ceiling via the
  // measured bytes-per-point of resident tiles (falling back to an estimate
  // until enough points are resident to measure). Whatever budget the host
  // asks for, this ceiling is what keeps selection inside GPU memory.
  const memoryPool: MemoryPool =
    typeof options.memory === "object"
      ? options.memory
      : createMemoryPool(
          typeof options.memory === "number"
            ? { totalBytes: wholeAtLeast("memory", options.memory, 1) }
            : {},
        );
  let poolMember: MemoryPoolMember | null = null;
  const joinMemoryPool = (): void => {
    if (poolMember !== null) return;
    poolMember = memoryPool.register(() => {
      if (disposed || !active) return;
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

  /**
   * This controller's byte share. A pool handed in by the host is not ours to
   * trust, and a non-finite share would make every budget comparison and both
   * memory statistics NaN, so nonsense reads as no memory at all.
   */
  const memoryBudgetBytes = (): number => {
    const bytes = poolMember?.budgetBytes() ?? 0;
    return notPositive(bytes) ? 0 : bytes;
  };

  const memoryCeilingPoints = (): number => {
    const bytes = memoryBudgetBytes();
    return bytes === 0 ? 0 : Math.max(1, Math.floor(bytes / bytesPerPoint()));
  };

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
    readyTerminalFrontier: emptyReadyTerminalFrontier(),
  };
  let budgetSkipped = new Set<string>();
  /**
   * The one physical read a key may have running. It is created when the read
   * starts and removed only when the promise settles, cancelled or not: while
   * the entry is here nobody may start a rival read of the same key, and a
   * reselect adopts this operation by flipping `wanted` back on instead of
   * paying for the same bytes twice.
   */
  const tileReads = new Map<
    string,
    { readonly abort: AbortController; wanted: boolean }
  >();
  let queue: string[] = [];

  // Cancellation is advisory wherever it matters: `abort()` marks a result
  // unwanted, but the COPC getter takes no signal, so the range read and the
  // decode keep burning I/O and CPU until the promise settles. Concurrency is
  // therefore counted on the physical operation — otherwise a
  // look-away/look-back storm starts a fresh read per gesture while every
  // abandoned one is still running, and the ceilings bound nothing at all.
  // These drop only when a promise settles, epoch changes included: ignoring a
  // result is not the same as stopping the work behind it.
  let physicalTileOperations = 0;
  let physicalHierarchyOperations = 0;

  // Decoded single ownership: a key's payload lives in exactly one place —
  // a live request, the CPU cache, or renderer residency. A second copy would
  // double-count decoded bytes and spend cache capacity on a tile the
  // renderer already holds. Every transition goes through these three.
  const takeResident = (keyString: string, tile: TileData): void => {
    cache.delete(keyString);
    resident.set(keyString, tile);
    residentPoints += tile.pointCount;
    residentBytes += tileBytes(tile);
  };

  /**
   * Leave renderer/GPU residency. The decoded payload stays in the
   * byte-bounded CPU cache for cheap reactivation; no dormant actor is kept.
   */
  const releaseResident = (keyString: string): void => {
    const tile = resident.get(keyString);
    if (tile === undefined) return;
    resident.delete(keyString);
    residentPoints -= tile.pointCount;
    residentBytes -= tileBytes(tile);
    cache.set(keyString, tile, tileBytes(tile));
  };

  /** Park a decoded payload only when nothing else owns the key. */
  const cacheDecoded = (keyString: string, tile: TileData): void => {
    if (resident.has(keyString) || tileReads.has(keyString)) return;
    cache.set(keyString, tile, tileBytes(tile));
  };

  /** Reads whose result the controller still wants. */
  const wantedTileReads = (): number => {
    let count = 0;
    for (const read of tileReads.values()) if (read.wanted) count += 1;
    return count;
  };

  /** Promote a cached payload into residency. */
  const promoteCached = (keyString: string): boolean => {
    const cached = cache.get(keyString);
    if (cached === undefined) return false;
    takeResident(keyString, cached);
    return true;
  };

  // The renderer's state as of the last batch it was handed. Every batch is
  // the delta from here to current residency, so an add and a remove of the
  // same key inside one microtask window cancel out instead of arriving as a
  // contradictory batch the consumer has to guess the order of.
  let submitted = new Map<string, TileData>();
  let flushScheduled = false;

  const submittedDelta = (): TileBatch => {
    const added: { key: VoxelKey; tile: TileData }[] = [];
    const removed: VoxelKey[] = [];
    for (const [keyString, tile] of resident) {
      // A key whose payload was replaced is an addition, never a remove/add
      // pair: added and removed stay disjoint within one batch.
      if (submitted.get(keyString) === tile) continue;
      added.push({ key: keyFromString(keyString), tile });
    }
    for (const keyString of submitted.keys()) {
      if (!resident.has(keyString)) removed.push(keyFromString(keyString));
    }
    return { added, removed };
  };

  const scheduleFlush = (): void => {
    if (flushScheduled) return;
    flushScheduled = true;
    queueMicrotask(() => {
      flushScheduled = false;
      if (disposed) return;
      const batch = submittedDelta();
      if (batch.added.length === 0 && batch.removed.length === 0) return;
      submitted = new Map(resident);
      onTiles(batch);
      scheduleRender();
    });
  };

  /** Worth asking for: not held, not already asked for, not resting. */
  const pageWanted = (keyString: string): boolean =>
    !pagesLoaded.has(keyString) &&
    !pagesInFlight.has(keyString) &&
    !resting(pageFailures, keyString);

  /**
   * Replace the page queue with what selection needs now, highest priority
   * first. Like the tile queue this is rebuilt rather than appended to: a page
   * the camera has moved past must not keep its slot reservation.
   */
  const queuePages = (keyStrings: readonly string[]): void => {
    pageQueue = [...new Set(keyStrings)]
      .filter(pageWanted)
      .sort(byFrontierPriority);
    pumpPages();
  };

  const pumpPages = (): void => {
    while (
      !disposed &&
      physicalHierarchyOperations < hierarchyConcurrency &&
      pageQueue.length > 0
    ) {
      const keyString = pageQueue.shift()!;
      if (!pageWanted(keyString)) continue;
      const abort = new AbortController();
      pagesInFlight.set(keyString, abort);
      const requestEpoch = epoch;
      physicalHierarchyOperations += 1;
      source.nodes(keyFromString(keyString), { signal: abort.signal }).then(
        (infos) => {
          physicalHierarchyOperations -= 1;
          if (pagesInFlight.get(keyString) === abort) {
            pagesInFlight.delete(keyString);
          }
          if (disposed || requestEpoch !== epoch) {
            pumpPages();
            return;
          }
          pagesLoaded.add(keyString);
          pageFailures.delete(keyString);
          for (const info of infos) {
            hierarchy.set(keyToString(info.key), {
              pointCount: info.pointCount,
              bounds: info.bounds,
              spacing: info.spacing,
              children: info.children ?? null,
              pageRef: info.pageRef === true,
            });
          }
          // A page reshapes the subtree below it, so the next queue is the one
          // selection derives from it; pump again for the inactive/no-camera
          // case where selection cannot run.
          runSelection();
          pumpPages();
        },
        (error) => {
          physicalHierarchyOperations -= 1;
          if (pagesInFlight.get(keyString) === abort) {
            pagesInFlight.delete(keyString);
          }
          if (disposed || requestEpoch !== epoch) {
            pumpPages();
            return;
          }
          if (!isAbortError(error)) {
            recordFailure(pageFailures, keyString);
            onError(error);
          }
          pumpPages();
        },
      );
    }
  };

  const childrenOf = (
    key: VoxelKey,
    entry: HierarchyEntry,
  ): readonly VoxelKey[] =>
    entry.children ??
    childKeys(key).filter((child) => hierarchy.has(keyToString(child)));

  const isEntryReady = (keyString: string, entry: HierarchyEntry): boolean =>
    entry.pointCount === 0 || resident.has(keyString);

  const updateReadyTerminalFrontier = (): void => {
    const currentView = view;
    if (currentView === null || !active || !target.has(keyToString(ROOT_KEY))) {
      selectionStats = {
        ...selectionStats,
        readyTerminalFrontier: emptyReadyTerminalFrontier(),
      };
      return;
    }

    const planes = frustumPlanes(currentView.viewProj);
    const values: number[] = [];
    const terminalKeys = new Set<string>();
    let leafNodes = 0;
    let cutoffNodes = 0;
    let hierarchyBlockedNodes = 0;
    let tileBlockedNodes = 0;
    let budgetBlockedNodes = 0;

    const addTerminal = (
      keyString: string,
      entry: HierarchyEntry,
      reasons: {
        leaf?: boolean;
        cutoff?: boolean;
        hierarchy?: boolean;
        tile?: boolean;
        budget?: boolean;
      },
    ): void => {
      // A structural node has no samples with which to cover a blocked region.
      if (entry.pointCount === 0 || !resident.has(keyString)) return;
      if (!terminalKeys.has(keyString)) {
        terminalKeys.add(keyString);
        values.push(sseFor(keyString));
      }
      if (reasons.leaf) leafNodes += 1;
      if (reasons.cutoff) cutoffNodes += 1;
      if (reasons.hierarchy) hierarchyBlockedNodes += 1;
      if (reasons.tile) tileBlockedNodes += 1;
      if (reasons.budget) budgetBlockedNodes += 1;
    };

    const walk = (key: VoxelKey): void => {
      const keyString = keyToString(key);
      if (!target.has(keyString)) return;
      const entry = hierarchy.get(keyString);
      if (entry === undefined || !isEntryReady(keyString, entry)) return;

      const children = childrenOf(key, entry);
      if (children.length === 0) {
        addTerminal(keyString, entry, { leaf: true });
        return;
      }
      if (sseFor(keyString) < refinementCutoffPx) {
        addTerminal(keyString, entry, { cutoff: true });
        return;
      }

      let hierarchyBlocked = false;
      let tileBlocked = false;
      let budgetBlocked = false;
      const readyChildren: VoxelKey[] = [];
      for (const child of children) {
        const childString = keyToString(child);
        const childEntry = hierarchy.get(childString);
        if (childEntry === undefined) {
          hierarchyBlocked = true;
          continue;
        }
        if (!boundsIntersectsFrustum(planes, childEntry.bounds)) continue;
        // Matches selection: an invisible page reference is not requested, so
        // it is not blocking anything either.
        if (childEntry.pageRef && !pagesLoaded.has(childString)) {
          hierarchyBlocked = true;
          continue;
        }
        if (!target.has(childString)) {
          // A visible, available child of a selected parent can only be absent
          // because the breadth-first point budget rejected it.
          budgetBlocked = budgetSkipped.has(childString) || budgetBlocked;
          continue;
        }
        if (!isEntryReady(childString, childEntry)) {
          tileBlocked = true;
          continue;
        }
        readyChildren.push(child);
      }

      if (hierarchyBlocked || tileBlocked || budgetBlocked) {
        addTerminal(keyString, entry, {
          hierarchy: hierarchyBlocked,
          tile: tileBlocked,
          budget: budgetBlocked,
        });
      }
      for (const child of readyChildren) walk(child);
    };

    walk(ROOT_KEY);
    const frontier: LodSelectionStats["readyTerminalFrontier"] = {
      count: terminalKeys.size,
      leafNodes,
      cutoffNodes,
      hierarchyBlockedNodes,
      tileBlockedNodes,
      budgetBlockedNodes,
      projectedSpacingCssPx: {
        p25: percentileOrNull(values, 0.25),
        p50: percentileOrNull(values, 0.5),
        p75: percentileOrNull(values, 0.75),
        p95: percentileOrNull(values, 0.95),
        max: values.length > 0 ? Math.max(...values) : null,
      },
    };
    selectionStats = { ...selectionStats, readyTerminalFrontier: frontier };

    if (presentation.mode === "auto") {
      const p75 = frontier.projectedSpacingCssPx.p75;
      if (p75 !== null) {
        const min =
          presentation.minDiameterCssPx ?? DEFAULT_AUTO_MIN_DIAMETER_CSS_PX;
        const max =
          presentation.maxDiameterCssPx ?? DEFAULT_AUTO_MAX_DIAMETER_CSS_PX;
        const target =
          presentation.userScale * Math.min(max, Math.max(min, p75));
        emitDiameter(target);
      }
    }
  };

  const pump = (): void => {
    while (
      !disposed &&
      physicalTileOperations < fetchConcurrency &&
      queue.length > 0
    ) {
      const keyString = queue.shift()!;
      if (
        !target.has(keyString) ||
        resident.has(keyString) ||
        tileReads.has(keyString)
      ) {
        continue;
      }
      // The payload can have landed in the cache while this key waited for a
      // slot — a cancelled read that resolved anyway, or a deselect/reselect.
      // Reading it again would be pure duplicate I/O, and a failure of that
      // read would settle the cloud with a hole over a payload it already has.
      if (promoteCached(keyString)) {
        updateReadyTerminalFrontier();
        scheduleFlush();
        continue;
      }
      const abort = new AbortController();
      tileReads.set(keyString, { abort, wanted: true });
      const requestEpoch = epoch;
      const key = keyFromString(keyString);
      physicalTileOperations += 1;
      source.loadTile(key, { signal: abort.signal }).then(
        (tile) => {
          physicalTileOperations -= 1;
          // An epoch change is the only thing that takes a key's read entry
          // away while the read runs, and it also makes the payload worthless:
          // it came from a source nobody is displaying any more.
          if (disposed || requestEpoch !== epoch) {
            pump();
            return;
          }
          tileReads.delete(keyString);
          tileFailures.delete(keyString);
          // Cancellation is advisory: the COPC getter takes no signal, so a
          // cancelled read still delivers. If the key was reselected while it
          // ran, this payload is exactly what the selection is waiting for.
          if (target.has(keyString) && !resident.has(keyString)) {
            takeResident(keyString, tile);
            updateReadyTerminalFrontier();
            scheduleFlush();
          } else {
            cacheDecoded(keyString, tile);
          }
          pump();
        },
        (error) => {
          physicalTileOperations -= 1;
          if (disposed || requestEpoch !== epoch) {
            pump();
            return;
          }
          tileReads.delete(keyString);
          if (!isAbortError(error)) {
            recordFailure(tileFailures, keyString);
            onError(error);
          } else if (target.has(keyString) && !resident.has(keyString)) {
            // A source that honours the signal really stopped, and the key was
            // reselected while the read was cancelled: it needs a fresh one.
            queue.unshift(keyString);
          }
          updateReadyTerminalFrontier();
          pump();
        },
      );
    }
  };

  const runSelection = (): void => {
    if (disposed || !active || view === null) return;
    const budget = currentBudget();
    lastSelectionBudget = budget;
    const currentView = view;
    const planes = frustumPlanes(currentView.viewProj);
    const sse = (key: VoxelKey): number => sseFor(keyToString(key));

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
        // Culling comes first: a page reference carries the bounds of the
        // subtree it stands for, so an invisible one must not be requested at
        // all. Reading it would spend a hierarchy slot on a region no
        // selection can use, which is precisely the fan-out the page queue
        // exists to bound.
        if (!boundsIntersectsFrustum(planes, entry.bounds)) {
          frustumCulledNodes += 1;
          frustumCulledPoints += entry.pointCount;
          return undefined;
        }
        if (entry.pageRef && !pagesLoaded.has(keyString)) {
          neededPages.push(key);
          hierarchyPageBlockedNodes += 1;
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
    budgetSkipped = new Set(selection.budgetSkipped);
    if (
      previousTarget.size !== target.size ||
      [...target].some((keyString) => !previousTarget.has(keyString))
    ) {
      targetRevision += 1;
    }
    selectionGeneration += 1;
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
      readyTerminalFrontier: emptyReadyTerminalFrontier(),
    };
    // The root page bootstraps the hierarchy, so it can never come back
    // through neededPages: that path needs a hierarchy entry, and only the
    // root page can create one. Without this, a failed bootstrap leaves the
    // controller with nothing to draw and no way to ask again. Level 0 sorts
    // to the front, and queuePages drops anything already held, in flight, or
    // resting, so this costs nothing on the normal path.
    const rootString = keyToString(ROOT_KEY);
    queuePages(
      pagesLoaded.has(rootString)
        ? neededPages.map(keyToString)
        : [rootString, ...neededPages.map(keyToString)],
    );

    // Deselected submitted tiles leave renderer/GPU residency immediately.
    for (const keyString of [...resident.keys()]) {
      if (target.has(keyString)) continue;
      releaseResident(keyString);
    }

    // Cancel fetches that no longer matter. The read keeps its physical slot
    // until it settles: dropping it here would let a reselect race a second
    // read of the same key against work that is still running.
    for (const [keyString, read] of tileReads) {
      if (target.has(keyString)) continue;
      read.wanted = false;
      read.abort.abort();
    }

    // Reuse cached tiles immediately; queue the rest, coarse levels first.
    const toFetch: string[] = [];
    for (const keyString of target) {
      if (resident.has(keyString)) continue;
      const read = tileReads.get(keyString);
      if (read !== undefined) {
        // Adopt the live read instead of starting a rival one; its payload
        // claims residency when it lands.
        read.wanted = true;
        continue;
      }
      const entry = hierarchy.get(keyString);
      // Structural hierarchy nodes participate in selection but carry no tile.
      if (entry?.pointCount === 0) continue;
      if (!promoteCached(keyString) && !resting(tileFailures, keyString)) {
        toFetch.push(keyString);
      }
    }
    queue = toFetch.sort(byFrontierPriority);

    updateReadyTerminalFrontier();
    scheduleFlush();
    pump();
  };

  let lastSelection = Number.NEGATIVE_INFINITY;
  let selectionTimer: ReturnType<typeof setTimeout> | null = null;
  // Fires once the camera has been still for interactionSettleMs, so the
  // stationary refinement pass runs even when the host renders on demand and
  // goes quiet after the last interaction frame.
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
      // Explicit completion is also the stationary refinement trigger. In
      // inference mode a budget change is the only missing work.
      if (
        explicitInteractionSeen ||
        presentation.mode === "auto" ||
        currentBudget() !== lastSelectionBudget
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
    // The epoch bump is what makes every outstanding result irrelevant. The
    // physical operation counts are deliberately left alone: the reads and
    // decodes behind those results are still running, and pretending
    // otherwise is how a source swap under a moving camera doubles real I/O.
    epoch += 1;
    for (const read of tileReads.values()) {
      read.wanted = false;
      read.abort.abort();
    }
    tileReads.clear();
    for (const abort of pagesInFlight.values()) abort.abort();
    pagesInFlight.clear();
    queue = [];
    pageQueue = [];
    hierarchy.clear();
    sseByKey.clear();
    pagesLoaded.clear();
    pageFailures.clear();
    tileFailures.clear();
    cache.clear();
    if (target.size > 0) targetRevision += 1;
    target = new Set();
    budgetSkipped = new Set();
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
      readyTerminalFrontier: emptyReadyTerminalFrontier(),
    };
    resident.clear();
    residentPoints = 0;
    residentBytes = 0;
  };

  // Hosts feed the camera on every render, unconditionally; an unchanged view
  // must be a no-op. Stamping it as a camera change would mark any render as
  // "interacting" — including renders the settle-timer reselect itself
  // triggers — flipping the regime back and oscillating between the two
  // budgets forever.
  const sameView = (a: CameraView, b: CameraView): boolean => {
    if (
      a.projection !== b.projection ||
      projectionScalar(a) !== projectionScalar(b) ||
      a.viewportHeightCssPx !== b.viewportHeightCssPx ||
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

  /** Bootstrap path for the root page, used before any camera exists. */
  const queueRootPage = (): void => queuePages([keyToString(ROOT_KEY)]);

  // Bootstrap: hierarchy root page loads eagerly; selection waits for camera.
  if (active) queueRootPage();

  return {
    setCamera(nextView) {
      if (disposed || !isFiniteView(nextView)) return;
      if (view !== null && sameView(view, nextView)) return;
      view = nextView;
      sseByKey.clear();
      if (!active) return;
      // The settle timer is the fallback for hosts that drive the camera
      // without begin/endInteraction.
      if (!explicitInteractionSeen && presentation.mode === "auto") {
        armSettleTimer();
      }
      requestSelection();
    },

    beginInteraction() {
      if (disposed) return;
      explicitInteractionSeen = true;
      interactionDepth += 1;
      if (interactionDepth !== 1) return;
      clearSettleTimer();
      // Bypass camera debounce before the first expensive moving frame.
      runSelection();
    },

    endInteraction() {
      if (disposed || interactionDepth === 0) return;
      interactionDepth -= 1;
      if (interactionDepth !== 0) return;
      armSettleTimer();
    },

    setPointBudget(points) {
      if (disposed || outOfRange(points, 1)) return;
      pointBudget = Math.floor(points);
      runSelection();
    },

    setSource(nextSource) {
      if (disposed) return;
      clearSettleTimer();
      if (presentation.mode === "auto") {
        emitDiameter(INITIAL_AUTO_DIAMETER_CSS_PX);
      }
      dropEverything();
      source = nextSource;
      scheduleFlush();
      if (active) queueRootPage();
    },

    refresh() {
      if (disposed) return;
      runSelection();
    },

    setRefinementCutoffPx(pixels) {
      if (disposed || outOfRange(pixels, 0)) return;
      if (pixels === refinementCutoffPx) return;
      refinementCutoffPx = pixels;
      runSelection();
    },

    setPresentation(nextPresentation) {
      if (disposed) return;
      const checked = checkPresentation(nextPresentation);
      if ("error" in checked) return;
      const normalized = checked.presentation;
      if (samePresentation(normalized, presentation)) return;
      presentation = normalized;
      if (presentation.mode === "fixed") {
        emitDiameter(presentation.diameterCssPx);
      } else {
        updateReadyTerminalFrontier();
      }
    },

    setActive(nextActive) {
      if (disposed || nextActive === active) return;
      active = nextActive;
      if (active) {
        joinMemoryPool();
        queueRootPage();
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
      // No new work while hidden. Hierarchy pages already in flight are left
      // to land: unlike tiles they survive deactivation in `hierarchy`, so
      // cancelling one only buys a refetch of the same bytes on reactivation.
      queue = [];
      pageQueue = [];
      // Hiding cancels tile reads but keeps their physical slots: a hide/show
      // pair must adopt the read that is still running, not race it.
      for (const read of tileReads.values()) {
        read.wanted = false;
        read.abort.abort();
      }

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
        readyTerminalFrontier: emptyReadyTerminalFrontier(),
      };

      // Nothing stays resident while hidden; the flush turns that into
      // removals for exactly the actors the consumer was last handed.
      for (const keyString of [...resident.keys()]) releaseResident(keyString);
      scheduleFlush();
    },

    stats() {
      const cachedBytes = cache.totalBytes();
      return {
        active,
        residentTiles: resident.size,
        residentPoints,
        residentBytes,
        decodedTiles: resident.size + cache.count(),
        decodedBytes: residentBytes + cachedBytes,
        cachedBytes,
        inFlight: wantedTileReads(),
        queuedTiles: queue.length,
        physicalTileOperations,
        hierarchyInFlight: pagesInFlight.size,
        queuedPages: pageQueue.length,
        physicalHierarchyOperations,
        fetchConcurrency,
        hierarchyConcurrency,
        pointBudget: currentBudget(),
        memoryBudgetBytes: memoryBudgetBytes(),
        memoryCeilingPoints: memoryCeilingPoints(),
        interactionDepth,
        presentation: {
          config: presentation,
          diameterCssPx,
        },
        cachedTiles: cache.count(),
        cacheBytes,
        refinementCutoffPx,
        selection: selectionStats,
      };
    },

    activeKeys: () => ({
      resident: [...resident.keys()].sort(),
      submitted: [...submitted.keys()].sort(),
    }),

    dispose() {
      if (disposed) return;
      if (selectionTimer !== null) {
        clearTimeout(selectionTimer);
        selectionTimer = null;
      }
      clearSettleTimer();
      interactionDepth = 0;
      // The consumer still owns everything the last flush handed it —
      // including tiles a flush this teardown cancels was about to remove.
      // Take all of it back or it keeps actors nothing will ever drop.
      const removed = [...submitted.keys()].map(keyFromString);
      dropEverything();
      submitted = new Map();
      poolMember?.release();
      poolMember = null;
      disposed = true;
      if (removed.length > 0) {
        onTiles({ added: [], removed });
        scheduleRender();
      }
    },
  };
};
