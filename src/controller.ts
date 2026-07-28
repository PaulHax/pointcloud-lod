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
import { createLruCache } from "./lru";
import {
  finiteAtLeast,
  finiteNonNegative,
  finitePositive,
  wholeAtLeast,
} from "./numeric";
import {
  createMemoryPool,
  type MemoryPool,
  type MemoryPoolMember,
} from "./memoryPool";
import {
  pickPointInTiles,
  type PickTile,
  type PointPickResult,
} from "./picking";
import {
  ROOT_KEY,
  childKeys,
  keyFromString,
  keyToString,
  levelFromString,
  type Bounds,
  type VoxelKey,
} from "./octree";
import { tileBytes, type TileData, type TileSource } from "./tileSource";

export type TileBatch = {
  readonly added: readonly { key: VoxelKey; tile: TileData }[];
  readonly removed: readonly VoxelKey[];
};

export type FixedPointPresentation = {
  readonly mode: "fixed";
  readonly diameterCssPx: number;
};

export type AutoPointPresentation = {
  readonly mode: "auto";
  /** Multiplier for the Auto diameter; 1 matches the projected spacing. */
  readonly userScale: number;
  /** Bounds for the unscaled density-aware diameter. */
  readonly minDiameterCssPx?: number;
  readonly maxDiameterCssPx?: number;
};

export type PointPresentation = FixedPointPresentation | AutoPointPresentation;

/**
 * Every numeric option is a programmer error when it is not finite or falls
 * outside its documented range: construction throws naming the option and the
 * value. The live setters take the opposite side of the same policy — see
 * `LodController`.
 */
export type LodControllerOptions = {
  source: TileSource;
  /** Receives batched tile arrivals/removals (typically a renderer adapter). */
  onTiles: (batch: TileBatch) => void;
  /**
   * Coalescing render request — called once per applied batch, never once per
   * tile. Must not render synchronously more than once per event loop turn.
   */
  scheduleRender: () => void;
  /**
   * Whether hierarchy I/O, selection, and renderer submission start enabled.
   * Default true. An initially inactive controller retains camera updates and
   * starts from the latest view when activated.
   */
  active?: boolean;
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
};

export type LodControllerStats = {
  /** Whether selection, requests, and renderer submission are enabled. */
  readonly active: boolean;
  readonly residentTiles: number;
  readonly residentPoints: number;
  /** Decoded bytes backing currently submitted tiles. */
  readonly residentBytes: number;
  /** Decoded payloads across submitted tiles and the dormant CPU cache. */
  readonly decodedTiles: number;
  /**
   * Decoded bytes across submitted tiles and the dormant CPU cache.
   *
   * Do NOT add this to a renderer adapter's `gpuResidentBytes` to get a total:
   * a submitted tile's payload is one set of ArrayBuffers counted in both, so
   * the sum double-counts everything on screen. The two exist because they
   * bound different things — this one against `cacheBytes` on the CPU side,
   * the adapter's against its own ceiling on the GPU side — and the overlap is
   * exactly the submitted set. Real occupancy is
   * `decodedBytes + adapter.pooledBytes`.
   */
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
};

export type LodSelectionStats = {
  readonly generation: number;
  /** Increments only when the selected key set changes. */
  readonly targetRevision: number;
  readonly targetTiles: number;
  readonly targetPoints: number;
  /**
   * Selected tiles with no decoded payload anywhere — neither resident nor in
   * the CPU cache. Read live at `stats()` time, not snapshotted at selection:
   * it is the number that says whether quiet I/O is explained by everything
   * already being decoded, which a global decoded count cannot (tiles decoded
   * for other selections mask the deficit).
   */
  readonly targetUndecodedTiles: number;
  readonly consideredNodes: number;
  readonly availableNodes: number;
  readonly selectedNodes: number;
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
};

/**
 * Setters carry live wire input, so they validate rather than throw: a value
 * that is not finite or is out of range is ignored and changes no state.
 * Construction is where an invalid number is fatal.
 */
export type LodController = {
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
   * ceiling still caps it. Zero means draw nothing — it is the share a view
   * governor assigns a deactivated member, and rejecting it would leave the
   * previous budget silently in force while diagnostics report zero.
   * Negative and non-finite values are ignored.
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
   * Read-only pick against exactly the tile set last handed to the consumer
   * (`submitted` plus each tile's hierarchy bounds) — never `resident`, which
   * can run ahead of a pending renderer flush, and never the decoded cache.
   * The view and cursor are in the same coordinates `setCamera` takes:
   * whatever space the tiles live in, with the cursor in renderer-local css
   * pixels.
   *
   * Returns null when the query is unavailable — inactive or disposed
   * controller, an invalid view, unusable viewport dimensions, a singular
   * view-projection, or a non-finite cursor. A valid sweep that supports
   * nothing is `{status: "miss"}`, never null: only an explicit miss tells a
   * caller its fallback is authorized.
   */
  pickPoint(
    view: CameraView,
    cursorXCssPx: number,
    cursorYCssPx: number,
  ): PointPickResult | null;
  /**
   * Diagnostics: the tiles held on screen, and the set last handed to the
   * consumer. A pending flush is the only reason they differ, so once the
   * controller is quiet they agree — and `submitted` is then exactly what the
   * renderer adapter must hold, key for key.
   */
  activeKeys(): { readonly resident: string[]; readonly submitted: string[] };
  /**
   * Cancel everything and release all tiles. Idempotent.
   *
   * The consumer is handed every submitted tile back as a removal, which is
   * all this side can do: a renderer adapter answers a removal by pooling the
   * actor for reuse, so disposing a controller alone leaves that pool holding
   * GPU resources nothing will ever reclaim. Dispose the adapter too.
   */
  dispose(): void;
};

type HierarchyEntry = {
  pointCount: number;
  bounds: Bounds;
  spacing: number;
  children: readonly VoxelKey[] | null;
  pageRef: boolean;
};

const isAbortError = (error: unknown): boolean =>
  error instanceof Error && error.name === "AbortError";

/**
 * The frontier's five spacing statistics, from one sort of the sample. `at` is
 * the nearest-rank formula and the last element is the maximum, so every
 * reported number matches a per-statistic percentile call — but the walk this
 * feeds reruns on every tile arrival, and sorting once per burst instead of
 * five times per arrival is the difference during a moving camera. Reports
 * null for "nothing measured" rather than NaN.
 */
const spacingQuantiles = (
  values: number[],
): LodSelectionStats["readyTerminalFrontier"]["projectedSpacingCssPx"] => {
  if (values.length === 0) {
    return { p25: null, p50: null, p75: null, p95: null, max: null };
  }
  const sorted = values.sort((a, b) => a - b);
  const at = (p: number): number =>
    sorted[
      Math.min(Math.max(Math.ceil(p * sorted.length) - 1, 0), sorted.length - 1)
    ]!;
  return {
    p25: at(0.25),
    p50: at(0.5),
    p75: at(0.75),
    p95: at(0.95),
    max: sorted[sorted.length - 1]!,
  };
};

const DEFAULT_PRESENTATION: FixedPointPresentation = {
  mode: "fixed",
  diameterCssPx: 2,
};
const DEFAULT_AUTO_MIN_DIAMETER_CSS_PX = 1.5;
const DEFAULT_AUTO_MAX_DIAMETER_CSS_PX = 4;
const INITIAL_AUTO_DIAMETER_CSS_PX = 2;
/**
 * Keep a selected node while a same-level challenger is only marginally more
 * important. Without this band, nodes straddling the point-budget boundary
 * trade places under tiny camera moves, replacing large actors even though the
 * view has barely changed. Nodes outside the frustum or below the refinement
 * cutoff never become candidates, so those changes remain immediate.
 */
const SELECTION_PRIORITY_HYSTERESIS = 0.1;

/** What `checkPresentation` returns: Auto bounds are always resolved. */
type NormalizedPresentation =
  | FixedPointPresentation
  | Required<AutoPointPresentation>;

type PresentationCheck =
  | { readonly presentation: NormalizedPresentation }
  | { readonly error: string };

/** One validation body for both the throwing and the ignoring boundary. */
const checkPresentation = (
  value: PointPresentation | undefined,
): PresentationCheck => {
  const presentation = value ?? DEFAULT_PRESENTATION;
  if (presentation.mode === "fixed") {
    if (!finitePositive(presentation.diameterCssPx)) {
      return {
        error: `Fixed diameterCssPx must be finite and > 0, got ${presentation.diameterCssPx}`,
      };
    }
    return {
      presentation: {
        mode: "fixed",
        diameterCssPx: presentation.diameterCssPx,
      },
    };
  }
  const min = presentation.minDiameterCssPx ?? DEFAULT_AUTO_MIN_DIAMETER_CSS_PX;
  const max = presentation.maxDiameterCssPx ?? DEFAULT_AUTO_MAX_DIAMETER_CSS_PX;
  if (
    !finitePositive(presentation.userScale) ||
    !finitePositive(min) ||
    !Number.isFinite(max) ||
    max < min
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
): NormalizedPresentation => {
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
    !finitePositive(scalar) ||
    (view.projection === "perspective" && view.fovY >= Math.PI) ||
    !finitePositive(view.viewportWidthCssPx) ||
    !finitePositive(view.viewportHeightCssPx)
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
  left: NormalizedPresentation,
  right: NormalizedPresentation,
): boolean =>
  left.mode === "fixed"
    ? right.mode === "fixed" && left.diameterCssPx === right.diameterCssPx
    : right.mode === "auto" &&
      left.userScale === right.userScale &&
      left.minDiameterCssPx === right.minDiameterCssPx &&
      left.maxDiameterCssPx === right.maxDiameterCssPx;

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

/**
 * A selection that has not run, or has been thrown away. The explicit return
 * type is the point: a field added to `LodSelectionStats` becomes one compile
 * error here rather than three silent omissions at the reset sites.
 */
const emptySelectionStats = (
  generation: number,
  targetRevision: number,
): Omit<LodSelectionStats, "targetUndecodedTiles"> => ({
  generation,
  targetRevision,
  targetTiles: 0,
  targetPoints: 0,
  consideredNodes: 0,
  availableNodes: 0,
  selectedNodes: 0,
  leafNodes: 0,
  sseStoppedNodes: 0,
  budgetSkippedNodes: 0,
  budgetSkippedPoints: 0,
  projectedImportance: 0,
  readyTerminalFrontier: emptyReadyTerminalFrontier(),
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
   *
   * Each key's level and error are read once and sorted alongside it rather
   * than recomputed inside the comparator, which a comparison-sort calls
   * O(n log n) times — the queues are rebuilt from scratch on every selection
   * pass, which is exactly when the camera is moving fastest.
   */
  const byFrontierPriority = (keyStrings: readonly string[]): string[] =>
    keyStrings
      .map((keyString) => ({
        keyString,
        level: levelFromString(keyString),
        sse: sseFor(keyString),
      }))
      .sort((a, b) => (a.level !== b.level ? a.level - b.level : b.sse - a.sse))
      .map((entry) => entry.keyString);

  const pagesLoaded = new Set<string>();
  /** Hierarchy pages requested whose result is still wanted. */
  const pagesInFlight = new Map<string, AbortController>();
  let pageQueue: string[] = [];

  // Selection re-requests whatever it still needs, so an endpoint that always
  // fails would be re-issued on every pass forever. Non-abort failures are
  // counted per key and the key rests once it runs out of attempts. The rest
  // is a backoff, not an eviction: a transient outage must not blank a tile
  // for the life of the controller, so the allowance is restored once the key
  // has been quiet for RETRY_BACKOFF_MS.
  //
  // Restoring the allowance only makes a key fetchable — something still has
  // to ask. A selection pass cannot be that something: on a converged scene
  // the camera is still, the budget has settled and nothing lands, so no pass
  // ever runs and one HTTP 500 would hold a parent-density hole (or an
  // unrefinable subtree) open until the user touched the camera. Each
  // non-abort failure therefore arms a timer for that key — RETRY_SOON_MS
  // while attempts remain, the remainder of the backoff once the key is
  // resting — which re-queues it and pumps, so the ordinary arrival path
  // repaints. A key nothing wants any more when its timer fires is dropped
  // silently, and dropEverything/dispose cancel every armed timer, so no
  // retry can carry work across an epoch.
  // A new source (dropEverything) is a fresh start. Aborts never count —
  // deselection, setSource, and dispose cancel normally and stay retryable.
  const MAX_ATTEMPTS = 3;
  const RETRY_BACKOFF_MS = 30_000;
  // Long enough that three attempts at a struggling endpoint are not one
  // burst, short enough that a blip repaints without waiting on the user.
  const RETRY_SOON_MS = 1_000;
  type FailureRecord = {
    count: number;
    lastMs: number;
  };
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

  /** When the key may be asked for again: soon, or when its rest is over. */
  const retryDelayMs = (
    failures: Map<string, FailureRecord>,
    keyString: string,
  ): number => {
    const record = failures.get(keyString);
    if (record === undefined || record.count < MAX_ATTEMPTS) {
      return RETRY_SOON_MS;
    }
    return Math.max(0, record.lastMs + RETRY_BACKOFF_MS - Date.now());
  };

  const tileRetryTimers = new Map<string, ReturnType<typeof setTimeout>>();
  const pageRetryTimers = new Map<string, ReturnType<typeof setTimeout>>();

  const clearRetryTimers = (): void => {
    for (const timer of tileRetryTimers.values()) clearTimeout(timer);
    tileRetryTimers.clear();
    for (const timer of pageRetryTimers.values()) clearTimeout(timer);
    pageRetryTimers.clear();
  };

  /** At most one armed retry per key, always at the earliest legal moment. */
  const armRetry = (
    timers: Map<string, ReturnType<typeof setTimeout>>,
    failures: Map<string, FailureRecord>,
    keyString: string,
    run: (keyString: string) => void,
  ): void => {
    const pending = timers.get(keyString);
    if (pending !== undefined) clearTimeout(pending);
    timers.set(
      keyString,
      setTimeout(
        () => {
          timers.delete(keyString);
          if (!disposed) run(keyString);
        },
        retryDelayMs(failures, keyString),
      ),
    );
  };

  const scheduleTileRetry = (keyString: string): void =>
    armRetry(tileRetryTimers, tileFailures, keyString, retryTile);

  const retryTile = (keyString: string): void => {
    // Anything that already answers for the key — deselection, an adopted
    // read, a payload that landed anyway — makes this retry pointless.
    if (
      !target.has(keyString) ||
      resident.has(keyString) ||
      tileReads.has(keyString)
    ) {
      return;
    }
    // Reading the record is also what restores the allowance, and a key still
    // inside its backoff must wait rather than spend an attempt early.
    if (resting(tileFailures, keyString)) {
      scheduleTileRetry(keyString);
      return;
    }
    // At the head: this key is a hole in the current frame, not new detail.
    queue.unshift(keyString);
    pump();
  };

  const schedulePageRetry = (keyString: string): void =>
    armRetry(pageRetryTimers, pageFailures, keyString, retryPage);

  const retryPage = (keyString: string): void => {
    // Nothing is fetched while hidden; reactivation reselects from scratch.
    if (!active) return;
    if (pagesLoaded.has(keyString) || pagesInFlight.has(keyString)) return;
    if (resting(pageFailures, keyString)) {
      schedulePageRetry(keyString);
      return;
    }
    // Whether a page is still needed is selection's answer, not a set this
    // side keeps, so rerun it: it re-derives the page queue from the current
    // camera, which both re-requests this page and drops the ones the camera
    // has moved past. Before the first camera selection cannot run at all,
    // and the bootstrap page is then the only page there is to ask for.
    if (view !== null) runSelection();
    else if (keyString === keyToString(ROOT_KEY)) queueRootPage();
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
    return !finitePositive(bytes) ? 0 : bytes;
  };

  const memoryCeilingPoints = (): number => {
    const bytes = memoryBudgetBytes();
    return bytes === 0 ? 0 : Math.max(1, Math.floor(bytes / bytesPerPoint()));
  };

  let target: ReadonlySet<string> = new Set<string>();
  let targetRevision = 0;
  let selectionGeneration = 0;
  let selectionStats = emptySelectionStats(0, 0);
  let budgetSkipped: ReadonlySet<string> = new Set<string>();
  /**
   * The one physical read a key may have running. It is created when the read
   * starts and removed only when the promise settles, cancelled or not: while
   * the entry is here nobody may start a rival read of the same key, and a
   * reselect adopts this operation by flipping `wanted` back on.
   *
   * Adoption salvages the read only where cancellation was advisory — a source
   * that ignores the signal, or one that had already finished by the time it
   * fired. An `AbortSignal` cannot be un-aborted, so a source that honours it
   * (a `fetch`, or the COPC decode past its next abort check) still rejects,
   * and the error path re-queues the key at the head. Deferring the abort to
   * make adoption always work would be worse than it sounds: the read holds
   * its concurrency slot until it settles either way, so a read nobody wants
   * would go on occupying a slot the camera's current tiles need.
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
    pageQueue = byFrontierPriority([...new Set(keyStrings)].filter(pageWanted));
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
            const infoString = keyToString(info.key);
            // The cached error was computed from the entry being replaced. A
            // page reference and the node that supersedes it happen to carry
            // the same bounds and spacing in both shipped sources, but that is
            // the source's convention, not this cache's contract — dropping
            // the value costs one recomputation and removes the requirement.
            sseByKey.delete(infoString);
            hierarchy.set(infoString, {
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
            schedulePageRetry(keyString);
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

  /**
   * Size Auto points for the selected density, not the subset that has happened
   * to finish loading.
   *
   * The ready frontier is intentionally conservative: while any selected child
   * is missing, its ready parent remains a terminal so the diagnostic describes
   * what is on screen. Using that same frontier for one global point diameter
   * makes streaming unstable, though. Several fine tiles can land under the
   * coarse diameter, then the last sibling makes the parent cease to be a
   * terminal and every actor shrinks at once. More fine tiles subsequently land,
   * producing a dense → sparse → dense sequence with an unchanged selection.
   *
   * Selection already states the density the completed frame is converging on.
   * Following its terminals moves the diameter once, with the selection, and
   * leaves tile arrivals to add detail monotonically. Hierarchy- and
   * budget-blocked branches still retain their closest sampled parent because
   * no selected descendant describes a finer density there.
   */
  const updateAutoDiameter = (): void => {
    if (
      presentation.mode !== "auto" ||
      view === null ||
      !target.has(keyToString(ROOT_KEY))
    ) {
      return;
    }

    const planes = frustumPlanes(view.viewProj);
    let largestSpacing: number | null = null;
    const include = (entry: HierarchyEntry, keyString: string): void => {
      if (entry.pointCount === 0) return;
      largestSpacing = Math.max(largestSpacing ?? 0, sseFor(keyString));
    };

    const walk = (key: VoxelKey): void => {
      const keyString = keyToString(key);
      if (!target.has(keyString)) return;
      const entry = hierarchy.get(keyString);
      if (entry === undefined) return;

      const children = childrenOf(key, entry);
      if (children.length === 0 || sseFor(keyString) < refinementCutoffPx) {
        include(entry, keyString);
        return;
      }

      let blocked = false;
      const selectedChildren: VoxelKey[] = [];
      for (const child of children) {
        const childString = keyToString(child);
        const childEntry = hierarchy.get(childString);
        if (childEntry === undefined) {
          blocked = true;
          continue;
        }
        if (!boundsIntersectsFrustum(planes, childEntry.bounds)) continue;
        if (childEntry.pageRef && !pagesLoaded.has(childString)) {
          blocked = true;
          continue;
        }
        if (!target.has(childString)) {
          blocked = budgetSkipped.has(childString) || blocked;
          continue;
        }
        selectedChildren.push(child);
      }

      if (blocked) include(entry, keyString);
      for (const child of selectedChildren) walk(child);
    };

    walk(ROOT_KEY);
    if (largestSpacing !== null) {
      emitDiameter(
        presentation.userScale *
          Math.min(
            presentation.maxDiameterCssPx,
            Math.max(presentation.minDiameterCssPx, largestSpacing),
          ),
      );
    }
  };

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
      projectedSpacingCssPx: spacingQuantiles(values),
    };
    selectionStats = { ...selectionStats, readyTerminalFrontier: frontier };
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
            if (target.has(keyString) && !resident.has(keyString)) {
              scheduleTileRetry(keyString);
            }
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

  const runSelection = (seed?: {
    readonly selected: ReadonlySet<string>;
    readonly totalPoints: number;
  }): void => {
    if (disposed || !active || view === null) return;
    const budget = currentBudget();
    const currentView = view;
    const planes = frustumPlanes(currentView.viewProj);
    const sse = (key: VoxelKey): number => sseFor(keyToString(key));
    const previousTarget = target;
    const priority = (key: VoxelKey): number => {
      const value = sse(key);
      return previousTarget.has(keyToString(key))
        ? value * (1 + SELECTION_PRIORITY_HYSTERESIS)
        : value;
    };

    const neededPages: VoxelKey[] = [];
    let leafNodes = 0;
    let sseStoppedNodes = 0;
    const selection = selectNodes({
      root: ROOT_KEY,
      pointBudget: budget,
      priority,
      seed,
      getNode: (key) => {
        const keyString = keyToString(key);
        const entry = hierarchy.get(keyString);
        if (entry === undefined) return undefined;
        // Culling comes first: a page reference carries the bounds of the
        // subtree it stands for, so an invisible one must not be requested at
        // all. Reading it would spend a hierarchy slot on a region no
        // selection can use, which is precisely the fan-out the page queue
        // exists to bound.
        if (!boundsIntersectsFrustum(planes, entry.bounds)) return undefined;
        if (entry.pageRef && !pagesLoaded.has(keyString)) {
          neededPages.push(key);
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

    target = selection.selected;
    budgetSkipped = selection.budgetSkipped;
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
      leafNodes,
      sseStoppedNodes,
      budgetSkippedNodes: selection.budgetSkippedNodes,
      budgetSkippedPoints: selection.budgetSkippedPoints,
      projectedImportance: target.size > 0 ? sse(ROOT_KEY) : 0,
      readyTerminalFrontier: emptyReadyTerminalFrontier(),
    };
    updateAutoDiameter();
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
    // `releaseResident` deletes only the key it is handed, which is safe to do
    // while iterating the map it deletes from — no snapshot needed.
    for (const keyString of resident.keys()) {
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
        // Adopt the live read instead of starting a rival one. If it outran
        // its cancellation the payload claims residency when it lands; if the
        // source really stopped, the abort path re-queues this key at the
        // head. Either way there is never a second read of the same bytes.
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
    queue = byFrontierPriority(toFetch);

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

  const clearSelectionTimer = (): void => {
    if (selectionTimer !== null) {
      clearTimeout(selectionTimer);
      selectionTimer = null;
    }
  };

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
      runSelection();
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
    // Armed retries belong to the epoch that scheduled them: one firing after
    // the swap would re-queue a key of the old source against the new one.
    clearRetryTimers();
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
    selectionStats = emptySelectionStats(
      selectionStats.generation,
      targetRevision,
    );
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
      a.viewportWidthCssPx !== b.viewportWidthCssPx ||
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

  // Bootstrap: an active controller loads hierarchy eagerly; an inactive one
  // waits until activation so a hidden cloud performs no hierarchy I/O.
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
      if (disposed || !finiteNonNegative(points)) return;
      const previousBudget = currentBudget();
      pointBudget = Math.floor(points);
      const nextBudget = currentBudget();
      runSelection(
        nextBudget > previousBudget &&
          target.size > 0 &&
          selectionStats.targetPoints <= nextBudget
          ? {
              selected: target,
              totalPoints: selectionStats.targetPoints,
            }
          : undefined,
      );
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
      if (disposed || !finiteNonNegative(pixels)) return;
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
        updateAutoDiameter();
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

      clearSelectionTimer();
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
      selectionStats = emptySelectionStats(selectionGeneration, targetRevision);

      // Nothing stays resident while hidden; the flush turns that into
      // removals for exactly the actors the consumer was last handed.
      for (const keyString of resident.keys()) releaseResident(keyString);
      scheduleFlush();
    },

    stats() {
      const cachedBytes = cache.totalBytes();
      let targetUndecodedTiles = 0;
      for (const keyString of target) {
        if (!resident.has(keyString) && !cache.has(keyString)) {
          targetUndecodedTiles += 1;
        }
      }
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
        selection: {
          ...selectionStats,
          // Live, not a selection-time snapshot: how many selected tiles have
          // no decoded payload anywhere (neither resident nor cached). This
          // is the number that distinguishes "the burst issued no reads
          // because everything it wanted was already decoded" from "it
          // wanted tiles it does not hold and read nothing" — a global
          // decoded count cannot, because tiles decoded for *other*
          // selections mask the deficit.
          targetUndecodedTiles,
        },
      };
    },

    pickPoint(pickView, cursorXCssPx, cursorYCssPx) {
      if (disposed || !active || !isFiniteView(pickView)) return null;
      const tiles: PickTile[] = [];
      for (const [keyString, tile] of submitted) {
        if (tile.pointCount === 0) continue;
        tiles.push({
          origin: tile.origin,
          positions: tile.positions,
          pointCount: tile.pointCount,
          bounds: hierarchy.get(keyString)?.bounds,
        });
      }
      return pickPointInTiles(pickView, cursorXCssPx, cursorYCssPx, tiles);
    },

    activeKeys: () => ({
      resident: [...resident.keys()].sort(),
      submitted: [...submitted.keys()].sort(),
    }),

    dispose() {
      if (disposed) return;
      clearSelectionTimer();
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
