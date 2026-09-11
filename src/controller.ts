/**
 * LOD controller: turns camera movement into a bounded set of submitted tiles.
 *
 * Selection is a pure pass (frustum cull → centre-ray cone priority →
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
  boundsCenterRayOffset,
  modelFrameOf,
  nodeScreenSpaceError,
  boundsIntersectsFrustum,
  projectionScalar,
  sameCameraView,
  sameMatrix,
  transformPointBy,
  viewInModelFrame,
  type CameraView,
  type Mat16,
  type ModelFrame,
  type Plane,
} from "./camera";
import { selectNodes } from "./budget";
import { scenePoint } from "./frames";
import { createLruCache } from "./lru";
import {
  finiteAtLeast,
  finiteNonNegative,
  finitePositive,
  finiteWithin,
  wholeAtLeast,
} from "./numeric";
import { defaultMemoryBudgetBytes } from "./memoryPool";
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
import { projectedSpacingScale } from "./pointDensity";

export type TileBatch = {
  readonly added: readonly { key: VoxelKey; tile: TileData }[];
  readonly removed: readonly VoxelKey[];
};

export type TileDrawPlan = {
  readonly entries: readonly {
    readonly key: VoxelKey;
    readonly pointCount: number;
  }[];
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
  /** Receives per-tile progressive prefixes without changing tile residency. */
  onDrawPlan?: (plan: TileDrawPlan) => void;
  /**
   * Coalescing render request — called once per applied batch, never once per
   * tile. Must not render synchronously more than once per event loop turn.
   */
  scheduleRender: () => void;
  /**
   * Coalesced notification that asynchronous work state changed. Unlike
   * `scheduleRender`, this does not imply that presentation changed. Hosts
   * may use it to refresh convergence or diagnostics without repainting.
   */
  onWorkChange?: () => void;
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
  /** Initial fraction of selected points to distribute across tile prefixes. */
  densityFraction?: number;
  /**
   * GPU-memory budget for resident tile bytes, updated through
   * `setMemoryBudgetBytes`. Defaults to `defaultMemoryBudgetBytes()`. The
   * controller converts it into a point ceiling using the measured
   * bytes-per-point of resident tiles, so the frame-time loop can never climb
   * into an out-of-memory failure it has no way to sense.
   */
  memoryBudgetBytes?: number;
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
   * Trailing debounce for camera-driven reselection, ms. Default 150.
   * A camera-deselected tile read receives the same short grace before abort,
   * allowing a quick view reversal to adopt work already in progress.
   */
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
  /**
   * Monotonic revision of hierarchy, tile/decode, and renderer-batch work.
   * Comparing consecutive presentations catches work that began and finished
   * between them.
   */
  readonly workRevision: number;
  /** Required current-view work has not drained yet. */
  readonly workPending: boolean;
  /**
   * Selected tiles with no payload that nothing is going to fetch right now:
   * their retry allowance is spent and they are inside a rest.
   *
   * They are deliberately excluded from `workPending`. A caller waiting for
   * the view to converge would otherwise wait on an endpoint that is not
   * answering, and the view governor stops sampling capacity for every member
   * while any of them claims pending work — so one dead tile would freeze
   * adaptive quality view-wide. A host that wants to say so can report this.
   */
  readonly restingTiles: number;
  /** Hierarchy pages whose retry allowance is spent, inside a rest. */
  readonly restingPages: number;
  /** A trailing camera-driven selection pass is still scheduled. */
  readonly selectionPending: boolean;
  /** The ceiling `physicalTileOperations` is held under. */
  readonly fetchConcurrency: number;
  /** The ceiling `physicalHierarchyOperations` is held under. */
  readonly hierarchyConcurrency: number;
  /** Effective visible-point budget currently driving selection. */
  readonly pointBudget: number;
  /** Fraction of selected points supplied to the per-tile draw allocator. */
  readonly densityFraction: number;
  /** Points actually participating in drawing across the submitted set. */
  readonly drawnPoints: number;
  /** Current per-tile allocation and its uniform-density counterfactual. */
  readonly drawPlan: LodDrawPlanStats;
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

/**
 * What a host reports about this cloud on every frame.
 *
 * A view governor needs the memory ceiling to bound the aggregate budget
 * before splitting it, the projected importance to weight this cloud's share,
 * and the physical operation counts to know whether work is still landing; a
 * renderer needs the byte share to size its reuse pool. All of it is already
 * held, so this costs nothing to read — unlike {@link LodControllerStats},
 * which walks the selected set and the submitted set to answer questions no
 * frame asks. Every field here is the same number under the same name there.
 *
 * A governor member also takes `workPending`, which is deliberately not here:
 * answering it needs the walk of the selected set. It is also the one input
 * that cannot change without the controller reporting a work change, so it
 * should be recomputed only when `workRevision` changes, not on every frame.
 */
export type LodGovernorInputs = {
  /** Invalidates cached logical work state without constructing diagnostics. */
  readonly workRevision: number;
  readonly memoryBudgetBytes: number;
  readonly memoryCeilingPoints: number;
  /** The selection's root screen-space error; `selection.projectedImportance`. */
  readonly projectedImportance: number;
  /**
   * Points this cloud could use at the current camera: what it selected plus
   * what its budget turned away. Exact whenever the budget is not binding —
   * nothing is turned away, so this is the whole of what the camera asks for —
   * and otherwise it only has to exceed the share, which it does, because
   * something was turned away. A governor allocates no more than this, so a
   * small cloud stops holding points it has nothing to spend them on.
   */
  readonly demandPoints: number;
  readonly physicalTileOperations: number;
  readonly physicalHierarchyOperations: number;
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
  /**
   * The coarsest projected spacing any terminal of the selection leaves on
   * screen, null when nothing is drawn.
   */
  readonly projectedSpacingCssPx: number | null;
};

export type LodDrawPlanStats = {
  /** Increments only when at least one tile prefix changes. */
  readonly revision: number;
  /** The selection's points scaled by the density fraction. */
  readonly pointBudget: number;
  /** Points the per-tile prefixes actually draw, after rounding. */
  readonly plannedPoints: number;
};

/**
 * Setters carry live wire input, so they validate rather than throw: a value
 * that is not finite or is out of range is ignored and changes no state.
 * Construction is where an invalid number is fatal.
 */
export type LodController = {
  /**
   * Update the camera; selection reruns debounced (leading edge immediate).
   * A view with any non-finite number is ignored. With a model matrix set,
   * the view is in world coordinates and the controller restates it in the
   * tiles' local frame itself; while the current model matrix is unusable
   * (not a similarity), camera updates are held and the last good view stays
   * in force.
   */
  setCamera(view: CameraView): void;
  /**
   * The transform the tiles draw under (the host actor's model/user matrix),
   * or null for identity. The controller's frustum/SSE math needs a
   * uniform-scale transform, so a matrix that is not a similarity marks the
   * transform unusable: selection holds the last good camera and picks are
   * unavailable until a usable matrix arrives. An unchanged matrix is a
   * no-op, so hosts can forward it every pass; a change re-derives the
   * camera restatement once, not per frame.
   */
  setModelMatrix(matrix: Mat16 | null): void;
  /** Enter a camera interaction. Nested calls are reference counted. */
  beginInteraction(): void;
  /** Leave a camera interaction. Camera stability belongs to the view governor. */
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
   * Redistribute this fraction of selected points into progressive prefixes
   * without changing tile selection.
   * Values outside [0, 1] and non-finite values are ignored.
   */
  setDensityFraction(densityFraction: number): void;
  /** Update a coordinator-owned byte allowance (ignored in pool-owned mode). */
  setMemoryBudgetBytes(bytes: number): void;
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
  /**
   * The per-frame numbers for a view governor and a resource pool. Prefer it
   * to `stats()` in a render loop: `stats()` is a diagnostic snapshot and
   * builds the whole thing.
   */
  governorInputs(): LodGovernorInputs;
  stats(): LodControllerStats;
  /**
   * Read-only pick against exactly the tile set last handed to the consumer
   * (`submitted` plus each tile's hierarchy bounds) — never `resident`, which
   * can run ahead of a pending renderer flush, and never the decoded cache.
   * The view and cursor are in the same coordinates `setCamera` takes —
   * world coordinates when a model matrix is set — with the cursor in
   * renderer-local css pixels. A hit's `scenePoint` comes back in those same
   * world coordinates: the controller solves on the model-local ray and
   * transforms the answer back through the model matrix itself.
   *
   * Returns null when the query is unavailable — inactive or disposed
   * controller, an invalid view, an unusable (non-similarity) model matrix,
   * unusable viewport dimensions, a singular
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

const ROOT_KEY_STRING = keyToString(ROOT_KEY);

const DEFAULT_PRESENTATION: FixedPointPresentation = {
  mode: "fixed",
  diameterCssPx: 2,
};
const DEFAULT_AUTO_MIN_DIAMETER_CSS_PX = 1.5;
const DEFAULT_AUTO_MAX_DIAMETER_CSS_PX = 4;
const INITIAL_AUTO_DIAMETER_CSS_PX = 2;
/** Screen-centre improvement required to replace a selected boundary tile. */
const SELECTION_CENTER_HYSTERESIS_CSS_PX = 8;

const centerOffsetHysteresis = (view: CameraView): number => {
  const normalized =
    (2 * SELECTION_CENTER_HYSTERESIS_CSS_PX) / view.viewportHeightCssPx;
  return view.projection === "perspective"
    ? Math.atan(normalized * Math.tan(view.fovY / 2))
    : normalized;
};

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

/**
 * A selection that has not run, or has been thrown away. The explicit return
 * type is the point: a field added to `LodSelectionStats` becomes one compile
 * error here rather than three silent omissions at the reset sites.
 */
const emptySelectionStats = (
  generation: number,
  targetRevision: number,
): Omit<
  LodSelectionStats,
  "targetUndecodedTiles" | "projectedSpacingCssPx"
> => ({
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
});

export const createLodController = (
  options: LodControllerOptions,
): LodController => {
  const {
    onTiles,
    onDrawPlan = () => {},
    scheduleRender,
    onWorkChange = () => {},
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
  let source = options.source;
  let pointBudget = wholeAtLeast(
    "pointBudget",
    options.pointBudget ?? 2_000_000,
    1,
  );
  let densityFraction = finiteWithin(
    "densityFraction",
    options.densityFraction ?? 1,
    0,
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
  /** The camera in the tiles' local frame — what all selection math reads. */
  let view: CameraView | null = null;
  /** The camera as the host supplied it, kept to re-derive `view` when the
   * model matrix changes. */
  let worldView: CameraView | null = null;
  /**
   * The model transform tiles draw under, resolved once per change: the
   * matrix as applied (for the no-op guard), the frame (inverse + uniform
   * scale), and whether the current matrix is usable at all. Restating the
   * camera is the only per-frame cost left — everything matrix-derived is
   * cached here.
   */
  let appliedModelMatrix: number[] | null = null;
  let modelFrame: ModelFrame | null = null;
  let modelMatrixUsable = true;
  let disposed = false;
  onPointDiameterCssPx(diameterCssPx);

  const emitDiameter = (next: number): void => {
    if (next === diameterCssPx) return;
    diameterCssPx = next;
    onPointDiameterCssPx(next);
  };

  let interactionDepth = 0;
  const currentBudget = (): number =>
    active ? Math.min(pointBudget, memoryCeilingPoints()) : 0;

  // Bumped on setSource/dispose; every async continuation checks it.
  let epoch = 0;
  let workRevision = 0;
  let workChangeScheduled = false;
  const markWork = (): void => {
    workRevision += 1;
    if (workChangeScheduled) return;
    workChangeScheduled = true;
    queueMicrotask(() => {
      workChangeScheduled = false;
      if (!disposed) onWorkChange();
    });
  };

  const hierarchy = new Map<string, HierarchyEntry>();

  // Both priority metrics depend only on a node and the current view, so their
  // values stay valid until the view moves. Selection and the ready-frontier
  // walk both need them, and the frontier walk reruns on every tile arrival.
  // A node whose hierarchy entry has not landed yet is deliberately NOT
  // cached: it scores 0 now, and its page may arrive before the view moves.
  const sseByKey = new Map<string, number>();
  const centerOffsetByKey = new Map<string, number>();
  const sseFor = (keyString: string): number => {
    const cached = sseByKey.get(keyString);
    if (cached !== undefined) return cached;
    const entry = hierarchy.get(keyString);
    if (entry === undefined || view === null) return 0;
    const value = nodeScreenSpaceError(entry.bounds, entry.spacing, view);
    sseByKey.set(keyString, value);
    return value;
  };
  const centerOffsetFor = (keyString: string): number => {
    const cached = centerOffsetByKey.get(keyString);
    if (cached !== undefined) return cached;
    const entry = hierarchy.get(keyString);
    if (entry === undefined || view === null) return Number.POSITIVE_INFINITY;
    const value = boundsCenterRayOffset(entry.bounds, view);
    centerOffsetByKey.set(keyString, value);
    return value;
  };
  /**
   * Request order for both queues: coarse levels first, then the smallest
   * 3D centre-ray offset, then screen-space error. That makes the selected
   * frontier grow as concentric cones through the octree, so a
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
        centerOffset: centerOffsetFor(keyString),
        sse: sseFor(keyString),
      }))
      .sort((a, b) =>
        a.level !== b.level
          ? a.level - b.level
          : a.centerOffset - b.centerOffset ||
            b.sse - a.sse ||
            a.keyString.localeCompare(b.keyString),
      )
      .map((entry) => entry.keyString);

  const pagesLoaded = new Set<string>();
  /** Hierarchy pages requested whose result is still wanted. */
  const pagesInFlight = new Map<string, AbortController>();
  let pageQueue: string[] = [];

  // Selection re-requests whatever it still needs, so an endpoint that always
  // fails would be re-issued on every pass forever. Non-abort failures are
  // counted per key and the key rests once it runs out of attempts. The rest
  // is a backoff, not an eviction: a transient outage must not blank a tile
  // for the life of the controller, so a rested key gets its allowance back.
  //
  // Each spent allowance rests longer than the one before it, to MAX_REST_MS.
  // An endpoint that is permanently gone therefore costs three requests per
  // key every 30 s at first and three every five minutes in the steady state,
  // rather than three every 30 s for as long as the page is open, while an
  // outage that ends recovers on the first rest — the case worth being quick
  // about.
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
  const FIRST_REST_MS = 30_000;
  const MAX_REST_MS = 300_000;
  // Long enough that three attempts at a struggling endpoint are not one
  // burst, short enough that a blip repaints without waiting on the user.
  const RETRY_SOON_MS = 1_000;
  type FailureRecord = {
    /** Failures in the current round. The allowance is MAX_ATTEMPTS. */
    count: number;
    /** Allowances already spent on this key; each one lengthens the rest. */
    rounds: number;
    lastMs: number;
  };
  const pageFailures = new Map<string, FailureRecord>();
  const tileFailures = new Map<string, FailureRecord>();

  const restMs = (record: FailureRecord): number =>
    Math.min(MAX_REST_MS, FIRST_REST_MS * 2 ** record.rounds);

  const recordFailure = (
    failures: Map<string, FailureRecord>,
    keyString: string,
  ): void => {
    const prior = failures.get(keyString);
    // A failure arriving on a key whose allowance is already spent means its
    // rest elapsed and the attempt that followed failed too: open a new round,
    // and rest longer for it.
    const spent = prior !== undefined && prior.count >= MAX_ATTEMPTS;
    failures.set(keyString, {
      count: spent ? 1 : (prior?.count ?? 0) + 1,
      rounds: spent ? prior.rounds + 1 : (prior?.rounds ?? 0),
      lastMs: Date.now(),
    });
  };
  /**
   * Whether the key is inside its rest.
   *
   * Pure: the allowance comes back on the clock rather than by being read, so
   * a diagnostic can ask this without changing what the controller fetches
   * next. The record itself is dropped when the key succeeds, and by
   * dropEverything.
   */
  const resting = (
    failures: Map<string, FailureRecord>,
    keyString: string,
  ): boolean => {
    const record = failures.get(keyString);
    if (record === undefined || record.count < MAX_ATTEMPTS) return false;
    return Date.now() - record.lastMs < restMs(record);
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
    return Math.max(0, record.lastMs + restMs(record) - Date.now());
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
    // A key inside its rest waits rather than spending an attempt early.
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
    else if (keyString === ROOT_KEY_STRING) queueRootPage();
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
  let externalMemoryBudgetBytes =
    options.memoryBudgetBytes === undefined
      ? defaultMemoryBudgetBytes()
      : Math.floor(
          finiteAtLeast("memoryBudgetBytes", options.memoryBudgetBytes, 0),
        );

  const FALLBACK_BYTES_PER_POINT = 16;
  const MEASURE_MIN_POINTS = 100_000;

  const bytesPerPoint = (): number =>
    residentPoints >= MEASURE_MIN_POINTS
      ? residentBytes / residentPoints
      : FALLBACK_BYTES_PER_POINT;

  // An inactive controller holds no share: it has dropped its resident tiles
  // and must not reselect until reactivated.
  const memoryBudgetBytes = (): number =>
    active ? externalMemoryBudgetBytes : 0;

  const memoryCeilingPoints = (): number => {
    const bytes = memoryBudgetBytes();
    return bytes === 0 ? 0 : Math.max(1, Math.floor(bytes / bytesPerPoint()));
  };

  let target: ReadonlySet<string> = new Set<string>();
  let targetRevision = 0;
  let selectionGeneration = 0;
  let selectionStats = emptySelectionStats(0, 0);
  let drawPrefixes: ReadonlyMap<string, number> = new Map();
  let drawPlanStats: LodDrawPlanStats = {
    revision: 0,
    pointBudget: 0,
    plannedPoints: 0,
  };

  /**
   * What a submitted tile actually draws: the planned prefix, clamped to the
   * payload the consumer holds. The plan is allocated from hierarchy point
   * counts, which need not match the decoded tile.
   */
  const drawnPointsOf = (keyString: string, tile: TileData): number =>
    Math.min(tile.pointCount, drawPrefixes.get(keyString) ?? 0);

  type TileRead = {
    readonly abort: AbortController;
    wanted: boolean;
    cancelTimer: ReturnType<typeof setTimeout> | null;
  };

  /**
   * The one physical read a key may have running. Camera deselection gives it
   * one selection interval to be adopted again before aborting. That bounds
   * how long stale work can occupy a slot while avoiding a cancel/refetch pair
   * when an orbit or pan crosses back over a recent tile. Lifecycle changes
   * retain their existing immediate-cancellation semantics.
   */
  const tileReads = new Map<string, TileRead>();
  let queue: string[] = [];

  const clearReadCancellation = (read: TileRead): void => {
    if (read.cancelTimer === null) return;
    clearTimeout(read.cancelTimer);
    read.cancelTimer = null;
  };

  const cancelRead = (read: TileRead, immediately = false): void => {
    read.wanted = false;
    if (immediately || selectionDelayMs === 0) {
      clearReadCancellation(read);
      read.abort.abort();
      return;
    }
    if (read.cancelTimer !== null) return;
    read.cancelTimer = setTimeout(() => {
      read.cancelTimer = null;
      if (!read.wanted) read.abort.abort();
    }, selectionDelayMs);
  };

  const adoptRead = (read: TileRead): void => {
    read.wanted = true;
    clearReadCancellation(read);
  };

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
      markWork();
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
      markWork();
      source.nodes(keyFromString(keyString), { signal: abort.signal }).then(
        (infos) => {
          physicalHierarchyOperations -= 1;
          markWork();
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
            centerOffsetByKey.delete(infoString);
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
          markWork();
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

  const samePrefixes = (
    left: ReadonlyMap<string, number>,
    right: ReadonlyMap<string, number>,
  ): boolean => {
    if (left.size !== right.size) return false;
    for (const [key, count] of left) {
      if (right.get(key) !== count) return false;
    }
    return true;
  };

  /**
   * Thin every selected tile to the governor's draw allowance.
   *
   * Selection and residency stay unchanged: only the progressive VBO prefixes
   * move, and they move together. Every tile keeps the same fraction of its
   * points, so a reduced allowance reads as a uniformly sparser cloud rather
   * than as whole tiles dropping out of the deepest levels while their
   * neighbours stay dense.
   */
  const updateDrawPlan = (): void => {
    const prefixes = new Map<string, number>();
    let plannedPoints = 0;
    for (const keyString of target) {
      const entry = hierarchy.get(keyString);
      if (entry === undefined || entry.pointCount === 0) continue;
      const prefix = Math.min(
        entry.pointCount,
        Math.ceil(entry.pointCount * densityFraction),
      );
      prefixes.set(keyString, prefix);
      plannedPoints += prefix;
    }
    const changed = !samePrefixes(drawPrefixes, prefixes);
    drawPrefixes = prefixes;
    drawPlanStats = {
      revision: drawPlanStats.revision + (changed ? 1 : 0),
      pointBudget: Math.floor(selectionStats.targetPoints * densityFraction),
      plannedPoints,
    };
    if (!changed) return;
    onDrawPlan({
      entries: [...drawPrefixes].map(([keyString, pointCount]) => ({
        key: keyFromString(keyString),
        pointCount,
      })),
    });
  };

  /**
   * The projected spacing a terminal's drawn prefix leaves on screen, null
   * when it draws nothing. Thinning a tile to a prefix spreads its points, so
   * the node's own screen-space error is scaled by the prefix's density.
   */
  const terminalSpacing = (
    keyString: string,
    entry: HierarchyEntry,
  ): number | null => {
    const prefix = drawPrefixes.get(keyString) ?? 0;
    if (entry.pointCount === 0 || prefix <= 0) return null;
    return sseFor(keyString) * projectedSpacingScale(prefix / entry.pointCount);
  };

  /**
   * Walk the selected tree down to its terminals — the nodes where refinement
   * stopped, because they are leaves, fall under the refinement cutoff, or
   * have a visible child the selection did not take.
   *
   * The walk follows the selection rather than what has landed. Several fine
   * tiles arriving under a coarse diameter would otherwise make their parent
   * cease to be a terminal and shrink every actor at once, producing a
   * dense -> sparse -> dense sequence under an unchanged selection.
   */
  const walkTerminals = (
    planes: readonly Plane[],
    onTerminal: (keyString: string, entry: HierarchyEntry) => void,
  ): void => {
    const walk = (key: VoxelKey): void => {
      const keyString = keyToString(key);
      if (!target.has(keyString)) return;
      const entry = hierarchy.get(keyString);
      if (entry === undefined) return;

      const children = childrenOf(key, entry);
      if (children.length === 0 || sseFor(keyString) < refinementCutoffPx) {
        onTerminal(keyString, entry);
        return;
      }

      let blocked = false;
      const openChildren: VoxelKey[] = [];
      for (const child of children) {
        const childString = keyToString(child);
        const childEntry = hierarchy.get(childString);
        if (childEntry === undefined) {
          blocked = true;
          continue;
        }
        if (!boundsIntersectsFrustum(planes, childEntry.bounds)) continue;
        // Matches selection: an invisible page reference is not requested, so
        // it is not blocking anything either.
        if (childEntry.pageRef && !pagesLoaded.has(childString)) {
          blocked = true;
          continue;
        }
        // A visible, available child of a selected parent can only be absent
        // because the breadth-first point budget rejected it.
        if (!target.has(childString)) {
          blocked = true;
          continue;
        }
        openChildren.push(child);
      }

      if (blocked) onTerminal(keyString, entry);
      for (const child of openChildren) walk(child);
    };

    walk(ROOT_KEY);
  };

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
  /**
   * The coarsest projected spacing any terminal leaves on screen, or null when
   * nothing is drawn. Auto sizing derives the point diameter from it, and it
   * is the one continuous measure of how finely the current view is resolved.
   */
  let largestTerminalSpacingCssPx: number | null = null;

  const updateAutoDiameter = (): void => {
    if (
      view === null ||
      drawPlanStats.plannedPoints <= 0 ||
      !target.has(ROOT_KEY_STRING)
    ) {
      largestTerminalSpacingCssPx = null;
      return;
    }

    let largestSpacing: number | null = null;
    walkTerminals(frustumPlanes(view.viewProj), (keyString, entry) => {
      const spacing = terminalSpacing(keyString, entry);
      if (spacing !== null)
        largestSpacing = Math.max(largestSpacing ?? 0, spacing);
    });
    largestTerminalSpacingCssPx = largestSpacing;

    if (largestSpacing !== null && presentation.mode === "auto") {
      emitDiameter(
        presentation.userScale *
          Math.min(
            presentation.maxDiameterCssPx,
            Math.max(presentation.minDiameterCssPx, largestSpacing),
          ),
      );
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
        scheduleFlush();
        continue;
      }
      const abort = new AbortController();
      const read: TileRead = { abort, wanted: true, cancelTimer: null };
      tileReads.set(keyString, read);
      const requestEpoch = epoch;
      const key = keyFromString(keyString);
      physicalTileOperations += 1;
      markWork();
      source.loadTile(key, { signal: abort.signal }).then(
        (loadedTile) => {
          physicalTileOperations -= 1;
          markWork();
          // An epoch change is the only thing that takes a key's read entry
          // away while the read runs, and it also makes the payload worthless:
          // it came from a source nobody is displaying any more.
          if (disposed || requestEpoch !== epoch) {
            pump();
            return;
          }
          clearReadCancellation(read);
          tileReads.delete(keyString);
          tileFailures.delete(keyString);
          // Cancellation is advisory: the COPC getter takes no signal, so a
          // cancelled read still delivers. If the key was reselected while it
          // ran, this payload is exactly what the selection is waiting for.
          if (target.has(keyString) && !resident.has(keyString)) {
            takeResident(keyString, loadedTile);
            scheduleFlush();
          } else {
            cacheDecoded(keyString, loadedTile);
          }
          pump();
        },
        (error) => {
          physicalTileOperations -= 1;
          markWork();
          if (disposed || requestEpoch !== epoch) {
            pump();
            return;
          }
          clearReadCancellation(read);
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
          pump();
        },
      );
    }
  };

  const runSelection = (seed?: ReadonlySet<string>): void => {
    if (disposed || !active || view === null) return;
    const budget = currentBudget();
    const currentView = view;
    const planes = frustumPlanes(currentView.viewProj);
    const sse = (key: VoxelKey): number => sseFor(keyToString(key));
    const previousTarget = target;
    const centerHysteresis = centerOffsetHysteresis(currentView);
    const centerPriority = (key: VoxelKey): number => {
      const keyString = keyToString(key);
      return (
        -centerOffsetFor(keyString) +
        (previousTarget.has(keyString) ? centerHysteresis : 0)
      );
    };

    const neededPages: VoxelKey[] = [];
    let leafNodes = 0;
    let sseStoppedNodes = 0;
    const selection = selectNodes({
      root: ROOT_KEY,
      pointBudget: budget,
      priority: centerPriority,
      secondaryPriority: sse,
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
    };
    updateDrawPlan();
    updateAutoDiameter();
    // The root page bootstraps the hierarchy, so it can never come back
    // through neededPages: that path needs a hierarchy entry, and only the
    // root page can create one. Without this, a failed bootstrap leaves the
    // controller with nothing to draw and no way to ask again. Level 0 sorts
    // to the front, and queuePages drops anything already held, in flight, or
    // resting, so this costs nothing on the normal path.
    queuePages(
      pagesLoaded.has(ROOT_KEY_STRING)
        ? neededPages.map(keyToString)
        : [ROOT_KEY_STRING, ...neededPages.map(keyToString)],
    );

    // Deselected submitted tiles leave renderer/GPU residency immediately.
    // `releaseResident` deletes only the key it is handed, which is safe to do
    // while iterating the map it deletes from — no snapshot needed.
    for (const keyString of resident.keys()) {
      if (target.has(keyString)) continue;
      releaseResident(keyString);
    }

    // Give camera-deselected reads one selection interval to be adopted again.
    // The read keeps its physical slot until it settles: dropping its entry
    // would let a reselect race a second read against work still running.
    for (const [keyString, read] of tileReads) {
      if (target.has(keyString)) continue;
      cancelRead(read);
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
        adoptRead(read);
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

    scheduleFlush();
    pump();
  };

  let lastSelection = Number.NEGATIVE_INFINITY;
  let selectionTimer: ReturnType<typeof setTimeout> | null = null;

  const clearSelectionTimer = (): void => {
    if (selectionTimer !== null) {
      clearTimeout(selectionTimer);
      selectionTimer = null;
    }
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
      cancelRead(read, true);
    }
    tileReads.clear();
    for (const abort of pagesInFlight.values()) abort.abort();
    pagesInFlight.clear();
    queue = [];
    pageQueue = [];
    hierarchy.clear();
    sseByKey.clear();
    centerOffsetByKey.clear();
    pagesLoaded.clear();
    pageFailures.clear();
    tileFailures.clear();
    cache.clear();
    if (target.size > 0) targetRevision += 1;
    target = new Set();
    selectionStats = emptySelectionStats(
      selectionStats.generation,
      targetRevision,
    );
    resident.clear();
    residentPoints = 0;
    residentBytes = 0;
    updateDrawPlan();
  };

  // Hosts feed the camera on every render, unconditionally; an unchanged view
  // must be a no-op. Stamping it as a camera change would mark any render as
  // "interacting" — including renders the settle-timer reselect itself
  // triggers — flipping the regime back and oscillating between the two
  // budgets forever.
  /** Bootstrap path for the root page, used before any camera exists. */
  const queueRootPage = (): void => queuePages([ROOT_KEY_STRING]);

  // Bootstrap: an active controller loads hierarchy eagerly; an inactive one
  // waits until activation so a hidden cloud performs no hierarchy I/O.
  if (active) queueRootPage();

  /** Adopt the local restatement of the stored world camera, if usable. */
  const applyWorldView = (): void => {
    if (worldView === null || !modelMatrixUsable) return;
    view = modelFrame ? viewInModelFrame(worldView, modelFrame) : worldView;
    sseByKey.clear();
    centerOffsetByKey.clear();
    if (active) requestSelection();
  };

  return {
    setCamera(nextView) {
      if (disposed || !isFiniteView(nextView)) return;
      if (worldView !== null && sameCameraView(worldView, nextView)) return;
      worldView = nextView;
      applyWorldView();
    },

    setModelMatrix(matrix) {
      if (disposed || sameMatrix(appliedModelMatrix, matrix ?? null)) return;
      appliedModelMatrix = matrix ? Array.from(matrix) : null;
      modelFrame = matrix ? modelFrameOf(matrix) : null;
      modelMatrixUsable = matrix ? modelFrame !== null : true;
      applyWorldView();
    },

    beginInteraction() {
      if (disposed) return;
      interactionDepth += 1;
      if (interactionDepth !== 1) return;
      // Bypass camera debounce before the first expensive moving frame.
      runSelection();
    },

    endInteraction() {
      if (disposed || interactionDepth === 0) return;
      interactionDepth -= 1;
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
          ? target
          : undefined,
      );
    },

    setDensityFraction(nextDensityFraction) {
      if (
        disposed ||
        !finiteNonNegative(nextDensityFraction) ||
        nextDensityFraction > 1 ||
        nextDensityFraction === densityFraction
      ) {
        return;
      }
      densityFraction = nextDensityFraction;
      updateDrawPlan();
      updateAutoDiameter();
    },

    setMemoryBudgetBytes(bytes) {
      if (disposed || !finiteNonNegative(bytes)) return;
      const next = Math.floor(bytes);
      if (next === externalMemoryBudgetBytes) return;
      const previousBudget = currentBudget();
      externalMemoryBudgetBytes = next;
      const nextBudget = currentBudget();
      runSelection(
        nextBudget > previousBudget &&
          target.size > 0 &&
          selectionStats.targetPoints <= nextBudget
          ? target
          : undefined,
      );
    },

    setSource(nextSource) {
      if (disposed) return;
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
      }
    },

    setActive(nextActive) {
      if (disposed || nextActive === active) return;
      active = nextActive;
      if (active) {
        queueRootPage();
        runSelection();
        return;
      }

      clearSelectionTimer();
      // No new work while hidden. Hierarchy pages already in flight are left
      // to land: unlike tiles they survive deactivation in `hierarchy`, so
      // cancelling one only buys a refetch of the same bytes on reactivation.
      queue = [];
      pageQueue = [];
      // Hiding cancels tile reads but keeps their physical slots: a hide/show
      // pair must adopt the read that is still running, not race it.
      for (const read of tileReads.values()) {
        cancelRead(read, true);
      }

      if (target.size > 0) targetRevision += 1;
      target = new Set();
      selectionGeneration += 1;
      selectionStats = emptySelectionStats(selectionGeneration, targetRevision);
      updateDrawPlan();

      // Nothing stays resident while hidden; the flush turns that into
      // removals for exactly the actors the consumer was last handed.
      for (const keyString of resident.keys()) releaseResident(keyString);
      scheduleFlush();
    },

    governorInputs() {
      return {
        workRevision,
        memoryBudgetBytes: memoryBudgetBytes(),
        memoryCeilingPoints: memoryCeilingPoints(),
        projectedImportance: selectionStats.projectedImportance,
        demandPoints:
          selectionStats.targetPoints + selectionStats.budgetSkippedPoints,
        physicalTileOperations,
        physicalHierarchyOperations,
      };
    },

    stats() {
      const cachedBytes = cache.totalBytes();
      let targetUndecodedTiles = 0;
      let restingTiles = 0;
      for (const keyString of target) {
        if (
          hierarchy.get(keyString)?.pointCount !== 0 &&
          !resident.has(keyString) &&
          !cache.has(keyString)
        ) {
          targetUndecodedTiles += 1;
          if (resting(tileFailures, keyString)) restingTiles += 1;
        }
      }
      let restingPages = 0;
      for (const keyString of pageFailures.keys()) {
        if (resting(pageFailures, keyString)) restingPages += 1;
      }
      let drawnPoints = 0;
      for (const [keyString, tile] of submitted) {
        drawnPoints += drawnPointsOf(keyString, tile);
      }
      const workPending =
        physicalTileOperations > 0 ||
        physicalHierarchyOperations > 0 ||
        queue.length > 0 ||
        pageQueue.length > 0 ||
        targetUndecodedTiles > restingTiles;
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
        workRevision,
        workPending,
        restingTiles,
        restingPages,
        selectionPending: selectionTimer !== null,
        fetchConcurrency,
        hierarchyConcurrency,
        pointBudget: currentBudget(),
        densityFraction,
        drawnPoints,
        drawPlan: drawPlanStats,
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
          projectedSpacingCssPx: largestTerminalSpacingCssPx,
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
      if (!modelMatrixUsable) return null;
      const localView = modelFrame
        ? viewInModelFrame(pickView, modelFrame)
        : pickView;
      const tiles: PickTile[] = [];
      for (const [keyString, tile] of submitted) {
        const pointCount = drawnPointsOf(keyString, tile);
        if (pointCount === 0) continue;
        tiles.push({
          origin: tile.origin,
          positions: tile.positions,
          pointCount,
          bounds: hierarchy.get(keyString)?.bounds,
        });
      }
      const result = pickPointInTiles(
        localView,
        cursorXCssPx,
        cursorYCssPx,
        tiles,
      );
      // The sweep solved on the model-local ray; hand the answer back in the
      // caller's world coordinates through the same matrix.
      return result?.status === "hit" && modelFrame
        ? {
            ...result,
            rayDepth: result.rayDepth * modelFrame.scale,
            scenePoint: scenePoint(
              ...transformPointBy(modelFrame.matrix, result.scenePoint),
            ),
          }
        : result;
    },

    activeKeys: () => ({
      resident: [...resident.keys()].sort(),
      submitted: [...submitted.keys()].sort(),
    }),

    dispose() {
      if (disposed) return;
      clearSelectionTimer();
      interactionDepth = 0;
      // The consumer still owns everything the last flush handed it —
      // including tiles a flush this teardown cancels was about to remove.
      // Take all of it back or it keeps actors nothing will ever drop.
      const removed = [...submitted.keys()].map(keyFromString);
      dropEverything();
      submitted = new Map();
      disposed = true;
      if (removed.length > 0) {
        onTiles({ added: [], removed });
        scheduleRender();
      }
    },
  };
};
