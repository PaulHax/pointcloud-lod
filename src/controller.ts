/**
 * LOD controller: turns camera movement into a bounded set of submitted tiles.
 *
 * Its decisions are pure functions over read-only views of the state it
 * holds: `selectPoints` chooses the tiles, `planDraw` thins them,
 * `largestTerminalSpacing` sizes Auto points and `pointCapacity` bounds the
 * budget by memory. Two loaders read hierarchy pages and tiles, and a payload
 * residency holds what they deliver and hands the consumer deltas. The
 * controller keeps the configuration, the camera and model frame, and the
 * selection debounce, and runs one selection pass whenever any of them moves.
 */

import {
  copyCameraView,
  modelFrameOf,
  prepareView,
  sameCameraView,
  usableView,
  viewInModelFrame,
  type CameraView,
  type ModelFrame,
  type PreparedView,
} from "./camera";
import {
  EMPTY_DRAW_PLAN,
  largestTerminalSpacing,
  planDraw,
  samePrefixes,
  type DrawPlan,
} from "./drawPlan";
import { createLoader, type LoaderCounts } from "./loader";
import { sameMatrix, transformPoint, type Mat16 } from "./mat4";
import { scenePoint } from "./frames";
import { finiteAtLeast, finiteNonNegative, wholeAtLeast } from "./numeric";
import { defaultMemoryBudgetBytes } from "./memoryPool";
import {
  pickPointInTiles,
  type PickTile,
  type PointPickResult,
} from "./picking";
import { keyFromString, keyToString, type VoxelKey } from "./octree";
import { createPayloadResidency, type TileBatch } from "./payloadResidency";
import {
  INITIAL_AUTO_DIAMETER_CSS_PX,
  autoDiameterCssPx,
  checkPresentation,
  normalizePresentation,
  samePresentation,
  type PointPresentation,
} from "./pointPresentation";
import {
  EMPTY_SELECTION,
  ROOT_KEY_STRING,
  frontierOrder,
  nodePayloadBytes,
  selectPoints,
  type Hierarchy,
  type HierarchyEntry,
  type PointSelection,
} from "./pointSelection";
import { pointCapacity } from "./pointQuality";
import type { RetryPolicy } from "./retryPolicy";
import {
  tileBytes,
  type NodeInfo,
  type TileData,
  type TileSource,
} from "./tileSource";

export type TileDrawPlan = {
  readonly entries: readonly {
    readonly key: VoxelKey;
    readonly pointCount: number;
  }[];
};

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
  /** Clamp picking to prefixes admitted by the renderer. */
  getDrawnPointCount?: (keyString: string) => number;
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
 * and the physical operation counts and pending work to know whether work is
 * still landing; a renderer needs the byte share to size its reuse pool. None
 * of it walks the selected or submitted set, as {@link LodControllerStats}
 * does to answer questions no frame asks, so it is cheap to read on every
 * frame. Every field here is the same value under the same name there.
 */
export type LodGovernorInputs = {
  readonly workRevision: number;
  /** Read from counts the loaders keep, so it walks nothing. */
  readonly workPending: boolean;
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

/**
 * Three attempts a second apart, then a rest of 30 s that doubles with each
 * spent allowance, up to five minutes. A permanently dead endpoint then costs
 * three requests per key every five minutes, an outage that ends recovers on
 * the first rest, and a blip repaints without waiting on the user.
 */
const POINT_RETRY_POLICY: RetryPolicy = {
  attempts: 3,
  retryDelayMs: () => 1_000,
  restMs: (round) => Math.min(300_000, 30_000 * 2 ** round),
};

/**
 * Required current-view work has not drained: reads running or queued, or a
 * selected tile waiting out a retry delay. A tile inside its rest does not
 * count (see `restingTiles`), and neither does a page waiting to be retried.
 */
const workPendingIn = (tiles: LoaderCounts, pages: LoaderCounts): boolean =>
  tiles.physical > 0 ||
  pages.physical > 0 ||
  tiles.queued > 0 ||
  pages.queued > 0 ||
  tiles.retrying > 0;

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
  let densityFraction = 1;
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
  /** The camera in the tiles' local frame: what all selection math reads. */
  let view: PreparedView | null = null;
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

  const nodes = new Map<string, HierarchyEntry>();
  const loadedPages = new Set<string>();
  const hierarchy: Hierarchy = { nodes, loadedPages };

  /** Before the first camera only the root page can be asked for. */
  const byFrontierPriority = (keyStrings: readonly string[]): string[] =>
    view === null ? [...keyStrings] : frontierOrder(keyStrings, nodes, view);

  const residency = createPayloadResidency({ cacheBytes });

  // The byte share converts to a point ceiling at the measured bytes per point
  // of resident tiles. Whatever budget the host asks for, this ceiling is what
  // keeps selection inside GPU memory.
  let externalMemoryBudgetBytes =
    options.memoryBudgetBytes === undefined
      ? defaultMemoryBudgetBytes()
      : Math.floor(
          finiteAtLeast("memoryBudgetBytes", options.memoryBudgetBytes, 0),
        );

  // An inactive controller holds no share: it has dropped its resident tiles
  // and must not reselect until reactivated.
  const memoryBudgetBytes = (): number =>
    active ? externalMemoryBudgetBytes : 0;

  const memoryCeilingPoints = (): number => {
    const totals = residency.totals();
    return pointCapacity(
      memoryBudgetBytes(),
      totals.residentPoints,
      totals.residentBytes,
    );
  };

  let selection: PointSelection = EMPTY_SELECTION;
  /** Increments only when the selected key set changes. */
  let targetRevision = 0;
  let selectionGeneration = 0;
  let plan: DrawPlan = EMPTY_DRAW_PLAN;
  /** Increments only when at least one tile prefix changes. */
  let drawPlanRevision = 0;
  /** See `LodSelectionStats.projectedSpacingCssPx`. */
  let largestTerminalSpacingCssPx: number | null = null;
  let terminalSpacingInputs:
    | Parameters<typeof largestTerminalSpacing>[0]
    | null = null;
  let terminalSpacingDirty = false;

  /** Fixed presentation only needs this diagnostic when a host asks for it. */
  const readTerminalSpacing = (): number | null => {
    if (!terminalSpacingDirty) return largestTerminalSpacingCssPx;
    largestTerminalSpacingCssPx =
      terminalSpacingInputs === null
        ? null
        : largestTerminalSpacing(terminalSpacingInputs);
    terminalSpacingDirty = false;
    // A resolved diagnostic retains only its scalar, not an old frontier.
    terminalSpacingInputs = null;
    return largestTerminalSpacingCssPx;
  };

  /**
   * What a submitted tile actually draws: the planned prefix, clamped to the
   * payload the consumer holds. The plan is allocated from hierarchy point
   * counts, which need not match the decoded tile.
   */
  const drawnPointsOf = (keyString: string, tile: TileData): number =>
    Math.min(
      tile.pointCount,
      plan.prefixes.get(keyString) ?? 0,
      options.getDrawnPointCount?.(keyString) ?? Infinity,
    );

  /** Hand the consumer the residency delta once per microtask burst. */
  let flushScheduled = false;
  const scheduleFlush = (): void => {
    if (flushScheduled) return;
    flushScheduled = true;
    queueMicrotask(() => {
      flushScheduled = false;
      if (disposed) return;
      const batch = residency.flush();
      if (batch === null) return;
      onTiles(batch);
      markWork();
      scheduleRender();
    });
  };

  const pages = createLoader<readonly NodeInfo[]>({
    load: (keyString, signal) =>
      source.nodes(keyFromString(keyString), { signal }),
    concurrency: hierarchyConcurrency,
    retry: POINT_RETRY_POLICY,
    // A page survives deactivation in the hierarchy, so cancelling one would
    // only buy a refetch of the same bytes.
    cancelGraceMs: Number.POSITIVE_INFINITY,
    onLoaded: (keyString, infos) => {
      // Resolve a deferred diagnostic before changing its hierarchy inputs.
      // This avoids copying the hierarchy merely to preserve an old snapshot.
      readTerminalSpacing();
      loadedPages.add(keyString);
      for (const info of infos) {
        const nodeKeyString = keyToString(info.key);
        nodes.set(nodeKeyString, {
          pointCount: info.pointCount,
          bounds: info.bounds,
          spacing: info.spacing,
          children: info.children ?? null,
          pageRef: info.pageRef === true,
          tileBytes: nodes.get(nodeKeyString)?.tileBytes,
        });
      }
      // A page reshapes the subtree below it, so the next page queue is the
      // one selection derives from it.
      runSelection();
    },
    onFailed: (_keyString, error) => onError(error),
    onChange: markWork,
  });

  const tiles = createLoader<TileData>({
    load: (keyString, signal) =>
      source.loadTile(keyFromString(keyString), { signal }),
    concurrency: fetchConcurrency,
    retry: POINT_RETRY_POLICY,
    // One selection interval, in which an orbit or pan crossing back over a
    // recent tile adopts its read instead of cancelling and refetching it.
    cancelGraceMs: selectionDelayMs,
    onLoaded: (keyString, tile) => {
      const entry = nodes.get(keyString);
      const bytes = tileBytes(tile);
      const exceedsEstimate =
        entry !== undefined && bytes > nodePayloadBytes(entry);
      if (entry !== undefined)
        nodes.set(keyString, { ...entry, tileBytes: bytes });
      // A source may decode more points than its hierarchy advertised. Learn
      // that cost before admitting the payload, and keep it after cache
      // eviction so another selection cannot refetch the same oversized tile.
      if (exceedsEstimate && selection.target.has(keyString)) {
        runSelection(undefined, { keyString, tile });
        return;
      }
      // Cancellation is advisory, so a cancelled read can still deliver. If
      // the key was reselected while it ran, this payload is exactly what the
      // selection is waiting for.
      if (selection.target.has(keyString) && !residency.isResident(keyString)) {
        residency.hold(keyString, tile);
        scheduleFlush();
      } else {
        residency.park(keyString, tile);
      }
    },
    onFailed: (_keyString, error) => onError(error),
    onChange: markWork,
  });

  /** Ask for the pages selection needs, highest priority first. */
  const queuePages = (keyStrings: readonly string[]): void =>
    pages.want(
      byFrontierPriority(
        [...new Set(keyStrings)].filter(
          (keyString) => !loadedPages.has(keyString),
        ),
      ),
    );

  /** Re-plan the prefixes, and hand them on only when one of them moved. */
  const updateDrawPlan = (): void => {
    const next = planDraw(selection.target, nodes, densityFraction);
    const changed = !samePrefixes(plan.prefixes, next.prefixes);
    plan = next;
    if (!changed) return;
    drawPlanRevision += 1;
    onDrawPlan({
      entries: [...plan.prefixes].map(([keyString, pointCount]) => ({
        key: keyFromString(keyString),
        pointCount,
      })),
    });
  };

  /**
   * Size Auto points for the density the selection is converging on, not
   * the subset that has finished loading: the diameter moves once, with the
   * selection, and tile arrivals only add detail.
   */
  const updateAutoDiameter = (): void => {
    // Keep the camera and frontier of this update: a later debounced camera
    // must not change the diagnostic before another selection or density pass.
    terminalSpacingInputs =
      view === null
        ? null
        : {
            target: selection.target,
            hierarchy,
            view,
            refinementCutoffPx,
            plan,
          };
    terminalSpacingDirty = true;
    if (presentation.mode !== "auto") return;
    const spacing = readTerminalSpacing();
    if (spacing !== null)
      emitDiameter(autoDiameterCssPx(presentation, spacing));
  };

  const runSelection = (
    seed?: ReadonlySet<string>,
    arrival?: { readonly keyString: string; readonly tile: TileData },
  ): void => {
    if (disposed || !active || view === null) return;
    const previous = selection.target;
    selection = selectPoints({
      hierarchy,
      view,
      pointBudget: currentBudget(),
      memoryBudgetBytes: memoryBudgetBytes(),
      refinementCutoffPx,
      previous,
      seed,
    });
    const target = selection.target;
    if (
      previous.size !== target.size ||
      [...target].some((keyString) => !previous.has(keyString))
    ) {
      targetRevision += 1;
    }
    selectionGeneration += 1;
    updateDrawPlan();
    updateAutoDiameter();
    // The root page bootstraps the hierarchy, so it can never come back
    // through neededPages: that path needs a hierarchy entry, and only the
    // root page can create one. Without this, a failed bootstrap leaves the
    // controller with nothing to draw and no way to ask again. Level 0 sorts
    // to the front, and a page already held, in flight, or resting is not
    // read again, so this costs nothing on the normal path.
    queuePages(
      loadedPages.has(ROOT_KEY_STRING)
        ? selection.neededPages
        : [ROOT_KEY_STRING, ...selection.neededPages],
    );

    // Deselected tiles leave renderer residency immediately; their payloads
    // stay in the CPU cache.
    residency.releaseExcept(target);
    if (arrival !== undefined) {
      // Handle the just-decoded tile directly: parking it before selection
      // could evict it from a smaller CPU cache even when GPU residency fits.
      if (
        target.has(arrival.keyString) &&
        !residency.isResident(arrival.keyString)
      ) {
        residency.hold(arrival.keyString, arrival.tile);
      } else {
        residency.park(arrival.keyString, arrival.tile);
      }
    }

    // Reuse decoded payloads at once; ask for the rest, coarse levels first.
    // A key whose read is still running is adopted rather than read twice.
    const missing: string[] = [];
    for (const keyString of target) {
      if (residency.isResident(keyString)) continue;
      // Structural hierarchy nodes participate in selection but carry no tile.
      if (nodes.get(keyString)?.pointCount === 0) continue;
      if (residency.promote(keyString)) continue;
      missing.push(keyString);
    }
    scheduleFlush();
    tiles.want(byFrontierPriority(missing));
  };

  /**
   * Rerun selection after the budget moved. A larger budget that still covers
   * the current selection keeps it as a seed, so growth only adds detail.
   */
  const reselectForBudget = (previousBudget: number): void => {
    const nextBudget = currentBudget();
    runSelection(
      nextBudget > previousBudget &&
        selection.target.size > 0 &&
        selection.targetPoints <= nextBudget
        ? selection.target
        : undefined,
    );
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

  const clearSelection = (): void => {
    // Outstanding hierarchy pages may still land while the member is hidden.
    readTerminalSpacing();
    if (selection.target.size > 0) targetRevision += 1;
    selection = EMPTY_SELECTION;
  };

  const dropEverything = (): void => {
    readTerminalSpacing();
    // Every outstanding result becomes irrelevant, but the reads behind them
    // keep their slots until they settle: pretending otherwise is how a
    // source swap under a moving camera doubles real I/O.
    tiles.reset();
    pages.reset();
    nodes.clear();
    loadedPages.clear();
    residency.clear();
    clearSelection();
    updateDrawPlan();
  };

  /** Bootstrap path for the root page, used before any camera exists. */
  const queueRootPage = (): void => queuePages([ROOT_KEY_STRING]);

  // Bootstrap: an active controller loads hierarchy eagerly; an inactive one
  // waits until activation so a hidden cloud performs no hierarchy I/O.
  if (active) queueRootPage();

  /** Adopt the local restatement of the stored world camera, if usable. */
  const applyWorldView = (): void => {
    if (worldView === null || !modelMatrixUsable) return;
    view = prepareView(
      modelFrame ? viewInModelFrame(worldView, modelFrame) : worldView,
    );
    if (active) requestSelection();
  };

  return {
    setCamera(nextView) {
      if (disposed || !usableView(nextView)) return;
      // Hosts feed the camera on every render, so an unchanged view is a
      // no-op rather than a reason to rerun selection.
      if (worldView !== null && sameCameraView(worldView, nextView)) return;
      worldView = copyCameraView(nextView);
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
      // The same budget selects the same frontier: rerunning it would only
      // report work and ask for another allocation.
      if (currentBudget() === previousBudget) return;
      reselectForBudget(previousBudget);
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
      reselectForBudget(previousBudget);
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
      // to land: they survive deactivation in the hierarchy.
      pages.want([]);
      // Hiding aborts tile reads but keeps their physical slots: a hide/show
      // pair must adopt the read that is still running, not race it.
      tiles.abandon();

      clearSelection();
      selectionGeneration += 1;
      updateDrawPlan();

      // Nothing stays resident while hidden; the flush turns that into
      // removals for exactly the actors the consumer was last handed.
      residency.releaseExcept(selection.target);
      scheduleFlush();
    },

    governorInputs() {
      const tileCounts = tiles.counts();
      const pageCounts = pages.counts();
      return {
        workRevision,
        workPending: workPendingIn(tileCounts, pageCounts),
        memoryBudgetBytes: memoryBudgetBytes(),
        memoryCeilingPoints: memoryCeilingPoints(),
        projectedImportance: selection.rootSseCssPx,
        demandPoints: selection.targetPoints + selection.budgetSkippedPoints,
        physicalTileOperations: tileCounts.physical,
        physicalHierarchyOperations: pageCounts.physical,
      };
    },

    stats() {
      const tileCounts = tiles.counts();
      const pageCounts = pages.counts();
      const totals = residency.totals();
      let targetUndecodedTiles = 0;
      let restingTiles = 0;
      for (const keyString of selection.target) {
        if (
          nodes.get(keyString)?.pointCount !== 0 &&
          !residency.isDecoded(keyString)
        ) {
          targetUndecodedTiles += 1;
          if (tiles.resting(keyString)) restingTiles += 1;
        }
      }
      let drawnPoints = 0;
      for (const [keyString, tile] of residency.submitted()) {
        drawnPoints += drawnPointsOf(keyString, tile);
      }
      return {
        active,
        residentTiles: totals.residentTiles,
        residentPoints: totals.residentPoints,
        residentBytes: totals.residentBytes,
        decodedTiles: totals.residentTiles + totals.cachedTiles,
        decodedBytes: totals.residentBytes + totals.cachedBytes,
        cachedBytes: totals.cachedBytes,
        inFlight: tileCounts.wantedReading,
        queuedTiles: tileCounts.queued,
        physicalTileOperations: tileCounts.physical,
        hierarchyInFlight: pageCounts.reading,
        queuedPages: pageCounts.queued,
        physicalHierarchyOperations: pageCounts.physical,
        workRevision,
        workPending: workPendingIn(tileCounts, pageCounts),
        restingTiles,
        restingPages: pages.restingCount(),
        selectionPending: selectionTimer !== null,
        fetchConcurrency,
        hierarchyConcurrency,
        pointBudget: currentBudget(),
        densityFraction,
        drawnPoints,
        drawPlan: {
          revision: drawPlanRevision,
          pointBudget: Math.floor(selection.targetPoints * densityFraction),
          plannedPoints: plan.plannedPoints,
        },
        memoryBudgetBytes: memoryBudgetBytes(),
        memoryCeilingPoints: memoryCeilingPoints(),
        interactionDepth,
        presentation: {
          config: presentation,
          diameterCssPx,
        },
        cachedTiles: totals.cachedTiles,
        cacheBytes,
        refinementCutoffPx,
        selection: {
          generation: selectionGeneration,
          targetRevision,
          targetTiles: selection.target.size,
          targetPoints: selection.targetPoints,
          consideredNodes: selection.consideredNodes,
          availableNodes: selection.availableNodes,
          sseStoppedNodes: selection.sseStoppedNodes,
          budgetSkippedNodes: selection.budgetSkippedNodes,
          budgetSkippedPoints: selection.budgetSkippedPoints,
          projectedImportance: selection.rootSseCssPx,
          projectedSpacingCssPx: readTerminalSpacing(),
          targetUndecodedTiles,
        },
      };
    },

    pickPoint(pickView, cursorXCssPx, cursorYCssPx) {
      if (disposed || !active || !usableView(pickView)) return null;
      if (!modelMatrixUsable) return null;
      const localView = modelFrame
        ? viewInModelFrame(pickView, modelFrame)
        : pickView;
      const pickTiles: PickTile[] = [];
      for (const [keyString, tile] of residency.submitted()) {
        const pointCount = drawnPointsOf(keyString, tile);
        if (pointCount === 0) continue;
        pickTiles.push({
          origin: tile.origin,
          positions: tile.positions,
          pointCount,
          bounds: nodes.get(keyString)?.bounds,
        });
      }
      const result = pickPointInTiles(
        localView,
        cursorXCssPx,
        cursorYCssPx,
        pickTiles,
      );
      // The sweep solved on the model-local ray; hand the answer back in the
      // caller's world coordinates through the same matrix.
      return result?.status === "hit" && modelFrame
        ? {
            ...result,
            rayDepth: result.rayDepth * modelFrame.scale,
            scenePoint: scenePoint(
              ...transformPoint(modelFrame.matrix, result.scenePoint),
            ),
          }
        : result;
    },

    activeKeys: () => ({
      resident: [...residency.residentKeys()].sort(),
      submitted: [...residency.submitted().keys()].sort(),
    }),

    dispose() {
      if (disposed) return;
      clearSelectionTimer();
      interactionDepth = 0;
      // The consumer still owns everything the last flush handed it,
      // including tiles a flush this teardown cancels was about to remove.
      // Take all of it back or it keeps actors nothing will ever drop.
      const removed = residency.takeBack();
      dropEverything();
      disposed = true;
      if (removed.length > 0) {
        onTiles({ added: [], removed });
        scheduleRender();
      }
    },
  };
};
