/**
 * pointcloud-lod vtk.js example.
 *
 * Streams any COPC dataset — a local file through `File.slice()` range reads,
 * a remote URL through HTTP Range — using the same three pieces the trame
 * bridge wires together: one LOD controller per cloud, one renderer adapter,
 * and one view governor owning the point budget for the whole view. Every
 * knob (budget mode, frame-time targets, maximum points, projection) is a
 * runtime control, so a different cloud needs no code change.
 *
 * The governor never schedules a frame: this page paints, times the paint,
 * reports it, and asks `needsFrame()` whether another one is owed. That is the
 * entire adaptive contract, and it is why every render goes through the
 * interactor — its own animation loop paints the frames of a gesture, which
 * are exactly the frames the moving regime has to measure.
 */

import "@kitware/vtk.js/Rendering/Profiles/Geometry";
import vtkFullScreenRenderWindow from "@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow";
import vtkInteractorStyleManipulator from "@kitware/vtk.js/Interaction/Style/InteractorStyleManipulator";
import vtkMouseCameraTrackballPanManipulator from "@kitware/vtk.js/Interaction/Manipulators/MouseCameraTrackballPanManipulator";
import vtkMouseCameraTrackballRotateManipulator from "@kitware/vtk.js/Interaction/Manipulators/MouseCameraTrackballRotateManipulator";
import vtkCompositeMouseManipulator from "@kitware/vtk.js/Interaction/Manipulators/CompositeMouseManipulator";
import vtkCompositeCameraManipulator from "@kitware/vtk.js/Interaction/Manipulators/CompositeCameraManipulator";
import macro from "@kitware/vtk.js/macros";

import {
  ROOT_KEY,
  createCopcTileSource,
  createLodController,
  createViewGovernor,
  type CameraView,
  type LodController,
  type MotionReference,
  type TileSource,
  type ViewGovernor,
  type ViewGovernorMember,
  type ViewGovernorOptions,
  type ViewGovernorStats,
} from "../../src";
import {
  createRendererAdapter,
  type RendererAdapter,
} from "../../src/rendererAdapter";

const element = <T extends Element>(selector: string): T => {
  const found = document.querySelector<T>(selector);
  if (!found) throw new Error(`The example page is missing ${selector}`);
  return found;
};

const viewer = element<HTMLElement>("#viewer");
const sceneSelect = element<HTMLSelectElement>("#cloud-preset");
const fileInput = element<HTMLInputElement>("#cloud-file");
const urlInput = element<HTMLInputElement>("#cloud-url");
const loadUrlButton = element<HTMLButtonElement>("#load-url");
const budgetModeSelect = element<HTMLSelectElement>("#budget-mode");
const projectionSelect = element<HTMLSelectElement>("#projection");
const fixedControls = element<HTMLElement>("#fixed-controls");
const adaptiveControls = element<HTMLElement>("#adaptive-controls");
const pointBudgetInput = element<HTMLInputElement>("#point-budget");
const movingTargetInput = element<HTMLInputElement>("#moving-target");
const stationaryTargetInput = element<HTMLInputElement>("#stationary-target");
const maxPointsInput = element<HTMLInputElement>("#max-points");
const resetViewButton = element<HTMLButtonElement>("#reset-view");
const regimeBadge = element<HTMLElement>("#regime");
const message = element<HTMLOutputElement>("#message");
const stats = element<HTMLElement>("#stats");

const fullScreen = vtkFullScreenRenderWindow.newInstance({
  rootContainer: viewer,
  background: [0.035, 0.055, 0.075],
});
const renderer = fullScreen.getRenderer();
const renderWindow = fullScreen.getRenderWindow();
const interactor = renderWindow.getInteractor();
const camera = renderer.getActiveCamera();

/** The world's up axis. Point clouds here are z-up, and orbiting keeps it. */
const WORLD_UP: [number, number, number] = [0, 0, 1];

/** Fraction of the eye-to-focus distance a full viewport-height drag covers. */
const DOLLY_PER_VIEWPORT = 1.6;
/** Zoom ratio per wheel notch. */
const DOLLY_PER_NOTCH = 1.12;

/**
 * The cursor, in the display coordinates vtk.js works in.
 *
 * The interactor caches a mouse position only while a button is held, so a
 * wheel turn with nothing pressed — the ordinary case — reaches a manipulator
 * with no position at all, and vtk.js's own zoom-to-mouse silently does
 * nothing. Tracking the pointer here is what makes the wheel able to aim.
 */
let pointer: { x: number; y: number } | null = null;
viewer.addEventListener("pointermove", (event: PointerEvent) => {
  const canvas = fullScreen.getApiSpecificRenderWindow().getCanvas();
  if (!canvas) return;
  const bounds = canvas.getBoundingClientRect();
  if (bounds.width === 0 || bounds.height === 0) return;
  pointer = {
    x: (canvas.width / bounds.width) * (event.clientX - bounds.left),
    y:
      (canvas.height / bounds.height) *
      (bounds.height - event.clientY + bounds.top),
  };
});

/**
 * Drag-to-zoom, toward the point the gesture started on.
 *
 * The step is a fraction of the distance to what is being looked at, so it
 * behaves the same on a village and on a valley. vtk.js's own zoom manipulator
 * scales its step by the far clipping plane instead, which makes the speed
 * depend on how deep the scene is rather than on how close the camera is —
 * measured across two clouds, the same drag moved 45% of the scene radius on
 * one and 7% on the other.
 */
const createDollyManipulator = (initialValues: object): any => {
  const publicAPI = {};
  const model = { ...initialValues };
  macro.obj(publicAPI, model);
  vtkCompositeMouseManipulator.extend(publicAPI, model, initialValues);
  vtkCompositeCameraManipulator.extend(publicAPI, model, initialValues);

  // Where the gesture started, and how much of it has been applied. The zoom
  // is computed from the cursor's total displacement rather than accumulated
  // per event, so a move the interactor coalesces or drops while it is busy
  // painting cannot leave the gesture short of where the cursor actually is.
  let start = { x: 0, y: 0 };
  let applied = 1;

  const api = publicAPI as {
    onButtonDown: (i: any, r: any, p: { x: number; y: number }) => void;
    onMouseMove: (i: any, r: any, p: { x: number; y: number } | null) => void;
    onScroll: (
      i: any,
      r: any,
      delta: number,
      p?: { x: number; y: number } | null,
    ) => void;
  };
  api.onButtonDown = (_interactor, _renderer, position) => {
    start = position;
    applied = 1;
  };
  api.onScroll = (scrollInteractor, renderer, delta, position) => {
    if (!delta) return;
    const at = position ?? pointer;
    if (!at) return;
    vtkInteractorStyleManipulator.dollyToPosition(
      DOLLY_PER_NOTCH ** -delta,
      at,
      renderer,
      scrollInteractor,
    );
  };
  api.onMouseMove = (interactor, renderer, position) => {
    if (!position) return;
    const height = interactor.getView().getViewportSize(renderer)[1];
    // Exponential in the drag: equal drags are equal ratios, and dragging
    // back undoes exactly what dragging out did. Up zooms in.
    const wanted = Math.exp(
      ((position.y - start.y) / Math.max(height, 1)) * DOLLY_PER_VIEWPORT,
    );
    const step = wanted / applied;
    applied = wanted;
    // Toward where the gesture began, so the drag pulls that point in rather
    // than whatever happens to be at the centre.
    vtkInteractorStyleManipulator.dollyToPosition(
      step,
      start,
      renderer,
      interactor,
    );
  };
  return publicAPI;
};

/**
 * Left drag pans, right drag orbits, middle drag and the wheel zoom.
 *
 * The stock trackball rolls the camera freely, which loses the horizon on the
 * first diagonal drag over a landscape. Orbiting about the focal point with a
 * fixed world up keeps the scene the way up the data is.
 */
const interactorStyle = vtkInteractorStyleManipulator.newInstance();
interactorStyle.addMouseManipulator(
  vtkMouseCameraTrackballPanManipulator.newInstance({ button: 1 }),
);
interactorStyle.addMouseManipulator(
  vtkMouseCameraTrackballRotateManipulator.newInstance({
    button: 3,
    useWorldUpVec: true,
    worldUpVec: WORLD_UP,
    useFocalPointAsCenterOfRotation: true,
  }),
);
// Middle drag and the wheel both zoom toward the cursor: the camera moves
// along the ray through whatever is under the pointer, so that point stays
// roughly put and the view closes in on it rather than on the screen centre.
interactorStyle.addMouseManipulator(
  createDollyManipulator({ button: 2, scrollEnabled: true }),
);
interactor.setInteractorStyle(interactorStyle);

/** The adaptive floor is stated so an out-of-order maximum is caught here. */
const ADAPTIVE_MIN_BUDGET = 200_000;
/** How long the camera must hold still before inferred motion is released. */
const MOTION_DEBOUNCE_MS = 250;
/**
 * Relative, so the same threshold works for a cloud in metres and one in
 * degrees: recomputing the camera product every frame jitters in the last
 * bits, and no real camera change is that small.
 */
const MOTION_RELATIVE_EPSILON = 1e-9;
/** The panel is an instrument, so it repaints on its own slower clock. */
const DIAGNOSTICS_INTERVAL_MS = 100;

type BudgetMode = "adaptive" | "fixed";
type Projection = "perspective" | "orthographic";

let controller: LodController | null = null;
let adapter: RendererAdapter | null = null;
let governor: ViewGovernor | null = null;
let governorKey: string | null = null;
let member: ViewGovernorMember | null = null;
let explicitMotion: MotionReference | null = null;
let inferredMotion: MotionReference | null = null;
let inferredMotionTimer: ReturnType<typeof setTimeout> | null = null;
let lastRenderedView: CameraView | null = null;
let loadedName = "";
let loadedPointCount = 0;
let frameQueued = false;
let frameStartedAt: number | null = null;
let lastFrameMs = 0;
// When set, every frame reports this duration instead of its measured one, so
// a check can drive the budget loop at a frame time this machine cannot
// actually produce. It replaces the measurement rather than being added
// alongside it: the loop must see one number per frame.
let syntheticFrameMs: number | null = null;
/**
 * The scale the host wants, which is not always the window's own: loading a
 * cloud builds a new adapter, and reading `window.devicePixelRatio` there
 * would silently discard a ratio the host had set for this view.
 */
let currentDevicePixelRatio = window.devicePixelRatio;
let loadGeneration = 0;

const setMessage = (text: string, error = false): void => {
  message.textContent = text;
  message.classList.toggle("error", error);
};

const budgetMode = (): BudgetMode =>
  budgetModeSelect.value === "fixed" ? "fixed" : "adaptive";

const numberFrom = (input: HTMLInputElement): number | null => {
  const text = input.value.trim();
  if (!text) return null;
  const value = Number(text);
  return Number.isFinite(value) ? value : null;
};

const fixedPointBudget = (): number => {
  const value = numberFrom(pointBudgetInput);
  return value !== null && value >= 1 ? Math.floor(value) : 2_000_000;
};

const transpose = (matrix: ArrayLike<number>): number[] => [
  matrix[0]!,
  matrix[4]!,
  matrix[8]!,
  matrix[12]!,
  matrix[1]!,
  matrix[5]!,
  matrix[9]!,
  matrix[13]!,
  matrix[2]!,
  matrix[6]!,
  matrix[10]!,
  matrix[14]!,
  matrix[3]!,
  matrix[7]!,
  matrix[11]!,
  matrix[15]!,
];

/**
 * The projection mode is read from the camera, never guessed from a number:
 * the library projects a world spacing to pixels by a different law per mode,
 * and a parallel camera's view angle is meaningless rather than absent.
 */
const cameraView = (): CameraView => {
  const width = Math.max(1, viewer.clientWidth);
  const height = Math.max(1, viewer.clientHeight);
  const common = {
    viewProj: transpose(
      camera.getCompositeProjectionMatrix(width / height, -1, 1),
    ),
    position: [...camera.getPosition()] as [number, number, number],
    viewportHeightCssPx: height,
  };
  return camera.getParallelProjection()
    ? {
        ...common,
        projection: "orthographic",
        parallelScale: camera.getParallelScale(),
      }
    : {
        ...common,
        projection: "perspective",
        fovY: (camera.getViewAngle() * Math.PI) / 180,
      };
};

// ---------------------------------------------------------------------------
// Camera-motion inference
//
// The same policy the trame bridge applies at its rendered-camera boundary
// (`classifyCameraMotion` in pointCloudLod.js): compare the camera actually
// handed to LOD against the previous one, ignore floating-point jitter, hold
// one inferred reference for the whole burst, and release it after a quiet
// debounce. The bridge's copy is bound to its per-view registry bookkeeping,
// so it cannot be imported; this is that rule with one view's worth of state.
// ---------------------------------------------------------------------------

const movedBeyondJitter = (previous: number, next: number): boolean =>
  Math.abs(previous - next) >
  MOTION_RELATIVE_EPSILON * Math.max(1, Math.abs(previous), Math.abs(next));

/**
 * World-to-clip entries describing where the camera is looking, row-major as
 * `cameraView()` transposes it (index = row * 4 + column).
 *
 * The clip-z row is deliberately left out, exactly as the bridge leaves out
 * its own: a host folds a depth remap derived from the scene's visible bounds
 * into that row, so a tile arriving or being evicted rewrites those four
 * numbers while the camera stands perfectly still. Every camera move shows up
 * in the x, y and w rows; the one motion that lives only in clip z — dollying
 * an orthographic camera along its view axis — shows up in the eye point
 * instead, which is compared alongside.
 *
 * `viewProj` is column-major (`index = column * 4 + row`), so the clip-z row
 * is entries 2, 6, 10 and 14 — not 8..11, which are column 2 and move under
 * ordinary rotation.
 *
 * This page hands the projection a fixed z range, so the row never moves here
 * and excluding it changes nothing on screen. It is here because the example
 * exists to show a host what to implement, and a host reading its own
 * composite matrix will have the remap folded in.
 */
const MOTION_MATRIX_INDICES = [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15];

/** Everything about the camera that changes what LOD selects. */
const motionScalars = (view: CameraView): number[] => [
  ...MOTION_MATRIX_INDICES.map((index) => Number(view.viewProj[index])),
  ...view.position,
  view.viewportHeightCssPx,
  view.projection === "orthographic" ? view.parallelScale : view.fovY,
];

const cameraMoved = (
  previous: CameraView | null,
  next: CameraView,
): boolean => {
  // The first camera a view supplies is the baseline, not a movement.
  if (!previous) return false;
  if (previous.projection !== next.projection) return true;
  const before = motionScalars(previous);
  const after = motionScalars(next);
  return before.some((value, index) => movedBeyondJitter(value, after[index]!));
};

const classifyCameraMotion = (view: CameraView): void => {
  const moved = cameraMoved(lastRenderedView, view);
  lastRenderedView = view;
  if (!moved || !governor) return;
  // One reference per burst, never one per frame: the governor restarts its
  // moving track whenever the first reference is taken, so a per-frame
  // reference would keep resetting the window it needs to learn from.
  if (!inferredMotion) inferredMotion = governor.beginMotion("inferred");
  if (inferredMotionTimer !== null) clearTimeout(inferredMotionTimer);
  inferredMotionTimer = setTimeout(() => {
    inferredMotionTimer = null;
    inferredMotion?.release();
    inferredMotion = null;
    // The settled regime cannot refine quality it never measures, so hand it
    // one frame to start from.
    scheduleRender();
  }, MOTION_DEBOUNCE_MS);
};

// ---------------------------------------------------------------------------
// Budget mode and the view governor
// ---------------------------------------------------------------------------

/**
 * The governor fixes its options at construction and throws on an unusable
 * one rather than clamping, so the panel is validated here and a changed
 * target replaces the instance — the same reconciliation the bridge does,
 * without the multi-cloud bookkeeping.
 */
const readGovernorOptions = (): ViewGovernorOptions | null => {
  const interactionTargetMs = numberFrom(movingTargetInput);
  const stationaryTargetMs = numberFrom(stationaryTargetInput);
  const maxBudget = numberFrom(maxPointsInput);
  if (
    interactionTargetMs === null ||
    interactionTargetMs <= 0 ||
    stationaryTargetMs === null ||
    stationaryTargetMs <= 0 ||
    (maxBudget !== null && maxBudget < ADAPTIVE_MIN_BUDGET)
  ) {
    return null;
  }
  return {
    minBudget: ADAPTIVE_MIN_BUDGET,
    interactionTargetMs,
    stationaryTargetMs,
    // The default allows VTK 70% of a host frame, which is right for a view
    // compositing a basemap and video underneath it. This page paints nothing
    // but the point cloud, so VTK's whole-frame allowance is the whole frame;
    // leaving the default would normalise every honest 33 ms frame up to 47 ms
    // and walk the budget down to its floor while nothing was ever late.
    vtkFrameFraction: 1,
    ...(maxBudget === null ? {} : { maxBudget: Math.floor(maxBudget) }),
  };
};

/**
 * Which number the controller draws to, reconciled as a whole: adaptive means
 * the governor owns it, anything else means the panel does. Registering
 * distributes, so the governor's allocation reaches the controller inside this
 * call rather than a frame later.
 */
const applyBudgetMode = (): void => {
  if (!controller) return;
  if (budgetMode() === "adaptive" && governor) {
    member ??= governor.register({
      id: loadedName || "cloud",
      setPointBudget: (budget) => controller?.setPointBudget(budget),
    });
    return;
  }
  member?.release();
  member = null;
  controller.setPointBudget(fixedPointBudget());
};

const syncGovernor = (): void => {
  const adaptive = budgetMode() === "adaptive";
  const wanted = adaptive ? readGovernorOptions() : null;
  if (adaptive && !wanted) {
    // An unusable option would throw out of the constructor; keep drawing to
    // the governor the panel last described.
    setMessage(
      "Frame-time targets must be above 0 ms and any maximum at least " +
        `${ADAPTIVE_MIN_BUDGET.toLocaleString()} points.`,
      true,
    );
    return;
  }
  const key = wanted === null ? null : JSON.stringify(wanted);
  if (key === governorKey) return;
  member?.release();
  member = null;
  governor?.dispose();
  governor = wanted === null ? null : createViewGovernor(wanted);
  governorKey = key;
  // Motion belongs to the view, not to the instance: a governor replaced
  // mid-gesture inherits what the view was holding, or the swap reads as the
  // camera having stopped and quality jumps mid-drag.
  if (explicitMotion) {
    explicitMotion.release();
    explicitMotion = governor?.beginMotion("explicit") ?? null;
  }
  if (inferredMotion) {
    inferredMotion.release();
    inferredMotion = governor?.beginMotion("inferred") ?? null;
  }
  applyBudgetMode();
};

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

const points = (value: number): string => Math.round(value).toLocaleString();
const millis = (value: number | null): string =>
  value === null ? "—" : `${value.toFixed(1)} ms`;
const mib = (bytes: number): string => `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
/**
 * One diagnostics line, kept as label and value rather than as padded text.
 *
 * The panel refreshes ten times a second, and every value in it changes width
 * as it changes: a wrapped line would reflow the rows below it and the table
 * would never sit still long enough to read. Laying the two out as columns
 * that clip instead of wrap fixes every row's position for as long as the
 * section's shape holds.
 */
type StatRow = { label: string; value: string };
type StatSection = { title: string; rows: StatRow[] };

const row = (label: string, value: string): StatRow => ({ label, value });

const governorLines = (view: ViewGovernorStats): StatSection => {
  const cloud = view.members[0] ?? null;
  const adjustment = view.lastAdjustment;
  return {
    title: "Budget chain",
    rows: [
      row("regime", `${view.regime}${view.motion.settling ? " settling" : ""}`),
      row("motion", view.motion.source ?? "none"),
      row(
        "references",
        `${view.motion.explicitReferences} explicit, ` +
          `${view.motion.inferredReferences} inferred`,
      ),
      row("target", millis(view.targetFrameTimeMs)),
      row("estimate", `${millis(view.estimateMs)} of ${view.samples}`),
      row("track budget", points(view.trackBudget)),
      row(
        "maximum",
        view.configuredMaxPoints === null
          ? "none configured"
          : points(view.configuredMaxPoints),
      ),
      row(
        "memory ceiling",
        view.memoryCeilingPoints === null
          ? "not reported"
          : points(view.memoryCeilingPoints),
      ),
      row("aggregate", points(view.aggregateBudget)),
      row("cloud share", cloud ? points(cloud.effectiveBudget) : "no member"),
      row("constraint", view.activeConstraint),
      row(
        "adjustment",
        adjustment
          ? `${adjustment.direction} · ${adjustment.reason}`
          : "none yet",
      ),
      row(
        "moved budget",
        adjustment
          ? `${points(adjustment.fromBudget)} → ${points(adjustment.toBudget)}`
          : "—",
      ),
      row(
        "physical work",
        `${view.physicalTileOperations} tile, ` +
          `${view.physicalHierarchyOperations} page`,
      ),
      row("needs frame", view.needsFrame ? "yes" : "no"),
    ],
  };
};

const cloudLines = (): StatSection | null => {
  const cloud = controller?.stats();
  const drawn = adapter?.stats();
  if (!cloud || !drawn) return null;
  const { selection } = cloud;
  return {
    title: "Cloud",
    rows: [
      row("source", loadedName),
      row("points", points(loadedPointCount)),
      row(
        "selection",
        `${points(selection.targetPoints)} in ${selection.targetTiles} tiles`,
      ),
      row("point budget", points(cloud.pointBudget)),
      row("importance", selection.projectedImportance.toFixed(2)),
      row(
        "resident",
        `${points(cloud.residentPoints)} in ${cloud.residentTiles} tiles`,
      ),
      row("decoded", `${mib(cloud.decodedBytes)} in ${cloud.cachedTiles}`),
      row(
        "tile work",
        `${cloud.inFlight} wanted, ${cloud.queuedTiles} queued, ` +
          `${cloud.physicalTileOperations} physical`,
      ),
      row(
        "page work",
        `${cloud.hierarchyInFlight} wanted, ${cloud.queuedPages} queued, ` +
          `${cloud.physicalHierarchyOperations} physical`,
      ),
      row("drawn", `${points(drawn.drawnPoints)} in ${drawn.drawnTiles} tiles`),
      row("gpu", `${mib(drawn.gpuResidentBytes)} resident`),
      row("point diameter", `${drawn.diameterCssPx.toFixed(2)} css px`),
      row("last frame", millis(lastFrameMs)),
    ],
  };
};

const updateRegimeBadge = (view: ViewGovernorStats | null): void => {
  if (!controller) {
    regimeBadge.dataset.regime = "idle";
    regimeBadge.textContent = "no cloud";
    return;
  }
  if (!view) {
    regimeBadge.dataset.regime = "fixed";
    regimeBadge.textContent = "fixed budget";
    return;
  }
  regimeBadge.dataset.regime = view.regime;
  regimeBadge.textContent =
    view.regime === "interaction"
      ? `moving · ${view.motion.source ?? "settling"}`
      : "settled";
};

/**
 * Values are written into elements that already exist, and the table is only
 * rebuilt when its rows actually differ — otherwise ten rebuilds a second
 * would drop any text the reader had selected, and would lose the tooltip on
 * whichever row they were hovering.
 */
let renderedShape = "";
const renderedValues = new Map<string, HTMLElement>();

const renderStats = (sections: StatSection[]): void => {
  const shape = sections
    .map(({ title, rows }) => `${title}:${rows.map((r) => r.label).join(",")}`)
    .join("|");
  if (shape !== renderedShape) {
    renderedShape = shape;
    renderedValues.clear();
    stats.replaceChildren(
      ...sections.flatMap(({ title, rows }) => {
        const heading = document.createElement("h2");
        heading.textContent = title;
        const grid = document.createElement("dl");
        for (const { label } of rows) {
          const name = document.createElement("dt");
          name.textContent = label;
          const value = document.createElement("dd");
          renderedValues.set(`${title}/${label}`, value);
          grid.append(name, value);
        }
        return [heading, grid];
      }),
    );
  }
  for (const { title, rows } of sections) {
    for (const { label, value } of rows) {
      const cell = renderedValues.get(`${title}/${label}`);
      if (!cell || cell.textContent === value) continue;
      cell.textContent = value;
      // The column clips rather than wraps, so the full text has to stay
      // reachable for the rows that overflow it — a source URL, mostly.
      cell.title = value;
    }
  }
};

const updateDiagnostics = (): void => {
  const view = governor?.stats() ?? null;
  updateRegimeBadge(view);
  renderStats(
    [
      view
        ? governorLines(view)
        : { title: "Budget chain", rows: [row("mode", "fixed budget")] },
      cloudLines(),
    ].filter((section): section is StatSection => section !== null),
  );
};

// ---------------------------------------------------------------------------
// Frame loop
// ---------------------------------------------------------------------------

/**
 * Coalescing render request handed to the controller and the adapter. The
 * interactor paints its own animation frames, so a second paint in the same
 * frame would double the cost the budget loop is measuring.
 */
/**
 * The scene's extent changed, so the clipping range no longer bounds it.
 *
 * Tiles arrive after the camera is framed, so a range computed when the
 * renderer was empty clips away everything that streams in afterwards — the
 * view stays black until some gesture makes an interactor style recompute it.
 * Recomputing costs a pass over the actors, so it is done when the scene
 * actually changed rather than on every frame.
 */
let clippingDirty = true;

/** Where loading the current cloud framed it, so the view can go back. */
let framing: {
  center: number[];
  radius: number;
  bounds: [number, number, number, number, number, number];
} | null = null;

const scheduleRender = (): void => {
  if (frameQueued) return;
  frameQueued = true;
  requestAnimationFrame(() => {
    frameQueued = false;
    if (interactor.isAnimating()) return;
    if (clippingDirty && renderer.getActors().length > 0) {
      clippingDirty = false;
      renderer.resetCameraClippingRange();
    }
    frameStartedAt = performance.now();
    interactor.render();
    // Nothing painted (a render was already in progress): drop the stamp so it
    // cannot be charged to the next frame.
    frameStartedAt = null;
  });
};

const updateMember = (): void => {
  if (!controller || !adapter) return;
  const cloud = controller.stats();
  adapter.setResourceCeilingBytes(cloud.memoryBudgetBytes);
  // The governor needs the memory ceiling to bound the aggregate before it
  // splits it, and the physical work counts to know whether another frame is
  // still worth painting.
  member?.update({
    active: true,
    projectedImportance: cloud.selection.projectedImportance,
    memoryCeilingPoints: cloud.memoryCeilingPoints,
    physicalTileOperations: cloud.physicalTileOperations,
    physicalHierarchyOperations: cloud.physicalHierarchyOperations,
  });
};

// Every paint the interactor drives ends here — ours and the ones its own
// animation loop runs during a gesture — so one place measures the frame,
// feeds the rendered camera to LOD, and asks whether another frame is owed.
interactor.onRenderEvent(() => {
  const startedAt = frameStartedAt;
  frameStartedAt = null;
  if (startedAt !== null) {
    lastFrameMs = syntheticFrameMs ?? performance.now() - startedAt;
  }
  const view = cameraView();
  controller?.setCamera(view);
  updateMember();
  classifyCameraMotion(view);
  if (startedAt !== null) {
    // Host frame and VTK frame are the same measurement here because this page
    // paints nothing else. A host with a basemap under the canvas reports the
    // whole frame as hostFrameMs and the VTK paint alone as vtkFrameMs.
    governor?.recordHostFrame({
      hostFrameMs: lastFrameMs,
      vtkFrameMs: lastFrameMs,
    });
  }
  if (governor?.needsFrame()) scheduleRender();
});

interactor.onStartAnimation(() => {
  explicitMotion?.release();
  explicitMotion = governor?.beginMotion("explicit") ?? null;
  controller?.beginInteraction();
});

interactor.onAnimation(() => {
  frameStartedAt = performance.now();
});

interactor.onEndAnimation(() => {
  explicitMotion?.release();
  explicitMotion = null;
  controller?.endInteraction();
  // The interactor paints once more immediately after this event; that paint
  // is a real frame and gets timed like any other.
  frameStartedAt = performance.now();
});

// ---------------------------------------------------------------------------
// Cloud lifecycle
// ---------------------------------------------------------------------------

const disposeCloud = (): void => {
  // Teardown supersedes any load still in flight. Without this a dispose
  // landing between the source opening and the controller being built is
  // simply undone by the load it was meant to cancel, which then wires a
  // controller into a page that asked for nothing.
  loadGeneration += 1;
  member?.release();
  member = null;
  controller?.dispose();
  adapter?.dispose();
  controller = null;
  adapter = null;
  loadedName = "";
  loadedPointCount = 0;
  framing = null;
  // Framing the next cloud is not motion the user asked for.
  lastRenderedView = null;
};

const VIEW_ANGLE = 38;
/** Degrees above the horizon the opening view looks down from. */
const START_ELEVATION = 60;
/** Degrees around the up axis, so the opening view is not axis-aligned. */
const START_AZIMUTH = -60;
const RADIANS = Math.PI / 180;

/**
 * Frame the cloud and put the orbit point in the middle of it.
 *
 * The centre comes from the root tile's points rather than from the root
 * node's bounds, because a COPC root node is a cube spanning the octree, not
 * the data: for a tile far wider than it is tall, the cube's centre floats
 * high above the ground and every orbit would swing about a point in the sky.
 * The root tile is a uniform subsample of the whole cloud, so its extent is a
 * good estimate of the real one for the price of a tile that is about to be
 * read anyway.
 */
const frameRoot = async (source: TileSource): Promise<void> => {
  const entries = await source.nodes(ROOT_KEY);
  const root = entries.find(
    ({ key }) => key.level === 0 && key.x === 0 && key.y === 0 && key.z === 0,
  );
  if (!root) throw new Error("The COPC hierarchy has no root node");

  const { origin, positions, pointCount } = await source.loadTile(ROOT_KEY);
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < pointCount; i += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      const value = origin[axis]! + positions[i * 3 + axis]!;
      if (value < min[axis]!) min[axis] = value;
      if (value > max[axis]!) max[axis] = value;
    }
  }
  // An empty or degenerate root leaves the node's bounds as the only estimate.
  const usable = pointCount > 0 && min.every(Number.isFinite);
  const low = usable ? min : [...root.bounds.min];
  const high = usable ? max : [...root.bounds.max];

  framing = {
    center: low.map((value, axis) => (value + high[axis]!) / 2),
    radius: Math.max(
      Math.hypot(...high.map((value, axis) => value - low[axis]!)) / 2,
      1e-6,
    ),
    bounds: [low[0]!, high[0]!, low[1]!, high[1]!, low[2]!, high[2]!],
  };
  applyFraming();
};

/** Put the camera back where loading the current cloud put it. */
const applyFraming = (): void => {
  if (framing === null) return;
  const { center, radius, bounds } = framing;
  // Far enough back that a sphere of that radius fits the vertical field.
  const distance = radius / Math.tan((VIEW_ANGLE / 2) * RADIANS);
  const elevation = START_ELEVATION * RADIANS;
  const azimuth = START_AZIMUTH * RADIANS;
  const offset = [
    Math.cos(elevation) * Math.cos(azimuth),
    Math.cos(elevation) * Math.sin(azimuth),
    Math.sin(elevation),
  ];

  camera.setFocalPoint(center[0]!, center[1]!, center[2]!);
  camera.setPosition(
    center[0]! + offset[0]! * distance,
    center[1]! + offset[1]! * distance,
    center[2]! + offset[2]! * distance,
  );
  camera.setViewUp(...WORLD_UP);
  camera.setViewAngle(VIEW_ANGLE);
  // Panning reads its depth reference from this point. It defaults to the
  // world origin, which for data in a projected CRS is tens of kilometres
  // away, and a pan referenced from there moves the scene by kilometres per
  // pixel.
  interactorStyle.setCenterOfRotation(center[0]!, center[1]!, center[2]!);
  // A parallel camera has no eye distance to frame from, so the world height
  // the viewport spans is stated directly.
  camera.setParallelScale(radius);
  // Set the range from the data's own bounds rather than from the renderer's
  // props, which are empty until something is selected. Selection culls
  // against this frustum, so a range left over from a nearer scene rejects
  // every node of a wider one — and with nothing selected there is never an
  // actor to trigger a recompute. That deadlock is permanent: a blank view
  // that no amount of waiting resolves.
  renderer.resetCameraClippingRange(bounds);
  clippingDirty = true;
};

const loadSource = async (
  name: string,
  sourcePromise: Promise<TileSource>,
): Promise<void> => {
  setMessage(`Reading ${name}…`);
  // Tear down first, then claim the generation: disposeCloud() bumps it to
  // cancel whatever was in flight, so a number taken before that call would be
  // stale the moment it was read.
  disposeCloud();
  const generation = ++loadGeneration;
  try {
    const source = await sourcePromise;
    if (generation !== loadGeneration) return;
    await frameRoot(source);
    if (generation !== loadGeneration) return;
    loadedName = name;
    loadedPointCount = source.metadata().pointCount;
    adapter = createRendererAdapter({
      renderer,
      scheduleRender,
      devicePixelRatio: currentDevicePixelRatio,
    });
    controller = createLodController({
      source,
      // No pointBudget here: applyBudgetMode owns that number in both modes
      // and runs before the first camera reaches the controller, so nothing is
      // ever selected against a budget the panel did not ask for.
      fetchConcurrency: 6,
      cacheBytes: 256 * 1024 ** 2,
      refinementCutoffPx: 0.75,
      presentation: {
        mode: "auto",
        userScale: 1,
        minDiameterCssPx: 1.25,
        maxDiameterCssPx: 4,
      },
      onTiles: (batch) => {
        adapter?.applyBatch(batch);
        clippingDirty = true;
      },
      onPointDiameterCssPx: (diameter) =>
        adapter?.setPointDiameterCssPx(diameter),
      onError: (error) => {
        console.error(error);
        setMessage(String(error), true);
      },
      scheduleRender,
    });
    applyBudgetMode();
    controller.setCamera(cameraView());
    setMessage(`${name}: ${points(loadedPointCount)} points`);
    scheduleRender();
  } catch (error) {
    if (generation !== loadGeneration) return;
    disposeCloud();
    const text = error instanceof Error ? error.message : String(error);
    setMessage(`Could not load ${name}: ${text}`, true);
  }
};

const localFileSource = (file: File): Promise<TileSource> =>
  createCopcTileSource({
    source: async (begin, end) =>
      new Uint8Array(await file.slice(begin, end).arrayBuffer()),
  });

/**
 * Keep the address bar showing the cloud on screen, so the page can be
 * reloaded or the link handed to someone and land on the same scene. A local
 * file has no URL to share, so it clears the parameter instead.
 */
const showInAddressBar = (url: string | null): void => {
  const target = new URL(window.location.href);
  if (url === null) target.searchParams.delete("url");
  else target.searchParams.set("url", url);
  window.history.replaceState(null, "", target);
};

const loadUrl = (): void => {
  const url = urlInput.value.trim();
  if (!url) {
    setMessage("Enter a COPC URL.", true);
    return;
  }
  showInAddressBar(url);
  void loadSource(url, createCopcTileSource({ source: url }));
};

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

/**
 * Switching projection preserves the world height the viewport covers, so the
 * toggle changes the selection law without re-framing the scene: an
 * orthographic camera refines on zoom (`parallelScale`) rather than on
 * approach.
 */
const syncProjection = (): void => {
  const parallel = projectionSelect.value === "orthographic";
  const halfAngle = (camera.getViewAngle() * Math.PI) / 360;
  if (parallel && !camera.getParallelProjection()) {
    camera.setParallelScale(
      Math.max(camera.getDistance() * Math.tan(halfAngle), 1e-6),
    );
  } else if (!parallel && camera.getParallelProjection()) {
    const distance = camera.getParallelScale() / Math.tan(halfAngle);
    const focal = camera.getFocalPoint();
    const direction = camera.getDirectionOfProjection();
    camera.setPosition(
      focal[0] - direction[0] * distance,
      focal[1] - direction[1] * distance,
      focal[2] - direction[2] * distance,
    );
  }
  camera.setParallelProjection(parallel);
  renderer.resetCameraClippingRange();
  scheduleRender();
};

const syncBudgetControls = (): void => {
  const adaptive = budgetMode() === "adaptive";
  fixedControls.hidden = adaptive;
  adaptiveControls.hidden = !adaptive;
};

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  sceneSelect.value = "";
  urlInput.value = "";
  showInAddressBar(null);
  void loadSource(file.name, localFileSource(file));
});

resetViewButton.addEventListener("click", () => {
  if (framing === null) return;
  applyFraming();
  // The controller selects against the camera it was last given, so the new
  // one has to reach it before the next frame is drawn against the old one.
  controller?.setCamera(cameraView());
  scheduleRender();
});

loadUrlButton.addEventListener("click", loadUrl);
urlInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") loadUrl();
});

budgetModeSelect.addEventListener("change", () => {
  syncBudgetControls();
  syncGovernor();
  applyBudgetMode();
  scheduleRender();
});

pointBudgetInput.addEventListener("change", applyBudgetMode);

for (const input of [movingTargetInput, stationaryTargetInput, maxPointsInput]) {
  input.addEventListener("change", () => {
    syncGovernor();
    scheduleRender();
  });
}

projectionSelect.addEventListener("change", syncProjection);

window.addEventListener("resize", () => {
  currentDevicePixelRatio = window.devicePixelRatio;
  adapter?.setDevicePixelRatio(currentDevicePixelRatio);
  scheduleRender();
});

syncBudgetControls();
syncGovernor();
// The panel runs on its own clock: driving it from the render loop would leave
// it a frame stale exactly when the view stops painting, and would charge its
// cost to the budget it is reporting on.
updateDiagnostics();
setInterval(updateDiagnostics, DIAGNOSTICS_INTERVAL_MS);

const OPEN_LIDAR = "https://open-lidar-data.s3.eu-central-1.amazonaws.com/data";

/**
 * Public COPC clouds, all CORS-enabled and served over HTTP ranges.
 *
 * Every one carries colour. Intensity-only surveys — Dublin City, SoFi
 * Stadium — stream just as well but render as a white mass, which shows the
 * example's point off worse than a smaller cloud that you can actually read.
 */
const HOSTED_SCENES: { label: string; url: string }[] = [
  {
    label: "Luxembourg city — 16M pts, 65/m² (CC0)",
    url: `${OPEN_LIDAR}/LU/Gouvernement_LUX/Lidar_2019/copc/LIDAR2019_NdP_54500_98500_EPSG2169.copc.laz`,
  },
  {
    label: "Autzen Stadium — 11M pts (CC-BY-4.0)",
    url: "https://s3.amazonaws.com/hobu-lidar/autzen-classified.copc.laz",
  },
  {
    label: "Luxembourg village — 163k pts (CC0)",
    url: `${OPEN_LIDAR}/LU/Gouvernement_LUX/Lidar_2019/copc/LIDAR2019_NdP_100000_82500_EPSG2169.copc.laz`,
  },
];

for (const { label, url } of HOSTED_SCENES) {
  const option = document.createElement("option");
  option.value = url;
  option.textContent = label;
  sceneSelect.append(option);
}

sceneSelect.addEventListener("change", () => {
  if (!sceneSelect.value) return;
  urlInput.value = sceneSelect.value;
  loadUrl();
});

// A URL typed by hand is no longer whichever scene the list is showing.
urlInput.addEventListener("input", () => {
  if (urlInput.value !== sceneSelect.value) sceneSelect.value = "";
});

// Every browser check passes an explicit `?url=`, so the default below is the
// interactive opening scene only and nothing under test observes it.
const initialUrl =
  new URLSearchParams(window.location.search).get("url") ??
  HOSTED_SCENES[0]!.url;
urlInput.value = initialUrl;
sceneSelect.value = HOSTED_SCENES.some((scene) => scene.url === initialUrl)
  ? initialUrl
  : "";
loadUrl();

// Driving handles for browser checks. Everything here drives the page the way
// a user or a host would — the camera moves, the panel changes, a frame is
// requested — rather than reaching past it into the library, so a check that
// passes says the assembled page works, not that the modules do.
Object.assign(window, {
  pointCloudExample: {
    stats: () => ({
      controller: controller?.stats() ?? null,
      adapter: adapter?.stats() ?? null,
      governor: governor?.stats() ?? null,
      lastFrameMs,
      source: loadedName || null,
      sourcePoints: loadedPointCount,
    }),

    /**
     * Both sides' key sets, so a check can assert they agree rather than
     * inferring agreement from two counts that happen to match.
     */
    keys: () => ({
      controller: controller?.activeKeys() ?? null,
      adapter: adapter?.activeKeys() ?? null,
    }),

    /** The CSS size selection is computed against, after any resize. */
    viewport: () => ({
      width: viewer.clientWidth,
      height: viewer.clientHeight,
    }),

    /**
     * What the renderer itself is holding, read from the scene rather than
     * from the adapter that put it there.
     *
     * Every other statistic here comes from a handle that a leak takes with
     * it: a disposed adapter reports nothing, so an actor it abandoned in the
     * renderer is invisible to all of them. This is the one place a stale
     * actor can actually be seen, and the point size is read off the actor
     * that will draw it, so a device-pixel-ratio change is observed as an
     * effect on the scene instead of as a setter echoing its argument.
     */
    scene: () => {
      const actors = renderer.getActors();
      const first = actors[0];
      return {
        actors: actors.length,
        pointSizeDevicePx: first
          ? (first.getProperty().getPointSize() ?? null)
          : null,
        mapperScaleFactor: first
          ? ((first.getMapper() as { getScaleFactor?: () => number })
              .getScaleFactor?.() ?? null)
          : null,
      };
    },
    setProjection: (projection: Projection) => {
      projectionSelect.value = projection;
      syncProjection();
    },
    setBudgetMode: (mode: BudgetMode) => {
      budgetModeSelect.value = mode;
      syncBudgetControls();
      syncGovernor();
      applyBudgetMode();
      scheduleRender();
    },
    dispose: disposeCloud,

    /** Resolves once the source has opened and its first selection has run. */
    load: (url: string): Promise<void> => {
      urlInput.value = url;
      return loadSource(url, createCopcTileSource({ source: url }));
    },

    camera: {
      read: () => ({
        position: [...camera.getPosition()],
        focalPoint: [...camera.getFocalPoint()],
        // Included because motion inference reads the whole view-projection
        // product: view-up moves the view matrix without moving the eye, and
        // vtk.js's trackball style orthogonalises it on interaction, so a
        // check that only compared eye and target could not tell a camera
        // that stood still from one whose basis was rewritten under it.
        viewUp: [...camera.getViewUp()],
        parallelScale: camera.getParallelScale(),
        viewAngle: camera.getViewAngle(),
        parallelProjection: !!camera.getParallelProjection(),
      }),
      /** Absolute placement, for a check that needs a known camera. */
      place: (next: {
        position?: readonly number[];
        focalPoint?: readonly number[];
        parallelScale?: number;
      }) => {
        if (next.position) {
          camera.setPosition(
            next.position[0]!,
            next.position[1]!,
            next.position[2]!,
          );
        }
        if (next.focalPoint) {
          camera.setFocalPoint(
            next.focalPoint[0]!,
            next.focalPoint[1]!,
            next.focalPoint[2]!,
          );
        }
        if (next.parallelScale !== undefined) {
          camera.setParallelScale(next.parallelScale);
        }
        renderer.resetCameraClippingRange();
        scheduleRender();
      },
      /** Orbit about the focal point, degrees. */
      azimuth: (degrees: number) => {
        camera.azimuth(degrees);
        renderer.resetCameraClippingRange();
        scheduleRender();
      },
      /** Move along the view axis; >1 approaches. Parallel cameras zoom. */
      dolly: (factor: number) => {
        if (camera.getParallelProjection()) {
          camera.setParallelScale(camera.getParallelScale() / factor);
        } else {
          camera.dolly(factor);
        }
        renderer.resetCameraClippingRange();
        scheduleRender();
      },
    },

    /** The anchor's draw switch, as a host toggling a layer would use it. */
    setVisible: (visible: boolean) => {
      adapter?.setVisible(visible);
      scheduleRender();
    },
    /** Streaming activation — what actually releases a hidden cloud's memory. */
    setActive: (active: boolean) => {
      controller?.setActive(active);
      scheduleRender();
    },
    setDevicePixelRatio: (ratio: number) => {
      currentDevicePixelRatio = ratio;
      adapter?.setDevicePixelRatio(ratio);
      scheduleRender();
    },
    /** Report every frame as this duration; null restores real measurement. */
    setSyntheticFrameMs: (ms: number | null) => {
      syntheticFrameMs = ms;
      scheduleRender();
    },
    needsFrame: () => governor?.needsFrame() ?? false,
    render: () => scheduleRender(),
  },
});
