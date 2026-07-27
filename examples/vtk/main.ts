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
 * The clip-z row (8, 9, 10, 11) is deliberately left out, exactly as the
 * bridge leaves out its own: a host folds a depth remap derived from the
 * scene's visible bounds into that row, so a tile arriving or being evicted
 * rewrites those four numbers while the camera stands perfectly still. Every
 * camera move shows up in the x, y and w rows; the one motion that lives only
 * in clip z — dollying an orthographic camera along its view axis — shows up
 * in the eye point instead, which is compared alongside.
 *
 * This page hands the projection a fixed z range, so the row never moves here
 * and excluding it changes nothing on screen. It is here because the example
 * exists to show a host what to implement, and a host reading its own
 * composite matrix will have the remap folded in.
 */
const MOTION_MATRIX_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15];

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
const row = (name: string, value: string): string =>
  `${name.padEnd(15)}${value}`;

const governorLines = (view: ViewGovernorStats): string[] => {
  const cloud = view.members[0] ?? null;
  const adjustment = view.lastAdjustment;
  return [
    "BUDGET CHAIN",
    row(
      "regime",
      `${view.regime}${view.motion.settling ? " (settling)" : ""}`,
    ),
    row(
      "motion",
      `${view.motion.source ?? "none"} — explicit ${
        view.motion.explicitReferences
      }, inferred ${view.motion.inferredReferences}`,
    ),
    row("target", millis(view.targetFrameTimeMs)),
    row("estimate", `${millis(view.estimateMs)} of ${view.samples} samples`),
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
    row(
      "cloud share",
      cloud
        ? `${points(cloud.effectiveBudget)} (${cloud.activeConstraint})`
        : "no member registered",
    ),
    row("constraint", view.activeConstraint),
    row(
      "adjustment",
      adjustment
        ? `${adjustment.direction}/${adjustment.reason} ` +
            `${points(adjustment.fromBudget)} → ${points(adjustment.toBudget)}`
        : "none yet",
    ),
    row(
      "physical work",
      `${view.physicalTileOperations} tile, ` +
        `${view.physicalHierarchyOperations} hierarchy`,
    ),
    row("needs frame", view.needsFrame ? "yes" : "no"),
  ];
};

const cloudLines = (): string[] => {
  const cloud = controller?.stats();
  const drawn = adapter?.stats();
  if (!cloud || !drawn) return [];
  const { selection } = cloud;
  return [
    "CLOUD",
    row("source", `${loadedName} (${points(loadedPointCount)} points)`),
    row(
      "selection",
      `${points(selection.targetPoints)} points in ` +
        `${selection.targetTiles} tiles`,
    ),
    row("point budget", points(cloud.pointBudget)),
    row("importance", selection.projectedImportance.toFixed(2)),
    row(
      "resident",
      `${points(cloud.residentPoints)} points, ${cloud.residentTiles} tiles`,
    ),
    row(
      "decoded",
      `${mib(cloud.decodedBytes)} (${cloud.cachedTiles} cached tiles)`,
    ),
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
    row(
      "drawn",
      `${points(drawn.drawnPoints)} points, ${drawn.drawnTiles} tiles`,
    ),
    row("gpu", `${mib(drawn.gpuResidentBytes)} resident`),
    row("point diameter", `${drawn.diameterCssPx.toFixed(2)} css px`),
    row("last frame", millis(lastFrameMs)),
  ];
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

const updateDiagnostics = (): void => {
  const view = governor?.stats() ?? null;
  updateRegimeBadge(view);
  const sections = [
    view
      ? governorLines(view)
      : ["BUDGET CHAIN", row("mode", "fixed (no view governor)")],
    cloudLines(),
  ].filter((lines) => lines.length > 0);
  stats.textContent = sections.map((lines) => lines.join("\n")).join("\n\n");
};

// ---------------------------------------------------------------------------
// Frame loop
// ---------------------------------------------------------------------------

/**
 * Coalescing render request handed to the controller and the adapter. The
 * interactor paints its own animation frames, so a second paint in the same
 * frame would double the cost the budget loop is measuring.
 */
const scheduleRender = (): void => {
  if (frameQueued) return;
  frameQueued = true;
  requestAnimationFrame(() => {
    frameQueued = false;
    if (interactor.isAnimating()) return;
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
  // Framing the next cloud is not motion the user asked for.
  lastRenderedView = null;
};

const frameRoot = async (source: TileSource): Promise<void> => {
  const entries = await source.nodes(ROOT_KEY);
  const root = entries.find(
    ({ key }) => key.level === 0 && key.x === 0 && key.y === 0 && key.z === 0,
  );
  if (!root) throw new Error("The COPC hierarchy has no root node");
  const center = [
    (root.bounds.min[0] + root.bounds.max[0]) / 2,
    (root.bounds.min[1] + root.bounds.max[1]) / 2,
    (root.bounds.min[2] + root.bounds.max[2]) / 2,
  ] as const;
  const halfSize = (root.bounds.max[0] - root.bounds.min[0]) / 2;
  camera.setFocalPoint(...center);
  camera.setPosition(
    center[0] + halfSize * 1.15,
    center[1] - halfSize * 1.25,
    center[2] + halfSize * 1.05,
  );
  camera.setViewUp(0, 0, 1);
  camera.setViewAngle(38);
  // A parallel camera has no eye distance to frame from, so the world height
  // the viewport spans is stated directly.
  camera.setParallelScale(halfSize * 1.4);
  renderer.resetCameraClippingRange();
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
      onTiles: (batch) => adapter?.applyBatch(batch),
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

const loadUrl = (): void => {
  const url = urlInput.value.trim();
  if (!url) {
    setMessage("Enter a COPC URL.", true);
    return;
  }
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
  if (file) void loadSource(file.name, localFileSource(file));
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

const initialUrl = new URLSearchParams(window.location.search).get("url");
if (initialUrl) {
  urlInput.value = initialUrl;
  loadUrl();
}

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
