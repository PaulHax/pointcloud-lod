/**
 * pointcloud-lod vtk.js example.
 *
 * Streams any COPC dataset through a source worker — a local file through
 * `Blob.slice()` range reads, a remote URL through HTTP Range — using the same
 * three pieces the trame bridge wires together: one LOD controller per cloud,
 * one renderer adapter, and one view governor owning normalized view quality.
 * Every knob (budget mode, frame-time targets, maximum points, projection) is
 * a runtime control, so a different cloud needs no code change.
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
  createCopcWorkerTileSource,
  createLodController,
  createViewGovernor,
  keyToString,
  type CameraView,
  type LodController,
  type MotionReference,
  type PointPresentation,
  type TileSource,
  type ViewGovernor,
  type ViewGovernorOptions,
  type ViewGovernorStats,
} from "../../../src";
import { DEFAULT_MIN_POINT_BUDGET } from "../../../src/pointCloudMember";
import {
  createRendererAdapter,
  type RendererAdapter,
} from "../../../src/rendererAdapter";
import { WORLD_UP, installCameraControls } from "../harness/cameraControls";
import { createGpuFrameTimer, type GpuTimerResult } from "../harness/gpuTimer";
import {
  captureTelemetryEnvironment,
  createTelemetryRecorder,
  type TelemetryEnvironment,
  type TelemetryFrameEvent,
  type TelemetryRecorder,
  type TelemetryTrace,
} from "../harness/telemetryRecorder";
import {
  HOSTED_POINT_CLOUDS,
  installExampleSceneSelect,
} from "../scene/exampleScenes";
import {
  installRecorderOverlay,
  recordingRequested,
} from "../harness/captureOverlay";
import {
  createInputRecorder,
  type InputRecorder,
} from "../harness/inputRecorder";

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
const pointSizeModeSelect = element<HTMLSelectElement>("#point-size-mode");
const pointSizeInput = element<HTMLInputElement>("#point-size");
const pointSizeLabel = element<HTMLLabelElement>("#point-size-label");
const pointSizeValue = element<HTMLOutputElement>("#point-size-value");
const fixedControls = element<HTMLElement>("#fixed-controls");
const adaptiveControls = element<HTMLElement>("#adaptive-controls");
const pointBudgetInput = element<HTMLInputElement>("#point-budget");
const movingTargetInput = element<HTMLInputElement>("#moving-target");
const stationaryTargetInput = element<HTMLInputElement>("#stationary-target");
const maxPointsInput = element<HTMLInputElement>("#max-points");
const resetViewButton = element<HTMLButtonElement>("#reset-view");
const message = element<HTMLOutputElement>("#message");
const stats = element<HTMLElement>("#stats");
const frameRateChart = element<SVGSVGElement>("#frame-rate-chart");
const frameRateArea = element<SVGPathElement>("#frame-rate-area");
const frameRateLine = element<SVGPolylineElement>("#frame-rate-line");
const frameRateLatest = element<SVGCircleElement>("#frame-rate-latest");
const frameRateValue = element<HTMLOutputElement>("#frame-rate-value");
const frameRateFilteredValue = element<HTMLOutputElement>(
  "#frame-rate-filtered-value",
);
const frameRateTargetLine = element<SVGLineElement>("#frame-rate-target-line");
const frameRateTarget = element<HTMLElement>("#frame-rate-target");
const telemetryStatus = element<HTMLOutputElement>("#telemetry-status");
const telemetryToggle = element<HTMLButtonElement>("#telemetry-toggle");
const telemetryDownload = element<HTMLButtonElement>("#telemetry-download");
const telemetryClear = element<HTMLButtonElement>("#telemetry-clear");

const fullScreen = vtkFullScreenRenderWindow.newInstance({
  rootContainer: viewer,
  background: [0.035, 0.055, 0.075],
});
const renderer = fullScreen.getRenderer();
const renderWindow = fullScreen.getRenderWindow();
const interactor = renderWindow.getInteractor();
const camera = renderer.getActiveCamera();

/** Where loading the current cloud framed it, so the view can go back. */
let framing: {
  center: number[];
  radius: number;
  bounds: [number, number, number, number, number, number];
} | null = null;

const interactorStyle = installCameraControls({
  interactor,
  renderer,
  viewer,
  canvas: () => fullScreen.getApiSpecificRenderWindow().getCanvas(),
  sceneRadius: () => framing?.radius ?? null,
});

/** The panel is an instrument, so it repaints on its own slower clock. */
const DIAGNOSTICS_INTERVAL_MS = 100;

type BudgetMode = "adaptive" | "fixed";
type PointSizeMode = "fixed" | "auto";
type Projection = "perspective" | "orthographic";

let controller: LodController | null = null;
let adapter: RendererAdapter | null = null;
let loadedSource: TileSource | null = null;
let governor: ViewGovernor | null = null;
let governorKey: string | null = null;
let explicitMotion: MotionReference | null = null;
let loadedName = "";
let loadedPointCount = 0;
let controllerWorkPending = false;
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
let fixedPointDiameterCssPx = 2;
let autoPointScale = 0.5;
/**
 * The draw density fixed quality is asked for. Adaptive quality derives its
 * own from measured frame time and takes no preference.
 */
let fixedDensityFraction = 1;
/** Last settled quality retained as the moving selection/residency ceiling. */
let stationaryQualityFraction = 1;

const telemetryEnvironment = (): TelemetryEnvironment =>
  captureTelemetryEnvironment(
    fullScreen.getApiSpecificRenderWindow().get3DContext() as
      | WebGLRenderingContext
      | WebGL2RenderingContext
      | null,
  );

const telemetry: TelemetryRecorder = createTelemetryRecorder({
  environment: telemetryEnvironment,
});

const webGlContext = fullScreen.getApiSpecificRenderWindow().get3DContext() as
  | WebGLRenderingContext
  | WebGL2RenderingContext
  | null;
const webGl2Context =
  webGlContext !== null && "createQuery" in webGlContext
    ? (webGlContext as WebGL2RenderingContext)
    : null;
const timerContext = telemetryEnvironment().webgl.softwareRenderer
  ? null
  : webGl2Context;
let handleGpuTimerResult: (result: GpuTimerResult) => void = () => {};
const gpuTimer = createGpuFrameTimer(timerContext, {
  onResult: (result) => handleGpuTimerResult(result),
});

const setMessage = (text: string, error = false): void => {
  message.textContent = text;
  message.classList.toggle("error", error);
};

const budgetMode = (): BudgetMode =>
  budgetModeSelect.value === "fixed" ? "fixed" : "adaptive";

const pointSizeMode = (): PointSizeMode =>
  pointSizeModeSelect.value === "auto" ? "auto" : "fixed";

const pointPresentation = (): PointPresentation =>
  pointSizeMode() === "auto"
    ? { mode: "auto", userScale: autoPointScale }
    : { mode: "fixed", diameterCssPx: fixedPointDiameterCssPx };

const syncPointSizeControl = (): void => {
  const auto = pointSizeMode() === "auto";
  pointSizeLabel.textContent = auto ? "Auto scale" : "Size (CSS px)";
  pointSizeInput.min = "0.25";
  pointSizeInput.max = auto ? "2" : "8";
  pointSizeInput.step = auto ? "0.05" : "0.25";
  pointSizeInput.value = String(
    auto ? autoPointScale : fixedPointDiameterCssPx,
  );
  pointSizeValue.value = auto
    ? `${autoPointScale.toFixed(2)}×`
    : `${fixedPointDiameterCssPx.toFixed(2)} px`;
};

const applyPointPresentation = (): void => {
  controller?.setPresentation(pointPresentation());
  scheduleRender();
};

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
    viewportWidthCssPx: width,
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

const telemetryState = (): unknown => ({
  source: loadedName || null,
  sourcePoints: loadedPointCount,
  camera: {
    selectionView: cameraView(),
    focalPoint: [...camera.getFocalPoint()],
    viewUp: [...camera.getViewUp()],
  },
  settings: {
    budgetMode: budgetMode(),
    fixedPointBudget: fixedPointBudget(),
    interactionTargetMs: numberFrom(movingTargetInput),
    stationaryTargetMs: numberFrom(stationaryTargetInput),
    maximumPoints: numberFrom(maxPointsInput),
    pointPresentation: pointPresentation(),
    devicePixelRatio: currentDevicePixelRatio,
  },
  controller: controller?.stats() ?? null,
  adapter: adapter?.stats() ?? null,
  governor: governor?.stats() ?? null,
});

type CoreWorkSnapshot = {
  readonly controllerRevision: number;
  readonly rendererRevision: number;
};

let lastPresentedCoreWork: CoreWorkSnapshot | null = null;

/** Reasons this presentation cannot describe steady-state rendering cost. */
const frameContamination = (): string[] => {
  const cloud = controller?.stats();
  const rendererState = adapter?.stats();
  if (cloud === undefined || rendererState === undefined) {
    lastPresentedCoreWork = null;
    return ["no-controller"];
  }
  const reasons: string[] = [];
  const current = {
    controllerRevision: cloud.workRevision,
    rendererRevision: rendererState.workRevision,
  };
  if (lastPresentedCoreWork === null) {
    reasons.push("first-core-frame");
  } else {
    if (
      current.controllerRevision !== lastPresentedCoreWork.controllerRevision
    ) {
      reasons.push("controller-work-overlap");
    }
    if (current.rendererRevision !== lastPresentedCoreWork.rendererRevision) {
      reasons.push("renderer-resource-change");
    }
  }
  if (cloud.inFlight > 0 || cloud.queuedTiles > 0)
    reasons.push("tile-work-wanted");
  if (cloud.physicalTileOperations > 0) reasons.push("tile-work-physical");
  if (cloud.hierarchyInFlight > 0 || cloud.queuedPages > 0)
    reasons.push("hierarchy-work-wanted");
  if (cloud.physicalHierarchyOperations > 0)
    reasons.push("hierarchy-work-physical");
  if (cloud.selection.targetUndecodedTiles > 0)
    reasons.push("selected-tiles-undecoded");
  if (cloud.workPending) reasons.push("required-work-pending");
  lastPresentedCoreWork = current;
  return reasons;
};

// ---------------------------------------------------------------------------
// Camera-motion inference
//
// Hand the governor the camera this page actually rendered and let it classify
// the motion: it owns the jitter epsilon, the clip-z-row exclusion, the
// inferred motion reference and the trailing stability timer. A host with
// several views passes one entry per view; this page has one.
// ---------------------------------------------------------------------------

/** Any stable key identifies a view; this page's single renderer is one. */
const renderedCameras = new Map<unknown, CameraView>();

// ---------------------------------------------------------------------------
// Budget mode and the view governor
// ---------------------------------------------------------------------------

/**
 * The governor throws on an unusable option rather than clamping, so the panel
 * is validated here — the same reconciliation the bridge does, without the
 * multi-cloud bookkeeping.
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
    (maxBudget !== null && maxBudget < DEFAULT_MIN_POINT_BUDGET)
  ) {
    return null;
  }
  return {
    interactionTargetMs,
    stationaryTargetMs,
    // The default allows VTK 70% of a host frame, which is right for a view
    // compositing a basemap and video underneath it. This page paints nothing
    // but the point cloud, so VTK's whole-frame allowance is the whole frame;
    // leaving the default would normalise every honest 33 ms frame up to 47 ms
    // and walk the budget down to its floor while nothing was ever late.
    vtkFrameFraction: 1,
  };
};

/**
 * Map normalized view quality onto the point member's useful ceiling.
 *
 * A stationary allocation changes selection and draws it all. While moving,
 * the last stationary selection remains resident and only its point prefixes
 * thin, so returning to full quality does not fetch or rebuild existing tiles.
 */
const applyAdaptiveQuality = (): void => {
  if (!controller || !governor) return;
  const inputs = controller.governorInputs();
  const configuredMaximum = numberFrom(maxPointsInput);
  const fullCeiling = Math.max(
    0,
    Math.min(
      loadedPointCount,
      inputs.memoryCeilingPoints,
      configuredMaximum === null
        ? Number.POSITIVE_INFINITY
        : Math.floor(configuredMaximum),
    ),
  );
  const qualityFraction = Math.min(
    governor.qualityFraction(),
    fullCeiling > 0 && inputs.demandPoints > 0
      ? Math.min(1, inputs.demandPoints / fullCeiling)
      : 1,
  );
  const budgetAt = (fraction: number): number => {
    const requested = Math.floor(fullCeiling * fraction);
    const demandCapped =
      inputs.demandPoints > 0
        ? Math.min(requested, inputs.demandPoints)
        : requested;
    return Math.min(
      fullCeiling,
      Math.max(DEFAULT_MIN_POINT_BUDGET, demandCapped),
    );
  };
  if (governor.stats().regime === "stationary") {
    stationaryQualityFraction = qualityFraction;
    controller.setPointBudget(budgetAt(qualityFraction));
    controller.setDensityFraction(1);
    return;
  }
  const drawPoints = budgetAt(qualityFraction);
  const selectionPoints = budgetAt(
    Math.max(stationaryQualityFraction, qualityFraction),
  );
  controller.setPointBudget(selectionPoints);
  controller.setDensityFraction(
    selectionPoints > 0 ? drawPoints / selectionPoints : 0,
  );
};

/** Adaptive quality is normalized; fixed point controls remain point-specific. */
const applyBudgetMode = (): void => {
  if (!controller) return;
  if (budgetMode() === "adaptive" && governor) {
    applyAdaptiveQuality();
    return;
  }
  controller.setPointBudget(fixedPointBudget());
  controller.setDensityFraction(fixedDensityFraction);
};

const syncGovernor = (): void => {
  const adaptive = budgetMode() === "adaptive";
  const wanted = adaptive ? readGovernorOptions() : null;
  if (adaptive && !wanted) {
    // An unusable option would throw out of the constructor; keep drawing to
    // the governor the panel last described.
    setMessage(
      "Frame-time targets must be above 0 ms and any maximum at least " +
        `${DEFAULT_MIN_POINT_BUDGET.toLocaleString()} points.`,
      true,
    );
    return;
  }
  const key = wanted === null ? null : JSON.stringify(wanted);
  if (key === governorKey) return;
  governorKey = key;
  if (wanted !== null && governor) {
    // Re-target in place. Memberships, motion references, camera stability
    // and the clock offset all survive a re-configuration, so a governor
    // retargeted mid-drag never reads as a torn-down one and quality does not
    // jump. Only the adaptive tracks restart, which is the point: their
    // learned budgets measured the targets being replaced.
    governor.setOptions(wanted);
  } else {
    // Switching quality policy entirely: the governor goes away, or comes back.
    governor?.dispose();
    governor = wanted === null ? null : createViewGovernor(wanted);
    // A gesture in progress belongs to the view, not to the instance.
    if (explicitMotion) {
      explicitMotion.release();
      explicitMotion = governor?.beginMotion("explicit") ?? null;
    }
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

const FRAME_RATE_CHART_WIDTH = 320;
const FRAME_RATE_CHART_HEIGHT = 80;
const FRAME_RATE_CHART_MAX = 120;
const FRAME_RATE_SAMPLE_CAPACITY = 90;
const FRAME_RATE_FILTER_WEIGHT = 0.15;
const FRAME_RATE_IDLE_MS = 250;
const frameRateSamples: number[] = [];
let lastFramePresentedAt: number | null = null;
let filteredFrameIntervalMs: number | null = null;
let frameRateLastActiveAt: number | null = null;
let frameRateIdle = true;
let frameRateTickQueued = false;
let frameRateGeneration = 0;

const frameRateY = (fps: number): number =>
  FRAME_RATE_CHART_HEIGHT *
  (1 - Math.min(Math.max(fps, 0), FRAME_RATE_CHART_MAX) / FRAME_RATE_CHART_MAX);

const formatFrameRate = (fps: number): string =>
  `${fps < 100 ? fps.toFixed(1) : Math.round(fps).toLocaleString()} fps`;

/** Add one filtered cadence sample to the graph. */
const plotFrameRate = (filteredFps: number): void => {
  frameRateSamples.push(filteredFps);
  if (frameRateSamples.length > FRAME_RATE_SAMPLE_CAPACITY) {
    frameRateSamples.shift();
  }

  // Right-align a partial history so the newest frame is always at the live
  // edge of the graph. Once full, each new sample advances the line one slot.
  const firstSlot = FRAME_RATE_SAMPLE_CAPACITY - frameRateSamples.length;
  const plotted = frameRateSamples.map((sample, index) => {
    const slot = firstSlot + index;
    return {
      x:
        (slot / Math.max(FRAME_RATE_SAMPLE_CAPACITY - 1, 1)) *
        FRAME_RATE_CHART_WIDTH,
      y: frameRateY(sample),
    };
  });
  const pointsAttribute = plotted
    .map(({ x, y }) => `${x.toFixed(1)},${y.toFixed(1)}`)
    .join(" ");
  const first = plotted[0]!;
  const latest = plotted[plotted.length - 1]!;

  frameRateLine.setAttribute("points", pointsAttribute);
  frameRateArea.setAttribute(
    "d",
    `M ${first.x.toFixed(1)} ${FRAME_RATE_CHART_HEIGHT} ` +
      `L ${pointsAttribute.replaceAll(",", " ")} ` +
      `L ${latest.x.toFixed(1)} ${FRAME_RATE_CHART_HEIGHT} Z`,
  );
  frameRateLatest.setAttribute("cx", latest.x.toFixed(1));
  frameRateLatest.setAttribute("cy", latest.y.toFixed(1));
  frameRateLatest.removeAttribute("hidden");
};

/**
 * Measure displayed frame cadence, not reciprocal render cost.
 *
 * A 3 ms paint means the renderer has headroom; it does not mean a 60 Hz
 * display showed 333 frames. Multiple renders before the next animation tick
 * are coalesced because the display can present only their final result.
 */
const sampleFrameRate = (presentedAt: number): number | null => {
  frameRateLastActiveAt = presentedAt;
  frameRateIdle = false;
  const previous = lastFramePresentedAt;
  lastFramePresentedAt = presentedAt;
  if (previous === null) return null;
  const intervalMs = presentedAt - previous;
  if (!Number.isFinite(intervalMs) || intervalMs <= 0) return null;

  filteredFrameIntervalMs =
    filteredFrameIntervalMs === null
      ? intervalMs
      : filteredFrameIntervalMs * (1 - FRAME_RATE_FILTER_WEIGHT) +
        intervalMs * FRAME_RATE_FILTER_WEIGHT;
  const currentFps = 1000 / intervalMs;
  const filteredFps = 1000 / filteredFrameIntervalMs;
  plotFrameRate(filteredFps);
  frameRateValue.value = formatFrameRate(currentFps);
  frameRateFilteredValue.value = formatFrameRate(filteredFps);
  frameRateChart.setAttribute(
    "aria-label",
    `${formatFrameRate(currentFps)} current, ` +
      `${formatFrameRate(filteredFps)} average, plotted over the latest ` +
      `${frameRateSamples.length} rendered frames`,
  );
  return intervalMs;
};

/** Coalesce the synchronous costs of paints landing in one presentation. */
let pendingVtkFrameMs: number | null = null;
let pendingGpuQueryIds: number[] = [];

type GpuPresentation = {
  readonly remaining: Set<number>;
  readonly frameEvent: TelemetryFrameEvent | null;
  readonly measuredGovernor: ViewGovernor | null;
  readonly regime: "interaction" | "stationary";
  readonly eligible: boolean;
  readonly presentedAt: number;
  readonly fallbackFrameMs: number;
  status: "valid" | "disjoint" | "error";
  gpuMs: number;
};

const gpuPresentations = new Map<number, GpuPresentation>();
const earlyGpuResults = new Map<number, GpuTimerResult>();
const ignoredGpuQueryIds = new Set<number>();

const adjustmentChanged = (
  before: ViewGovernorStats["lastAdjustment"],
  after: ViewGovernorStats["lastAdjustment"],
): boolean =>
  after !== null &&
  (before === null ||
    after.atMs !== before.atMs ||
    after.reason !== before.reason ||
    after.toFraction !== before.toFraction);

const finishGpuPresentation = (presentation: GpuPresentation): void => {
  const valid = presentation.status === "valid";
  if (presentation.frameEvent !== null) {
    telemetry.resolveGpuFrame(presentation.frameEvent, {
      status: presentation.status,
      gpuMs: valid ? presentation.gpuMs : null,
    });
  }
  const measuredGovernor = presentation.measuredGovernor;
  if (measuredGovernor !== null) {
    const before = measuredGovernor.stats().lastAdjustment;
    measuredGovernor.recordCapacitySample({
      frameMs: valid ? presentation.gpuMs : presentation.fallbackFrameMs,
      regime: presentation.regime,
      eligible: presentation.eligible && valid,
      now: presentation.presentedAt,
    });
    const after = measuredGovernor.stats().lastAdjustment;
    if (adjustmentChanged(before, after)) {
      telemetry.recordState("governor-adjustment", telemetryState());
    }
    if (measuredGovernor.needsFrame()) scheduleRender();
  }
};

const applyGpuResult = (
  presentation: GpuPresentation,
  result: GpuTimerResult,
): void => {
  gpuPresentations.delete(result.id);
  presentation.remaining.delete(result.id);
  if (result.status !== "valid") presentation.status = result.status;
  else if (result.gpuMs !== null) {
    presentation.gpuMs += result.gpuMs;
  }
  if (presentation.remaining.size === 0) finishGpuPresentation(presentation);
};

handleGpuTimerResult = (result) => {
  if (ignoredGpuQueryIds.delete(result.id)) return;
  const presentation = gpuPresentations.get(result.id);
  if (presentation === undefined) {
    earlyGpuResults.set(result.id, result);
    return;
  }
  applyGpuResult(presentation, result);
};

const recordFrameRate = (
  vtkFrameMs: number,
  gpuQueryId: number | null,
): void => {
  pendingVtkFrameMs =
    pendingVtkFrameMs === null
      ? vtkFrameMs
      : Math.max(pendingVtkFrameMs, vtkFrameMs);
  if (gpuQueryId !== null) pendingGpuQueryIds.push(gpuQueryId);
  if (frameRateTickQueued) return;
  frameRateTickQueued = true;
  const generation = frameRateGeneration;
  requestAnimationFrame((presentedAt) => {
    // Loading a new cloud resets the graph. A presentation callback belonging
    // to the previous one must not put its last frame into the new history.
    if (generation !== frameRateGeneration) return;
    frameRateTickQueued = false;
    const hostFrameMs = sampleFrameRate(presentedAt);
    const measuredVtkFrameMs = pendingVtkFrameMs;
    const queryIds = pendingGpuQueryIds;
    pendingVtkFrameMs = null;
    pendingGpuQueryIds = [];
    if (hostFrameMs === null || measuredVtkFrameMs === null) {
      for (const id of queryIds) {
        earlyGpuResults.delete(id);
        ignoredGpuQueryIds.add(id);
      }
      return;
    }
    const governorFrameMs = syntheticFrameMs ?? hostFrameMs;
    const recording = telemetry.isActive();
    const contamination = frameContamination();
    const measuredGovernor = governor;
    const governorStats = measuredGovernor?.stats() ?? null;
    const capacityEligible =
      contamination.length === 0 &&
      governorStats?.activity.measurementEligible === true;
    const useGpuCapacity =
      gpuTimer.supported && queryIds.length > 0 && syntheticFrameMs === null;
    const beforeAdjustment = measuredGovernor?.stats().lastAdjustment ?? null;
    if (useGpuCapacity) {
      measuredGovernor?.recordTransientFrame({
        hostFrameMs: governorFrameMs,
        vtkFrameMs: measuredVtkFrameMs,
        now: presentedAt,
      });
    } else {
      measuredGovernor?.recordHostFrame({
        hostFrameMs: governorFrameMs,
        vtkFrameMs: measuredVtkFrameMs,
        capacitySampleEligible: capacityEligible,
        now: presentedAt,
      });
    }
    const frameEvent = recording
      ? telemetry.recordFrame({
          presentedAtMs: presentedAt,
          rafIntervalMs: hostFrameMs,
          vtkCpuMs: measuredVtkFrameMs,
          governorFrameMs,
          gpuPending: gpuTimer.supported && queryIds.length > 0,
          reportedToGovernor: measuredGovernor !== null,
          capacitySampleEligible: capacityEligible,
          capacitySamplePending: useGpuCapacity,
          contamination,
          state: telemetryState(),
        })
      : null;
    const afterAdjustment = measuredGovernor?.stats().lastAdjustment ?? null;
    if (adjustmentChanged(beforeAdjustment, afterAdjustment)) {
      telemetry.recordState("governor-adjustment", telemetryState());
    }

    if (gpuTimer.supported && queryIds.length > 0) {
      const presentation: GpuPresentation = {
        remaining: new Set(queryIds),
        frameEvent,
        measuredGovernor: useGpuCapacity ? measuredGovernor : null,
        regime: governorStats?.regime ?? "stationary",
        eligible: capacityEligible,
        presentedAt,
        fallbackFrameMs: Math.max(governorFrameMs, measuredVtkFrameMs),
        status: "valid",
        gpuMs: 0,
      };
      for (const id of queryIds) gpuPresentations.set(id, presentation);
      for (const id of queryIds) {
        const early = earlyGpuResults.get(id);
        if (early === undefined) continue;
        earlyGpuResults.delete(id);
        applyGpuResult(presentation, early);
      }
    }
    if (measuredGovernor?.needsFrame()) scheduleRender();
  });
};

const resetFrameRate = (): void => {
  frameRateGeneration += 1;
  frameRateTickQueued = false;
  pendingVtkFrameMs = null;
  for (const id of pendingGpuQueryIds) ignoredGpuQueryIds.add(id);
  pendingGpuQueryIds = [];
  lastPresentedCoreWork = null;
  frameRateSamples.length = 0;
  lastFramePresentedAt = null;
  filteredFrameIntervalMs = null;
  frameRateLastActiveAt = null;
  frameRateIdle = true;
  frameRateLine.setAttribute("points", "");
  frameRateArea.setAttribute("d", "");
  frameRateLatest.setAttribute("hidden", "");
  frameRateValue.value = "— fps";
  frameRateFilteredValue.value = "— fps";
  frameRateChart.setAttribute("aria-label", "No frame-rate samples yet");
};

const updateFrameRateIdle = (now: number): void => {
  if (
    frameRateIdle ||
    frameRateLastActiveAt === null ||
    now - frameRateLastActiveAt < FRAME_RATE_IDLE_MS
  ) {
    return;
  }
  frameRateIdle = true;
  lastFramePresentedAt = null;
  filteredFrameIntervalMs = null;
  plotFrameRate(0);
  frameRateValue.value = "0.0 fps";
  frameRateFilteredValue.value = "0.0 fps";
  frameRateChart.setAttribute(
    "aria-label",
    `0.0 fps, idle, plotted over the latest ${frameRateSamples.length} samples`,
  );
};

const updateFrameRateTarget = (frameMs: number | null): void => {
  if (frameMs === null || !Number.isFinite(frameMs) || frameMs <= 0) {
    frameRateTargetLine.setAttribute("hidden", "");
    frameRateTarget.textContent = "No target";
    return;
  }
  const targetFps = 1000 / frameMs;
  const y = frameRateY(targetFps).toFixed(1);
  frameRateTargetLine.setAttribute("y1", y);
  frameRateTargetLine.setAttribute("y2", y);
  frameRateTargetLine.removeAttribute("hidden");
  frameRateTarget.textContent = `Target ${formatFrameRate(targetFps)}`;
};

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
  const adjustment = view.lastAdjustment;
  return {
    title: "Budget chain",
    rows: [
      row(
        "status",
        view.activity.inputActive
          ? "moving"
          : !view.activity.cameraStable
            ? "stabilizing"
            : view.activity.workPending
              ? "refining"
              : "settled",
      ),
      row("motion", view.motion.source ?? "none"),
      row(
        "references",
        `${view.motion.explicitReferences} explicit, ` +
          `${view.motion.inferredReferences} inferred`,
      ),
      row("target", millis(view.targetFrameTimeMs)),
      row("estimate", `${millis(view.estimateMs)} of ${view.samples}`),
      row(
        "capacity samples",
        `${view.capacitySamples.eligible} clean, ` +
          `${view.capacitySamples.rejected} rejected`,
      ),
      row("view quality", `${(view.viewQualityFraction * 100).toFixed(1)}%`),
      row(
        "adjustment",
        adjustment
          ? `${adjustment.direction} · ${adjustment.reason}`
          : "none yet",
      ),
      row(
        "moved quality",
        adjustment
          ? `${(adjustment.fromFraction * 100).toFixed(1)}% → ` +
              `${(adjustment.toFraction * 100).toFixed(1)}%`
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
      row("density", `${(cloud.densityFraction * 100).toFixed(0)}%`),
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

const updateTelemetryStatus = (): void => {
  const summary = telemetry.summary();
  const webgl = summary.environment.webgl;
  const renderer =
    webgl.unmaskedRenderer ?? webgl.renderer ?? "renderer unavailable";
  const rendererKind = webgl.softwareRenderer ? "software" : "GPU";
  const state = summary.active
    ? `Recording ${summary.events.toLocaleString()} events`
    : summary.events > 0
      ? `Stopped ${summary.events.toLocaleString()} events`
      : "Off";
  telemetryStatus.value = `${state} · ${rendererKind}: ${renderer}`;
  telemetryStatus.title = telemetryStatus.value;
  telemetryToggle.textContent = summary.active ? "Stop" : "Start";
  telemetryDownload.disabled = summary.events === 0;
  telemetryClear.disabled = summary.events === 0;
};

const updateDiagnostics = (): void => {
  const view = governor?.stats() ?? null;
  updateFrameRateIdle(performance.now());
  updateFrameRateTarget(view?.targetFrameTimeMs ?? null);
  renderStats(
    [
      view
        ? governorLines(view)
        : { title: "Budget chain", rows: [row("mode", "fixed budget")] },
      cloudLines(),
    ].filter((section): section is StatSection => section !== null),
  );
  if (telemetry.isActive()) updateTelemetryStatus();
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
let activeGpuQueryId: number | null = null;

const beginGpuFrame = (): void => {
  activeGpuQueryId = gpuTimer.begin();
};

const endGpuFrame = (): number | null => {
  const id = activeGpuQueryId;
  activeGpuQueryId = null;
  gpuTimer.end();
  return id;
};

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
    beginGpuFrame();
    interactor.render();
    // Nothing painted (a render was already in progress): drop the stamp so it
    // cannot be charged to the next frame.
    if (frameStartedAt !== null) {
      const unusedQuery = endGpuFrame();
      if (unusedQuery !== null) ignoredGpuQueryIds.add(unusedQuery);
    }
    frameStartedAt = null;
  });
};

/**
 * The per-frame report. View quality is normalized, so point demand and the
 * memory ceiling stay in the point-member mapping above. The governor only
 * needs the physical work state that decides whether a capacity sample is
 * clean; `governorInputs()` provides it without a diagnostic selected-set walk.
 */
const updateMember = (): void => {
  if (!controller || !adapter) return;
  const inputs = controller.governorInputs();
  adapter.setResourceCeilingBytes(inputs.memoryBudgetBytes);
  governor?.setWorkState({
    workPending: controllerWorkPending,
    physicalTileOperations: inputs.physicalTileOperations,
    physicalHierarchyOperations: inputs.physicalHierarchyOperations,
  });
  if (budgetMode() === "adaptive") applyAdaptiveQuality();
};

/**
 * `workPending` is the one governor input that costs a walk of the selected
 * set, and it is also the one that cannot change without the controller
 * reporting a work change — so it rides the `onWorkChange` path instead of the
 * frame, and a member that has just been registered is told once.
 */
const reportWorkPending = (): void => {
  if (!controller) return;
  controllerWorkPending = controller.stats().workPending;
  const inputs = controller.governorInputs();
  governor?.setWorkState({
    workPending: controllerWorkPending,
    physicalTileOperations: inputs.physicalTileOperations,
    physicalHierarchyOperations: inputs.physicalHierarchyOperations,
  });
};

// Every paint the interactor drives ends here — ours and the ones its own
// animation loop runs during a gesture — so one place measures synchronous vtk
// work and feeds the rendered camera to LOD. The following presentation tick
// reports full displayed cadence to the governor because WebGL submission time
// alone cannot say whether the frame made presentation.
interactor.onRenderEvent(() => {
  const startedAt = frameStartedAt;
  const completedAt = performance.now();
  const gpuQueryId = endGpuFrame();
  frameStartedAt = null;
  if (startedAt !== null) {
    lastFrameMs = syntheticFrameMs ?? completedAt - startedAt;
  }
  const view = cameraView();
  controller?.setCamera(view);
  updateMember();
  renderedCameras.set(renderer, view);
  governor?.noteRenderedCameras(renderedCameras, scheduleRender);
  if (budgetMode() === "adaptive") applyAdaptiveQuality();
  if (startedAt !== null) {
    if (controller) recordFrameRate(lastFrameMs, gpuQueryId);
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
  beginGpuFrame();
});

interactor.onEndAnimation(() => {
  explicitMotion?.release();
  explicitMotion = null;
  controller?.endInteraction();
  const focalPoint = camera.getFocalPoint();
  interactorStyle.setCenterOfRotation(
    focalPoint[0],
    focalPoint[1],
    focalPoint[2],
  );
  // The interactor paints once more immediately after this event; that paint
  // is a real frame and gets timed like any other.
  frameStartedAt = performance.now();
  beginGpuFrame();
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
  controller?.dispose();
  adapter?.dispose();
  loadedSource?.dispose?.();
  controller = null;
  adapter = null;
  loadedSource = null;
  loadedName = "";
  loadedPointCount = 0;
  controllerWorkPending = false;
  stationaryQualityFraction = 1;
  framing = null;
  resetFrameRate();
  // Framing the next cloud is not motion the user asked for: this view stops
  // feeding cameras, so its baseline has to go with it.
  renderedCameras.clear();
  governor?.resetMotionBaselines();
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
 * The extent comes from the source's stated data bounds, never from the root
 * node's: a COPC root node is a cube spanning the octree, not the data, so for
 * a survey far wider than it is tall the cube's centre floats high above the
 * ground and every orbit would swing about a point in the sky. A source that
 * states no bounds leaves that cube as the only estimate — and nothing here
 * reads a tile, so opening a cloud costs one hierarchy page and the tiles the
 * first selection actually wants.
 */
const frameRoot = async (source: TileSource): Promise<void> => {
  const stated = source.metadata().bounds;
  const entries = await source.nodes(ROOT_KEY);
  const root = entries.find(
    ({ key }) => key.level === 0 && key.x === 0 && key.y === 0 && key.z === 0,
  );
  if (!root) throw new Error("The COPC hierarchy has no root node");

  // A header carrying an unset or inverted extent is worse than none: it would
  // put the camera somewhere no data is and leave the view blank.
  const usable =
    stated !== undefined &&
    stated.min.every((value, axis) => {
      const top = stated.max[axis]!;
      return Number.isFinite(value) && Number.isFinite(top) && top >= value;
    });
  const extent = usable ? stated : root.bounds;
  const low = [...extent.min];
  const high = [...extent.max];

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

const instrumentSource = (source: TileSource): TileSource => ({
  metadata: () => source.metadata(),
  dispose: () => source.dispose?.(),
  async nodes(key, options) {
    const finish = telemetry.isActive()
      ? telemetry.beginWork("hierarchy", { key: keyToString(key) })
      : null;
    try {
      const nodes = await source.nodes(key, options);
      finish?.("ok", { entries: nodes.length });
      return nodes;
    } catch (error) {
      finish?.(options?.signal?.aborted ? "cancelled" : "error", {
        error: error instanceof Error ? error.name : "unknown",
      });
      throw error;
    }
  },
  async loadTile(key, options) {
    const finish = telemetry.isActive()
      ? telemetry.beginWork("tile-load", { key: keyToString(key) })
      : null;
    try {
      const tile = await source.loadTile(key, options);
      finish?.("ok", {
        points: tile.pointCount,
        bytes: tile.positions.byteLength + (tile.rgb?.byteLength ?? 0),
      });
      return tile;
    } catch (error) {
      finish?.(options?.signal?.aborted ? "cancelled" : "error", {
        error: error instanceof Error ? error.name : "unknown",
      });
      throw error;
    }
  },
});

const loadSource = async (
  name: string,
  sourcePromise: Promise<TileSource>,
): Promise<void> => {
  setMessage(`Reading ${name}…`);
  const finishSourceOpen = telemetry.isActive()
    ? telemetry.beginWork("source-open", { source: name })
    : null;
  // Tear down first, then claim the generation: disposeCloud() bumps it to
  // cancel whatever was in flight, so a number taken before that call would be
  // stale the moment it was read.
  disposeCloud();
  const generation = ++loadGeneration;
  try {
    const openedSource = await sourcePromise;
    if (generation !== loadGeneration) {
      finishSourceOpen?.("cancelled");
      openedSource.dispose?.();
      return;
    }
    finishSourceOpen?.("ok", {
      points: openedSource.metadata().pointCount,
    });
    const source = instrumentSource(openedSource);
    loadedSource = source;
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
      presentation: pointPresentation(),
      onTiles: (batch) => {
        const finish = telemetry.isActive()
          ? telemetry.beginWork("renderer-batch", {
              addedTiles: batch.added.length,
              removedTiles: batch.removed.length,
              addedPoints: batch.added.reduce(
                (total, { tile }) => total + tile.pointCount,
                0,
              ),
            })
          : null;
        try {
          adapter?.applyBatch(batch);
          finish?.(adapter === null ? "cancelled" : "ok");
        } catch (error) {
          finish?.("error", {
            error: error instanceof Error ? error.name : "unknown",
          });
          throw error;
        }
        clippingDirty = true;
      },
      onPointDiameterCssPx: (diameter) =>
        adapter?.setPointDiameterCssPx(diameter),
      onDrawPlan: (plan) => adapter?.applyDrawPlan(plan),
      onWorkChange: () => {
        updateMember();
        reportWorkPending();
        if (governor?.needsFrame()) scheduleRender();
      },
      onError: (error) => {
        console.error(error);
        setMessage(String(error), true);
      },
      scheduleRender,
    });
    applyBudgetMode();
    controller.setCamera(cameraView());
    setMessage(`${name}: ${points(loadedPointCount)} points`);
    if (telemetry.isActive()) {
      telemetry.recordState("source-loaded", telemetryState());
    }
    scheduleRender();
  } catch (error) {
    finishSourceOpen?.("error", {
      error: error instanceof Error ? error.name : "unknown",
    });
    if (generation !== loadGeneration) return;
    disposeCloud();
    const text = error instanceof Error ? error.message : String(error);
    setMessage(`Could not load ${name}: ${text}`, true);
    if (telemetry.isActive()) {
      telemetry.recordState("source-error", {
        source: name,
        error: text,
      });
    }
  }
};

const copcSource = (source: string | Blob): Promise<TileSource> =>
  createCopcWorkerTileSource({
    source,
    createWorker: () =>
      new Worker(new URL("../copc.worker.ts", import.meta.url), {
        type: "module",
      }),
    lazPerfWasmUrl: new URL("/laz-perf.wasm", window.location.href).href,
  });

const localFileSource = (file: File): Promise<TileSource> => copcSource(file);

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
  void loadSource(url, copcSource(url));
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

const startTelemetry = (): void => {
  telemetry.start();
  telemetry.recordState("recording-started", telemetryState());
  updateTelemetryStatus();
};

const stopTelemetry = (): void => {
  telemetry.recordState("recording-stopped", telemetryState());
  telemetry.stop();
  updateTelemetryStatus();
};

const markTelemetry = (label: string): void => {
  const normalized = label.trim().slice(0, 128);
  if (normalized.length === 0 || !telemetry.isActive()) return;
  telemetry.recordState(`marker:${normalized}`, telemetryState());
};

telemetryToggle.addEventListener("click", () => {
  if (telemetry.isActive()) stopTelemetry();
  else startTelemetry();
});

telemetryDownload.addEventListener("click", () => telemetry.download());

telemetryClear.addEventListener("click", () => {
  telemetry.clear();
  if (telemetry.isActive()) {
    telemetry.recordState("recording-cleared", telemetryState());
  }
  updateTelemetryStatus();
});

budgetModeSelect.addEventListener("change", () => {
  syncBudgetControls();
  syncGovernor();
  applyBudgetMode();
  scheduleRender();
});

pointBudgetInput.addEventListener("change", applyBudgetMode);

pointSizeModeSelect.addEventListener("change", () => {
  syncPointSizeControl();
  applyPointPresentation();
});

pointSizeInput.addEventListener("input", () => {
  const value = numberFrom(pointSizeInput);
  if (value === null || value <= 0) return;
  if (pointSizeMode() === "auto") autoPointScale = value;
  else fixedPointDiameterCssPx = value;
  syncPointSizeControl();
  applyPointPresentation();
});

for (const input of [
  movingTargetInput,
  stationaryTargetInput,
  maxPointsInput,
]) {
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
syncPointSizeControl();
syncGovernor();
// The panel runs on its own clock: driving it from the render loop would leave
// it a frame stale exactly when the view stops painting, and would charge its
// cost to the budget it is reporting on.
updateDiagnostics();
setInterval(updateDiagnostics, DIAGNOSTICS_INTERVAL_MS);

// Every browser check passes an explicit `?url=`, so the default below is the
// interactive opening scene only and nothing under test observes it.
const initialParameters = new URLSearchParams(window.location.search);
const initialUrl = initialParameters.get("url") ?? HOSTED_POINT_CLOUDS[0]!.url;
urlInput.value = initialUrl;
const exampleSceneSelect = installExampleSceneSelect(sceneSelect, {
  kind: "points",
  currentValue: initialUrl,
  onSelect: (url) => {
    urlInput.value = url;
    loadUrl();
  },
});
// A URL typed by hand is no longer whichever scene the list is showing.
urlInput.addEventListener("input", () =>
  exampleSceneSelect.setCurrent({ kind: "points", value: urlInput.value }),
);
if (initialParameters.get("telemetry") === "1") startTelemetry();
loadUrl();

/**
 * Gesture capture, so a benchmark can replay a hand-driven camera path rather
 * than one this file described. It reads the same camera the page renders
 * from, and records nothing until asked, so an ordinary session is unchanged.
 */
const inputRecorder: InputRecorder = createInputRecorder({
  viewer,
  pose: () => ({
    position: [...camera.getPosition()] as [number, number, number],
    focalPoint: [...camera.getFocalPoint()] as [number, number, number],
    viewUp: [...camera.getViewUp()] as [number, number, number],
    viewAngle: camera.getViewAngle(),
    parallelScale: camera.getParallelScale(),
    parallelProjection: !!camera.getParallelProjection(),
  }),
  environment: telemetryEnvironment,
});
if (recordingRequested()) {
  installRecorderOverlay(inputRecorder, {
    telemetry: {
      start: startTelemetry,
      stop: stopTelemetry,
      isActive: () => telemetry.isActive(),
      download: (filename) => telemetry.download(filename),
      summary: () => telemetry.summary(),
    },
  });
}

// Driving handles for browser checks. Everything here drives the page the way
// a user or a host would — the camera moves, the panel changes, a frame is
// requested — rather than reaching past it into the library, so a check that
// passes says the assembled page works, not that the modules do.
Object.assign(window, {
  pointCloudRecorder: inputRecorder,
  pointCloudExample: {
    stats: () => ({
      controller: controller?.stats() ?? null,
      adapter: adapter?.stats() ?? null,
      governor: governor?.stats() ?? null,
      lastFrameMs,
      source: loadedName || null,
      sourcePoints: loadedPointCount,
    }),

    telemetry: {
      start: startTelemetry,
      stop: stopTelemetry,
      clear: () => {
        telemetry.clear();
        if (telemetry.isActive()) {
          telemetry.recordState("recording-cleared", telemetryState());
        }
        updateTelemetryStatus();
      },
      isActive: () => telemetry.isActive(),
      mark: markTelemetry,
      environment: (): TelemetryEnvironment => telemetry.summary().environment,
      summary: () => telemetry.summary(),
      trace: (): TelemetryTrace => telemetry.trace(),
      download: () => telemetry.download(),
    },

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
          ? ((
              first.getMapper() as { getScaleFactor?: () => number }
            ).getScaleFactor?.() ?? null)
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
      return loadSource(url, copcSource(url));
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
      /** Pick a support depth from exactly the point prefixes being drawn. */
      pick: (xCssPx: number, yCssPx: number) =>
        controller?.pickPoint(cameraView(), xCssPx, yCssPx) ?? null,
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
        const focalPoint = camera.getFocalPoint();
        interactorStyle.setCenterOfRotation(
          focalPoint[0],
          focalPoint[1],
          focalPoint[2],
        );
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
    /**
     * Fixed point quality bypasses the normalized adaptive governor, so this
     * point-specific presentation control writes the controller directly.
     */
    setDensityFraction: (fraction: number) => {
      fixedDensityFraction = fraction;
      if (budgetMode() === "fixed") {
        controller?.setDensityFraction(fraction);
      }
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
