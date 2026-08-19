/**
 * The vtk.js half of a streamed scene: one render window, one coordinator, and
 * the frame loop that ties them together.
 *
 * Both streamed-scene examples build on this because sharing a budget is the
 * whole point of the coordinator — every member registered here divides one
 * memory pool, one submission scheduler and one view governor, and sees the
 * same camera in the same scene frame.
 *
 * The loop obeys two contracts. `prepareFrame()` is one bounded admission
 * drain per painted frame, so it runs once immediately before each paint —
 * ours and the ones the interactor's own animation loop runs during a gesture,
 * which are exactly the frames the moving regime has to measure. And the
 * coordinator never schedules a frame: the page paints, times the paint,
 * reports it, and asks `needsFrame()` whether another one is owed.
 */

import "@kitware/vtk.js/Rendering/Profiles/Geometry";
import vtkFullScreenRenderWindow from "@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow";
import { getCompressedTextureCapabilities } from "@kitware/vtk.js/Rendering/OpenGL/Texture/compressedFormats";

import {
  createGpuFrameTimer,
  createMemoryPool,
  createStreamedSceneCoordinator,
  isSoftwareRenderer,
  type CameraView,
  type GpuFrameTimer,
  type StreamedSceneCoordinator,
  type TextureCapabilities,
} from "../../../src";
import { WORLD_UP, installCameraControls } from "../../../src/vtk";
import { createDecodeWorkers } from "./decodeAssets";

/**
 * One presented frame, as the host measured it.
 *
 * `hostFrameMs` is null exactly when the interval was rejected as idle and
 * never reached the governor, which a measurement must be able to tell apart
 * from a frame that cost nothing.
 */
export type FrameReport = {
  readonly id: number;
  readonly presentedAtMs: number;
  readonly hostFrameMs: number | null;
  readonly vtkFrameMs: number;
  readonly paints: number;
  readonly capacitySampleEligible: boolean;
  readonly reportedToGovernor: boolean;
  /** A GPU timer query is outstanding; its duration arrives on resolution. */
  readonly gpuPending: boolean;
};

export type GpuFrameResolution = {
  /** The `FrameReport.id` this duration belongs to. */
  readonly id: number;
  readonly status: "valid" | "disjoint" | "error";
  readonly gpuMs: number | null;
};

/**
 * An observer of painted frames.
 *
 * Attaching one is the only thing that makes the host issue GPU timer
 * queries, so an ordinary session runs the same code path it always did — a
 * measurement that changed the frame loop would be measuring itself.
 */
export type FrameProbe = {
  readonly onFrame: (report: FrameReport) => void;
  readonly onGpuResolved: (resolution: GpuFrameResolution) => void;
};

export type SceneHost = {
  readonly coordinator: StreamedSceneCoordinator;
  readonly renderer: unknown;
  readonly camera: any;
  readonly textureCapabilities: TextureCapabilities;
  readonly rendererName: string;
  /** The live WebGL context, for callers that report on the machine drawing. */
  glContext(): WebGL2RenderingContext | null;
  /** Coalesced repaint request; the only way anything here paints. */
  scheduleRender(): void;
  cameraView(): CameraView;
  /** Notified immediately before member preparation for the frame. */
  onBeforeFrame(
    listener: (view: CameraView, devicePixelRatio: number) => void,
  ): void;
  /** Notified after every painted frame, with the camera that was drawn. */
  onFrame(listener: (view: CameraView) => void): void;
  /** Notified when a painted frame reaches the browser presentation clock. */
  onPresentation(listener: (presentedAt: number) => void): void;
  /** Attach or clear the frame observer. Null restores the unmeasured loop. */
  setFrameProbe(probe: FrameProbe | null): void;
  /** True when this context can time the GPU rather than only the CPU. */
  gpuTimingSupported(): boolean;
  /** Frame the camera on a scene-space sphere. */
  lookAt(
    center: readonly [number, number, number],
    distanceMeters: number,
  ): void;
  /** Frame every currently visible prop; false while the scene is empty. */
  frameVisible(): boolean;
  lastFrameMs(): number;
  paintCount(): number;
  devicePixelRatio(): number;
};

const transpose = (m: ArrayLike<number>): number[] => [
  m[0]!,
  m[4]!,
  m[8]!,
  m[12]!,
  m[1]!,
  m[5]!,
  m[9]!,
  m[13]!,
  m[2]!,
  m[6]!,
  m[10]!,
  m[14]!,
  m[3]!,
  m[7]!,
  m[11]!,
  m[15]!,
];

/**
 * A software rasterizer reports compressed formats it will transcode on the
 * CPU, so probing it would pick a format for a GPU that is not there. Those
 * contexts take the RGBA path, which is the same colour either way.
 */
const probeTextureCapabilities = (
  gl: WebGL2RenderingContext | null,
): { capabilities: TextureCapabilities; rendererName: string } => {
  if (!gl) {
    return {
      capabilities: {
        capabilityKey: "compressed-texture-v1:rgba",
        compressedFormats: [],
      },
      rendererName: "unknown",
    };
  }
  const debug = gl.getExtension("WEBGL_debug_renderer_info");
  const rendererName = String(
    (debug && gl.getParameter(debug.UNMASKED_RENDERER_WEBGL)) ??
      gl.getParameter(gl.RENDERER),
  );
  return {
    capabilities: isSoftwareRenderer(rendererName)
      ? { capabilityKey: "compressed-texture-v1:rgba", compressedFormats: [] }
      : getCompressedTextureCapabilities(gl),
    rendererName,
  };
};

export const createSceneHost = (container: HTMLElement): SceneHost => {
  const fullScreen = vtkFullScreenRenderWindow.newInstance({
    rootContainer: container,
    containerStyle: { height: "100%", width: "100%", position: "relative" },
    background: [0.08, 0.09, 0.11],
  });
  const renderer = fullScreen.getRenderer();
  const renderWindow = fullScreen.getRenderWindow();
  const openGlRenderWindow = fullScreen.getApiSpecificRenderWindow();
  const interactor = renderWindow.getInteractor();
  const camera = renderer.getActiveCamera();

  /**
   * Half the diagonal of what is drawn, which is what bounds how close the eye
   * may dolly. A streamed scene has nothing in it when the controls are
   * installed and grows with every tile, so this is read per gesture.
   */
  const sceneRadius = (): number | null => {
    if (renderer.getActors().length === 0) return null;
    const b = renderer.computeVisiblePropBounds() as number[];
    const radius = Math.hypot(b[1]! - b[0]!, b[3]! - b[2]!, b[5]! - b[4]!) / 2;
    return Number.isFinite(radius) && radius > 0 ? radius : null;
  };

  const style = installCameraControls({
    interactor,
    renderer,
    viewer: container,
    canvas: () => openGlRenderWindow.getCanvas(),
    sceneRadius,
  });

  const { capabilities, rendererName } = probeTextureCapabilities(
    openGlRenderWindow.getContext?.() ?? null,
  );

  let currentDevicePixelRatio = window.devicePixelRatio;
  let frameQueued = false;
  let paintStartedAt: number | null = null;
  let lastFrameMs = 0;
  let paints = 0;
  let lastPresentedAt: number | null = null;
  let presentationQueued = false;
  let frameSerial = 0;
  let probe: FrameProbe | null = null;
  let presentationSerial = 0;
  let paintsSincePresentation = 0;
  let pendingQueryIds: number[] = [];
  const frameListeners: ((view: CameraView) => void)[] = [];
  const presentationListeners: ((presentedAt: number) => void)[] = [];
  const beforeFrameListeners: ((
    view: CameraView,
    devicePixelRatio: number,
  ) => void)[] = [];
  const renderedCameras = new Map<unknown, CameraView>();

  /**
   * A streamed scene's extent is not known when the camera is first placed and
   * grows with every tile, so the clipping range is re-derived each frame
   * rather than at load. Leaving it at the default keeps a far plane of 1000
   * metres in front of a city that is kilometres deep, and the view shows a
   * strip of ground with everything behind it clipped away.
   */
  const refreshClippingRange = (): void => {
    if (renderer.getActors().length > 0) renderer.resetCameraClippingRange();
  };

  /**
   * Outstanding GPU queries, grouped by the presentation they were painted
   * for. A presentation can carry more than one paint, and its duration is
   * their sum; the group resolves once every query in it has come back.
   */
  const gpuGroups = new Map<
    number,
    {
      readonly remaining: Set<number>;
      status: GpuFrameResolution["status"];
      gpuMs: number;
    }
  >();
  const gpuGroupByQuery = new Map<number, number>();

  const gpuTimer: GpuFrameTimer = createGpuFrameTimer(
    openGlRenderWindow.getContext?.() ?? null,
    {
      onResult: (result) => {
        const groupId = gpuGroupByQuery.get(result.id);
        if (groupId === undefined) return;
        gpuGroupByQuery.delete(result.id);
        const group = gpuGroups.get(groupId);
        if (group === undefined) return;
        group.remaining.delete(result.id);
        if (result.status !== "valid") group.status = result.status;
        else group.gpuMs += result.gpuMs ?? 0;
        if (group.remaining.size > 0) return;
        gpuGroups.delete(groupId);
        probe?.onGpuResolved({
          id: groupId,
          status: group.status,
          gpuMs: group.status === "valid" ? group.gpuMs : null,
        });
      },
    },
  );

  /** Open a query around the paint that is about to run, when observed. */
  const beginPaint = (): void => {
    paintStartedAt = performance.now();
    if (probe === null) return;
    const id = gpuTimer.begin();
    if (id !== null) pendingQueryIds.push(id);
  };

  const scheduleRender = (): void => {
    if (frameQueued) return;
    frameQueued = true;
    requestAnimationFrame(() => {
      frameQueued = false;
      // A gesture's own animation loop is already painting, and it drains
      // admission itself; a second paint here would double the drain.
      if (interactor.isAnimating()) return;
      const view = cameraView();
      for (const listener of beforeFrameListeners)
        listener(view, currentDevicePixelRatio);
      coordinator.prepareFrame(++frameSerial);
      refreshClippingRange();
      beginPaint();
      interactor.render();
    });
  };

  const coordinator = createStreamedSceneCoordinator({
    scheduleRender,
    memory: createMemoryPool(),
    workers: createDecodeWorkers(),
    textureCapabilities: capabilities,
    devicePixelRatio: currentDevicePixelRatio,
  });

  const cameraView = (): CameraView => {
    const width = Math.max(1, container.clientWidth);
    const height = Math.max(1, container.clientHeight);
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

  /**
   * Displayed cadence is what the governor learns from, and WebGL submission
   * time cannot say whether a frame made presentation, so the report waits for
   * the next presentation tick and carries the interval it measured there.
   *
   * Two things must not reach the governor. An interval spanning idle time is
   * not a frame cost — a page that painted, sat still, and painted again has
   * not slowed down — and reporting it drives quality to the floor over an
   * idle page that is in fact drawing in under a millisecond. An interval far
   * longer than the work done inside it is that idle case, so it only
   * re-anchors the clock, while a genuinely expensive frame still reports.
   * Separately, a frame that carried tile work or a resource change was not
   * measuring steady-state cost, which the governor's activity gate tracks.
   */
  const reportFrame = (vtkFrameMs: number): void => {
    if (presentationQueued) return;
    presentationQueued = true;
    requestAnimationFrame((presentedAt) => {
      presentationQueued = false;
      for (const listener of presentationListeners) listener(presentedAt);
      const interval =
        lastPresentedAt === null ? null : presentedAt - lastPresentedAt;
      lastPresentedAt = presentedAt;
      const contiguous = Math.max(80, vtkFrameMs * 4);
      const usable =
        interval !== null && interval > 0 && interval <= contiguous;
      const eligible =
        coordinator.stats().governor.activity.measurementEligible;
      if (usable) {
        coordinator.recordHostFrame({
          hostFrameMs: interval,
          vtkFrameMs,
          capacitySampleEligible: eligible,
          now: presentedAt,
        });
      }
      // Reported after the governor has been told, so an observer sees the
      // same ordering the adaptive loop ran in, and reported for rejected
      // intervals too: a benchmark counts every frame the display showed.
      if (probe !== null) {
        const queryIds = pendingQueryIds;
        pendingQueryIds = [];
        const id = ++presentationSerial;
        if (queryIds.length > 0) {
          gpuGroups.set(id, {
            remaining: new Set(queryIds),
            status: "valid",
            gpuMs: 0,
          });
          for (const queryId of queryIds) gpuGroupByQuery.set(queryId, id);
        }
        probe.onFrame({
          id,
          presentedAtMs: presentedAt,
          hostFrameMs: usable ? interval : null,
          vtkFrameMs,
          paints: Math.max(1, paintsSincePresentation),
          capacitySampleEligible: eligible,
          reportedToGovernor: usable,
          gpuPending: queryIds.length > 0,
        });
      }
      paintsSincePresentation = 0;
      if (usable && coordinator.needsFrame()) scheduleRender();
    });
  };

  interactor.onRenderEvent(() => {
    const startedAt = paintStartedAt;
    paintStartedAt = null;
    if (probe !== null) gpuTimer.end();
    if (startedAt !== null) {
      lastFrameMs = performance.now() - startedAt;
      paints += 1;
      paintsSincePresentation += 1;
    }
    const view = cameraView();
    renderedCameras.set(renderer, view);
    coordinator.noteRenderedCameras(renderedCameras);
    for (const listener of frameListeners) listener(view);
    if (startedAt !== null) reportFrame(lastFrameMs);
    if (coordinator.needsFrame()) scheduleRender();
  });

  interactor.onStartAnimation(() => coordinator.beginInteraction());
  interactor.onAnimation(() => {
    // Fires before the gesture frame paints, which is where its drain belongs.
    const view = cameraView();
    for (const listener of beforeFrameListeners)
      listener(view, currentDevicePixelRatio);
    coordinator.prepareFrame(++frameSerial);
    refreshClippingRange();
    beginPaint();
  });
  interactor.onEndAnimation(() => {
    coordinator.endInteraction();
    paintStartedAt = performance.now();
  });

  window.addEventListener("resize", () => {
    currentDevicePixelRatio = window.devicePixelRatio;
    scheduleRender();
  });

  return {
    coordinator,
    renderer,
    camera,
    textureCapabilities: capabilities,
    rendererName,
    glContext: () => openGlRenderWindow.getContext?.() ?? null,
    scheduleRender,
    cameraView,
    onBeforeFrame: (listener) => beforeFrameListeners.push(listener),
    onFrame: (listener) => frameListeners.push(listener),
    onPresentation: (listener) => presentationListeners.push(listener),
    setFrameProbe(next) {
      probe = next;
      if (next !== null) return;
      pendingQueryIds = [];
      gpuGroups.clear();
      gpuGroupByQuery.clear();
    },
    gpuTimingSupported: () => gpuTimer.supported,
    lookAt(center, distanceMeters) {
      const [x, y, z] = center;
      camera.setFocalPoint(x, y, z);
      // From the north-east, tilted enough to show facades rather than roofs,
      // which is where mesh refinement is legible. Looking south-west also
      // keeps the AHN sheet the combined example streams inside the view: its
      // northern edge runs close behind every place's origin.
      camera.setPosition(
        x + distanceMeters * 0.6,
        y + distanceMeters * 0.7,
        z + distanceMeters * 0.45,
      );
      camera.setViewUp(...WORLD_UP);
      refreshClippingRange();
      style.setCenterOfRotation(x, y, z);
      scheduleRender();
    },
    frameVisible() {
      if (renderer.getActors().length === 0) return false;
      const bounds = renderer.computeVisiblePropBounds() as number[];
      if (
        bounds.length < 6 ||
        !bounds.slice(0, 6).every(Number.isFinite) ||
        bounds[1]! < bounds[0]! ||
        bounds[3]! < bounds[2]! ||
        bounds[5]! < bounds[4]!
      ) {
        return false;
      }
      const center: [number, number, number] = [
        (bounds[0]! + bounds[1]!) / 2,
        (bounds[2]! + bounds[3]!) / 2,
        (bounds[4]! + bounds[5]!) / 2,
      ];
      const radius =
        Math.hypot(
          bounds[1]! - bounds[0]!,
          bounds[3]! - bounds[2]!,
          bounds[5]! - bounds[4]!,
        ) / 2;
      if (!(radius > 0)) return false;
      const distance =
        (radius / Math.tan((camera.getViewAngle() * Math.PI) / 360)) * 1.25;
      const [x, y, z] = center;
      camera.setFocalPoint(x, y, z);
      camera.setPosition(
        x + distance * 0.6,
        y + distance * 0.7,
        z + distance * 0.45,
      );
      camera.setViewUp(...WORLD_UP);
      refreshClippingRange();
      style.setCenterOfRotation(x, y, z);
      scheduleRender();
      return true;
    },
    lastFrameMs: () => lastFrameMs,
    paintCount: () => paints,
    devicePixelRatio: () => currentDevicePixelRatio,
  };
};
