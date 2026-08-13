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
import vtkInteractorStyleManipulator from "@kitware/vtk.js/Interaction/Style/InteractorStyleManipulator";
import vtkMouseCameraTrackballPanManipulator from "@kitware/vtk.js/Interaction/Manipulators/MouseCameraTrackballPanManipulator";
import vtkMouseCameraTrackballRotateManipulator from "@kitware/vtk.js/Interaction/Manipulators/MouseCameraTrackballRotateManipulator";
import vtkMouseCameraTrackballZoomManipulator from "@kitware/vtk.js/Interaction/Manipulators/MouseCameraTrackballZoomManipulator";
import { getCompressedTextureCapabilities } from "@kitware/vtk.js/Rendering/OpenGL/Texture/compressedFormats";

import {
  createMemoryPool,
  createStreamedSceneCoordinator,
  isSoftwareRenderer,
  type CameraView,
  type StreamedSceneCoordinator,
  type TextureCapabilities,
} from "../../../src";
import { createDecodeWorkers } from "./decodeAssets";

export type SceneHost = {
  readonly coordinator: StreamedSceneCoordinator;
  readonly renderer: unknown;
  readonly camera: any;
  readonly textureCapabilities: TextureCapabilities;
  readonly rendererName: string;
  /** Coalesced repaint request; the only way anything here paints. */
  scheduleRender(): void;
  cameraView(): CameraView;
  /** Notified after every painted frame, with the camera that was drawn. */
  onFrame(listener: (view: CameraView) => void): void;
  /** Frame the camera on a scene-space sphere. */
  lookAt(
    center: readonly [number, number, number],
    distanceMeters: number,
  ): void;
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

  const style = vtkInteractorStyleManipulator.newInstance();
  style.addMouseManipulator(
    vtkMouseCameraTrackballRotateManipulator.newInstance({ button: 1 }),
  );
  style.addMouseManipulator(
    vtkMouseCameraTrackballPanManipulator.newInstance({ button: 2 }),
  );
  style.addMouseManipulator(
    vtkMouseCameraTrackballZoomManipulator.newInstance({
      scrollEnabled: true,
      button: 3,
    }),
  );
  interactor.setInteractorStyle(style);

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
  const frameListeners: ((view: CameraView) => void)[] = [];
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

  const scheduleRender = (): void => {
    if (frameQueued) return;
    frameQueued = true;
    requestAnimationFrame(() => {
      frameQueued = false;
      // A gesture's own animation loop is already painting, and it drains
      // admission itself; a second paint here would double the drain.
      if (interactor.isAnimating()) return;
      coordinator.prepareFrame();
      refreshClippingRange();
      paintStartedAt = performance.now();
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
      const interval =
        lastPresentedAt === null ? null : presentedAt - lastPresentedAt;
      lastPresentedAt = presentedAt;
      const contiguous = Math.max(80, vtkFrameMs * 4);
      if (interval === null || interval <= 0 || interval > contiguous) return;
      coordinator.recordHostFrame({
        hostFrameMs: interval,
        vtkFrameMs,
        capacitySampleEligible:
          coordinator.stats().governor.activity.measurementEligible,
        now: presentedAt,
      });
      if (coordinator.needsFrame()) scheduleRender();
    });
  };

  interactor.onRenderEvent(() => {
    const startedAt = paintStartedAt;
    paintStartedAt = null;
    if (startedAt !== null) {
      lastFrameMs = performance.now() - startedAt;
      paints += 1;
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
    coordinator.prepareFrame();
    refreshClippingRange();
    paintStartedAt = performance.now();
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
    scheduleRender,
    cameraView,
    onFrame: (listener) => frameListeners.push(listener),
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
      camera.setViewUp(0, 0, 1);
      refreshClippingRange();
      style.setCenterOfRotation(x, y, z);
      scheduleRender();
    },
    lastFrameMs: () => lastFrameMs,
    paintCount: () => paints,
    devicePixelRatio: () => currentDevicePixelRatio,
  };
};
