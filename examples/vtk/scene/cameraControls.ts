/**
 * The camera controls every example in this app shares.
 *
 * Left drag pans, right drag orbits, middle drag and the wheel zoom.
 *
 * The stock trackball rolls the camera freely, which loses the horizon on the
 * first diagonal drag over a landscape. Orbiting about the focal point with a
 * fixed world up keeps the scene the way up the data is — which matters as
 * much for a city of buildings as for the point clouds these were written for.
 */

import vtkInteractorStyleManipulator from "@kitware/vtk.js/Interaction/Style/InteractorStyleManipulator";
import vtkMouseCameraTrackballPanManipulator from "@kitware/vtk.js/Interaction/Manipulators/MouseCameraTrackballPanManipulator";
import vtkCompositeMouseManipulator from "@kitware/vtk.js/Interaction/Manipulators/CompositeMouseManipulator";
import vtkCompositeCameraManipulator from "@kitware/vtk.js/Interaction/Manipulators/CompositeCameraManipulator";
import macro from "@kitware/vtk.js/macros";

/** The world's up axis. The scenes here are z-up, and orbiting keeps it. */
export const WORLD_UP: [number, number, number] = [0, 0, 1];

/** Fraction of the eye-to-focus distance a full viewport-height drag covers. */
const DOLLY_PER_VIEWPORT = 1.6;
/** Zoom ratio per wheel notch. */
const DOLLY_PER_NOTCH = 1.12;
/** Orbit travel across one whole viewport. Matches vtk.js's trackball. */
const ORBIT_RADIANS_PER_VIEWPORT = 2 * Math.PI;
/** Keep a small stable margin on either side of the world-up singularity. */
const MAX_ORBIT_PITCH = (89 * Math.PI) / 180;
/** Closest a perspective eye may get, relative to the loaded scene. */
const MIN_DOLLY_DISTANCE_RATIO = 1e-5;
/** Enough coordinate increments to keep eye and focus numerically distinct. */
const MIN_DOLLY_DISTANCE_ULPS = 1024;

export type CameraControlsOptions = {
  readonly interactor: any;
  readonly renderer: any;
  /** Element the pointer is tracked over, for wheel aiming. */
  readonly viewer: HTMLElement;
  readonly canvas: () => HTMLCanvasElement | null;
  /**
   * Extent of what is on screen, in world units. It sets how close the eye may
   * get; a streamed scene's grows as tiles arrive, so it is read per gesture
   * rather than captured. Falls back to the camera's own distance when the
   * scene has nothing in it yet.
   */
  readonly sceneRadius: () => number | null;
};

type Position = { x: number; y: number };

/**
 * Dolly without letting a perspective camera collapse through its focus.
 *
 * vtk.js protects only at 1e-20 world units. Repeated zoom steps can reach
 * that numerical limit, at which point its reconstructed focal point may land
 * behind the eye and make the next pan appear reversed. The usable limit has
 * to be relative to both the scene and the magnitude of its coordinates:
 * scenes here range from local metres to large projected coordinates.
 *
 * The returned factor is what was actually applied. Drag zoom uses it to keep
 * its reversible accumulated ratio honest when the limit clips a step.
 */
const dollyToPosition = (
  requestedFactor: number,
  position: Position,
  renderer: any,
  interactor: any,
  sceneRadius: () => number | null,
): number => {
  if (Number.isNaN(requestedFactor) || requestedFactor <= 0) return 1;

  const dollyCamera = renderer.getActiveCamera();
  let appliedFactor = requestedFactor;
  let minimumDistance = 0;
  let approachDirection: number[] | null = null;
  if (!dollyCamera.getParallelProjection()) {
    const cameraPosition = dollyCamera.getPosition() as number[];
    const focalPoint = dollyCamera.getFocalPoint() as number[];
    const offset = focalPoint.map(
      (value, axis) => value - cameraPosition[axis]!,
    );
    const distance = Math.hypot(...offset);
    if (Number.isFinite(distance) && distance > 0) {
      approachDirection = offset.map((value) => value / distance);
    }
    const sceneScale = sceneRadius() ?? dollyCamera.getDistance();
    const coordinateScale = Math.max(
      sceneScale,
      ...cameraPosition.map(Math.abs),
      ...focalPoint.map(Math.abs),
    );
    minimumDistance = Math.max(
      sceneScale * MIN_DOLLY_DISTANCE_RATIO,
      coordinateScale * Number.EPSILON * MIN_DOLLY_DISTANCE_ULPS,
      Number.MIN_VALUE,
    );
    if (requestedFactor > 1) {
      const maximumFactor = Math.max(
        dollyCamera.getDistance() / minimumDistance,
        1,
      );
      appliedFactor = Math.min(requestedFactor, maximumFactor);
    }
  }

  if (!Number.isFinite(appliedFactor) || appliedFactor === 1) return 1;
  vtkInteractorStyleManipulator.dollyToPosition(
    appliedFactor,
    position,
    renderer,
    interactor,
  );
  if (approachDirection !== null) {
    const focalPoint = dollyCamera.getFocalPoint() as number[];
    const cameraPosition = dollyCamera.getPosition() as number[];
    const nextOffset = focalPoint.map(
      (value, axis) => value - cameraPosition[axis]!,
    );
    const signedDistance = nextOffset.reduce(
      (sum, value, axis) => sum + value * approachDirection![axis]!,
      0,
    );
    // Clamp the result as well as the requested factor. Zoom-to-cursor moves
    // the focal point before dollying, and large projected coordinates can
    // round an exactly capped step onto or through it.
    if (!Number.isFinite(signedDistance) || signedDistance < minimumDistance) {
      dollyCamera.setPosition(
        focalPoint[0]! - approachDirection[0]! * minimumDistance,
        focalPoint[1]! - approachDirection[1]! * minimumDistance,
        focalPoint[2]! - approachDirection[2]! * minimumDistance,
      );
      renderer.resetCameraClippingRange();
    }
    const panCenter = dollyCamera.getFocalPoint() as number[];
    interactor
      .getInteractorStyle()
      .setCenterOfRotation(panCenter[0]!, panCenter[1]!, panCenter[2]!);
  }
  return appliedFactor;
};

/**
 * World-up orbit with pitch constrained before either pole.
 *
 * The focal point is explicitly reprojected onto the screen-center ray when
 * an orbit begins. Zoom-to-cursor may translate it, but orbit never inherits
 * an off-center pivot from another interaction.
 *
 * vtk.js's world-up trackball keeps the horizon level but lets elevation pass
 * through ±90°. At the pole the screen-right axis is undefined; crossing it
 * reverses the horizontal basis and makes the next orbit or pan feel flipped.
 * Keeping pitch as an explicit spherical coordinate removes that singular
 * transition while retaining vtk.js's one-turn-per-viewport sensitivity.
 */
const createOrbitManipulator = (initialValues: object): any => {
  const publicAPI = {};
  const model = { ...initialValues };
  macro.obj(publicAPI, model);
  vtkCompositeMouseManipulator.extend(publicAPI, model, initialValues);
  vtkCompositeCameraManipulator.extend(publicAPI, model, initialValues);

  let previous = { x: 0, y: 0 };
  const api = publicAPI as {
    onButtonDown: (i: any, r: any, p: Position) => void;
    onMouseMove: (i: any, r: any, p: Position | null) => void;
  };
  api.onButtonDown = (interactor, renderer, position) => {
    const orbitCamera = renderer.getActiveCamera();
    const focalPoint = orbitCamera.getFocalPoint() as number[];
    const style = interactor.getInteractorStyle();
    const displayFocal = style.computeWorldToDisplay(
      renderer,
      focalPoint[0],
      focalPoint[1],
      focalPoint[2],
    ) as number[];
    const viewport = renderer.getViewport() as number[];
    const [viewWidth, viewHeight] = interactor.getView().getSize() as number[];
    const centeredFocal = style.computeDisplayToWorld(
      renderer,
      ((viewport[0]! + viewport[2]!) * viewWidth!) / 2,
      ((viewport[1]! + viewport[3]!) * viewHeight!) / 2,
      displayFocal[2],
    ) as number[];
    if (centeredFocal.slice(0, 3).every(Number.isFinite)) {
      orbitCamera.setFocalPoint(
        centeredFocal[0],
        centeredFocal[1],
        centeredFocal[2],
      );
      style.setCenterOfRotation(
        centeredFocal[0],
        centeredFocal[1],
        centeredFocal[2],
      );
    }
    previous = position;
  };
  api.onMouseMove = (interactor, renderer, position) => {
    if (!position) return;
    const orbitCamera = renderer.getActiveCamera();
    const cameraPosition = orbitCamera.getPosition() as number[];
    const focalPoint = orbitCamera.getFocalPoint() as number[];
    const offset = cameraPosition.map(
      (value, axis) => value - focalPoint[axis]!,
    );
    const distance = Math.hypot(...offset);
    const horizontalDistance = Math.hypot(offset[0]!, offset[1]!);
    if (distance === 0 || horizontalDistance === 0) {
      previous = position;
      return;
    }

    const [width, height] = interactor
      .getView()
      .getViewportSize(renderer) as number[];
    const yaw =
      ((previous.x - position.x) / Math.max(width!, 1)) *
      ORBIT_RADIANS_PER_VIEWPORT;
    const pitchStep =
      ((previous.y - position.y) / Math.max(height!, 1)) *
      ORBIT_RADIANS_PER_VIEWPORT;
    const pitch = Math.max(
      -MAX_ORBIT_PITCH,
      Math.min(
        MAX_ORBIT_PITCH,
        Math.asin(Math.max(-1, Math.min(1, offset[2]! / distance))) + pitchStep,
      ),
    );

    // Yaw the horizontal eye direction around world up, then rebuild the eye
    // offset at the constrained pitch without changing orbit distance.
    const horizontalX = offset[0]! / horizontalDistance;
    const horizontalY = offset[1]! / horizontalDistance;
    const cosineYaw = Math.cos(yaw);
    const sineYaw = Math.sin(yaw);
    const yawedX = horizontalX * cosineYaw - horizontalY * sineYaw;
    const yawedY = horizontalX * sineYaw + horizontalY * cosineYaw;
    const horizontalRadius = Math.cos(pitch) * distance;
    orbitCamera.setPosition(
      focalPoint[0]! + yawedX * horizontalRadius,
      focalPoint[1]! + yawedY * horizontalRadius,
      focalPoint[2]! + Math.sin(pitch) * distance,
    );
    orbitCamera.setViewUp(...WORLD_UP);
    orbitCamera.orthogonalizeViewUp();
    renderer.resetCameraClippingRange();
    if (interactor.getLightFollowCamera()) {
      renderer.updateLightsGeometryToFollowCamera();
    }
    previous = position;
  };
  return publicAPI;
};

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
const createDollyManipulator = (
  initialValues: object,
  pointerPosition: () => Position | null,
  sceneRadius: () => number | null,
): any => {
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
    onButtonDown: (i: any, r: any, p: Position) => void;
    onMouseMove: (i: any, r: any, p: Position | null) => void;
    onScroll: (i: any, r: any, delta: number, p?: Position | null) => void;
  };
  api.onButtonDown = (_interactor, _renderer, position) => {
    start = position;
    applied = 1;
  };
  api.onScroll = (scrollInteractor, renderer, delta, position) => {
    if (!delta) return;
    const at = position ?? pointerPosition();
    if (!at) return;
    dollyToPosition(
      DOLLY_PER_NOTCH ** -delta,
      at,
      renderer,
      scrollInteractor,
      sceneRadius,
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
    // Toward where the gesture began, so the drag pulls that point in rather
    // than whatever happens to be at the centre.
    applied *= dollyToPosition(step, start, renderer, interactor, sceneRadius);
  };
  return publicAPI;
};

/**
 * The cursor, in the display coordinates vtk.js works in.
 *
 * The interactor caches a mouse position only while a button is held, so a
 * wheel turn with nothing pressed — the ordinary case — reaches a manipulator
 * with no position at all, and vtk.js's own zoom-to-mouse silently does
 * nothing. Tracking the pointer here is what makes the wheel able to aim.
 */
const trackPointer = (
  viewer: HTMLElement,
  canvas: () => HTMLCanvasElement | null,
): (() => Position | null) => {
  let pointer: Position | null = null;
  viewer.addEventListener("pointermove", (event: PointerEvent) => {
    const element = canvas();
    if (!element) return;
    const bounds = element.getBoundingClientRect();
    if (bounds.width === 0 || bounds.height === 0) return;
    pointer = {
      x: (element.width / bounds.width) * (event.clientX - bounds.left),
      y:
        (element.height / bounds.height) *
        (bounds.height - event.clientY + bounds.top),
    };
  });
  return () => pointer;
};

/** Installs the controls and returns the style, whose centre of rotation the
 * page sets whenever it reframes. */
export const installCameraControls = (options: CameraControlsOptions): any => {
  const pointerPosition = trackPointer(options.viewer, options.canvas);
  const style = vtkInteractorStyleManipulator.newInstance();
  style.addMouseManipulator(
    vtkMouseCameraTrackballPanManipulator.newInstance({ button: 1 }),
  );
  style.addMouseManipulator(createOrbitManipulator({ button: 3 }));
  // Middle drag and the wheel both zoom toward the cursor: the camera moves
  // along the ray through whatever is under the pointer, so that point stays
  // roughly put and the view closes in on it rather than on the screen centre.
  style.addMouseManipulator(
    createDollyManipulator(
      { button: 2, scrollEnabled: true },
      pointerPosition,
      options.sceneRadius,
    ),
  );
  options.interactor.setInteractorStyle(style);
  return style;
};
