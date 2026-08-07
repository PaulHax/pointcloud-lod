/**
 * Minimal vtk.js integration: one worker-backed COPC source, one fixed-budget
 * LOD controller, and one renderer adapter. The complete example one directory
 * up adds adaptive quality, custom interaction, diagnostics, and telemetry.
 */

import "@kitware/vtk.js/Rendering/Profiles/Geometry";
import vtkFullScreenRenderWindow from "@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow";

import {
  ROOT_KEY,
  createCopcWorkerTileSource,
  createLodController,
  type Bounds,
  type CameraView,
  type LodController,
  type TileSource,
} from "../../../src";
import {
  createRendererAdapter,
  type RendererAdapter,
} from "../../../src/rendererAdapter";

const required = <T extends Element>(selector: string): T => {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing example element: ${selector}`);
  return element;
};

const viewer = required<HTMLElement>("#viewer");
const urlInput = required<HTMLInputElement>("#cloud-url");
const loadUrlButton = required<HTMLButtonElement>("#load-url");
const fileInput = required<HTMLInputElement>("#cloud-file");
const status = required<HTMLOutputElement>("#status");

const fullScreen = vtkFullScreenRenderWindow.newInstance({
  rootContainer: viewer,
  background: [0.035, 0.055, 0.075],
});
const renderer = fullScreen.getRenderer();
const renderWindow = fullScreen.getRenderWindow();
const interactor = renderWindow.getInteractor();
const camera = renderer.getActiveCamera();

let source: TileSource | null = null;
let controller: LodController | null = null;
let adapter: RendererAdapter | null = null;
let loadGeneration = 0;
let frameQueued = false;
let clippingDirty = false;

const setStatus = (message: string, error = false): void => {
  status.value = message;
  status.classList.toggle("error", error);
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

const cameraView = (): CameraView => {
  const width = Math.max(1, viewer.clientWidth);
  const height = Math.max(1, viewer.clientHeight);
  return {
    projection: "perspective",
    viewProj: transpose(
      camera.getCompositeProjectionMatrix(width / height, -1, 1),
    ),
    position: [...camera.getPosition()] as [number, number, number],
    fovY: (camera.getViewAngle() * Math.PI) / 180,
    viewportWidthCssPx: width,
    viewportHeightCssPx: height,
  };
};

const scheduleRender = (): void => {
  if (frameQueued) return;
  frameQueued = true;
  requestAnimationFrame(() => {
    frameQueued = false;
    if (clippingDirty && renderer.getActors().length > 0) {
      clippingDirty = false;
      renderer.resetCameraClippingRange();
    }
    interactor.render();
  });
};

const usableBounds = (bounds: Bounds | undefined): bounds is Bounds =>
  bounds !== undefined &&
  bounds.min.every((minimum, axis) => {
    const maximum = bounds.max[axis]!;
    return (
      Number.isFinite(minimum) && Number.isFinite(maximum) && maximum >= minimum
    );
  });

const sourceBounds = async (nextSource: TileSource): Promise<Bounds> => {
  const stated = nextSource.metadata().bounds;
  if (usableBounds(stated)) return stated;
  const root = (await nextSource.nodes(ROOT_KEY)).find(
    ({ key }) => key.level === 0 && key.x === 0 && key.y === 0 && key.z === 0,
  );
  if (!root) throw new Error("The COPC hierarchy has no root node");
  return root.bounds;
};

const frame = (bounds: Bounds): void => {
  const center = bounds.min.map(
    (minimum, axis) => (minimum + bounds.max[axis]!) / 2,
  );
  const radius = Math.max(
    Math.hypot(
      ...bounds.max.map((maximum, axis) => maximum - bounds.min[axis]!),
    ) / 2,
    1e-6,
  );
  const distance = radius / Math.tan((38 * Math.PI) / 360);
  const offset = [0.25, -0.433, 0.866];
  camera.setFocalPoint(center[0]!, center[1]!, center[2]!);
  camera.setPosition(
    center[0]! + offset[0]! * distance,
    center[1]! + offset[1]! * distance,
    center[2]! + offset[2]! * distance,
  );
  camera.setViewUp(0, 0, 1);
  camera.setViewAngle(38);
  renderer.resetCameraClippingRange([
    bounds.min[0],
    bounds.max[0],
    bounds.min[1],
    bounds.max[1],
    bounds.min[2],
    bounds.max[2],
  ]);
};

const disposeCloud = (): void => {
  loadGeneration += 1;
  controller?.dispose();
  adapter?.dispose();
  source?.dispose?.();
  controller = null;
  adapter = null;
  source = null;
};

const loadCloud = async (input: string | Blob, name: string): Promise<void> => {
  disposeCloud();
  const generation = ++loadGeneration;
  setStatus(`Reading ${name}…`);
  try {
    const opened = await createCopcWorkerTileSource({
      source: input,
      createWorker: () =>
        new Worker(new URL("../copc.worker.ts", import.meta.url), {
          type: "module",
        }),
      lazPerfWasmUrl: new URL("/laz-perf.wasm", window.location.href).href,
    });
    if (generation !== loadGeneration) {
      opened.dispose?.();
      return;
    }
    source = opened;
    frame(await sourceBounds(opened));
    adapter = createRendererAdapter({
      renderer,
      scheduleRender,
      devicePixelRatio: window.devicePixelRatio,
    });
    controller = createLodController({
      source: opened,
      pointBudget: 2_000_000,
      presentation: { mode: "fixed", diameterCssPx: 2 },
      onTiles: (batch) => {
        adapter?.applyBatch(batch);
        clippingDirty = true;
      },
      onDrawPlan: (plan) => adapter?.applyDrawPlan(plan),
      onPointDiameterCssPx: (diameter) =>
        adapter?.setPointDiameterCssPx(diameter),
      onError: (error) => setStatus(String(error), true),
      scheduleRender,
    });
    controller.setCamera(cameraView());
    setStatus(
      `${name}: ${opened.metadata().pointCount.toLocaleString()} points`,
    );
    scheduleRender();
  } catch (error) {
    if (generation !== loadGeneration) return;
    disposeCloud();
    setStatus(error instanceof Error ? error.message : String(error), true);
  }
};

interactor.onRenderEvent(() => controller?.setCamera(cameraView()));

loadUrlButton.addEventListener("click", () => {
  const url = urlInput.value.trim();
  if (!url) return setStatus("Enter a COPC URL.", true);
  void loadCloud(url, url);
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) void loadCloud(file, file.name);
});

window.addEventListener("resize", () => {
  adapter?.setDevicePixelRatio(window.devicePixelRatio);
  controller?.setCamera(cameraView());
  scheduleRender();
});
window.addEventListener("beforeunload", () => disposeCloud());

const initialUrl = new URL(window.location.href).searchParams.get("url");
if (initialUrl) {
  urlInput.value = initialUrl;
  void loadCloud(initialUrl, initialUrl);
}
