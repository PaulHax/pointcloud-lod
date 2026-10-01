import "@kitware/vtk.js/Rendering/Profiles/Geometry";
import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";

import { createSubmissionScheduler } from "../../src/submissionScheduler";
import { createMeshAdapter } from "../../src/tiles3d/meshAdapter";
import { createMeshPickSet } from "../../src/tiles3d/meshPicking";

const params = new URLSearchParams(window.location.search);
const mode = params.get("mode");
if (
  mode !== "MASK" &&
  mode !== "OPAQUE" &&
  mode !== "BLEND" &&
  mode !== "UNLIT" &&
  mode !== "VERTEX_MASK"
) {
  throw new Error(`unsupported alpha mode ${mode}`);
}

const generic = vtkGenericRenderWindow.newInstance({ background: [0, 0, 0] });
generic.setContainer(document.querySelector("#view"));
generic.resize();
const renderer = generic.getRenderer();
const renderWindow = generic.getRenderWindow();
const scheduler = createSubmissionScheduler({
  scheduleRender: () => {},
  maxBytesPerFrame: 1_000_000,
  maxTimeMsPerFrame: 100,
  now: () => 0,
});
const adapterErrors = [];
const adapter = createMeshAdapter({
  renderer,
  scheduleRender: () => {},
  submissions: scheduler,
  onError: (error) => adapterErrors.push(String(error)),
});
const sampler = {
  magFilter: 9728,
  minFilter: 9728,
  wrapS: 33071,
  wrapT: 33071,
};
const primitive = (
  z,
  color,
  alphaMode,
  texture,
  unlit = false,
  xRange = [-2, 2],
  normalZ = 1,
) => ({
  positions: new Float32Array([
    xRange[0],
    -1,
    z,
    xRange[1],
    -1,
    z,
    xRange[1],
    1,
    z,
    xRange[0],
    1,
    z,
  ]),
  normals: new Float32Array([
    0,
    0,
    normalZ,
    0,
    0,
    normalZ,
    0,
    0,
    normalZ,
    0,
    0,
    normalZ,
  ]),
  uvs: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]),
  indices: new Uint16Array([0, 1, 2, 0, 2, 3]),
  material: {
    baseColorFactor: color,
    ...(texture ? { baseColorTexture: texture } : {}),
    raw: {
      version: 1,
      kind: "gltf-material",
      alphaMode,
      alphaCutoff: 0.5,
      doubleSided: false,
      unlit,
      metallicFactor: 0,
      roughnessFactor: 1,
      emissiveFactor: [0, 0, 0],
    },
  },
});
const background = {
  origin: [0, 0, 0],
  primitives: [primitive(-0.2, [0, 0, 1, 1], "OPAQUE")],
  byteEstimate: { geometry: 0, textures: 0 },
};
const texture = {
  kind: "rgba",
  // A nonzero alpha below the authored cutoff catches reliance on vtk's
  // built-in alpha==0 discard instead of glTF MASK semantics.
  rgba: new Uint8Array([255, 0, 0, 64, 255, 0, 0, 255]),
  width: 2,
  height: 1,
  colorSpace: "srgb",
  sampler,
};
const vertexMaskPrimitives = () => {
  const factorAlpha = 0.8;
  const zero = params.get("vertexAlpha") === "zero";
  const opaqueTexture = {
    ...texture,
    rgba: new Uint8Array([255, 0, 0, 255]),
    width: 1,
  };
  return [zero ? 0 : 0.4999 / factorAlpha, zero ? 1 : 0.5001 / factorAlpha].map(
    (alpha, side) => {
      const result = primitive(
        0,
        [1, 1, 1, factorAlpha],
        "MASK",
        opaqueTexture,
        true,
        side === 0 ? [-2, 0] : [0, 2],
      );
      result.colors = new Float32Array(
        Array.from({ length: 4 }, () => [1, 1, 1, alpha]).flat(),
      );
      result.material.raw.alphaCutoff = Number(params.get("cutoff") ?? 0.5);
      return result;
    },
  );
};
const foreground = {
  origin: [0, 0, 0],
  primitives:
    mode === "UNLIT"
      ? [
          primitive(0, [1, 0, 0, 1], "OPAQUE", undefined, true, [-2, 0], -1),
          primitive(0, [1, 0, 0, 1], "OPAQUE", undefined, false, [0, 2], -1),
        ]
      : mode === "VERTEX_MASK"
        ? vertexMaskPrimitives()
        : [primitive(0, [1, 1, 1, 1], mode, texture)],
  byteEstimate: { geometry: 0, textures: mode === "UNLIT" ? 0 : 8 },
};
adapter.submitTile("background", background);
adapter.submitTile("foreground", foreground);
scheduler.prepareFrame();
adapter.setDrawnTiles(["background", "foreground"]);
const camera = renderer.getActiveCamera();
camera.setParallelProjection(true);
camera.setPosition(0, 0, 1);
camera.setFocalPoint(0, 0, 0);
camera.setViewUp(0, 1, 0);
camera.setParallelScale(1);
renderer.resetCameraClippingRange();
renderWindow.render();
if (mode === "VERTEX_MASK") {
  const matrix = camera.getCompositeProjectionMatrix(2, -1, 1);
  const view = {
    projection: "orthographic",
    viewProj: Array.from(
      { length: 16 },
      (_, i) => matrix[(i % 4) * 4 + Math.floor(i / 4)],
    ),
    position: [...camera.getPosition()],
    parallelScale: camera.getParallelScale(),
    viewportWidthCssPx: 96,
    viewportHeightCssPx: 48,
  };
  const picks = createMeshPickSet();
  picks.replaceDrawn(adapter.submittedTiles());
  const samples = [24, 72];
  document.documentElement.dataset.picks = JSON.stringify(
    samples.map((x) => picks.pick(view, x, 24)),
  );
  document.documentElement.dataset.occlusion = JSON.stringify(
    samples.map((x) => picks.occlusionDepth(view, x, 24)),
  );
}
document.documentElement.dataset.stats = JSON.stringify(adapter.stats());
document.documentElement.dataset.adapterErrors = JSON.stringify(adapterErrors);
document.documentElement.dataset.ready = "true";
