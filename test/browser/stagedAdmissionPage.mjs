import "@kitware/vtk.js/Rendering/Profiles/Geometry";
import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import { createMemoryPool } from "../../src/memoryPool";
import { createPointCloudMember } from "../../src/pointCloudMember";
import { createStreamedSceneCoordinator } from "../../src/streamedSceneCoordinator";

const total = 20_000;
const positions = new Float32Array(total * 3);
const rgb = new Uint8Array(total * 3);
const rgba = new Uint8Array(total * 4);
for (let i = 0; i < total - 1; i++) {
  const angle = i * 2.399963229728653;
  const radius = 0.75 + (i % 17) / 320;
  positions.set([radius * Math.cos(angle), radius * Math.sin(angle), 0], i * 3);
  rgb.set([i % 251, (i * 7) % 251, (i * 13) % 251], i * 3);
}
rgb.set([255, 0, 0], (total - 1) * 3);
for (let i = 0; i < total; i++)
  rgba.set([...rgb.subarray(i * 3, i * 3 + 3), 255], i * 4);
const source = {
  metadata: () => ({ pointCount: total }),
  nodes: async () => [
    {
      key: { level: 0, x: 0, y: 0, z: 0 },
      pointCount: total,
      bounds: { min: [-0.9, -0.9, -0.1], max: [0.9, 0.9, 0.1] },
      spacing: 0.01,
      children: [],
    },
  ],
  loadTile: async () => ({
    origin: [0, 0, 0],
    positions,
    rgb,
    pointCount: total,
  }),
};
const generic = vtkGenericRenderWindow.newInstance({ background: [0, 0, 0] });
generic.setContainer(document.querySelector("#view"));
generic.resize();
const renderer = generic.getRenderer();
const renderWindow = generic.getRenderWindow();
const camera = renderer.getActiveCamera();
camera.setParallelProjection(true);
camera.setPosition(0, 0, 1);
camera.setFocalPoint(0, 0, 0);
camera.setViewUp(0, 1, 0);
camera.setParallelScale(1);
camera.setClippingRange(0.01, 10);
renderWindow.render();
const gl = renderWindow.getViews()[0].getContext();
const expected = {
  positions: new Uint8Array(positions.buffer),
  colors: rgba,
};
const buffers = new Map();
const allocations = { positions: 0, colors: 0 };
const writes = [];
const bufferData = gl.bufferData.bind(gl);
gl.bufferData = (target, data, ...rest) => {
  if (target === gl.ARRAY_BUFFER && typeof data === "number") {
    const kind = Object.keys(expected).find(
      (key) => expected[key].length === data,
    );
    if (kind) {
      buffers.set(gl.getParameter(gl.ARRAY_BUFFER_BINDING), kind);
      allocations[kind]++;
    }
  }
  return bufferData(target, data, ...rest);
};
const bufferSubData = gl.bufferSubData.bind(gl);
gl.bufferSubData = (target, offset, data, ...rest) => {
  const result = bufferSubData(target, offset, data, ...rest);
  const kind =
    target === gl.ARRAY_BUFFER
      ? buffers.get(gl.getParameter(gl.ARRAY_BUFFER_BINDING))
      : undefined;
  if (kind) {
    const actual = new Uint8Array(data.byteLength);
    gl.getBufferSubData(target, offset, actual);
    const reference = expected[kind].subarray(offset, offset + actual.length);
    writes.push({
      kind,
      offset,
      bytes: actual.length,
      mismatches: actual.reduce(
        (count, value, i) => count + (value !== reference[i]),
        0,
      ),
    });
  }
  return result;
};
const coordinator = createStreamedSceneCoordinator({
  scheduleRender() {},
  memory: createMemoryPool({ totalBytes: 64_000_000 }),
});
const config = {
  source,
  adaptive: false,
  pointBudget: total,
  selectionDelayMs: 0,
  presentation: { mode: "fixed", diameterCssPx: 2 },
};
const context = coordinator.context(renderer);
const member = createPointCloudMember(context, config);
coordinator.register(member);
const matrix = camera.getCompositeProjectionMatrix(1, -1, 1);
const view = {
  projection: "orthographic",
  viewProj: Array.from(
    { length: 16 },
    (_, i) => matrix[(i % 4) * 4 + Math.floor(i / 4)],
  ),
  position: [0, 0, 1],
  parallelScale: 1,
  viewportWidthCssPx: 512,
  viewportHeightCssPx: 512,
};
member.setCamera(view);
let serial = 1;
coordinator.prepareFrame(serial);
const decoded = async (member) => {
  for (let i = 0; i < 40 && member.stats().renderer.submittedTiles === 0; i++)
    await Promise.resolve();
  if (member.stats().renderer.submittedTiles !== 1)
    throw new Error("point fixture did not decode");
};
await decoded(member);

const snapshot = () => {
  const stats = member.stats();
  const count = stats.renderer.drawnPoints;
  const binding = gl.getParameter(gl.ARRAY_BUFFER_BINDING);
  let mismatches = 0;
  for (const [buffer, kind] of buffers) {
    const data = new Uint8Array(count * (kind === "positions" ? 12 : 4));
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.getBufferSubData(gl.ARRAY_BUFFER, 0, data);
    mismatches += data.reduce(
      (sum, value, i) => sum + (value !== expected[kind][i]),
      0,
    );
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, binding);
  const pixel = new Uint8Array(4);
  gl.readPixels(256, 256, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
  return {
    drawnPoints: count,
    controllerPoints: stats.controller.drawnPoints,
    pending: member.governorInputs().work.operations,
    pick: member.pick(view, 256, 256),
    allocations: { ...allocations },
    writes: [...writes],
    mismatches,
    pixel: [...pixel],
    submission: coordinator.stats().submissions,
  };
};
window.stagedAdmission = {
  snapshot,
  paint() {
    writes.length = 0;
    coordinator.prepareFrame(++serial);
    renderWindow.render();
    return snapshot();
  },
  dispose() {
    coordinator.dispose();
    generic.delete();
  },
};
document.documentElement.dataset.ready = "true";
