import type { PerspectiveCameraView } from "../src/camera";
import { IDENTITY } from "../src/mat4";

/** An identity view-projection at the origin over a 100x100 css-px viewport. */
export const perspectiveView = (
  overrides: Partial<PerspectiveCameraView> = {},
): PerspectiveCameraView => ({
  projection: "perspective",
  viewProj: IDENTITY,
  position: [0, 0, 0],
  fovY: Math.PI / 2,
  viewportWidthCssPx: 100,
  viewportHeightCssPx: 100,
  ...overrides,
});
