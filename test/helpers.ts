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

/** Column-major perspective matrix (symmetric frustum, looking down -Z). */
export const perspective = (
  fovY: number,
  aspect: number,
  near: number,
  far: number,
): number[] => {
  const f = 1 / Math.tan(fovY / 2);
  return [
    f / aspect,
    0,
    0,
    0,
    0,
    f,
    0,
    0,
    0,
    0,
    (far + near) / (near - far),
    -1,
    0,
    0,
    (2 * far * near) / (near - far),
    0,
  ];
};
