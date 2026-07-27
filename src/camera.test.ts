import { describe, expect, it } from "vitest";

import {
  boundsIntersectsFrustum,
  distanceToBounds,
  frustumPlanes,
  nodeScreenSpaceError,
  orthographicScreenSpaceError,
  perspectiveScreenSpaceError,
  type OrthographicCameraView,
  type PerspectiveCameraView,
} from "./camera";
import type { Bounds } from "./octree";

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** Column-major perspective matrix (symmetric frustum, looking down -Z). */
const perspective = (
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

/**
 * Column-major symmetric orthographic matrix (half-height `parallelScale`,
 * looking down -Z) — the projection a vtk.js parallel camera renders.
 */
const orthographic = (
  parallelScale: number,
  aspect: number,
  near: number,
  far: number,
): number[] => {
  const halfWidth = parallelScale * aspect;
  return [
    1 / halfWidth,
    0,
    0,
    0,
    0,
    1 / parallelScale,
    0,
    0,
    0,
    0,
    -2 / (far - near),
    0,
    0,
    0,
    -(far + near) / (far - near),
    1,
  ];
};

const bounds = (
  center: [number, number, number],
  halfExtents: [number, number, number],
): Bounds => ({
  min: [
    center[0] - halfExtents[0],
    center[1] - halfExtents[1],
    center[2] - halfExtents[2],
  ],
  max: [
    center[0] + halfExtents[0],
    center[1] + halfExtents[1],
    center[2] + halfExtents[2],
  ],
});

describe("frustumPlanes", () => {
  it("extracts the six NDC half-spaces from the identity matrix", () => {
    const planes = frustumPlanes(IDENTITY);
    expect(planes).toHaveLength(6);
    // Every plane of the NDC cube passes through |coord| = 1.
    for (const p of planes) {
      expect(p.d).toBeCloseTo(1);
      expect(Math.hypot(...p.normal)).toBeCloseTo(1);
    }
  });
});

describe("boundsIntersectsFrustum with the identity matrix (NDC cube)", () => {
  const planes = frustumPlanes(IDENTITY);

  it("keeps a cube at the origin", () => {
    expect(
      boundsIntersectsFrustum(planes, bounds([0, 0, 0], [0.5, 0.2, 0.1])),
    ).toBe(true);
  });

  it("rejects a cube fully outside", () => {
    expect(
      boundsIntersectsFrustum(planes, bounds([3, 0, 0], [0.5, 0.2, 0.1])),
    ).toBe(false);
    expect(
      boundsIntersectsFrustum(planes, bounds([0, -3, 0], [0.5, 0.2, 0.1])),
    ).toBe(false);
  });

  it("keeps a cube straddling a plane", () => {
    expect(
      boundsIntersectsFrustum(planes, bounds([1, 0, 0], [0.5, 0.2, 0.1])),
    ).toBe(true);
  });
});

describe("boundsIntersectsFrustum with a perspective matrix", () => {
  const planes = frustumPlanes(perspective(Math.PI / 2, 1, 0.1, 100));

  it("keeps a cube in front of the camera", () => {
    expect(
      boundsIntersectsFrustum(planes, bounds([0, 0, -5], [1, 0.5, 0.25])),
    ).toBe(true);
  });

  it("rejects a cube behind the camera", () => {
    expect(
      boundsIntersectsFrustum(planes, bounds([0, 0, 5], [1, 0.5, 0.25])),
    ).toBe(false);
  });

  it("rejects a cube far outside the side planes", () => {
    // At z = -5 with fov 90° the frustum half-width is 5.
    expect(
      boundsIntersectsFrustum(planes, bounds([20, 0, -5], [1, 0.5, 0.25])),
    ).toBe(false);
  });

  it("rejects a cube beyond the far plane", () => {
    expect(
      boundsIntersectsFrustum(planes, bounds([0, 0, -500], [1, 0.5, 0.25])),
    ).toBe(false);
  });
});

describe("distanceToBounds", () => {
  it("is zero inside", () => {
    expect(
      distanceToBounds([0.2, -0.3, 0], bounds([0, 0, 0], [0.5, 0.5, 0.5])),
    ).toBe(0);
  });

  it("measures face distance", () => {
    expect(
      distanceToBounds([2, 0, 0], bounds([0, 0, 0], [0.5, 1, 2])),
    ).toBeCloseTo(1.5);
  });

  it("measures corner distance", () => {
    const d = distanceToBounds([2, 2, 0], bounds([0, 0, 0], [1, 1, 1]));
    expect(d).toBeCloseTo(Math.hypot(1, 1));
  });
});

describe("boundsIntersectsFrustum with an orthographic matrix", () => {
  // Half-height 10, aspect 2 → the visible box spans x ±20, y ±10, z -1..-100.
  const planes = frustumPlanes(orthographic(10, 2, 1, 100));

  it("keeps a cube inside the parallel box", () => {
    expect(
      boundsIntersectsFrustum(planes, bounds([15, 5, -50], [1, 1, 1])),
    ).toBe(true);
  });

  it("rejects a cube outside the side planes even though it is dead ahead", () => {
    // A perspective frustum would widen with depth and keep this; a parallel
    // one never does.
    expect(
      boundsIntersectsFrustum(planes, bounds([30, 0, -90], [1, 1, 1])),
    ).toBe(false);
  });

  it("rejects a cube above the top plane", () => {
    expect(
      boundsIntersectsFrustum(planes, bounds([0, 15, -50], [1, 1, 1])),
    ).toBe(false);
  });

  it("rejects cubes outside the near and far planes", () => {
    expect(boundsIntersectsFrustum(planes, bounds([0, 0, 5], [1, 1, 1]))).toBe(
      false,
    );
    expect(
      boundsIntersectsFrustum(planes, bounds([0, 0, -200], [1, 1, 1])),
    ).toBe(false);
  });
});

describe("perspectiveScreenSpaceError", () => {
  it("projects spacing to pixels", () => {
    // fov 90° → tan(fov/2) = 1: 1 m spacing at 10 m over 1000 px = 50 px.
    expect(perspectiveScreenSpaceError(1, 10, 1000, Math.PI / 2)).toBeCloseTo(
      50,
    );
  });

  it("halves with double distance", () => {
    const near = perspectiveScreenSpaceError(1, 10, 1000, Math.PI / 2);
    const far = perspectiveScreenSpaceError(1, 20, 1000, Math.PI / 2);
    expect(far).toBeCloseTo(near / 2);
  });

  it("stays finite when the camera touches the node", () => {
    const touching = perspectiveScreenSpaceError(1, 0, 1000, Math.PI / 2);
    expect(touching).toBeGreaterThan(1e9);
    expect(Number.isFinite(touching)).toBe(true);
  });
});

describe("orthographicScreenSpaceError", () => {
  it("projects spacing to pixels from the parallel scale alone", () => {
    // Half-height 10 m over 1000 px = 50 px/m: a 0.5 m spacing is 25 px.
    expect(orthographicScreenSpaceError(0.5, 1000, 10)).toBeCloseTo(25);
  });

  it("halves when the camera zooms out by two", () => {
    expect(orthographicScreenSpaceError(0.5, 1000, 20)).toBeCloseTo(12.5);
  });

  it("stays finite at a degenerate parallel scale", () => {
    const collapsed = orthographicScreenSpaceError(1, 1000, 0);
    expect(collapsed).toBeGreaterThan(1e9);
    expect(Number.isFinite(collapsed)).toBe(true);
  });
});

const perspectiveView: PerspectiveCameraView = {
  projection: "perspective",
  viewProj: IDENTITY,
  position: [0, 0, 10.5],
  fovY: Math.PI / 2,
  viewportHeightCssPx: 1000,
};

const orthographicView: OrthographicCameraView = {
  projection: "orthographic",
  viewProj: IDENTITY,
  position: [0, 0, 10.5],
  parallelScale: 10,
  viewportHeightCssPx: 1000,
};

describe("nodeScreenSpaceError", () => {
  const cube = bounds([0, 0, 0], [0.5, 0.25, 0.5]);

  it("combines cube distance with a perspective view", () => {
    // Cube surface is 10 m from the camera.
    expect(nodeScreenSpaceError(cube, 1, perspectiveView)).toBeCloseTo(50);
  });

  it("is distance-invariant under parallel projection", () => {
    const near = nodeScreenSpaceError(cube, 0.5, orthographicView);
    const far = nodeScreenSpaceError(cube, 0.5, {
      ...orthographicView,
      position: [0, 0, 5000],
    });
    expect(near).toBeCloseTo(25);
    expect(far).toBeCloseTo(near);
  });

  it("is distance-dependent under perspective projection", () => {
    const near = nodeScreenSpaceError(cube, 1, perspectiveView);
    const far = nodeScreenSpaceError(cube, 1, {
      ...perspectiveView,
      position: [0, 0, 20.5],
    });
    expect(far).toBeCloseTo(near / 2);
  });

  it("refines an orthographic view only when the camera zooms in", () => {
    const wide = nodeScreenSpaceError(cube, 0.5, orthographicView);
    const zoomed = nodeScreenSpaceError(cube, 0.5, {
      ...orthographicView,
      parallelScale: 2.5,
    });
    expect(zoomed).toBeCloseTo(wide * 4);
  });
});
