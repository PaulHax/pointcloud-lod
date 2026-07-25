import { describe, expect, it } from "vitest";

import {
  boundsIntersectsFrustum,
  distanceToBounds,
  frustumPlanes,
  nodeScreenSpaceError,
  screenSpaceError,
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

describe("screenSpaceError", () => {
  it("projects spacing to pixels", () => {
    // fov 90° → tan(fov/2) = 1: 1 m spacing at 10 m over 1000 px = 50 px.
    expect(screenSpaceError(1, 10, 1000, Math.PI / 2)).toBeCloseTo(50);
  });

  it("halves with double distance", () => {
    const near = screenSpaceError(1, 10, 1000, Math.PI / 2);
    const far = screenSpaceError(1, 20, 1000, Math.PI / 2);
    expect(far).toBeCloseTo(near / 2);
  });

  it("stays finite when the camera touches the node", () => {
    expect(screenSpaceError(1, 0, 1000, Math.PI / 2)).toBeGreaterThan(1e9);
    expect(Number.isFinite(screenSpaceError(1, 0, 1000, Math.PI / 2))).toBe(
      true,
    );
  });
});

describe("nodeScreenSpaceError", () => {
  it("combines cube distance with the view", () => {
    const view = {
      viewProj: IDENTITY,
      position: [0, 0, 10.5] as [number, number, number],
      fovY: Math.PI / 2,
      viewportHeightCssPx: 1000,
    };
    // Cube surface is 10 m from the camera.
    expect(
      nodeScreenSpaceError(bounds([0, 0, 0], [0.5, 0.25, 0.5]), 1, view),
    ).toBeCloseTo(50);
  });
});
