import { describe, expect, it } from "vitest";

import {
  boundsIntersectsFrustum,
  boundsCenterRayOffset,
  cursorRay,
  distanceToBounds,
  frustumPlanes,
  modelFrameOf,
  nodeScreenSpaceError,
  orthographicScreenSpaceError,
  perspectiveScreenSpaceError,
  projectPointToCssPx,
  viewInModelFrame,
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
  viewportWidthCssPx: 1000,
  viewportHeightCssPx: 1000,
};

const orthographicView: OrthographicCameraView = {
  projection: "orthographic",
  viewProj: IDENTITY,
  position: [0, 0, 10.5],
  parallelScale: 10,
  viewportWidthCssPx: 1000,
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

describe("projectPointToCssPx", () => {
  it("maps NDC into css pixels with y down, using both viewport dimensions", () => {
    // Identity view-projection: world coordinates are NDC directly.
    const center = projectPointToCssPx(IDENTITY, [0, 0, 0], 200, 100);
    expect(center).not.toBeNull();
    expect(center!.xCssPx).toBeCloseTo(100);
    expect(center!.yCssPx).toBeCloseTo(50);
    expect(center!.ndcZ).toBeCloseTo(0);

    // +x is right, +y is up on screen — so css y decreases as world y grows.
    const offset = projectPointToCssPx(IDENTITY, [0.5, 0.5, 0.25], 200, 100);
    expect(offset!.xCssPx).toBeCloseTo(150);
    expect(offset!.yCssPx).toBeCloseTo(25);
    expect(offset!.ndcZ).toBeCloseTo(0.25);
  });

  it("projects an off-center point through a perspective matrix", () => {
    // fov 90°, aspect 2: at z = -10 the half-extents are 20 x 10.
    const m = perspective(Math.PI / 2, 2, 0.1, 100);
    const projected = projectPointToCssPx(m, [10, 5, -10], 800, 400);
    expect(projected!.xCssPx).toBeCloseTo(600);
    expect(projected!.yCssPx).toBeCloseTo(100);
  });

  it("refuses a point at or behind the camera plane", () => {
    const m = perspective(Math.PI / 2, 2, 0.1, 100);
    expect(projectPointToCssPx(m, [0, 0, 10], 800, 400)).toBeNull();
    expect(projectPointToCssPx(m, [0, 0, 0], 800, 400)).toBeNull();
  });

  it("refuses unusable viewport dimensions and non-finite input", () => {
    expect(projectPointToCssPx(IDENTITY, [0, 0, 0], 0, 100)).toBeNull();
    expect(
      projectPointToCssPx(IDENTITY, [0, 0, 0], 200, Number.NaN),
    ).toBeNull();
    expect(projectPointToCssPx(IDENTITY, [0, 0, 0], -200, 100)).toBeNull();
    expect(
      projectPointToCssPx(IDENTITY, [Number.NaN, 0, 0], 200, 100),
    ).toBeNull();
    const broken = [...IDENTITY.slice(0, 15), Number.NaN];
    expect(projectPointToCssPx(broken, [0, 0, 0], 200, 100)).toBeNull();
  });
});

describe("boundsCenterRayOffset", () => {
  const perspectiveView: PerspectiveCameraView = {
    projection: "perspective",
    viewProj: perspective(Math.PI / 2, 1, 0.1, 100),
    position: [0, 0, 0],
    fovY: Math.PI / 2,
    viewportWidthCssPx: 400,
    viewportHeightCssPx: 400,
  };

  it("prefers a distant volume on the 3D centre ray over a nearby side volume", () => {
    const distantCenter = bounds([0, 0, -20], [0.1, 0.1, 0.1]);
    const nearbyBottom = bounds([0, -2, -5], [0.1, 0.1, 0.1]);

    expect(boundsCenterRayOffset(distantCenter, perspectiveView)).toBe(0);
    expect(
      boundsCenterRayOffset(nearbyBottom, perspectiveView),
    ).toBeGreaterThan(0);
  });

  it("orders same-sized 3D volumes by their centre-ray angle", () => {
    const inner = bounds([0.5, 0, -5], [0.1, 0.1, 0.1]);
    const outer = bounds([2, 0, -5], [0.1, 0.1, 0.1]);
    expect(boundsCenterRayOffset(inner, perspectiveView)).toBeLessThan(
      boundsCenterRayOffset(outer, perspectiveView),
    );
  });

  it("orders parallel volumes by perpendicular clearance from the centre ray", () => {
    const view: OrthographicCameraView = {
      projection: "orthographic",
      viewProj: orthographic(5, 1, 0.1, 100),
      position: [0, 0, 0],
      parallelScale: 5,
      viewportWidthCssPx: 400,
      viewportHeightCssPx: 400,
    };
    const center = bounds([0, 0, -20], [0.1, 0.1, 0.1]);
    const side = bounds([2, 0, -5], [0.1, 0.1, 0.1]);

    expect(boundsCenterRayOffset(center, view)).toBe(0);
    expect(boundsCenterRayOffset(side, view)).toBeGreaterThan(0);
  });
});

describe("cursorRay", () => {
  const roundTrips = (
    m: number[],
    cursor: [number, number],
    width: number,
    height: number,
  ): void => {
    const ray = cursorRay(m, cursor[0], cursor[1], width, height);
    expect(ray).not.toBeNull();
    expect(Math.hypot(...ray!.direction)).toBeCloseTo(1);
    // Every point along the ray projects back onto the cursor.
    for (const t of [1, 5]) {
      const point: [number, number, number] = [
        ray!.origin[0] + ray!.direction[0] * t,
        ray!.origin[1] + ray!.direction[1] * t,
        ray!.origin[2] + ray!.direction[2] * t,
      ];
      const projected = projectPointToCssPx(m, point, width, height);
      expect(projected).not.toBeNull();
      expect(projected!.xCssPx).toBeCloseTo(cursor[0]);
      expect(projected!.yCssPx).toBeCloseTo(cursor[1]);
    }
  };

  it("builds a centered perspective ray in a non-square viewport", () => {
    const m = perspective(Math.PI / 2, 2, 0.1, 100);
    const ray = cursorRay(m, 400, 200, 800, 400);
    expect(ray!.origin[0]).toBeCloseTo(0);
    expect(ray!.origin[1]).toBeCloseTo(0);
    expect(ray!.origin[2]).toBeCloseTo(-0.1); // near plane
    expect(ray!.direction[0]).toBeCloseTo(0);
    expect(ray!.direction[1]).toBeCloseTo(0);
    expect(ray!.direction[2]).toBeCloseTo(-1);
  });

  it("round-trips off-center perspective cursors", () => {
    const m = perspective(Math.PI / 2, 2, 0.1, 100);
    roundTrips(m, [600, 100], 800, 400);
    roundTrips(m, [37, 311], 800, 400);
  });

  it("builds parallel orthographic rays whose origin tracks the cursor", () => {
    // Half-height 10, aspect 2, near 1: the viewport spans x ±20, y ±10.
    const m = orthographic(10, 2, 1, 100);
    const ray = cursorRay(m, 600, 100, 800, 400);
    expect(ray!.origin[0]).toBeCloseTo(10);
    expect(ray!.origin[1]).toBeCloseTo(5);
    expect(ray!.origin[2]).toBeCloseTo(-1); // near plane
    expect(ray!.direction[0]).toBeCloseTo(0);
    expect(ray!.direction[1]).toBeCloseTo(0);
    expect(ray!.direction[2]).toBeCloseTo(-1);
    roundTrips(m, [600, 100], 800, 400);
  });

  it("refuses singular and non-finite matrices", () => {
    const singular = IDENTITY.map(() => 0);
    expect(cursorRay(singular, 10, 10, 100, 100)).toBeNull();
    const broken = [...IDENTITY.slice(0, 15), Number.NaN];
    expect(cursorRay(broken, 10, 10, 100, 100)).toBeNull();
  });

  it("refuses unusable viewport dimensions and non-finite cursors", () => {
    expect(cursorRay(IDENTITY, 10, 10, 0, 100)).toBeNull();
    expect(cursorRay(IDENTITY, 10, 10, 100, Number.NaN)).toBeNull();
    expect(cursorRay(IDENTITY, 10, 10, -100, 100)).toBeNull();
    expect(cursorRay(IDENTITY, Number.NaN, 10, 100, 100)).toBeNull();
    expect(
      cursorRay(IDENTITY, 10, Number.POSITIVE_INFINITY, 100, 100),
    ).toBeNull();
  });
});

describe("viewInModelFrame", () => {
  // Rotation by 90 degrees about z, uniform scale 2, translation (5, 6, 7).
  // prettier-ignore
  const SIMILARITY = [
    0, 2, 0, 0,
    -2, 0, 0, 0,
    0, 0, 2, 0,
    5, 6, 7, 1,
  ];
  const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const frame = modelFrameOf(SIMILARITY)!;

  it("resolves a similarity's frame and refuses anything else", () => {
    expect(frame.scale).toBeCloseTo(2);
    // prettier-ignore
    const anisotropic = [
      1, 0, 0, 0,
      0, 2, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ];
    expect(modelFrameOf(anisotropic)).toBeNull();
    expect(modelFrameOf([1, 0, 0])).toBeNull();
  });

  it("restates the eye through the inverse and keeps the field of view", () => {
    const view = {
      projection: "perspective" as const,
      viewProj: IDENTITY,
      position: [5, 8, 7] as [number, number, number],
      fovY: Math.PI / 3,
      viewportWidthCssPx: 800,
      viewportHeightCssPx: 600,
    };
    const local = viewInModelFrame(view, frame);
    // world (5, 8, 7) = M * local: local = (1, 0, 0).
    expect(local.position[0]).toBeCloseTo(1);
    expect(local.position[1]).toBeCloseTo(0);
    expect(local.position[2]).toBeCloseTo(0);
    // viewProj folds the model matrix in: I * M = M.
    expect(Array.from(local.viewProj)).toEqual(SIMILARITY);
    // A field of view is an angle: the uniform model scale cancels out.
    expect(local.projection).toBe("perspective");
    if (local.projection === "perspective") {
      expect(local.fovY).toBe(view.fovY);
    }
    expect(local.viewportHeightCssPx).toBe(600);
  });

  it("restates an orthographic parallelScale in model units", () => {
    const view = {
      projection: "orthographic" as const,
      viewProj: IDENTITY,
      position: [5, 8, 7] as [number, number, number],
      parallelScale: 8,
      viewportWidthCssPx: 800,
      viewportHeightCssPx: 600,
    };
    const local = viewInModelFrame(view, frame);
    // parallelScale is a world height, and the model scales local units by
    // 2, so the same viewport spans half as many model units.
    expect(local.projection).toBe("orthographic");
    if (local.projection === "orthographic") {
      expect(local.parallelScale).toBe(4);
    }
  });
});
