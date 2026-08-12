/** Renderer-neutral ray/triangle picking over the exact submitted draw set. */

import {
  cursorRay,
  projectPointToCssPx,
  type CameraView,
  type Mat16,
} from "../camera";
import type { MemberPickResult, OcclusionResult } from "../streamedMember";
import type { Bounds } from "../octree";
import type { DecodedPrimitive } from "./decode";

export type SubmittedMeshPrimitive = Pick<
  DecodedPrimitive,
  "positions" | "indices"
>;

export type SubmittedMeshTile = {
  readonly id: string;
  readonly origin: readonly [number, number, number];
  readonly primitives: readonly SubmittedMeshPrimitive[];
  readonly bounds?: Bounds;
};

export type MeshPickSet = {
  replaceDrawn(tiles: readonly SubmittedMeshTile[]): void;
  setModelMatrix(matrix: Mat16 | null): void;
  pick(view: CameraView, cssX: number, cssY: number): MemberPickResult | null;
  occlusionDepth(
    view: CameraView,
    cssX: number,
    cssY: number,
  ): OcclusionResult | null;
  dispose(): void;
};

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const TRIANGLE_EPSILON = 1e-10;

type Vec3 = readonly [number, number, number];

const finiteMatrix = (matrix: Mat16 | null): readonly number[] => {
  if (matrix === null) return IDENTITY;
  if (
    matrix.length !== 16 ||
    Array.from(matrix).some((value) => !Number.isFinite(value))
  ) {
    throw new TypeError("mesh model matrix must contain 16 finite numbers");
  }
  return Array.from(matrix);
};

const worldPoint = (
  matrix: readonly number[],
  origin: readonly [number, number, number],
  positions: Float32Array,
  vertexIndex: number,
): Vec3 => {
  const x = origin[0] + positions[vertexIndex * 3]!;
  const y = origin[1] + positions[vertexIndex * 3 + 1]!;
  const z = origin[2] + positions[vertexIndex * 3 + 2]!;
  return [
    matrix[0]! * x + matrix[4]! * y + matrix[8]! * z + matrix[12]!,
    matrix[1]! * x + matrix[5]! * y + matrix[9]! * z + matrix[13]!,
    matrix[2]! * x + matrix[6]! * y + matrix[10]! * z + matrix[14]!,
  ];
};

const subtract = (left: Vec3, right: Vec3): Vec3 => [
  left[0] - right[0],
  left[1] - right[1],
  left[2] - right[2],
];

const cross = (left: Vec3, right: Vec3): Vec3 => [
  left[1] * right[2] - left[2] * right[1],
  left[2] * right[0] - left[0] * right[2],
  left[0] * right[1] - left[1] * right[0],
];

const dot = (left: Vec3, right: Vec3): number =>
  left[0] * right[0] + left[1] * right[1] + left[2] * right[2];

const triangleDepth = (
  rayOrigin: Vec3,
  rayDirection: Vec3,
  a: Vec3,
  b: Vec3,
  c: Vec3,
): number | null => {
  const edge1 = subtract(b, a);
  const edge2 = subtract(c, a);
  const p = cross(rayDirection, edge2);
  const determinant = dot(edge1, p);
  if (
    !Number.isFinite(determinant) ||
    Math.abs(determinant) <= TRIANGLE_EPSILON
  )
    return null;
  const inverse = 1 / determinant;
  const t = subtract(rayOrigin, a);
  const u = dot(t, p) * inverse;
  if (!(u >= 0 && u <= 1)) return null;
  const q = cross(t, edge1);
  const v = dot(rayDirection, q) * inverse;
  if (!(v >= 0 && u + v <= 1)) return null;
  const depth = dot(edge2, q) * inverse;
  return Number.isFinite(depth) && depth > TRIANGLE_EPSILON ? depth : null;
};

const vertexAt = (
  primitive: SubmittedMeshPrimitive,
  triangleIndex: number,
  corner: number,
): number =>
  primitive.indices?.[triangleIndex * 3 + corner] ?? triangleIndex * 3 + corner;

const projectedBoundsContain = (
  view: CameraView,
  cssX: number,
  cssY: number,
  bounds: Bounds | undefined,
  matrix: readonly number[],
): boolean => {
  if (!bounds) return true;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const x of [bounds.min[0], bounds.max[0]]) {
    for (const y of [bounds.min[1], bounds.max[1]]) {
      for (const z of [bounds.min[2], bounds.max[2]]) {
        const projected = projectPointToCssPx(
          view.viewProj,
          [
            matrix[0]! * x + matrix[4]! * y + matrix[8]! * z + matrix[12]!,
            matrix[1]! * x + matrix[5]! * y + matrix[9]! * z + matrix[13]!,
            matrix[2]! * x + matrix[6]! * y + matrix[10]! * z + matrix[14]!,
          ],
          view.viewportWidthCssPx,
          view.viewportHeightCssPx,
        );
        if (!projected) return true;
        minX = Math.min(minX, projected.xCssPx);
        minY = Math.min(minY, projected.yCssPx);
        maxX = Math.max(maxX, projected.xCssPx);
        maxY = Math.max(maxY, projected.yCssPx);
      }
    }
  }
  return cssX >= minX && cssX <= maxX && cssY >= minY && cssY <= maxY;
};

export const pickSubmittedTriangles = (
  view: CameraView,
  cssX: number,
  cssY: number,
  tiles: readonly SubmittedMeshTile[],
  modelMatrix: Mat16 | null = null,
): MemberPickResult | null => {
  const ray = cursorRay(
    view.viewProj,
    cssX,
    cssY,
    view.viewportWidthCssPx,
    view.viewportHeightCssPx,
  );
  if (ray === null) return null;
  let matrix: readonly number[];
  try {
    matrix = finiteMatrix(modelMatrix);
  } catch {
    return null;
  }
  let bestDepth = Number.POSITIVE_INFINITY;
  for (const tile of tiles) {
    if (!projectedBoundsContain(view, cssX, cssY, tile.bounds, matrix))
      continue;
    for (const primitive of tile.primitives) {
      const availableVertices = Math.floor(primitive.positions.length / 3);
      const triangleCount = Math.floor(
        (primitive.indices?.length ?? availableVertices) / 3,
      );
      for (let triangle = 0; triangle < triangleCount; triangle += 1) {
        const ia = vertexAt(primitive, triangle, 0);
        const ib = vertexAt(primitive, triangle, 1);
        const ic = vertexAt(primitive, triangle, 2);
        if (
          ia < 0 ||
          ib < 0 ||
          ic < 0 ||
          ia >= availableVertices ||
          ib >= availableVertices ||
          ic >= availableVertices
        ) {
          continue;
        }
        const depth = triangleDepth(
          ray.origin,
          ray.direction,
          worldPoint(matrix, tile.origin, primitive.positions, ia),
          worldPoint(matrix, tile.origin, primitive.positions, ib),
          worldPoint(matrix, tile.origin, primitive.positions, ic),
        );
        if (depth !== null && depth < bestDepth) bestDepth = depth;
      }
    }
  }
  if (!Number.isFinite(bestDepth)) return { status: "miss" };
  const pointOnRay: [number, number, number] = [
    ray.origin[0] + ray.direction[0] * bestDepth,
    ray.origin[1] + ray.direction[1] * bestDepth,
    ray.origin[2] + ray.direction[2] * bestDepth,
  ];
  // The ray itself is exact, but this projection also rejects intersections
  // outside the rendered near/far interval after arbitrary anchor transforms.
  const projected = projectPointToCssPx(
    view.viewProj,
    pointOnRay,
    view.viewportWidthCssPx,
    view.viewportHeightCssPx,
  );
  if (
    projected === null ||
    !Number.isFinite(projected.ndcZ) ||
    projected.ndcZ < -1 ||
    projected.ndcZ > 1
  ) {
    return { status: "miss" };
  }
  return {
    status: "hit",
    rayDepth: bestDepth,
    pointOnRay,
    distancePx: Math.hypot(projected.xCssPx - cssX, projected.yCssPx - cssY),
  };
};

export const createMeshPickSet = (): MeshPickSet => {
  let drawn: readonly SubmittedMeshTile[] = [];
  let modelMatrix: Mat16 | null = null;
  let disposed = false;
  return {
    replaceDrawn(tiles) {
      if (!disposed) drawn = [...tiles];
    },
    setModelMatrix(matrix) {
      if (!disposed) modelMatrix = matrix === null ? null : Array.from(matrix);
    },
    pick(view, cssX, cssY) {
      return disposed
        ? null
        : pickSubmittedTriangles(view, cssX, cssY, drawn, modelMatrix);
    },
    occlusionDepth(view, cssX, cssY) {
      const result = disposed
        ? null
        : pickSubmittedTriangles(view, cssX, cssY, drawn, modelMatrix);
      return result === null
        ? null
        : result.status === "hit"
          ? { status: "hit", rayDepth: result.rayDepth }
          : { status: "clear" };
    },
    dispose() {
      disposed = true;
      drawn = [];
      modelMatrix = null;
    },
  };
};
