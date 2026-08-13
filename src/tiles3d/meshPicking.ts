/** Renderer-neutral ray/triangle picking over the exact submitted draw set. */

import {
  cursorRay,
  projectPointToCssPx,
  type CameraView,
  type Mat16,
} from "../camera";
import {
  occlusionFromPick,
  type MemberPickResult,
  type OcclusionResult,
} from "../streamedMember";
import type { Bounds } from "../octree";
import { scenePoint as makeScenePoint } from "../frames";
import type { DecodedPrimitive, SerializableSampler } from "./decode";

export type PickAlphaTexture = {
  readonly width: number;
  readonly height: number;
  readonly alpha: Uint8Array;
  readonly sampler: SerializableSampler;
};

export type SubmittedMeshPrimitive = Pick<
  DecodedPrimitive,
  "positions" | "indices" | "uvs"
> & {
  readonly alphaMask?:
    | {
        readonly kind: "known";
        readonly factorAlpha: number;
        readonly cutoff: number;
        readonly texture?: PickAlphaTexture;
      }
    | { readonly kind: "unknown" };
};

export type SubmittedMeshTile = {
  readonly id: string;
  readonly origin: readonly [number, number, number];
  readonly primitives: readonly SubmittedMeshPrimitive[];
  readonly bounds?: Bounds;
};

/**
 * Where the member's geometry sits, in the two frames that differ.
 *
 * `drawn` is what the renderer paints — it carries vertical exaggeration, so
 * it is the only frame in which a screen ray means anything. `scene` is the
 * canonical scene coordinates the rest of the app reasons in: the same placement
 * without exaggeration. Picking happens in `drawn` and reports in `scene`;
 * collapsing the two is what let exaggerated z values reach saved control
 * points.
 */
export type MeshPlacement = {
  readonly drawn: Mat16 | null;
  readonly scene: Mat16 | null;
};

export type MeshPickSet = {
  replaceDrawn(tiles: readonly SubmittedMeshTile[]): void;
  setPlacement(placement: MeshPlacement | null): void;
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

/** Tile-local vertex lifted into the member's unplaced coordinate frame. */
const localPoint = (
  origin: readonly [number, number, number],
  positions: Float32Array,
  vertexIndex: number,
): Vec3 => [
  origin[0] + positions[vertexIndex * 3]!,
  origin[1] + positions[vertexIndex * 3 + 1]!,
  origin[2] + positions[vertexIndex * 3 + 2]!,
];

const transformPoint = (matrix: readonly number[], point: Vec3): Vec3 => [
  matrix[0]! * point[0] +
    matrix[4]! * point[1] +
    matrix[8]! * point[2] +
    matrix[12]!,
  matrix[1]! * point[0] +
    matrix[5]! * point[1] +
    matrix[9]! * point[2] +
    matrix[13]!,
  matrix[2]! * point[0] +
    matrix[6]! * point[1] +
    matrix[10]! * point[2] +
    matrix[14]!,
];

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

/**
 * Möller–Trumbore hit, keeping the barycentric coordinates.
 *
 * `u`/`v` are what let the caller rebuild the intersection in a frame other
 * than the one the ray was cast in: the same (u, v) name the same point on the
 * same triangle under any placement, so a hit found against drawn geometry can
 * be re-expressed in scene space without inverting a matrix.
 */
const triangleHit = (
  rayOrigin: Vec3,
  rayDirection: Vec3,
  a: Vec3,
  b: Vec3,
  c: Vec3,
): { depth: number; u: number; v: number } | null => {
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
  return Number.isFinite(depth) && depth > TRIANGLE_EPSILON
    ? { depth, u, v }
    : null;
};

const vertexAt = (
  primitive: SubmittedMeshPrimitive,
  triangleIndex: number,
  corner: number,
): number =>
  primitive.indices?.[triangleIndex * 3 + corner] ?? triangleIndex * 3 + corner;

const wrapped = (value: number, size: number, mode: number): number => {
  if (mode === 10497) return ((value % size) + size) % size;
  if (mode === 33648) {
    const period = size * 2;
    const repeated = ((value % period) + period) % period;
    return repeated < size ? repeated : period - repeated - 1;
  }
  return Math.min(size - 1, Math.max(0, value));
};

const textureAlpha = (
  texture: PickAlphaTexture,
  u: number,
  v: number,
): number => {
  const read = (x: number, y: number): number =>
    texture.alpha[
      wrapped(y, texture.height, texture.sampler.wrapT) * texture.width +
        wrapped(x, texture.width, texture.sampler.wrapS)
    ]! / 255;
  if (texture.sampler.magFilter === 9728) {
    return read(Math.floor(u * texture.width), Math.floor(v * texture.height));
  }
  const x = u * texture.width - 0.5;
  const y = v * texture.height - 0.5;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const tx = x - x0;
  const ty = y - y0;
  return (
    read(x0, y0) * (1 - tx) * (1 - ty) +
    read(x0 + 1, y0) * tx * (1 - ty) +
    read(x0, y0 + 1) * (1 - tx) * ty +
    read(x0 + 1, y0 + 1) * tx * ty
  );
};

const alphaAtHit = (
  primitive: SubmittedMeshPrimitive,
  triangle: number,
  barycentricU: number,
  barycentricV: number,
): "opaque" | "transparent" | "unknown" => {
  const mask = primitive.alphaMask;
  if (!mask) return "opaque";
  if (mask.kind === "unknown") return "unknown";
  let alpha = mask.factorAlpha;
  if (mask.texture && primitive.uvs) {
    const ia = vertexAt(primitive, triangle, 0);
    const ib = vertexAt(primitive, triangle, 1);
    const ic = vertexAt(primitive, triangle, 2);
    const weightA = 1 - barycentricU - barycentricV;
    const u =
      primitive.uvs[ia * 2]! * weightA +
      primitive.uvs[ib * 2]! * barycentricU +
      primitive.uvs[ic * 2]! * barycentricV;
    const v =
      primitive.uvs[ia * 2 + 1]! * weightA +
      primitive.uvs[ib * 2 + 1]! * barycentricU +
      primitive.uvs[ic * 2 + 1]! * barycentricV;
    alpha *= textureAlpha(mask.texture, u, v);
  }
  return alpha < mask.cutoff ? "transparent" : "opaque";
};

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
  placement: MeshPlacement | null = null,
): MemberPickResult | null => {
  const ray = cursorRay(
    view.viewProj,
    cssX,
    cssY,
    view.viewportWidthCssPx,
    view.viewportHeightCssPx,
  );
  if (ray === null) return null;
  let drawnMatrix: readonly number[];
  let sceneMatrix: readonly number[];
  try {
    drawnMatrix = finiteMatrix(placement === null ? null : placement.drawn);
    // A null `scene` means an identity scene placement, NOT "same as drawn" —
    // coalescing the two would silently restore the exaggerated point.
    sceneMatrix = finiteMatrix(placement === null ? null : placement.scene);
  } catch {
    return null;
  }
  let bestDepth = Number.POSITIVE_INFINITY;
  let nearestUnknownDepth = Number.POSITIVE_INFINITY;
  // The winning triangle in the member's unplaced frame, plus where on it the
  // ray landed. Kept so the hit can be reported in scene space even though it
  // had to be found in drawn space.
  let bestLocal: { a: Vec3; b: Vec3; c: Vec3; u: number; v: number } | null =
    null;
  for (const tile of tiles) {
    if (!projectedBoundsContain(view, cssX, cssY, tile.bounds, drawnMatrix))
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
        const a = localPoint(tile.origin, primitive.positions, ia);
        const b = localPoint(tile.origin, primitive.positions, ib);
        const c = localPoint(tile.origin, primitive.positions, ic);
        const hit = triangleHit(
          ray.origin,
          ray.direction,
          transformPoint(drawnMatrix, a),
          transformPoint(drawnMatrix, b),
          transformPoint(drawnMatrix, c),
        );
        if (hit === null) continue;
        const alpha = alphaAtHit(primitive, triangle, hit.u, hit.v);
        if (alpha === "transparent") continue;
        if (alpha === "unknown") {
          nearestUnknownDepth = Math.min(nearestUnknownDepth, hit.depth);
          continue;
        }
        if (hit.depth < bestDepth) {
          bestDepth = hit.depth;
          bestLocal = { a, b, c, u: hit.u, v: hit.v };
        }
      }
    }
  }
  // Compressed MASK alpha is not CPU-readable. If it is in front of the first
  // certain hit, report unavailable rather than inventing either a hit or a
  // clear path through an unknown texel.
  if (nearestUnknownDepth < bestDepth) return null;
  if (!Number.isFinite(bestDepth) || bestLocal === null) {
    return { status: "miss" };
  }
  const drawnPoint: Vec3 = [
    ray.origin[0] + ray.direction[0] * bestDepth,
    ray.origin[1] + ray.direction[1] * bestDepth,
    ray.origin[2] + ray.direction[2] * bestDepth,
  ];
  // The ray itself is exact, but this projection also rejects intersections
  // outside the rendered near/far interval after arbitrary anchor transforms.
  // It must use the drawn point: that is the one the cursor actually sits on.
  const projected = projectPointToCssPx(
    view.viewProj,
    drawnPoint,
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
  const { a, b, c, u, v } = bestLocal;
  const scenePoint = transformPoint(sceneMatrix, [
    a[0] + u * (b[0] - a[0]) + v * (c[0] - a[0]),
    a[1] + u * (b[1] - a[1]) + v * (c[1] - a[1]),
    a[2] + u * (b[2] - a[2]) + v * (c[2] - a[2]),
  ]);
  return {
    status: "hit",
    rayDepth: bestDepth,
    scenePoint: makeScenePoint(scenePoint[0], scenePoint[1], scenePoint[2]),
    distancePx: Math.hypot(projected.xCssPx - cssX, projected.yCssPx - cssY),
  };
};

export const createMeshPickSet = (): MeshPickSet => {
  let drawn: readonly SubmittedMeshTile[] = [];
  let placement: MeshPlacement | null = null;
  let disposed = false;
  return {
    replaceDrawn(tiles) {
      if (!disposed) drawn = [...tiles];
    },
    setPlacement(next) {
      if (disposed) return;
      placement =
        next === null
          ? null
          : {
              drawn:
                next.drawn === null ? null : (Array.from(next.drawn) as Mat16),
              scene:
                next.scene === null ? null : (Array.from(next.scene) as Mat16),
            };
    },
    pick(view, cssX, cssY) {
      return disposed
        ? null
        : pickSubmittedTriangles(view, cssX, cssY, drawn, placement);
    },
    occlusionDepth(view, cssX, cssY) {
      return occlusionFromPick(
        disposed
          ? null
          : pickSubmittedTriangles(view, cssX, cssY, drawn, placement),
      );
    },
    dispose() {
      disposed = true;
      drawn = [];
      placement = null;
    },
  };
};
