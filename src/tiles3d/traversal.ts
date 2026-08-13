/** Pure camera-driven traversal for explicit REPLACE 3D Tiles hierarchies. */

import {
  frustumPlanes,
  orthographicScreenSpaceError,
  perspectiveScreenSpaceError,
  type CameraView,
  type Plane,
} from "../camera";
import {
  multiplyTilesetMatrices,
  type TilesetBox,
  type TilesetTile,
} from "./tilesetSource";

export type TileReadiness =
  | "unloaded"
  | "loading"
  | "decoded"
  | "submitted"
  | "failed";

export type TilesetTraversalOptions = {
  readonly root: TilesetTile;
  readonly camera: CameraView;
  /** The finest threshold configured for this member. */
  readonly maximumScreenSpaceErrorPx: number;
  /** Normalized allocation; the threshold relaxes by `1 / max(q, 0.05)`. */
  readonly qualityFraction?: number;
  /** Caller-owned anchor × ECEF-to-scene transform. */
  readonly modelMatrix?: readonly number[];
  readonly readiness: (tileId: string) => TileReadiness;
};

export type TilesetTraversalResult = {
  /** Finest camera-selected content, excluding ancestors held for readiness. */
  readonly desiredTileIds: readonly string[];
  /** Content that should be available: desired tiles plus required fallbacks. */
  readonly requestedTileIds: readonly string[];
  /** Submitted draw set with REPLACE fallback applied. */
  readonly drawnTileIds: readonly string[];
  readonly culledTileIds: readonly string[];
  readonly effectiveScreenSpaceErrorPx: number;
  /**
   * Screen-space error of the root's own content, before refinement.
   *
   * This is the member's demand in the one unit that is comparable across
   * formats: how wrong the coarsest thing it can draw currently looks. Zero
   * when the root is culled or carries no content.
   */
  readonly rootScreenSpaceErrorPx: number;
};

type Vec3 = readonly [number, number, number];

type WorldBox = {
  readonly center: Vec3;
  readonly axes: readonly [Vec3, Vec3, Vec3];
};

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

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

const transformVector = (matrix: readonly number[], vector: Vec3): Vec3 => [
  matrix[0]! * vector[0] + matrix[4]! * vector[1] + matrix[8]! * vector[2],
  matrix[1]! * vector[0] + matrix[5]! * vector[1] + matrix[9]! * vector[2],
  matrix[2]! * vector[0] + matrix[6]! * vector[1] + matrix[10]! * vector[2],
];

const worldBox = (box: TilesetBox, matrix: readonly number[]): WorldBox => {
  const h = box.halfAxes;
  return {
    center: transformPoint(matrix, box.center),
    axes: [
      transformVector(matrix, [h[0], h[1], h[2]]),
      transformVector(matrix, [h[3], h[4], h[5]]),
      transformVector(matrix, [h[6], h[7], h[8]]),
    ],
  };
};

const dot = (left: Vec3, right: Vec3): number =>
  left[0] * right[0] + left[1] * right[1] + left[2] * right[2];

/** Exact plane-vs-oriented-box rejection. */
const boxIntersectsFrustum = (
  box: WorldBox,
  planes: readonly Plane[],
): boolean => {
  for (const plane of planes) {
    const radius =
      Math.abs(dot(plane.normal, box.axes[0])) +
      Math.abs(dot(plane.normal, box.axes[1])) +
      Math.abs(dot(plane.normal, box.axes[2]));
    if (dot(plane.normal, box.center) + plane.d + radius < 0) return false;
  }
  return true;
};

const boxDistance = (box: WorldBox, point: Vec3): number => {
  // A conservative world AABB is sufficient for SSE: it may refine early,
  // but never drops detail because an oriented or sheared box was too near.
  const radius: Vec3 = [
    Math.abs(box.axes[0][0]) +
      Math.abs(box.axes[1][0]) +
      Math.abs(box.axes[2][0]),
    Math.abs(box.axes[0][1]) +
      Math.abs(box.axes[1][1]) +
      Math.abs(box.axes[2][1]),
    Math.abs(box.axes[0][2]) +
      Math.abs(box.axes[1][2]) +
      Math.abs(box.axes[2][2]),
  ];
  const dx = Math.max(Math.abs(point[0] - box.center[0]) - radius[0], 0);
  const dy = Math.max(Math.abs(point[1] - box.center[1]) - radius[1], 0);
  const dz = Math.max(Math.abs(point[2] - box.center[2]) - radius[2], 0);
  return Math.hypot(dx, dy, dz);
};

const maximumScale = (matrix: readonly number[]): number =>
  Math.max(
    Math.hypot(matrix[0]!, matrix[1]!, matrix[2]!),
    Math.hypot(matrix[4]!, matrix[5]!, matrix[6]!),
    Math.hypot(matrix[8]!, matrix[9]!, matrix[10]!),
  );

const screenSpaceError = (
  tile: TilesetTile,
  matrix: readonly number[],
  box: WorldBox,
  camera: CameraView,
): number => {
  const error = tile.geometricError * maximumScale(matrix);
  return camera.projection === "orthographic"
    ? orthographicScreenSpaceError(
        error,
        camera.viewportHeightCssPx,
        camera.parallelScale,
      )
    : perspectiveScreenSpaceError(
        error,
        boxDistance(box, camera.position),
        camera.viewportHeightCssPx,
        camera.fovY,
      );
};

type VisitResult = {
  readonly visible: boolean;
  readonly coverageSubmitted: boolean;
  readonly desired: string[];
  readonly requested: string[];
  readonly drawn: string[];
};

const hasContent = (tile: TilesetTile): boolean =>
  tile.contentUrl !== undefined;

export const traverseTileset = (
  options: TilesetTraversalOptions,
): TilesetTraversalResult => {
  if (
    !Number.isFinite(options.maximumScreenSpaceErrorPx) ||
    options.maximumScreenSpaceErrorPx <= 0
  ) {
    throw new RangeError("maximumScreenSpaceErrorPx must be finite and > 0");
  }
  const quality = options.qualityFraction ?? 1;
  if (!Number.isFinite(quality) || quality < 0 || quality > 1) {
    throw new RangeError("qualityFraction must be finite and in [0, 1]");
  }
  const effective = options.maximumScreenSpaceErrorPx / Math.max(quality, 0.05);
  const planes = frustumPlanes(options.camera.viewProj);
  const culled: string[] = [];

  const hasSubmittedDescendant = (tile: TilesetTile): boolean =>
    tile.children.some(
      (child) =>
        (hasContent(child) && options.readiness(child.id) === "submitted") ||
        hasSubmittedDescendant(child),
    );

  const visit = (
    tile: TilesetTile,
    parentTransform: readonly number[],
  ): VisitResult => {
    const accumulated = multiplyTilesetMatrices(
      parentTransform,
      tile.transform,
    );
    const bounds = worldBox(tile.boundingVolume, accumulated);
    if (!boxIntersectsFrustum(bounds, planes)) {
      culled.push(tile.id);
      return {
        visible: false,
        coverageSubmitted: true,
        desired: [],
        requested: [],
        drawn: [],
      };
    }

    const contentful = hasContent(tile);
    // Contentless tiles are hierarchy nodes, not drawable levels of detail.
    // Their descendants remain reachable regardless of the node's own SSE.
    const refine =
      tile.children.length > 0 &&
      (!contentful ||
        screenSpaceError(tile, accumulated, bounds, options.camera) >
          effective);
    const childResults = refine
      ? tile.children.map((child) => visit(child, accumulated))
      : [];
    const visibleChildren = childResults.filter((child) => child.visible);

    if (!refine || visibleChildren.length === 0) {
      if (!contentful) {
        return {
          visible: true,
          // A contentless leaf (or a hierarchy branch whose descendants are
          // all culled) has no draw obligation of its own.
          coverageSubmitted: true,
          desired: [],
          requested: [],
          drawn: [],
        };
      }
      const submitted = options.readiness(tile.id) === "submitted";
      // Reverse refinement is paced too: if a coarser desired parent is not
      // submitted yet, retain only descendants that were already submitted.
      // Fresh coarse startup does not request the entire hierarchy.
      const fallbacks =
        !submitted && hasSubmittedDescendant(tile)
          ? tile.children.map((child) => visit(child, accumulated))
          : [];
      const visibleFallbacks = fallbacks.filter((child) => child.visible);
      return {
        visible: true,
        coverageSubmitted:
          submitted ||
          (visibleFallbacks.length > 0 &&
            visibleFallbacks.every((child) => child.coverageSubmitted)),
        desired: [tile.id],
        requested: [tile.id, ...fallbacks.flatMap((child) => child.requested)],
        drawn: submitted
          ? [tile.id]
          : fallbacks.flatMap((child) => child.drawn),
      };
    }

    const desired = visibleChildren.flatMap((child) => child.desired);
    if (visibleChildren.every((child) => child.coverageSubmitted)) {
      return {
        visible: true,
        coverageSubmitted: true,
        desired,
        requested: visibleChildren.flatMap((child) => child.requested),
        drawn: visibleChildren.flatMap((child) => child.drawn),
      };
    }

    const submitted = contentful && options.readiness(tile.id) === "submitted";
    return {
      visible: true,
      coverageSubmitted: submitted,
      desired,
      requested: [
        ...(contentful ? [tile.id] : []),
        ...visibleChildren.flatMap((child) => child.requested),
      ],
      // Never partially replace a submitted parent. If it has not arrived yet,
      // any submitted descendants are still useful while initial coverage loads.
      drawn: submitted
        ? [tile.id]
        : visibleChildren.flatMap((child) => child.drawn),
    };
  };

  const rootParent = options.modelMatrix ?? IDENTITY;
  const result = visit(options.root, rootParent);
  const rootAccumulated = multiplyTilesetMatrices(
    rootParent,
    options.root.transform,
  );
  const rootSse = result.visible
    ? screenSpaceError(
        options.root,
        rootAccumulated,
        worldBox(options.root.boundingVolume, rootAccumulated),
        options.camera,
      )
    : 0;
  return Object.freeze({
    desiredTileIds: Object.freeze(result.desired),
    requestedTileIds: Object.freeze(result.requested),
    drawnTileIds: Object.freeze(result.drawn),
    culledTileIds: Object.freeze(culled),
    effectiveScreenSpaceErrorPx: effective,
    rootScreenSpaceErrorPx: Number.isFinite(rootSse) && rootSse > 0 ? rootSse : 0,
  });
};
