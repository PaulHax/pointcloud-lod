/** Pure camera-driven traversal for explicit and implicit REPLACE hierarchies. */

import {
  IDENTITY,
  frustumPlanes,
  orthographicScreenSpaceError,
  perspectiveScreenSpaceError,
  transformPointBy,
  type CameraView,
  type Plane,
} from "../camera";
import {
  multiplyTilesetMatrices,
  substituteImplicitTemplate,
  type ImplicitTileAddress,
  type TilesetBox,
  type TilesetImplicitTiling,
  type TilesetTile,
} from "./tilesetSource";
import {
  quadtreeMortonIndex,
  subtreeTileIndex,
  type ParsedSubtree,
} from "./subtree";
import type {
  ContentQueueEntrySnapshot,
  TileContentRequest,
} from "./contentQueue";
import type { GeometricErrorScale } from "./memberTypes";

/** Immutable implicit-hierarchy state sampled at the start of one pass. */
export type SubtreeHierarchyState = {
  readonly revision: string;
  readonly configGeneration: number;
  readonly entries: readonly ContentQueueEntrySnapshot[];
  readonly subtreeById: ReadonlyMap<string, ParsedSubtree>;
};

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
  /** Caller-owned anchor × tileset-to-scene transform. */
  readonly modelMatrix?: readonly number[];
  /** Maximum-axis 3D Tiles convention, or local XY for terrain-derived error. */
  readonly geometricErrorScale?: GeometricErrorScale;
  readonly readiness: (tileId: string) => TileReadiness;
  /**
   * Whether a contentful tile may still be refined past, default true.
   *
   * Answers "is there any point asking for this tile's children", which is a
   * different question from whether they have arrived. A caller that knows a
   * tile's replacement set can never be drawn — it does not fit the memory
   * allowance — says so here, and the tile is selected as the frontier instead
   * of being refined past into content that will be requested forever.
   */
  readonly refinable?: (tileId: string) => boolean;
  /** Retain previously drawn refinement at the configured SSE, within the
   * caller's small-frontier budget. Does not override memory-blocked groups. */
  readonly retainRefinement?: (tileId: string) => boolean;
  /** Immutable hierarchy state sampled at the beginning of this pass. */
  readonly subtrees?: SubtreeHierarchyState | null;
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
  /** Visible unknown boundaries. The member owns fetching and retry policy. */
  readonly neededSubtreeRequests: readonly TileContentRequest[];
  /** Explicit source tiles plus implicit tiles materialized for this pass. */
  readonly tileById: ReadonlyMap<string, TilesetTile>;
};

type Vec3 = readonly [number, number, number];

type WorldBox = {
  readonly center: Vec3;
  readonly axes: readonly [Vec3, Vec3, Vec3];
};

type TilePlacement = {
  transform: readonly number[];
  bounds: WorldBox;
};

const transformVector = (matrix: readonly number[], vector: Vec3): Vec3 => [
  matrix[0]! * vector[0] + matrix[4]! * vector[1] + matrix[8]! * vector[2],
  matrix[1]! * vector[0] + matrix[5]! * vector[1] + matrix[9]! * vector[2],
  matrix[2]! * vector[0] + matrix[6]! * vector[1] + matrix[10]! * vector[2],
];

const worldBox = (box: TilesetBox, matrix: readonly number[]): WorldBox => {
  const h = box.halfAxes;
  return {
    center: transformPointBy(matrix, box.center),
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

const horizontalScale = (matrix: readonly number[]): number =>
  Math.max(
    Math.hypot(matrix[0]!, matrix[1]!, matrix[2]!),
    Math.hypot(matrix[4]!, matrix[5]!, matrix[6]!),
  );

const screenSpaceError = (
  tile: TilesetTile,
  matrix: readonly number[],
  box: WorldBox,
  camera: CameraView,
  errorScale: GeometricErrorScale,
): number => {
  const scale =
    errorScale === "horizontal"
      ? horizontalScale(matrix)
      : maximumScale(matrix);
  const error = tile.geometricError * scale;
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

type MaterializedHierarchy = {
  readonly root: TilesetTile;
  readonly tileById: ReadonlyMap<string, TilesetTile>;
  readonly unknownRequestByTileId: ReadonlyMap<string, TileContentRequest>;
};

const implicitTileId = (address: ImplicitTileAddress): string => {
  if (address.level === 0) return "root";
  const parts = ["root"];
  for (let bit = address.level - 1; bit >= 0; bit -= 1) {
    parts.push(String(((address.x >> bit) & 1) + 2 * ((address.y >> bit) & 1)));
  }
  return parts.join("/");
};

const childAddress = (
  parent: ImplicitTileAddress,
  quadrant: number,
): ImplicitTileAddress =>
  Object.freeze({
    level: parent.level + 1,
    x: parent.x * 2 + (quadrant & 1),
    y: parent.y * 2 + ((quadrant >> 1) & 1),
  });

const implicitRequest = (
  descriptor: TilesetImplicitTiling,
  address: ImplicitTileAddress,
): TileContentRequest =>
  Object.freeze({
    id: `subtree/${address.level}/${address.x}/${address.y}`,
    url: substituteImplicitTemplate(descriptor.subtreeUrlTemplate, address),
  });

const implicitContent = (
  descriptor: TilesetImplicitTiling,
  address: ImplicitTileAddress,
): { readonly contentUri: string; readonly contentUrl: string } =>
  Object.freeze({
    contentUri: substituteImplicitTemplate(
      descriptor.contentUriTemplate,
      address,
    ),
    contentUrl: substituteImplicitTemplate(
      descriptor.contentUrlTemplate,
      address,
    ),
  });

const explicitHierarchy = (root: TilesetTile): MaterializedHierarchy => {
  const tileById = new Map<string, TilesetTile>();
  const collect = (tile: TilesetTile): void => {
    tileById.set(tile.id, tile);
    for (const child of tile.children) collect(child);
  };
  collect(root);
  return {
    root,
    tileById,
    unknownRequestByTileId: new Map(),
  };
};

const implicitHierarchy = (
  documentRoot: TilesetTile,
  snapshot: SubtreeHierarchyState | null | undefined,
): MaterializedHierarchy => {
  const descriptor = documentRoot.implicitTiling!;
  const tileById = new Map<string, TilesetTile>();
  const unknownRequestByTileId = new Map<string, TileContentRequest>();
  const entryById = new Map(
    snapshot?.entries.map((entry) => [entry.id, entry]) ?? [],
  );

  const proxy = (
    address: ImplicitTileAddress,
    boundingVolume: TilesetBox,
  ): TilesetTile => {
    const id = implicitTileId(address);
    unknownRequestByTileId.set(id, implicitRequest(descriptor, address));
    const tile = Object.freeze({
      id,
      geometricError: documentRoot.geometricError / 2 ** address.level,
      boundingVolume,
      transform: address.level === 0 ? documentRoot.transform : IDENTITY,
      worldTransform: documentRoot.worldTransform,
      children: Object.freeze([]),
      implicitAddress: address,
    });
    tileById.set(id, tile);
    return tile;
  };

  const buildSubtree = (
    subtreeRoot: ImplicitTileAddress,
    fallbackBounds: TilesetBox,
  ): TilesetTile => {
    const request = implicitRequest(descriptor, subtreeRoot);
    const parsed = snapshot?.subtreeById.get(request.id);
    if (!parsed) return proxy(subtreeRoot, fallbackBounds);

    const buildTile = (
      localLevel: number,
      localX: number,
      localY: number,
      address: ImplicitTileAddress,
    ): TilesetTile => {
      const tileIndex = subtreeTileIndex(localLevel, localX, localY);
      const bounds = parsed.tileBoundingBoxes[tileIndex];
      if (!bounds) {
        // parseSubtree enforces metadata for every available tile. This guard
        // keeps custom test parsers from materializing an under-bounded tile.
        throw new Error(
          `available implicit tile ${tileIndex} has no metadata bounds`,
        );
      }
      const children: TilesetTile[] = [];
      if (address.level + 1 < descriptor.availableLevels) {
        if (localLevel + 1 < descriptor.subtreeLevels) {
          for (let quadrant = 0; quadrant < 4; quadrant += 1) {
            const childLocalX = localX * 2 + (quadrant & 1);
            const childLocalY = localY * 2 + ((quadrant >> 1) & 1);
            const childIndex = subtreeTileIndex(
              localLevel + 1,
              childLocalX,
              childLocalY,
            );
            if (!parsed.tileAvailability.isAvailable(childIndex)) continue;
            children.push(
              buildTile(
                localLevel + 1,
                childLocalX,
                childLocalY,
                childAddress(address, quadrant),
              ),
            );
          }
        } else {
          for (let quadrant = 0; quadrant < 4; quadrant += 1) {
            const childLocalX = localX * 2 + (quadrant & 1);
            const childLocalY = localY * 2 + ((quadrant >> 1) & 1);
            const childIndex = quadtreeMortonIndex(
              descriptor.subtreeLevels,
              childLocalX,
              childLocalY,
            );
            if (!parsed.childSubtreeAvailability.isAvailable(childIndex))
              continue;
            const nextAddress = childAddress(address, quadrant);
            children.push(buildSubtree(nextAddress, bounds));
          }
        }
      }
      const content = parsed.contentAvailability.isAvailable(tileIndex)
        ? implicitContent(descriptor, address)
        : undefined;
      const id = implicitTileId(address);
      const tile: TilesetTile = Object.freeze({
        id,
        geometricError: documentRoot.geometricError / 2 ** address.level,
        boundingVolume: bounds,
        transform: address.level === 0 ? documentRoot.transform : IDENTITY,
        worldTransform: documentRoot.worldTransform,
        ...content,
        children: Object.freeze(children),
        implicitAddress: address,
        ...(address.level === 0 ? { implicitTiling: descriptor } : {}),
      });
      tileById.set(id, tile);
      return tile;
    };

    // A failed selected entry remains an unknown boundary and is deliberately
    // not converted to empty availability.
    if (entryById.get(request.id)?.status === "failed") {
      return proxy(subtreeRoot, fallbackBounds);
    }
    return buildTile(0, 0, 0, subtreeRoot);
  };

  const rootAddress = Object.freeze({ level: 0, x: 0, y: 0 });
  return {
    root: buildSubtree(rootAddress, documentRoot.boundingVolume),
    tileById,
    unknownRequestByTileId,
  };
};

const hasContent = (tile: TilesetTile): boolean =>
  tile.contentUrl !== undefined;

const sameHierarchy = (
  left: SubtreeHierarchyState | null | undefined,
  right: SubtreeHierarchyState | null | undefined,
): boolean => {
  if (left === right) return true;
  if (
    !left ||
    !right ||
    left.revision !== right.revision ||
    left.configGeneration !== right.configGeneration ||
    left.subtreeById.size !== right.subtreeById.size
  )
    return false;
  for (const [id, subtree] of left.subtreeById) {
    if (right.subtreeById.get(id) !== subtree) return false;
  }
  const failed = (snapshot: SubtreeHierarchyState) =>
    new Set(
      snapshot.entries
        .filter((entry) => entry.status === "failed")
        .map((entry) => entry.id),
    );
  const a = failed(left),
    b = failed(right);
  return a.size === b.size && [...a].every((id) => b.has(id));
};

/** Per-member cache of one hierarchy, bounded by the current subtree store.
 * Camera, readiness and quality are still evaluated on every traversal.
 */
export const createTilesetTraversal = () => {
  let root: TilesetTile | undefined;
  let snapshot: SubtreeHierarchyState | null | undefined;
  let hierarchy: MaterializedHierarchy | undefined;
  let modelMatrix: readonly number[] | undefined;
  const placements = new Map<TilesetTile, TilePlacement>();
  return (options: TilesetTraversalOptions): TilesetTraversalResult => {
    if (root !== options.root || !sameHierarchy(snapshot, options.subtrees)) {
      hierarchy = undefined;
      placements.clear();
    }
    const nextMatrix = options.modelMatrix ?? IDENTITY;
    if (
      !modelMatrix ||
      nextMatrix.some((value, i) => value !== modelMatrix![i])
    ) {
      modelMatrix = Array.from(nextMatrix);
      placements.clear();
    }
    root = options.root;
    snapshot = options.subtrees;
    hierarchy ??= root.implicitTiling
      ? implicitHierarchy(root, snapshot)
      : explicitHierarchy(root);
    return traverseHierarchy(options, hierarchy, placements);
  };
};

export const traverseTileset = (
  options: TilesetTraversalOptions,
): TilesetTraversalResult => traverseHierarchy(options);

const traverseHierarchy = (
  options: TilesetTraversalOptions,
  materialized?: MaterializedHierarchy,
  placements?: Map<TilesetTile, TilePlacement>,
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
  const errorScale = options.geometricErrorScale ?? "maximum";
  if (errorScale !== "maximum" && errorScale !== "horizontal") {
    throw new RangeError(
      "geometricErrorScale must be 'maximum' or 'horizontal'",
    );
  }
  const effective = options.maximumScreenSpaceErrorPx / Math.max(quality, 0.05);
  const planes = frustumPlanes(options.camera.viewProj);
  const culled: string[] = [];
  const hierarchy =
    materialized ??
    (options.root.implicitTiling
      ? implicitHierarchy(options.root, options.subtrees)
      : explicitHierarchy(options.root));
  const neededSubtrees = new Map<string, TileContentRequest>();

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
    let placement = placements?.get(tile);
    if (!placement) {
      const transform = multiplyTilesetMatrices(
        parentTransform,
        tile.transform,
      );
      placement = {
        transform,
        bounds: worldBox(tile.boundingVolume, transform),
      };
      placements?.set(tile, placement);
    }
    const { transform: accumulated, bounds } = placement;
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

    const unknownRequest = hierarchy.unknownRequestByTileId.get(tile.id);
    if (unknownRequest) {
      neededSubtrees.set(unknownRequest.id, unknownRequest);
      const submitted = options.readiness(tile.id) === "submitted";
      return {
        visible: true,
        coverageSubmitted: submitted,
        desired: [],
        requested: submitted ? [tile.id] : [],
        drawn: submitted ? [tile.id] : [],
      };
    }

    const contentful = hasContent(tile);
    // Contentless tiles are hierarchy nodes, not drawable levels of detail.
    // Their descendants remain reachable regardless of the node's own SSE, and
    // regardless of `refinable` — such a node has no content of its own to
    // stand as a frontier, so refusing to descend would draw nothing at all.
    const refine =
      tile.children.length > 0 &&
      (!contentful ||
        (screenSpaceError(
          tile,
          accumulated,
          bounds,
          options.camera,
          errorScale,
        ) >
          (options.retainRefinement?.(tile.id)
            ? options.maximumScreenSpaceErrorPx
            : effective) &&
          (options.refinable?.(tile.id) ?? true)));
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
  const result = visit(hierarchy.root, rootParent);
  const rootAccumulated = multiplyTilesetMatrices(
    rootParent,
    hierarchy.root.transform,
  );
  const rootSse = result.visible
    ? screenSpaceError(
        hierarchy.root,
        rootAccumulated,
        worldBox(hierarchy.root.boundingVolume, rootAccumulated),
        options.camera,
        errorScale,
      )
    : 0;
  return Object.freeze({
    desiredTileIds: Object.freeze(result.desired),
    requestedTileIds: Object.freeze(result.requested),
    drawnTileIds: Object.freeze(result.drawn),
    culledTileIds: Object.freeze(culled),
    effectiveScreenSpaceErrorPx: effective,
    rootScreenSpaceErrorPx:
      Number.isFinite(rootSse) && rootSse > 0 ? rootSse : 0,
    neededSubtreeRequests: Object.freeze([...neededSubtrees.values()]),
    tileById: hierarchy.tileById,
  });
};
