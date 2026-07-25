export {
  ROOT_KEY,
  keyToString,
  keyFromString,
  childKeys,
  nodeBounds,
  nodeCube,
  pointSpacing,
  type VoxelKey,
  type Vec3,
  type Cube,
  type Bounds,
} from "./octree";

export { createLruCache, type LruCache, type LruCacheOptions } from "./lru";

export {
  selectNodes,
  type HierarchyNode,
  type SelectNodesOptions,
  type NodeSelection,
} from "./budget";

export type {
  TileSource,
  TileSourceMetadata,
  NodeInfo,
  TileData,
  LoadTileOptions,
} from "./tileSource";

export {
  frustumPlanes,
  boundsIntersectsFrustum,
  distanceToBounds,
  screenSpaceError,
  nodeScreenSpaceError,
  type CameraView,
  type Mat16,
  type Plane,
} from "./camera";

export {
  createLodController,
  type LodController,
  type LodControllerOptions,
  type LodControllerStats,
  type LodSelectionStats,
  type PointPresentation,
  type FixedPointPresentation,
  type AutoPointPresentation,
  type TileBatch,
} from "./controller";

export {
  createMemoryPool,
  defaultMemoryBudgetBytes,
  DEFAULT_MEMORY_BUDGET_BYTES,
  type MemoryPool,
  type MemoryPoolMember,
  type MemoryPoolOptions,
} from "./memoryPool";

// The adaptive budget loop is the view governor's internal. Only the two
// types that appear in the governor's own surface are re-exported.
export type {
  AdaptiveBudgetOptions,
  AdaptiveBudgetStats,
} from "./adaptiveBudget";

export {
  createViewGovernor,
  type HostFrameMetrics,
  type ViewGovernor,
  type ViewGovernorMember,
  type ViewGovernorMemberOptions,
  type ViewGovernorOptions,
  type ViewGovernorStats,
} from "./viewGovernor";

export {
  createHttpTileSource,
  parsePct1,
  RevisionGoneError,
  PCT1_HEADER_BYTES,
  type HttpTileSourceOptions,
} from "./httpTileSource";

export {
  createCopcTileSource,
  type CopcTileSourceOptions,
  type RangeGetter,
} from "./copcTileSource";

// The vtk.js renderer adapter is deliberately NOT re-exported here. It imports
// vtk.js at module scope, so re-exporting it would make this entry point throw
// for consumers that only want the octree/controller core. Import it from
// `pointcloud-lod/vtk` instead.
