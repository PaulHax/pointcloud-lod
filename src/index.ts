export {
  ROOT_KEY,
  keyToString,
  keyFromString,
  parentKey,
  childKeys,
  nodeBounds,
  nodeCube,
  pointSpacing,
  type VoxelKey,
  type Vec3,
  type Cube,
  type Bounds,
} from './octree';

export {
  createLruCache,
  type LruCache,
  type LruCacheOptions,
} from './lru';

export {
  selectNodes,
  allChildren,
  type HierarchyNode,
  type SelectNodesOptions,
  type NodeSelection,
} from './budget';

export type {
  TileSource,
  TileSourceMetadata,
  NodeInfo,
  TileData,
  LoadTileOptions,
} from './tileSource';

export {
  frustumPlanes,
  cubeIntersectsFrustum,
  distanceToCube,
  screenSpaceError,
  nodeScreenSpaceError,
  type CameraView,
  type Mat16,
  type Plane,
} from './camera';

export {
  createLodController,
  type LodController,
  type LodControllerOptions,
  type LodControllerStats,
  type TileBatch,
} from './controller';

export {
  createMemoryPool,
  defaultMemoryBudgetBytes,
  DEFAULT_MEMORY_BUDGET_BYTES,
  type MemoryPool,
  type MemoryPoolMember,
  type MemoryPoolOptions,
} from './memoryPool';

export {
  createAdaptiveBudget,
  percentile,
  type AdaptiveBudget,
  type AdaptiveBudgetOptions,
  type AdaptiveBudgetStats,
  type AdaptiveBudgetTrackStats,
  type BudgetRegime,
  type RecordFrameOptions,
} from './adaptiveBudget';

export {
  createHttpTileSource,
  parsePct1,
  RevisionGoneError,
  PCT1_HEADER_BYTES,
  type HttpTileSourceOptions,
} from './httpTileSource';

export {
  createCopcTileSource,
  type CopcTileSourceOptions,
  type RangeGetter,
} from './copcTileSource';

// The vtk.js renderer adapter is deliberately NOT re-exported here. It imports
// vtk.js at module scope, so re-exporting it would make this entry point throw
// for consumers that only want the octree/controller core. Import it from
// `pointcloud-lod/vtk` instead.
