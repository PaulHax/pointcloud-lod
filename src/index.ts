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

export type {
  TileSource,
  TileSourceMetadata,
  NodeInfo,
  TileData,
  LoadOptions,
} from "./tileSource";

export {
  frustumPlanes,
  boundsIntersectsFrustum,
  distanceToBounds,
  perspectiveScreenSpaceError,
  orthographicScreenSpaceError,
  nodeScreenSpaceError,
  type CameraView,
  type PerspectiveCameraView,
  type OrthographicCameraView,
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

// The adaptive budget loop is the view governor's internal. Only the types
// that appear in the governor's own surface are re-exported.
export {
  // The adaptive policy's numbers are part of the public contract: a host
  // that validates a configured maximum against the floor must read the floor
  // from here, not restate it — a restated copy drifts the first time the
  // policy moves.
  DEFAULTS,
  type AdaptiveBudgetOptions,
  type BudgetAdjustment,
  type BudgetAdjustmentDirection,
  type BudgetAdjustmentReason,
  type BudgetRegime,
} from "./adaptiveBudget";

export {
  createViewGovernor,
  type BudgetConstraint,
  type HostFrameMetrics,
  type MotionReference,
  type MotionSourceKind,
  type ViewGovernor,
  type ViewGovernorMember,
  type ViewGovernorMemberOptions,
  type ViewGovernorMemberStats,
  type ViewGovernorMemberUpdate,
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
