export {
  ROOT_KEY,
  keyToString,
  keyFromString,
  childKeys,
  type VoxelKey,
  type Vec3,
  type Bounds,
} from "./octree";

export type {
  TileSource,
  TileSourceMetadata,
  NodeInfo,
  TileData,
  LoadOptions,
} from "./tileSource";

export { orderTileForProgressiveDrawing } from "./progressiveOrder";

export {
  cursorRay,
  type CameraView,
  type PerspectiveCameraView,
  type OrthographicCameraView,
  type Mat16,
  type CursorRay,
} from "./camera";

// The picking machinery itself (ray building, prefilter, sweep) is internal:
// the public query is `LodController.pickPoint`, and only its result shape and
// the bucket radii — mirrored from telesculptor-web's `scene/ray_depth.py` —
// are contract.
export {
  DEFAULT_PICK_PIXEL_RADIUS,
  DEFAULT_PICK_PIXEL_RADIUS_MULTIPLIERS,
  PICK_RADII_CSS_PX,
  type PointPickResult,
} from "./picking";

export {
  createLodController,
  type LodController,
  type LodControllerOptions,
  type LodControllerStats,
  type LodGovernorInputs,
  type LodSelectionStats,
  type LodDrawPlanStats,
  type PointPresentation,
  type FixedPointPresentation,
  type AutoPointPresentation,
  type TileBatch,
  type TileDrawPlan,
} from "./controller";

export {
  createMemoryPool,
  defaultMemoryBudgetBytes,
  DEFAULT_MEMORY_BUDGET_BYTES,
  type MemoryPool,
  type MemoryPoolMember,
  type MemoryPoolOptions,
} from "./memoryPool";

export type {
  Submission,
  SubmissionJob,
  SubmissionScheduler,
  SubmissionSchedulerStats,
} from "./submissionScheduler";

export {
  createStreamedMemberFactoryRegistry,
  type MemberFactoryRegistration,
  type StreamedMemberFactory,
  type StreamedMemberFactoryRegistry,
} from "./memberFactoryRegistry";

export type {
  Allocation,
  AllocationRegime,
  GovernorInputs,
  Importance,
  MemberPickResult,
  OcclusionResult,
  OutstandingWork,
  StreamedMember,
  StreamedMemberContext,
} from "./streamedMember";

export { scenePoint, type ScenePoint } from "./frames";

export { CULLED, importanceFromRootSseCssPx } from "./streamedMember";

export {
  createStreamedSceneCoordinator,
  type AdaptiveQualityTargets,
  type StreamedCoordinatorMemberStats,
  type StreamedMemberRegistration,
  type StreamedMemberRegistrationOptions,
  type StreamedSceneCoordinator,
  type StreamedSceneCoordinatorOptions,
  type StreamedSceneCoordinatorStats,
} from "./streamedSceneCoordinator";

// The adaptive quality loop is the view governor's internal. Its defaults stay
// public on this vtk-free entry so a host, and a node-side check of its
// configuration, reads the budget floor (`minBudget`) instead of restating it.
// Otherwise only the types in the governor's own surface are re-exported.
export {
  ADAPTIVE_QUALITY_DEFAULTS,
  type AdaptiveQualityOptions,
  type QualityAdjustment,
  type QualityAdjustmentDirection,
  type QualityAdjustmentReason,
  type QualityRegime,
} from "./adaptiveBudget";

export {
  createViewGovernor,
  type CapacitySampleMetrics,
  type FrameVerdict,
  type GovernorWorkState,
  type HostFrameMetrics,
  type MotionReference,
  type MotionSourceKind,
  type TransientFrameMetrics,
  type ViewGovernor,
  type ViewGovernorOptions,
  type ViewGovernorStats,
} from "./viewGovernor";

export {
  createHttpTileSource,
  RevisionGoneError,
  type HttpTileSourceOptions,
} from "./httpTileSource";

export {
  createCopcTileSource,
  type CopcTileSourceOptions,
  type RangeGetter,
} from "./copcTileSource";

export {
  createCopcWorkerTileSource,
  type CopcWorkerTileSourceOptions,
} from "./copcWorkerTileSource";

export {
  TilesetFetchError,
  TilesetProfileError,
  TilesetUnsupportedError,
  TilesetValidationError,
  type TilesetFetch,
  type TilesetFetchResponse,
} from "./tiles3d/tilesetSource";

export {
  DEFAULT_MAXIMUM_SCREEN_SPACE_ERROR_PX,
  DEFAULT_TILES3D_CACHE_BYTES,
  DEFAULT_TILES3D_CONCURRENCY,
  DEFAULT_VERTICAL_EXAGGERATION,
  DEFAULT_VERTICAL_PIVOT_Z,
  type ContentQueueStats,
  type Tiles3dMemberConfig,
  type Tiles3dMemberStats,
} from "./tiles3d/memberTypes";

export type {
  ContentQueueFetch,
  ContentQueueFetchResponse,
} from "./tiles3d/contentQueue";

export {
  DecodeWorkerError,
  DecodeWorkerPool,
  DecodeWorkerPoolDisposedError,
  TileDecodeError,
  TileUnsupportedExtensionError,
  type CompressedTextureFormat,
  type DecodeWasmUrls,
  type DecodeWorkerLike,
  type DecodeWorkerPoolHandle,
  type DecodeWorkerPoolOptions,
  type TextureCapabilities,
  type TileDecodeStage,
} from "./tiles3d/decode";

export {
  createEcefToEnuTransform,
  wgs84ToEcef,
  type Mat4,
} from "./tiles3d/rtc";

// The vtk.js renderer adapter is deliberately NOT re-exported here. It imports
// vtk.js at module scope, so re-exporting it would make this entry point throw
// for consumers that only want the octree/controller core. Import it from
// `pointcloud-lod/vtk` instead.
