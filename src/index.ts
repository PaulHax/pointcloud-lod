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
  cursorRay,
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

export {
  createSubmissionScheduler,
  DEFAULT_SUBMISSION_BYTES_PER_FRAME,
  DEFAULT_SUBMISSION_TIME_MS_PER_FRAME,
  type Submission,
  type SubmissionJob,
  type SubmissionScheduler,
  type SubmissionSchedulerOptions,
  type SubmissionSchedulerStats,
} from "./submissionScheduler";

export {
  MIN_VIEW_QUALITY_FRACTION,
  MAX_VIEW_QUALITY_FRACTION,
  allocateViewQuality,
  type ViewQualityAllocation,
  type ViewQualityContender,
} from "./viewBudget";

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

// The adaptive quality loop is the view governor's internal. Only the types
// that appear in the governor's own surface are re-exported.
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
  createGpuFrameTimer,
  type GpuFrameTimer,
  type GpuFrameTimerOptions,
  type GpuTimerResult,
} from "./gpuTimer";

export {
  captureTelemetryEnvironment,
  createTelemetryRecorder,
  isSoftwareRenderer,
  type TelemetryDetail,
  type TelemetryEnvironment,
  type TelemetryEvent,
  type TelemetryFrameEvent,
  type TelemetryLongTaskEvent,
  type TelemetryRecorder,
  type TelemetrySessionEvent,
  type TelemetryStateEvent,
  type TelemetrySummary,
  type TelemetryTrace,
  type TelemetryWorkEvent,
  type TelemetryWorkFinish,
  type TelemetryWorkKind,
} from "./telemetry";

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

export {
  createCopcWorkerTileSource,
  type CopcWorkerTileSourceOptions,
} from "./copcWorkerTileSource";

export {
  loadTileset,
  resolveTilesetContentUri,
  multiplyTilesetMatrices,
  TilesetFetchError,
  TilesetProfileError,
  TilesetUnsupportedError,
  TilesetValidationError,
  type LoadTilesetOptions,
  type TilesetBox,
  type TilesetFetch,
  type TilesetFetchResponse,
  type TilesetSource,
  type TilesetTile,
} from "./tiles3d/tilesetSource";

export {
  traverseTileset,
  type TileReadiness,
  type TilesetTraversalOptions,
  type TilesetTraversalResult,
} from "./tiles3d/traversal";

export {
  DEFAULT_MAXIMUM_SCREEN_SPACE_ERROR_PX,
  DEFAULT_TILES3D_CACHE_BYTES,
  DEFAULT_TILES3D_MIN_CONCURRENCY,
  DEFAULT_TILES3D_MAX_CONCURRENCY,
  DEFAULT_VERTICAL_EXAGGERATION,
  DEFAULT_VERTICAL_PIVOT_Z,
  type Tiles3dMemberConfig,
  type Tiles3dMemberStats,
} from "./tiles3d/memberTypes";

export {
  createContentQueue,
  ContentQueueDecodeError,
  ContentQueueError,
  ContentQueueFetchError,
  ContentQueueRetryError,
  type ContentDecodeContext,
  type ContentQueue,
  type ContentQueueClock,
  type ContentQueueConfiguration,
  type ContentQueueEntrySnapshot,
  type ContentQueueEntryStatus,
  type ContentQueueFetch,
  type ContentQueueFetchResponse,
  type ContentQueueOptions,
  type ContentQueueSnapshot,
  type TileContentRequest,
} from "./tiles3d/contentQueue";

export {
  buildDecodeCacheKey,
  buildTransferList,
  capabilityTarget,
  DecodeWorkerError,
  DecodeWorkerPool,
  DecodeWorkerPoolDisposedError,
  type BasisTargetTimingStats,
  type CompressedTextureLevel,
  type CompressedTextureFormat,
  type DecodeCacheIdentity,
  type DecodeJob,
  type DecodeTextureTarget,
  type DecodeTileRequest,
  type DecodedCompressedTexture,
  type DecodedMaterial,
  type DecodedPrimitive,
  type DecodedRgbaTexture,
  type DecodedTexture,
  type DecodedTileContent,
  type DecodeWasmUrls,
  type DecodeWorkerLike,
  type DecodeWorkerPoolHandle,
  type DecodeWorkerPoolOptions,
  type SerializableMaterial,
  type SerializableSampler,
  type TextureCapabilities,
} from "./tiles3d/decode";

export {
  composeSceneTransform,
  composeVerticalExaggeratedSceneTransform,
  createVerticalExaggerationTransform,
  createEcefToEnuTransform,
  flattenPrimitiveToRtc,
  multiplyMat4,
  transformPoint,
  wgs84ToEcef,
  type Mat4,
  type RtcPrimitiveInput,
  type RtcPrimitiveResult,
} from "./tiles3d/rtc";

// The vtk.js renderer adapter is deliberately NOT re-exported here. It imports
// vtk.js at module scope, so re-exporting it would make this entry point throw
// for consumers that only want the octree/controller core. Import it from
// `pointcloud-lod/vtk` instead.
