import type { CameraView, Mat16 } from "./camera";
import type { MemoryPool } from "./memoryPool";
import type { Vec3 } from "./octree";
import type { SubmissionScheduler } from "./submissionScheduler";
import type {
  DecodeWorkerPoolHandle,
  TextureCapabilities,
} from "./tiles3d/decode";

export type DecodeWorkerPool = DecodeWorkerPoolHandle;
export type { TextureCapabilities } from "./tiles3d/decode";

export type StreamedMemberContext = {
  /** The renderer that owns the streamed anchor. Kept neutral at this seam. */
  readonly renderer: unknown;
  /** Coalescing view render request. Members must never render synchronously. */
  readonly scheduleRender: () => void;
  /** The one page-wide residency pool. The coordinator owns registrations. */
  readonly memory: MemoryPool;
  readonly workers: DecodeWorkerPool;
  readonly submissions: SubmissionScheduler;
  readonly textureCapabilities: TextureCapabilities;
  readonly devicePixelRatio: number;
  /**
   * Report that asynchronous work state changed without implying new pixels.
   * This is what keeps retry backoff visible to capacity-sample eligibility.
   */
  readonly onWorkChange?: () => void;
};

export type GovernorInputs = {
  /** Finite, non-negative projected importance. Zero means culled. */
  readonly projectedImportance: number;
  /** Fraction of full member quality useful to the current view. */
  readonly qualityDemand: number;
  /** Required fetch, decode, retry, or submission work has not drained. */
  readonly workPending: boolean;
  readonly physicalTileOperations: number;
  readonly physicalHierarchyOperations: number;
  readonly residentBytes: number;
};

export type AllocationRegime = "moving" | "stationary";

export type Allocation = {
  readonly qualityFraction: number;
  readonly memoryBudgetBytes: number;
  readonly regime: AllocationRegime;
};

export type MemberPickResult =
  | {
      readonly status: "hit";
      /** Positive distance from ray origin along its normalized direction. */
      readonly rayDepth: number;
      readonly pointOnRay: Vec3;
      readonly distancePx: number;
    }
  | { readonly status: "miss" };

export type OcclusionResult =
  | { readonly status: "hit"; readonly rayDepth: number }
  | { readonly status: "clear" };

/** The one pick-to-occlusion projection every member shares. */
export const occlusionFromPick = (
  result: MemberPickResult | null,
): OcclusionResult | null =>
  result === null
    ? null
    : result.status === "hit"
      ? { status: "hit", rayDepth: result.rayDepth }
      : { status: "clear" };

/** Renderer-neutral lifecycle every streamed scene format implements. */
export interface StreamedMember {
  setCamera(view: CameraView): void;
  setModelMatrix(matrix: Mat16 | null): void;
  setDevicePixelRatio(devicePixelRatio: number): void;
  setActive(active: boolean): void;
  setConfig(kindConfig: object): void;
  beginInteraction(): void;
  endInteraction(): void;
  prepareFrame(): void;
  governorInputs(): GovernorInputs;
  applyAllocation(allocation: Allocation): void;
  pick(view: CameraView, cssX: number, cssY: number): MemberPickResult | null;
  occlusionDepth(
    view: CameraView,
    cssX: number,
    cssY: number,
  ): OcclusionResult | null;
  stats(): object;
  dispose(): void;
}
