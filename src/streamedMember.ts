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

declare const IMPORTANCE_BRAND: unique symbol;

/**
 * A member's share-of-view weight, normalized to [0, 1].
 *
 * Branded so it can only be produced by the constructors below. A bare
 * `number` here was satisfied equally by a screen-space error in CSS pixels
 * (tens to hundreds) and by a literal `1`, and the allocator — which can only
 * read them as relative weights — then handed the pixel-scaled member ~99.9%
 * of the view budget and pinned the other at its floor. The unit is a property
 * of the value, so it has to be carried by the type.
 */
export type Importance = number & { readonly [IMPORTANCE_BRAND]: true };

/** Nothing of this member is on screen; it contends for no view quality. */
export const CULLED: Importance = 0 as Importance;

/**
 * Weigh a member by how far its root sits above its own refinement cutoff.
 *
 * Both members express demand the same way — "my coarsest visible content is
 * this many times worse than the error I am willing to draw" — so the ratio is
 * comparable across formats even though the underlying content is not.
 */
export const importanceFromRootSseCssPx = (
  rootSseCssPx: number,
  refinementCutoffCssPx: number,
): Importance => {
  if (
    !Number.isFinite(rootSseCssPx) ||
    rootSseCssPx <= 0 ||
    !Number.isFinite(refinementCutoffCssPx) ||
    refinementCutoffCssPx <= 0
  ) {
    return CULLED;
  }
  return Math.min(1, rootSseCssPx / refinementCutoffCssPx) as Importance;
};

export type GovernorInputs = {
  /** Normalized [0, 1] share-of-view weight. Zero means culled. */
  readonly projectedImportance: Importance;
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
      /**
       * Positive distance from ray origin along its normalized direction, in
       * the drawn frame. Comparable across members, so a nearer one can veto.
       */
      readonly rayDepth: number;
      /**
       * The hit in canonical scene ENU — NOT `origin + direction * rayDepth`
       * whenever the member draws in a transformed frame (terrain vertical
       * exaggeration). This is the value the app may store.
       */
      readonly scenePoint: Vec3;
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
