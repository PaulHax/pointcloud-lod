import type { TilesetFetch } from "./tilesetSource";
import type { ContentQueueFetch } from "./contentQueue";
import type { SubtreeFetch } from "./subtreeStore";
import type { DecodeWasmUrls } from "./decode";
import type { Allocation } from "../streamedMember";
import type { MeshAdapterStats } from "./meshAdapter";

export const DEFAULT_MAXIMUM_SCREEN_SPACE_ERROR_PX = 16;
export const DEFAULT_TILES3D_CACHE_BYTES = 128 * 1024 * 1024;
export const DEFAULT_TILES3D_MIN_CONCURRENCY = 1;
export const DEFAULT_TILES3D_MAX_CONCURRENCY = 4;
export const DEFAULT_TILES3D_SUBTREE_CACHE_BYTES = 8 * 1024 * 1024;
export const DEFAULT_VERTICAL_EXAGGERATION = 1;
export const DEFAULT_VERTICAL_PIVOT_Z = 0;

export type GeometricErrorScale = "maximum" | "horizontal";

export type Tiles3dMemberConfig = {
  readonly endpoint: string;
  readonly revision: string;
  readonly tilesetToScene: readonly number[];
  /** Finite positive scale applied to the scene Z axis. */
  readonly verticalExaggeration?: number;
  /** Finite scene Z coordinate held fixed by exaggeration. */
  readonly verticalPivotZ?: number;
  /** Scale convention for tile geometric error; terrain error is horizontal. */
  readonly geometricErrorScale?: GeometricErrorScale;
  readonly maximumScreenSpaceErrorPx?: number;
  readonly wasm?: DecodeWasmUrls;
  readonly cacheBytes?: number;
  readonly minConcurrency?: number;
  readonly maxConcurrency?: number;
  readonly maxAttempts?: number;
  readonly retryBackoffMs?: (failedAttempt: number) => number;
  readonly fetchTileset?: TilesetFetch;
  readonly fetchContent?: ContentQueueFetch;
  readonly fetchSubtree?: SubtreeFetch;
  readonly onError?: (error: unknown) => void;
};

export type Tiles3dMemberStats = {
  readonly kind: "tiles3d";
  readonly active: boolean;
  readonly disposed: boolean;
  readonly sourceState: "idle" | "loading" | "ready" | "failed" | "disposed";
  /**
   * The root does not fit the member's byte allowance, so nothing is drawn
   * and no further tile work is outstanding. Clears when the allowance grows.
   */
  readonly irreducibleBudget: boolean;
  readonly revision: string;
  readonly capabilityKey: string;
  readonly devicePixelRatio: number;
  readonly interactionDepth: number;
  /** Currency of placement-affecting source configuration. */
  readonly configGeneration: number;
  readonly verticalExaggeration: number;
  readonly verticalPivotZ: number;
  readonly geometricErrorScale: GeometricErrorScale;
  readonly allocation: Allocation;
  readonly maximumScreenSpaceErrorPx: number;
  readonly effectiveScreenSpaceErrorPx: number;
  readonly sseMultiplier: number;
  readonly memoryConstrained: boolean;
  readonly selectedTiles: number;
  readonly requestedTiles: number;
  /** Traversal passes, exposed so hosts can verify stationary frame idempotence. */
  readonly selectionPasses: number;
  readonly errorCount: number;
  readonly lastError: string | null;
  readonly queue: {
    readonly selected: number;
    readonly active: number;
    readonly queued: number;
    readonly retrying: number;
    readonly ready: number;
    readonly failed: number;
    readonly decodedBytes: number;
    readonly workPending: boolean;
    readonly cacheHits: number;
    readonly cacheMisses: number;
    readonly cacheEvictions: number;
    readonly cacheRevisits: number;
    readonly entries: readonly {
      readonly id: string;
      readonly url: string;
      readonly status: "queued" | "fetching" | "retrying" | "ready" | "failed";
      readonly attempt: number;
    }[];
  } | null;
  readonly subtrees: {
    readonly selected: number;
    readonly active: number;
    readonly queued: number;
    readonly retrying: number;
    readonly ready: number;
    readonly failed: number;
    readonly cached: number;
    readonly cachedBytes: number;
    readonly workPending: boolean;
    readonly cacheHits: number;
    readonly cacheMisses: number;
    readonly cacheEvictions: number;
  } | null;
  readonly decode: ReturnType<
    NonNullable<import("./decode").DecodeWorkerPoolHandle["stats"]>
  > | null;
  readonly renderer: MeshAdapterStats;
  readonly submissions: {
    readonly queuedJobs: number;
    readonly queuedBytes: number;
    readonly lastFrameAdmittedJobs: number;
    readonly lastFrameAdmittedBytes: number;
    readonly admittedJobs: number;
    readonly admittedBytes: number;
    readonly peakQueuedJobs: number;
    readonly peakQueuedBytes: number;
    readonly peakFrameAdmittedBytes: number;
    readonly peakFrameElapsedMs: number;
    readonly admissionFrames: number;
  };
};
