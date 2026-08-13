import type { Mat4 } from "../rtc";

export type CompressedTextureFormat =
  | "astc-4x4"
  | "bc7"
  | "etc2-rgba8"
  | "s3tc-dxt5";

export type DecodeTextureTarget = CompressedTextureFormat | "rgba";

export interface TextureCapabilities {
  /** Stable identity supplied by the live render context. */
  capabilityKey: string;
  /** Canonical formats supported by that context; order is ignored. */
  compressedFormats: readonly CompressedTextureFormat[];
}

export interface DecodeWasmUrls {
  draco?: {
    wrapperUrl: string;
    wasmUrl: string;
  };
  basis?: {
    encoderUrl: string;
    wasmUrl: string;
  };
}

export interface SerializableSampler {
  magFilter: 9728 | 9729;
  minFilter: 9728 | 9729 | 9984 | 9985 | 9986 | 9987;
  wrapS: 33071 | 33648 | 10497;
  wrapT: 33071 | 33648 | 10497;
}

export interface SerializableMaterial {
  version: 1;
  kind: "gltf-material";
  name?: string;
  alphaMode: "OPAQUE" | "MASK" | "BLEND";
  alphaCutoff: number;
  doubleSided: boolean;
  metallicFactor: number;
  roughnessFactor: number;
  emissiveFactor: [number, number, number];
  baseColorTexture?: {
    texture: number;
    texCoord: number;
    orientation: string;
    sourceColorSpace: "srgb" | "linear";
  };
}

export interface CompressedTextureLevel {
  width: number;
  height: number;
  data: Uint8Array;
}

export interface DecodedCompressedTexture {
  kind: "compressed";
  format: CompressedTextureFormat;
  width: number;
  height: number;
  colorSpace: "srgb";
  levels: CompressedTextureLevel[];
  sampler: SerializableSampler;
  capabilityKey: string;
}

export interface DecodedRgbaTexture {
  kind: "rgba";
  rgba: Uint8Array;
  width: number;
  height: number;
  colorSpace: "srgb";
  sampler: SerializableSampler;
}

export type DecodedTexture = DecodedCompressedTexture | DecodedRgbaTexture;

export interface DecodedMaterial {
  baseColorFactor: [number, number, number, number];
  baseColorTexture?: DecodedTexture;
  raw: SerializableMaterial;
}

export interface DecodedPrimitive {
  positions: Float32Array;
  normals?: Float32Array;
  uvs?: Float32Array;
  indices?: Uint16Array | Uint32Array;
  material: DecodedMaterial;
}

export interface DecodedTileContent {
  primitives: DecodedPrimitive[];
  /** Float64 scene-local origin retained as JavaScript numbers. */
  origin: [number, number, number];
  byteEstimate: {
    geometry: number;
    textures: number;
  };
  /** Worker-side aggregate timings retained for public operational diagnostics. */
  diagnostics?: {
    totalDecodeMs: number;
    /** One-time worker-local Basis wrapper/WASM initialization time. */
    basisRuntimeInitializationMs: number;
    /** Sum of per-texture KTX2 transcodes, excluding runtime initialization. */
    basisTranscodeMs: number;
    basisTranscodeSamplesMs: number[];
    basisTextures: number;
    basisTarget: DecodeTextureTarget | null;
  };
}

export interface BasisTargetTimingStats {
  readonly count: number;
  readonly totalMs: number;
  /** Bounded tail of individual texture transcode durations. */
  readonly samplesMs: readonly number[];
}

export interface DecodeTileRequest {
  content: ArrayBuffer;
  contentUrl: string;
  /** Root below which external glTF dependencies may resolve. */
  dependencyRootUrl?: string;
  revision: string;
  accumulatedTransform: Mat4;
  tilesetToScene: Mat4;
  textureCapabilities: TextureCapabilities;
  wasm?: DecodeWasmUrls;
}

export interface DecodeCacheIdentity {
  contentUrl: string;
  revision: string;
  capabilityKey: string;
}

export interface DecodeJob {
  promise: Promise<DecodedTileContent>;
  cancel(): void;
}

/** Minimal pool surface supplied to renderer-neutral members. */
export interface DecodeWorkerPoolHandle {
  readonly size: number;
  decode(request: DecodeTileRequest): DecodeJob;
  stats?(): {
    readonly size: number;
    readonly queuedJobs: number;
    readonly activeJobs: number;
    readonly completedJobs: number;
    readonly failedJobs: number;
    readonly cancelledJobs: number;
    readonly workerElapsedMs: number;
    readonly decodedGeometryBytes: number;
    readonly decodedTextureBytes: number;
    readonly basisRuntimeInitializationMs: number;
    readonly basisTranscodeMs: number;
    readonly basisTextures: number;
    readonly basisTargets: Readonly<Record<string, number>>;
    readonly basisTargetTimings: Readonly<
      Record<string, BasisTargetTimingStats>
    >;
  };
  invalidate?(): void;
  dispose?(): void;
}
