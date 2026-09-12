import { parse } from "@loaders.gl/core";
import { GLTFLoader } from "@loaders.gl/gltf";
import { BasisLoader } from "@loaders.gl/textures";
import { loadLibrary } from "@loaders.gl/worker-utils";
import { read as readKtx2 } from "ktx-parse";

import { IDENTITY } from "../../camera";
import {
  composeSceneTransform,
  flattenPrimitiveToRtc,
  multiplyMat4,
  Y_UP_TO_Z_UP,
  type Mat4,
  type RtcPrimitiveResult,
} from "../rtc";
import { capabilityTarget, loadersBasisFormat } from "./capabilities";
import { repairMeshoptFallbackOffsets } from "./meshoptFallback";
import { collectContentBuffers } from "./transfer";
import type {
  CompressedTextureFormat,
  DecodeTextureTarget,
  DecodeTileRequest,
  DecodedMaterial,
  DecodedPrimitive,
  DecodedTexture,
  DecodedTileContent,
  DecodeWasmUrls,
  SerializableMaterial,
  SerializableSampler,
} from "./types";
import { TileDecodeError, TileUnsupportedExtensionError } from "./types";

type BasisMetadata = {
  width: number;
  height: number;
  levels: readonly { byteLength: number }[];
  orientation: string;
  srgb: boolean;
};

type BasisLevel = {
  width: number;
  height: number;
  data: Uint8Array;
  compressed: boolean;
};

type DecodeDiagnosticsAccumulator = {
  basisRuntimeInitializationMs: number;
  basisTranscodeMs: number;
  basisTranscodeSamplesMs: number[];
  basisTextures: number;
  basisTarget: DecodeTextureTarget | null;
};

const monotonicNow = (): number =>
  globalThis.performance?.now?.() ?? Date.now();

type BasisRuntimeProviderResult = {
  basisEncoder: unknown;
  initializedNow: boolean;
};

export type DecodeTileOptions = {
  modules?: Record<string, unknown>;
  fetchDependency?: (url: string) => Promise<ArrayBuffer>;
  decodeRasterImage?: (
    data: Uint8Array,
    mimeType: string,
  ) => Promise<{ rgba: Uint8Array; width: number; height: number }>;
  transcodeBasis?: (
    data: Uint8Array,
    loadersFormat: "astc-4x4" | "bc7-m5" | "etc2" | "bc3" | "rgba32",
    metadata: BasisMetadata,
    modules: Record<string, unknown>,
  ) => Promise<BasisLevel[]>;
  /** Test/integration seam; production initializes loaders.gl's Basis runtime. */
  basisRuntimeProvider?: (
    modules: Record<string, unknown>,
  ) => Promise<BasisRuntimeProviderResult>;
  now?: () => number;
};

type ExpandedAccessor = {
  value: ArrayBufferView;
  count?: number;
  type?: string;
  componentType?: number;
};

type GltfAccessor = {
  bufferView?: number;
  byteOffset?: number;
  componentType: number;
  count: number;
  type: string;
  normalized?: boolean;
  sparse?: unknown;
};

type GltfBufferView = {
  buffer: number;
  byteOffset?: number;
  byteLength: number;
  byteStride?: number;
};

type GltfBuffer = {
  arrayBuffer: ArrayBuffer;
  byteOffset: number;
  byteLength: number;
};

type GltfTextureInfo = {
  index: number;
  texCoord?: number;
};

type GltfMaterial = {
  name?: string;
  pbrMetallicRoughness?: {
    baseColorFactor?: number[];
    baseColorTexture?: GltfTextureInfo;
    metallicFactor?: number;
    roughnessFactor?: number;
  };
  emissiveFactor?: number[];
  alphaMode?: string;
  alphaCutoff?: number;
  doubleSided?: boolean;
  extensions?: {
    KHR_materials_unlit?: Record<string, never>;
  };
};

/**
 * Required extensions this decoder can actually honour.
 *
 * The gate exists so an asset that needs something unimplemented fails by
 * name instead of drawing something wrong. `EXT_meshopt_compression` is on the
 * list because loaders.gl decompresses it and this decoder repairs the one
 * authoring mistake that decoding in place turns into silent corruption; see
 * `meshoptFallback.ts`.
 */
const REQUIRED_EXTENSION_ALLOWLIST = new Set([
  "KHR_draco_mesh_compression",
  "KHR_texture_basisu",
  "KHR_materials_unlit",
  "KHR_mesh_quantization",
  "EXT_meshopt_compression",
]);

const validateRequiredExtensions = (json: GltfJson, tileUri: string): void => {
  if (json.extensionsRequired === undefined) return;
  if (
    !Array.isArray(json.extensionsRequired) ||
    json.extensionsRequired.some(
      (extension) => typeof extension !== "string" || extension.length === 0,
    )
  ) {
    throw new TileDecodeError(
      tileUri,
      "profile",
      "extensionsRequired must contain non-empty strings",
    );
  }
  for (const extension of json.extensionsRequired) {
    if (!REQUIRED_EXTENSION_ALLOWLIST.has(extension)) {
      throw new TileUnsupportedExtensionError(tileUri, extension);
    }
  }
};

type GltfPrimitive = {
  attributes: Record<string, number | ExpandedAccessor>;
  indices?: number | ExpandedAccessor;
  material?: number;
  mode?: number;
};

type GltfMesh = {
  primitives: GltfPrimitive[];
};

type GltfNode = {
  mesh?: number;
  children?: number[];
  matrix?: number[];
  translation?: number[];
  rotation?: number[];
  scale?: number[];
};

type GltfScene = {
  nodes?: number[];
};

type GltfImage = {
  uri?: string;
  bufferView?: number;
  mimeType?: string;
};

type GltfTexture = {
  source?: number;
  sampler?: number;
};

type GltfSampler = {
  magFilter?: number;
  minFilter?: number;
  wrapS?: number;
  wrapT?: number;
};

type GltfJson = {
  scene?: number;
  scenes?: GltfScene[];
  nodes?: GltfNode[];
  meshes?: GltfMesh[];
  accessors?: GltfAccessor[];
  bufferViews?: GltfBufferView[];
  images?: GltfImage[];
  textures?: GltfTexture[];
  samplers?: GltfSampler[];
  materials?: GltfMaterial[];
  extensionsUsed?: string[];
  extensionsRequired?: string[];
};

const GLTF_MAG_FILTERS = new Set([9728, 9729]);
const GLTF_MIN_FILTERS = new Set([9728, 9729, 9984, 9985, 9986, 9987]);
const GLTF_WRAP_MODES = new Set([33071, 33648, 10497]);

const samplerValue = <T extends number>(
  candidate: number | undefined,
  fallback: T,
  allowed: ReadonlySet<number>,
  label: string,
): T => {
  const resolved = candidate ?? fallback;
  if (!allowed.has(resolved)) {
    throw new Error(`glTF sampler has an invalid ${label}: ${resolved}`);
  }
  return resolved as T;
};

type ParsedGltf = {
  json: GltfJson;
  buffers: GltfBuffer[];
};

type TextureResult = {
  texture: DecodedTexture;
  orientation: string;
  sourceColorSpace: "srgb" | "linear";
};

type PendingPrimitive = {
  rtc: RtcPrimitiveResult;
  materialIndex?: number;
};

const GLB_MAGIC = 0x46546c67;
const GLB_JSON_CHUNK = 0x4e4f534a;
const COMPONENTS: Readonly<Record<string, number>> = {
  SCALAR: 1,
  VEC2: 2,
  VEC3: 3,
  VEC4: 4,
  MAT2: 4,
  MAT3: 9,
  MAT4: 16,
};
const COMPONENT_BYTES: Readonly<Record<number, number>> = {
  5120: 1,
  5121: 1,
  5122: 2,
  5123: 2,
  5125: 4,
  5126: 4,
};

/**
 * Fetch bytes, refusing to treat an HTTP error body as content.
 *
 * Without the status check a 410 for a retired revision hands ~30 bytes of
 * `text/plain` to the glTF/KTX2 parser, which reports a buffer-overrun or
 * corrupt-container error — sending debugging at the exporter for what was a
 * routine retirement.
 */
const fetchOkBytes = async (url: string): Promise<ArrayBuffer> => {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `dependency fetch failed: ${response.status} ${response.statusText} (${url})`,
    );
  }
  return response.arrayBuffer();
};

const requireIndex = <T>(
  array: readonly T[] | undefined,
  index: number,
  label: string,
): T => {
  const value = array?.[index];
  if (!value) throw new Error(`${label} ${index} is missing`);
  return value;
};

const copyArrayBuffer = (data: Uint8Array): ArrayBuffer => {
  const copy = new Uint8Array(data.byteLength);
  copy.set(data);
  return copy.buffer;
};

const contentJson = (content: ArrayBuffer): GltfJson => {
  if (content.byteLength < 4) throw new Error("glTF content is truncated");
  const view = new DataView(content);
  let jsonBytes: Uint8Array;
  if (view.getUint32(0, true) === GLB_MAGIC) {
    if (
      content.byteLength < 20 ||
      view.getUint32(4, true) !== 2 ||
      view.getUint32(12, true) + 20 > content.byteLength ||
      view.getUint32(16, true) !== GLB_JSON_CHUNK
    ) {
      throw new Error("GLB header or JSON chunk is invalid");
    }
    jsonBytes = new Uint8Array(content, 20, view.getUint32(12, true));
  } else {
    jsonBytes = new Uint8Array(content);
  }
  let text = new TextDecoder("utf-8", { fatal: true }).decode(jsonBytes);
  while (text.endsWith("\0") || text.endsWith(" ")) text = text.slice(0, -1);
  const parsed: unknown = JSON.parse(text);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("glTF JSON root must be an object");
  }
  return parsed as GltfJson;
};

const hasExtension = (json: GltfJson, extension: string): boolean =>
  [...(json.extensionsUsed ?? []), ...(json.extensionsRequired ?? [])].includes(
    extension,
  );

const validUrl = (value: string, label: string): string => {
  if (
    !value ||
    [...value].some((character) => character.charCodeAt(0) < 0x20)
  ) {
    throw new Error(`${label} must be a non-empty URL`);
  }
  try {
    return new URL(value).href;
  } catch {
    throw new Error(`${label} must be an absolute URL`);
  }
};

type DataUriDescriptor = {
  mediaType?: string;
  payload: string;
  base64: boolean;
};

const dataUriDescriptor = (value: string): DataUriDescriptor | null => {
  if (!/^data:/iu.test(value)) return null;
  if ([...value].some((character) => character.charCodeAt(0) < 0x20)) {
    throw new Error("embedded glTF data URI contains a control character");
  }
  const comma = value.indexOf(",");
  if (comma < 5) throw new Error("embedded glTF data URI has no payload");
  const metadata = value.slice(5, comma);
  const fields = metadata.split(";");
  const mediaType = fields[0] || undefined;
  if (
    mediaType !== undefined &&
    !/^[!#$&^_.+\-\w]+\/[!#$&^_.+\-\w]+$/u.test(mediaType)
  ) {
    throw new Error("embedded glTF data URI has an invalid media type");
  }
  const parameters = fields.slice(1);
  const base64 = parameters.at(-1)?.toLowerCase() === "base64";
  const regularParameters = base64 ? parameters.slice(0, -1) : parameters;
  if (
    regularParameters.some(
      (parameter) =>
        !/^[!#$&^_.+\-\w]+=[^;\s]*$/u.test(parameter) ||
        parameter.toLowerCase() === "base64",
    )
  ) {
    throw new Error("embedded glTF data URI has invalid parameters");
  }
  const payload = value.slice(comma + 1);
  if (base64) {
    if (
      payload.length % 4 !== 0 ||
      !/^(?:[A-Za-z\d+/]{4})*(?:[A-Za-z\d+/]{2}==|[A-Za-z\d+/]{3}=)?$/u.test(
        payload,
      )
    ) {
      throw new Error("embedded glTF data URI has invalid base64 data");
    }
  } else if (
    /%(?![\dA-Fa-f]{2})/u.test(payload) ||
    /[^\x20-\x7e]/u.test(payload) ||
    /[#?]/u.test(payload)
  ) {
    throw new Error("embedded glTF data URI has invalid percent-encoded data");
  }
  return {
    ...(mediaType ? { mediaType: mediaType.toLowerCase() } : {}),
    payload,
    base64,
  };
};

const decodeDataUri = (
  value: string,
): { bytes: Uint8Array; mediaType?: string } | null => {
  const descriptor = dataUriDescriptor(value);
  if (!descriptor) return null;
  let bytes: Uint8Array;
  if (descriptor.base64) {
    const decoded = atob(descriptor.payload);
    bytes = Uint8Array.from(decoded, (character) => character.charCodeAt(0));
  } else {
    const output: number[] = [];
    for (let index = 0; index < descriptor.payload.length; index += 1) {
      const character = descriptor.payload[index]!;
      if (character === "%") {
        output.push(
          Number.parseInt(descriptor.payload.slice(index + 1, index + 3), 16),
        );
        index += 2;
      } else {
        output.push(character.charCodeAt(0));
      }
    }
    bytes = Uint8Array.from(output);
  }
  return {
    bytes,
    ...(descriptor.mediaType ? { mediaType: descriptor.mediaType } : {}),
  };
};

const wasmModules = (
  wasm: DecodeWasmUrls | undefined,
): Record<string, unknown> => {
  const modules: Record<string, unknown> = {};
  if (wasm?.draco) {
    modules["draco_wasm_wrapper.js"] = validUrl(
      wasm.draco.wrapperUrl,
      "Draco wrapper URL",
    );
    modules["draco_decoder.wasm"] = validUrl(
      wasm.draco.wasmUrl,
      "Draco WASM URL",
    );
  }
  if (wasm?.basis) {
    modules["basis_encoder.js"] = validUrl(
      wasm.basis.encoderUrl,
      "Basis encoder URL",
    );
    modules["basis_encoder.wasm"] = validUrl(
      wasm.basis.wasmUrl,
      "Basis encoder WASM URL",
    );
  }
  return modules;
};

const dependencyUrl = (request: DecodeTileRequest, value: string): string => {
  if (!request.dependencyRootUrl || !value) {
    throw new Error(
      `external glTF dependency URI is not same-root relative: ${value}`,
    );
  }
  const forms = [value];
  for (;;) {
    const current = forms.at(-1)!;
    if (/%(?![\dA-Fa-f]{2})/u.test(current)) {
      throw new Error(
        `external glTF dependency URI contains a malformed percent escape: ${value}`,
      );
    }
    const decoded = decodeURIComponent(current);
    if (decoded.includes("\0")) {
      throw new Error(
        `external glTF dependency URI contains a NUL byte: ${value}`,
      );
    }
    if (decoded === current) break;
    forms.push(decoded);
  }
  for (const form of forms) {
    let parsed: URL;
    try {
      parsed = new URL(form, "https://relative.invalid/");
    } catch {
      throw new Error(
        `external glTF dependency URI is not same-root relative: ${value}`,
      );
    }
    if (
      /^[a-z][a-z\d+.-]*:/iu.test(form) ||
      form.startsWith("//") ||
      form.startsWith("/") ||
      form.includes("\\") ||
      parsed.search ||
      parsed.hash
    ) {
      throw new Error(
        `external glTF dependency URI is not same-root relative: ${value}`,
      );
    }
  }
  if (forms.some((form) => /%(?:2f|5c)/iu.test(form))) {
    throw new Error(
      `external glTF dependency URI contains an encoded path separator: ${value}`,
    );
  }
  const canonical = forms.at(-1)!;
  const parts = canonical.split("/");
  if (parts.some((part) => part === "..")) {
    throw new Error(
      `external glTF dependency URI contains invalid path segments: ${value}`,
    );
  }
  const root = new URL(request.dependencyRootUrl);
  const base = new URL(request.contentUrl);
  const normalized = parts
    .filter((part) => part !== "" && part !== ".")
    .join("/");
  if (!normalized) {
    throw new Error(
      `external glTF dependency URI contains invalid path segments: ${value}`,
    );
  }
  const resolved = new URL(normalized, base);
  const rootPath = root.pathname.endsWith("/")
    ? root.pathname
    : `${root.pathname}/`;
  if (
    resolved.origin !== root.origin ||
    !resolved.pathname.startsWith(rootPath)
  ) {
    throw new Error(
      `external glTF dependency escapes its registered root: ${value}`,
    );
  }
  return resolved.href;
};

const dependencyRequestForms = (
  contentUrl: string,
  value: string,
): readonly string[] => {
  const urls: string[] = [];
  let current = value;
  for (;;) {
    urls.push(new URL(current, contentUrl).href);
    const decoded = decodeURIComponent(current);
    if (decoded === current) return urls;
    current = decoded;
  }
};

const dependencyAllowlist = (
  request: DecodeTileRequest,
  json: GltfJson,
): {
  external: ReadonlyMap<string, string>;
  embedded: ReadonlyMap<string, string>;
} => {
  const external = new Map<string, string>();
  const embedded = new Map<string, string>();
  const candidates: string[] = [];
  const buffers = (json as GltfJson & { buffers?: { uri?: string }[] }).buffers;
  for (const buffer of buffers ?? []) {
    if (buffer.uri) candidates.push(buffer.uri);
  }
  for (const image of json.images ?? []) {
    if (image.uri) candidates.push(image.uri);
  }
  for (const candidate of candidates) {
    if (dataUriDescriptor(candidate)) {
      embedded.set(candidate, candidate);
      embedded.set(new URL(candidate).href, candidate);
      // loaders.gl 4 recognizes only a lower-case `data:` prefix before its
      // URL resolver. URI schemes are case-insensitive, so retain the locally
      // authorized bytes when it resolves an upper-case spelling against the
      // content directory instead.
      embedded.set(
        `${new URL(".", request.contentUrl).href}${candidate}`,
        candidate,
      );
      continue;
    }
    const canonical = dependencyUrl(request, candidate);
    for (const requestUrl of dependencyRequestForms(
      request.contentUrl,
      candidate,
    )) {
      external.set(requestUrl, canonical);
    }
    external.set(canonical, canonical);
  }
  return { external, embedded };
};

const defaultRasterDecoder = async (
  data: Uint8Array,
  mimeType: string,
): Promise<{ rgba: Uint8Array; width: number; height: number }> => {
  if (
    typeof createImageBitmap !== "function" ||
    typeof OffscreenCanvas !== "function"
  ) {
    throw new Error(
      "raster decode requires createImageBitmap and OffscreenCanvas",
    );
  }
  const bitmap = await createImageBitmap(
    new Blob([copyArrayBuffer(data)], { type: mimeType }),
  );
  try {
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("2D OffscreenCanvas context is unavailable");
    context.drawImage(bitmap, 0, 0);
    const image = context.getImageData(0, 0, bitmap.width, bitmap.height);
    return {
      rgba: Uint8Array.from(image.data),
      width: bitmap.width,
      height: bitmap.height,
    };
  } finally {
    bitmap.close();
  }
};

const defaultBasisTranscoder = async (
  data: Uint8Array,
  loadersFormat: "astc-4x4" | "bc7-m5" | "etc2" | "bc3" | "rgba32",
  _metadata: BasisMetadata,
  modules: Record<string, unknown>,
): Promise<BasisLevel[]> => {
  const result = (await parse(copyArrayBuffer(data), BasisLoader, {
    core: {
      worker: false,
      CDN: null,
      useLocalLibraries: true,
      mimeType: "image/ktx2",
    },
    modules,
    basis: {
      format: loadersFormat,
      containerFormat: "auto",
      module: "encoder",
    },
  })) as unknown;
  if (!Array.isArray(result) || !Array.isArray(result[0])) {
    throw new Error("Basis loader returned an invalid mip hierarchy");
  }
  return result[0].map((value: unknown, level: number) => {
    if (!value || typeof value !== "object") {
      throw new Error(`Basis mip ${level} is invalid`);
    }
    const candidate = value as {
      width?: unknown;
      height?: unknown;
      data?: unknown;
      compressed?: unknown;
    };
    if (
      typeof candidate.width !== "number" ||
      typeof candidate.height !== "number" ||
      !(candidate.data instanceof Uint8Array) ||
      typeof candidate.compressed !== "boolean"
    ) {
      throw new Error(`Basis mip ${level} has an invalid shape`);
    }
    return {
      width: candidate.width,
      height: candidate.height,
      data: candidate.data.slice(),
      compressed: candidate.compressed,
    };
  });
};

let basisRuntimePromise: Promise<unknown> | null = null;

const initializeBasisRuntime = async (
  modules: Record<string, unknown>,
): Promise<unknown> => {
  if (modules.basisEncoder) return modules.basisEncoder;
  const wrapperUrl = modules["basis_encoder.js"];
  const wasmUrl = modules["basis_encoder.wasm"];
  if (typeof wrapperUrl !== "string" || typeof wasmUrl !== "string") {
    throw new Error("Basis runtime requires injected encoder and WASM URLs");
  }
  const loadOptions = {
    useLocalLibraries: false,
    CDN: null,
    modules,
  };
  const [loadedFactory, wasmBinary] = await Promise.all([
    loadLibrary(wrapperUrl, "textures", loadOptions),
    loadLibrary(wasmUrl, "textures", loadOptions),
  ]);
  const globalFactory = (globalThis as { BASIS?: unknown }).BASIS;
  const factory = loadedFactory ?? globalFactory;
  if (typeof factory !== "function") {
    throw new Error("Basis encoder wrapper did not install a module factory");
  }
  const runtime = (await factory({ wasmBinary })) as {
    BasisFile?: unknown;
    KTX2File?: unknown;
    BasisEncoder?: unknown;
    initializeBasis?: () => void;
  };
  if (!runtime || typeof runtime.KTX2File !== "function") {
    throw new Error("Basis encoder runtime is missing KTX2File");
  }
  runtime.initializeBasis?.();
  return {
    BasisFile: runtime.BasisFile,
    KTX2File: runtime.KTX2File,
    BasisEncoder: runtime.BasisEncoder,
  };
};

const defaultBasisRuntimeProvider = async (
  modules: Record<string, unknown>,
): Promise<BasisRuntimeProviderResult> => {
  if (modules.basisEncoder) {
    return { basisEncoder: modules.basisEncoder, initializedNow: false };
  }
  const initializedNow = basisRuntimePromise === null;
  if (initializedNow) basisRuntimePromise = initializeBasisRuntime(modules);
  try {
    return {
      basisEncoder: await basisRuntimePromise!,
      initializedNow,
    };
  } catch (error) {
    if (initializedNow) basisRuntimePromise = null;
    throw error;
  }
};

const componentValue = (
  data: DataView,
  offset: number,
  componentType: number,
): number => {
  switch (componentType) {
    case 5120:
      return data.getInt8(offset);
    case 5121:
      return data.getUint8(offset);
    case 5122:
      return data.getInt16(offset, true);
    case 5123:
      return data.getUint16(offset, true);
    case 5125:
      return data.getUint32(offset, true);
    case 5126:
      return data.getFloat32(offset, true);
    default:
      throw new Error(`glTF component type ${componentType} is unsupported`);
  }
};

const normalizedComponent = (value: number, componentType: number): number => {
  switch (componentType) {
    case 5120:
      return Math.max(value / 127, -1);
    case 5121:
      return value / 255;
    case 5122:
      return Math.max(value / 32_767, -1);
    case 5123:
      return value / 65_535;
    case 5125:
      return value / 4_294_967_295;
    default:
      return value;
  }
};

const accessorFloats = (
  reference: number | ExpandedAccessor | undefined,
  expectedType: "VEC2" | "VEC3",
  gltf: ParsedGltf,
): Float32Array | undefined => {
  if (reference === undefined) return undefined;
  if (typeof reference !== "number") {
    if (!ArrayBuffer.isView(reference.value)) {
      throw new Error("expanded glTF accessor has no typed value");
    }
    return Float32Array.from(reference.value as unknown as ArrayLike<number>);
  }
  const accessor = requireIndex(gltf.json.accessors, reference, "accessor");
  if (accessor.type !== expectedType || accessor.sparse) {
    throw new Error(
      `glTF accessor ${reference} must be a non-sparse ${expectedType}`,
    );
  }
  const components = COMPONENTS[accessor.type];
  const componentBytes = COMPONENT_BYTES[accessor.componentType];
  if (!components || !componentBytes) {
    throw new Error(`glTF accessor ${reference} has an unsupported type`);
  }
  const bufferView = requireIndex(
    gltf.json.bufferViews,
    accessor.bufferView ?? -1,
    "bufferView",
  );
  const buffer = requireIndex(gltf.buffers, bufferView.buffer, "buffer");
  const stride = bufferView.byteStride ?? componentBytes * components;
  if (stride < componentBytes * components) {
    throw new Error(`glTF accessor ${reference} has an invalid byte stride`);
  }
  const start =
    buffer.byteOffset +
    (bufferView.byteOffset ?? 0) +
    (accessor.byteOffset ?? 0);
  const end =
    start +
    Math.max(0, accessor.count - 1) * stride +
    componentBytes * components;
  if (start < 0 || end > buffer.arrayBuffer.byteLength) {
    throw new Error(`glTF accessor ${reference} exceeds its buffer`);
  }
  const data = new DataView(buffer.arrayBuffer);
  const output = new Float32Array(accessor.count * components);
  for (let element = 0; element < accessor.count; element += 1) {
    for (let component = 0; component < components; component += 1) {
      const value = componentValue(
        data,
        start + element * stride + component * componentBytes,
        accessor.componentType,
      );
      output[element * components + component] = accessor.normalized
        ? normalizedComponent(value, accessor.componentType)
        : value;
    }
  }
  return output;
};

const accessorIndices = (
  reference: number | ExpandedAccessor | undefined,
  gltf: ParsedGltf,
): Uint16Array | Uint32Array | undefined => {
  if (reference === undefined) return undefined;
  if (typeof reference !== "number") {
    const source = reference.value;
    if (source instanceof Uint32Array) return source.slice();
    if (
      source instanceof Uint16Array ||
      source instanceof Uint8Array ||
      source instanceof Int8Array ||
      source instanceof Int16Array ||
      source instanceof Int32Array
    ) {
      const values = Array.from(source);
      if (values.some((value) => value < 0)) {
        throw new Error("glTF indices must be unsigned");
      }
      const max = Math.max(0, ...values);
      return max <= 65_535
        ? Uint16Array.from(values)
        : Uint32Array.from(values);
    }
    throw new Error("expanded glTF indices have an invalid typed value");
  }
  const accessor = requireIndex(gltf.json.accessors, reference, "accessor");
  if (
    accessor.type !== "SCALAR" ||
    accessor.sparse ||
    ![5121, 5123, 5125].includes(accessor.componentType)
  ) {
    throw new Error(`glTF index accessor ${reference} is unsupported`);
  }
  const bufferView = requireIndex(
    gltf.json.bufferViews,
    accessor.bufferView ?? -1,
    "bufferView",
  );
  const buffer = requireIndex(gltf.buffers, bufferView.buffer, "buffer");
  const bytes = COMPONENT_BYTES[accessor.componentType]!;
  const stride = bufferView.byteStride ?? bytes;
  const start =
    buffer.byteOffset +
    (bufferView.byteOffset ?? 0) +
    (accessor.byteOffset ?? 0);
  const end = start + Math.max(0, accessor.count - 1) * stride + bytes;
  if (stride < bytes || start < 0 || end > buffer.arrayBuffer.byteLength) {
    throw new Error(`glTF index accessor ${reference} exceeds its buffer`);
  }
  const data = new DataView(buffer.arrayBuffer);
  const output =
    accessor.componentType === 5125
      ? new Uint32Array(accessor.count)
      : new Uint16Array(accessor.count);
  for (let index = 0; index < accessor.count; index += 1) {
    output[index] = componentValue(
      data,
      start + index * stride,
      accessor.componentType,
    );
  }
  return output;
};

/** Preserve strip winding; repeated indices join strips without drawing faces. */
const triangulateStrip = (
  indices: Uint16Array | Uint32Array | undefined,
  vertexCount: number,
): Uint16Array | Uint32Array => {
  const count = indices?.length ?? vertexCount;
  const at = (index: number): number => indices?.[index] ?? index;
  let triangles = 0;
  for (let i = 2; i < count; i += 1) {
    const a = at(i - 2),
      b = at(i - 1),
      c = at(i);
    if (a !== b && b !== c && a !== c) triangles += 1;
  }
  const result =
    vertexCount <= 65_536
      ? new Uint16Array(triangles * 3)
      : new Uint32Array(triangles * 3);
  let offset = 0;
  for (let i = 2; i < count; i += 1) {
    const a = at(i - 2),
      b = at(i - 1),
      c = at(i);
    if (a === b || b === c || a === c) continue;
    result[offset++] = i % 2 === 0 ? a : b;
    result[offset++] = i % 2 === 0 ? b : a;
    result[offset++] = c;
  }
  return result;
};

const nodeMatrix = (node: GltfNode): Mat4 => {
  if (node.matrix) {
    if (
      node.matrix.length !== 16 ||
      node.matrix.some((value) => !Number.isFinite(value))
    ) {
      throw new Error("glTF node matrix must contain 16 finite numbers");
    }
    return [...node.matrix] as Mat4;
  }
  const translation = node.translation ?? [0, 0, 0];
  const scale = node.scale ?? [1, 1, 1];
  const rotation = node.rotation ?? [0, 0, 0, 1];
  if (
    translation.length !== 3 ||
    scale.length !== 3 ||
    rotation.length !== 4 ||
    [...translation, ...scale, ...rotation].some(
      (value) => !Number.isFinite(value),
    )
  ) {
    throw new Error("glTF node TRS values must be finite and correctly sized");
  }
  const length = Math.hypot(...rotation);
  if (length <= 1e-12) throw new Error("glTF node quaternion is degenerate");
  const [x, y, z, w] = rotation.map((value) => value / length) as [
    number,
    number,
    number,
    number,
  ];
  const sx = scale[0]!;
  const sy = scale[1]!;
  const sz = scale[2]!;
  return [
    (1 - 2 * (y * y + z * z)) * sx,
    2 * (x * y + z * w) * sx,
    2 * (x * z - y * w) * sx,
    0,
    2 * (x * y - z * w) * sy,
    (1 - 2 * (x * x + z * z)) * sy,
    2 * (y * z + x * w) * sy,
    0,
    2 * (x * z + y * w) * sz,
    2 * (y * z - x * w) * sz,
    (1 - 2 * (x * x + y * y)) * sz,
    0,
    translation[0]!,
    translation[1]!,
    translation[2]!,
    1,
  ];
};

const applySceneRtc = (
  gltf: ParsedGltf,
  sceneTransform: Mat4,
): PendingPrimitive[] => {
  const scene = requireIndex(
    gltf.json.scenes,
    gltf.json.scene ?? 0,
    "active scene",
  );
  const pending: PendingPrimitive[] = [];
  const visiting = new Set<number>();
  const visit = (nodeIndex: number, parent: readonly number[]): void => {
    if (visiting.has(nodeIndex))
      throw new Error("glTF node graph contains a cycle");
    const node = requireIndex(gltf.json.nodes, nodeIndex, "node");
    visiting.add(nodeIndex);
    const world = multiplyMat4(parent, nodeMatrix(node));
    if (node.mesh !== undefined) {
      const mesh = requireIndex(gltf.json.meshes, node.mesh, "mesh");
      for (const primitive of mesh.primitives) {
        if (![4, 5].includes(primitive.mode ?? 4)) {
          throw new Error(
            "only glTF TRIANGLES and TRIANGLE_STRIP primitives are supported",
          );
        }
        const positions = accessorFloats(
          primitive.attributes.POSITION,
          "VEC3",
          gltf,
        );
        if (!positions)
          throw new Error("glTF primitive has no POSITION attribute");
        const normals = accessorFloats(
          primitive.attributes.NORMAL,
          "VEC3",
          gltf,
        );
        const material =
          primitive.material === undefined
            ? undefined
            : requireIndex(gltf.json.materials, primitive.material, "material");
        const textureInfo = material?.pbrMetallicRoughness?.baseColorTexture;
        const texCoord = textureInfo?.texCoord ?? 0;
        if (!Number.isInteger(texCoord) || texCoord < 0) {
          throw new Error(
            "glTF baseColorTexture texCoord must be a non-negative integer",
          );
        }
        const uvs = accessorFloats(
          primitive.attributes[`TEXCOORD_${texCoord}`],
          "VEC2",
          gltf,
        );
        if (textureInfo && !uvs) {
          throw new Error(
            `glTF primitive has no TEXCOORD_${texCoord} attribute`,
          );
        }
        let indices = accessorIndices(primitive.indices, gltf);
        const vertexCount = positions.length / 3;
        if (
          indices?.some(
            (index) => !Number.isSafeInteger(index) || index >= vertexCount,
          )
        ) {
          throw new Error(
            `glTF primitive index exceeds its ${vertexCount}-vertex POSITION accessor`,
          );
        }
        if (primitive.mode === 5)
          indices = triangulateStrip(indices, vertexCount);
        pending.push({
          rtc: flattenPrimitiveToRtc(
            {
              positions,
              ...(normals ? { normals } : {}),
              ...(uvs ? { uvs } : {}),
              ...(indices ? { indices } : {}),
            },
            sceneTransform,
            world,
          ),
          ...(primitive.material !== undefined
            ? { materialIndex: primitive.material }
            : {}),
        });
      }
    }
    for (const child of node.children ?? []) visit(child, world);
    visiting.delete(nodeIndex);
  };
  for (const root of scene.nodes ?? []) visit(root, IDENTITY);
  if (pending.length === 0)
    throw new Error("active glTF scene has no primitives");
  return pending;
};

const commonOrigin = (
  primitives: readonly PendingPrimitive[],
): [number, number, number] => {
  const minimum: [number, number, number] = [Infinity, Infinity, Infinity];
  const maximum: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (const { rtc } of primitives) {
    for (let offset = 0; offset < rtc.positions.length; offset += 3) {
      for (let axis = 0; axis < 3; axis += 1) {
        const absolute = rtc.origin[axis]! + rtc.positions[offset + axis]!;
        minimum[axis] = Math.min(minimum[axis]!, absolute);
        maximum[axis] = Math.max(maximum[axis]!, absolute);
      }
    }
  }
  return [
    minimum[0] + (maximum[0] - minimum[0]) / 2,
    minimum[1] + (maximum[1] - minimum[1]) / 2,
    minimum[2] + (maximum[2] - minimum[2]) / 2,
  ];
};

const rebasePositions = (
  rtc: RtcPrimitiveResult,
  origin: readonly [number, number, number],
): Float32Array => {
  const result = new Float32Array(rtc.positions.length);
  for (let offset = 0; offset < result.length; offset += 3) {
    result[offset] = rtc.origin[0] + rtc.positions[offset]! - origin[0];
    result[offset + 1] = rtc.origin[1] + rtc.positions[offset + 1]! - origin[1];
    result[offset + 2] = rtc.origin[2] + rtc.positions[offset + 2]! - origin[2];
  }
  return result;
};

const imageBytes = async (
  gltf: ParsedGltf,
  image: GltfImage,
  request: DecodeTileRequest,
  fetchDependency: (url: string) => Promise<ArrayBuffer>,
): Promise<{ bytes: Uint8Array; mediaType?: string }> => {
  if (image.bufferView !== undefined) {
    const view = requireIndex(
      gltf.json.bufferViews,
      image.bufferView,
      "image bufferView",
    );
    const buffer = requireIndex(gltf.buffers, view.buffer, "image buffer");
    const start = buffer.byteOffset + (view.byteOffset ?? 0);
    if (start < 0 || start + view.byteLength > buffer.arrayBuffer.byteLength) {
      throw new Error("glTF image exceeds its buffer");
    }
    return {
      bytes: new Uint8Array(buffer.arrayBuffer, start, view.byteLength).slice(),
      ...(image.mimeType ? { mediaType: image.mimeType } : {}),
    };
  }
  if (image.uri) {
    const embedded = decodeDataUri(image.uri);
    if (embedded) return embedded;
    const url = dependencyUrl(request, image.uri);
    return {
      bytes: new Uint8Array(await fetchDependency(url)),
      ...(image.mimeType ? { mediaType: image.mimeType } : {}),
    };
  }
  throw new Error("glTF image has neither a bufferView nor a URI");
};

const samplerFor = (
  json: GltfJson,
  samplerIndex: number | undefined,
): SerializableSampler => {
  const sampler =
    samplerIndex === undefined
      ? undefined
      : requireIndex(json.samplers, samplerIndex, "sampler");
  return {
    magFilter: samplerValue(
      sampler?.magFilter,
      9729,
      GLTF_MAG_FILTERS,
      "magFilter",
    ),
    minFilter: samplerValue(
      sampler?.minFilter,
      9987,
      GLTF_MIN_FILTERS,
      "minFilter",
    ),
    wrapS: samplerValue(sampler?.wrapS, 10497, GLTF_WRAP_MODES, "wrapS"),
    wrapT: samplerValue(sampler?.wrapT, 10497, GLTF_WRAP_MODES, "wrapT"),
  };
};

const orientationValue = (value: string | Uint8Array | undefined): string => {
  let orientation =
    value instanceof Uint8Array
      ? new TextDecoder().decode(value)
      : (value ?? "rd");
  while (orientation.endsWith("\0")) orientation = orientation.slice(0, -1);
  if (!/^[rl][du]$/u.test(orientation)) {
    throw new Error(`KTXorientation ${orientation} is unsupported`);
  }
  return orientation;
};

/** What every step of one tile's texture and material decode shares. */
type DecodeContext = {
  readonly gltf: ParsedGltf;
  readonly request: DecodeTileRequest;
  readonly options: DecodeTileOptions;
  readonly modules: Record<string, unknown>;
  readonly diagnostics: DecodeDiagnosticsAccumulator;
  /** One decode per image+sampler pair, shared by every material using it. */
  readonly textureCache: Map<string, Promise<TextureResult>>;
};

const basisMetadata = (bytes: Uint8Array): BasisMetadata => {
  const ktx = readKtx2(bytes);
  const descriptor = ktx.dataFormatDescriptor[0];
  if (!descriptor) throw new Error("KTX2 has no data format descriptor");
  const metadata: BasisMetadata = {
    width: ktx.pixelWidth,
    height: ktx.pixelHeight,
    levels: ktx.levels.map((level) => ({
      byteLength: level.levelData.byteLength,
    })),
    orientation: orientationValue(ktx.keyValue.KTXorientation),
    srgb: descriptor.transferFunction === 2,
  };
  if (
    !Number.isInteger(metadata.width) ||
    !Number.isInteger(metadata.height) ||
    metadata.width <= 0 ||
    metadata.height <= 0 ||
    metadata.levels.length === 0
  ) {
    throw new Error("KTX2 dimensions or mip chain are invalid");
  }
  return metadata;
};

const transcodeBasisLevels = async (
  bytes: Uint8Array,
  metadata: BasisMetadata,
  target: DecodeTextureTarget,
  { options, modules, diagnostics }: DecodeContext,
): Promise<BasisLevel[]> => {
  const transcode = options.transcodeBasis ?? defaultBasisTranscoder;
  const now = options.now ?? monotonicNow;
  const runtimeProvider =
    options.basisRuntimeProvider ??
    (options.transcodeBasis ? null : defaultBasisRuntimeProvider);
  let transcodeModules = modules;
  if (runtimeProvider) {
    const runtimeStarted = now();
    const runtime = await runtimeProvider(modules);
    const runtimeElapsed = Math.max(0, now() - runtimeStarted);
    if (runtime.initializedNow) {
      diagnostics.basisRuntimeInitializationMs += runtimeElapsed;
    }
    transcodeModules = {
      ...modules,
      basisEncoder: runtime.basisEncoder,
    };
  }
  const started = now();
  const levels = await transcode(
    bytes,
    loadersBasisFormat(target),
    metadata,
    transcodeModules,
  );
  const elapsed = Math.max(0, now() - started);
  diagnostics.basisTranscodeMs += elapsed;
  diagnostics.basisTranscodeSamplesMs.push(elapsed);
  diagnostics.basisTextures += 1;
  diagnostics.basisTarget = target;
  if (levels.length !== metadata.levels.length) {
    throw new Error(
      "Basis transcode did not retain the complete source mip chain",
    );
  }
  return levels;
};

const decodeKtx2Texture = async (
  bytes: Uint8Array,
  sampler: SerializableSampler,
  context: DecodeContext,
): Promise<TextureResult> => {
  const metadata = basisMetadata(bytes);
  const target = capabilityTarget(context.request.textureCapabilities);
  const levels = await transcodeBasisLevels(bytes, metadata, target, context);
  const sourceColorSpace = metadata.srgb ? "srgb" : "linear";
  if (target === "rgba") {
    const base = levels[0]!;
    if (
      base.compressed ||
      base.data.byteLength !== metadata.width * metadata.height * 4
    ) {
      throw new Error("Basis RGBA fallback did not return full RGBA32 pixels");
    }
    return {
      texture: {
        kind: "rgba",
        rgba: base.data.slice(),
        width: metadata.width,
        height: metadata.height,
        colorSpace: "srgb",
        sampler,
      },
      orientation: metadata.orientation,
      sourceColorSpace,
    };
  }
  if (levels.some((level) => !level.compressed)) {
    throw new Error("native Basis target returned an uncompressed mip");
  }
  return {
    texture: {
      kind: "compressed",
      format: target as CompressedTextureFormat,
      width: metadata.width,
      height: metadata.height,
      colorSpace: "srgb",
      levels: levels.map((level, index) => ({
        // loaders.gl clamps dimensions to the compressed block. Preserve
        // the logical KTX dimensions used by texture upload and sampling.
        width: Math.max(1, metadata.width >> index),
        height: Math.max(1, metadata.height >> index),
        data: level.data.slice(),
      })),
      sampler,
      capabilityKey: context.request.textureCapabilities.capabilityKey,
    },
    orientation: metadata.orientation,
    sourceColorSpace,
  };
};

const decodeRasterTexture = async (
  bytes: Uint8Array,
  mimeType: string | undefined,
  sampler: SerializableSampler,
  options: DecodeTileOptions,
): Promise<TextureResult> => {
  const decoded = await (options.decodeRasterImage ?? defaultRasterDecoder)(
    bytes,
    mimeType ?? "application/octet-stream",
  );
  if (
    !Number.isInteger(decoded.width) ||
    !Number.isInteger(decoded.height) ||
    decoded.width <= 0 ||
    decoded.height <= 0 ||
    decoded.rgba.byteLength !== decoded.width * decoded.height * 4
  ) {
    throw new Error("raster decoder returned an invalid RGBA image");
  }
  return {
    texture: {
      kind: "rgba",
      rgba: decoded.rgba.slice(),
      width: decoded.width,
      height: decoded.height,
      colorSpace: "srgb",
      sampler,
    },
    orientation: "rd",
    sourceColorSpace: "srgb",
  };
};

const decodeTexture = async (
  textureIndex: number,
  context: DecodeContext,
): Promise<TextureResult> => {
  const { gltf, request, options } = context;
  const texture = requireIndex(gltf.json.textures, textureIndex, "texture");
  const image = requireIndex(gltf.json.images, texture.source ?? -1, "image");
  const { bytes, mediaType } = await imageBytes(
    gltf,
    image,
    request,
    options.fetchDependency ?? fetchOkBytes,
  );
  const sampler = samplerFor(gltf.json, texture.sampler);
  const mimeType = image.mimeType ?? mediaType;
  return mimeType === "image/ktx2"
    ? decodeKtx2Texture(bytes, sampler, context)
    : decodeRasterTexture(bytes, mimeType, sampler, options);
};

export const normalizeTextureCoordinates = (
  uvs: Float32Array | undefined,
  orientation: string,
): void => {
  if (!/^[rl][du]$/u.test(orientation)) {
    throw new Error(`texture orientation ${orientation} is unsupported`);
  }
  if (!uvs || orientation === "rd") return;
  for (let offset = 0; offset < uvs.length; offset += 2) {
    if (orientation[0] === "l") uvs[offset] = 1 - uvs[offset]!;
    if (orientation[1] === "u") uvs[offset + 1] = 1 - uvs[offset + 1]!;
  }
};

const serializableMaterial = (
  material: GltfMaterial | undefined,
): SerializableMaterial => {
  const pbr = material?.pbrMetallicRoughness;
  return {
    version: 1,
    kind: "gltf-material",
    ...(material?.name ? { name: material.name } : {}),
    alphaMode:
      material?.alphaMode === "MASK" || material?.alphaMode === "BLEND"
        ? material.alphaMode
        : "OPAQUE",
    alphaCutoff: material?.alphaCutoff ?? 0.5,
    doubleSided: material?.doubleSided ?? false,
    unlit: material?.extensions?.KHR_materials_unlit !== undefined,
    metallicFactor: pbr?.metallicFactor ?? 1,
    roughnessFactor: pbr?.roughnessFactor ?? 1,
    emissiveFactor: [
      material?.emissiveFactor?.[0] ?? 0,
      material?.emissiveFactor?.[1] ?? 0,
      material?.emissiveFactor?.[2] ?? 0,
    ],
  };
};

const cachedTexture = (
  textureIndex: number,
  context: DecodeContext,
): Promise<TextureResult> => {
  const gltfTexture = requireIndex(
    context.gltf.json.textures,
    textureIndex,
    "texture",
  );
  const key = `${gltfTexture.source ?? -1}:${gltfTexture.sampler ?? -1}`;
  const cached = context.textureCache.get(key);
  if (cached) return cached;
  const decoding = decodeTexture(textureIndex, context);
  context.textureCache.set(key, decoding);
  return decoding;
};

const materialFor = async (
  materialIndex: number | undefined,
  primitive: RtcPrimitiveResult,
  context: DecodeContext,
): Promise<DecodedMaterial> => {
  const material =
    materialIndex === undefined
      ? undefined
      : requireIndex(context.gltf.json.materials, materialIndex, "material");
  const factor = material?.pbrMetallicRoughness?.baseColorFactor ?? [
    1, 1, 1, 1,
  ];
  if (factor.length !== 4 || factor.some((value) => !Number.isFinite(value))) {
    throw new Error("glTF baseColorFactor must contain four finite numbers");
  }
  const baseColorFactor = [...factor] as [number, number, number, number];
  const raw = serializableMaterial(material);
  const textureInfo = material?.pbrMetallicRoughness?.baseColorTexture;
  if (!textureInfo) return { baseColorFactor, raw };
  const decoded = await cachedTexture(textureInfo.index, context);
  raw.baseColorTexture = {
    texture: textureInfo.index,
    texCoord: textureInfo.texCoord ?? 0,
    orientation: decoded.orientation,
    sourceColorSpace: decoded.sourceColorSpace,
  };
  normalizeTextureCoordinates(primitive.uvs, decoded.orientation);
  return { baseColorFactor, baseColorTexture: decoded.texture, raw };
};

const totalByteLength = (buffers: ReadonlySet<ArrayBuffer>): number => {
  let total = 0;
  for (const buffer of buffers) total += buffer.byteLength;
  return total;
};

const actualBytes = (
  primitives: readonly DecodedPrimitive[],
): DecodedTileContent["byteEstimate"] => {
  const { geometry, textures } = collectContentBuffers(primitives);
  return {
    geometry: totalByteLength(geometry),
    textures: totalByteLength(textures),
  };
};

const decodeTileContentInner = async (
  request: DecodeTileRequest,
  options: DecodeTileOptions = {},
): Promise<DecodedTileContent> => {
  const now = options.now ?? monotonicNow;
  const decodeStarted = now();
  const diagnostics: DecodeDiagnosticsAccumulator = {
    basisRuntimeInitializationMs: 0,
    basisTranscodeMs: 0,
    basisTranscodeSamplesMs: [],
    basisTextures: 0,
    basisTarget: null,
  };
  // Repaired before anything reads it, so the JSON validated here and the
  // bytes handed to the parser describe the same buffer layout.
  const content = repairMeshoptFallbackOffsets(request.content);
  const originalJson = contentJson(content);
  validateRequiredExtensions(originalJson, request.contentUrl);
  const allowedDependencies = dependencyAllowlist(request, originalJson);
  const needsDraco = hasExtension(originalJson, "KHR_draco_mesh_compression");
  const needsBasis = hasExtension(originalJson, "KHR_texture_basisu");
  if (needsDraco && !request.wasm?.draco && !options.modules?.draco3d) {
    throw new Error("Draco content requires injected wrapper and WASM URLs");
  }
  if (needsBasis && !request.wasm?.basis && !options.transcodeBasis) {
    throw new Error(
      "KTX2 content requires injected Basis encoder and WASM URLs",
    );
  }

  const modules = { ...wasmModules(request.wasm), ...options.modules };
  const fetchDependency = async (value: string): Promise<Response> => {
    const embeddedCandidate = allowedDependencies.embedded.get(value);
    if (embeddedCandidate) {
      const embedded = decodeDataUri(embeddedCandidate)!;
      return new Response(copyArrayBuffer(embedded.bytes), { status: 200 });
    }
    if (decodeDataUri(value)) {
      throw new Error(
        "glTF parser requested an unauthorized embedded dependency",
      );
    }
    const requestedUrl = new URL(value, request.contentUrl).href;
    const canonicalUrl = allowedDependencies.external.get(requestedUrl);
    if (!canonicalUrl) {
      throw new Error(
        `glTF parser requested an unauthorized dependency: ${value}`,
      );
    }
    const data = await (options.fetchDependency ?? fetchOkBytes)(canonicalUrl);
    return new Response(data, { status: 200 });
  };
  const parsed = (await parse(content.slice(0), GLTFLoader, {
    core: {
      worker: false,
      CDN: null,
      useLocalLibraries: true,
      baseUrl: request.contentUrl,
      fetch: fetchDependency,
    },
    modules,
    // Extract the stored index buffer; the authored glTF primitive mode still
    // determines its topology. Generating new Draco strips changes that buffer.
    draco: { topology: "triangle-list" },
    gltf: {
      normalize: false,
      loadBuffers: true,
      loadImages: false,
      decompressMeshes: true,
    },
  })) as unknown as ParsedGltf;
  // loaders.gl normalizes material objects and currently drops the empty
  // KHR_materials_unlit marker. Material indices remain stable, so restore
  // that renderer-significant authored flag from the validated source JSON.
  for (
    let index = 0;
    index < (originalJson.materials?.length ?? 0);
    index += 1
  ) {
    if (
      originalJson.materials?.[index]?.extensions?.KHR_materials_unlit !==
      undefined
    ) {
      const material = parsed.json.materials?.[index];
      if (material) {
        material.extensions = { KHR_materials_unlit: {} };
      }
    }
  }

  const sceneTransform = multiplyMat4(
    composeSceneTransform(request.tilesetToScene, request.accumulatedTransform),
    Y_UP_TO_Z_UP,
  );
  const pending = applySceneRtc(parsed, sceneTransform);
  const origin = commonOrigin(pending);
  const context: DecodeContext = {
    gltf: parsed,
    request,
    options,
    modules,
    diagnostics,
    textureCache: new Map(),
  };
  const primitives: DecodedPrimitive[] = await Promise.all(
    pending.map(async ({ rtc, materialIndex }) => ({
      positions: rebasePositions(rtc, origin),
      ...(rtc.normals ? { normals: rtc.normals } : {}),
      ...(rtc.uvs ? { uvs: rtc.uvs } : {}),
      ...(rtc.indices ? { indices: rtc.indices } : {}),
      material: await materialFor(materialIndex, rtc, context),
    })),
  );
  return {
    primitives,
    origin,
    byteEstimate: actualBytes(primitives),
    diagnostics: {
      totalDecodeMs: Math.max(0, now() - decodeStarted),
      ...diagnostics,
    },
  };
};

export const decodeTileContent = async (
  request: DecodeTileRequest,
  options: DecodeTileOptions = {},
): Promise<DecodedTileContent> => {
  try {
    return await decodeTileContentInner(request, options);
  } catch (error) {
    if (error instanceof TileDecodeError) throw error;
    const reason = error instanceof Error ? error.message : String(error);
    throw new TileDecodeError(request.contentUrl, "decode", reason, {
      cause: error,
    });
  }
};

// Keep this exported for focused URI-policy tests without exposing a fetch
// implementation object on the worker protocol.
export const resolveGltfDependencyUrl = dependencyUrl;
