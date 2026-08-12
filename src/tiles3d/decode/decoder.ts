import { parse } from "@loaders.gl/core";
import { GLTFLoader } from "@loaders.gl/gltf";
import { BasisLoader } from "@loaders.gl/textures";
import { loadLibrary } from "@loaders.gl/worker-utils";
import { read as readKtx2 } from "ktx-parse";

import {
  composeSceneTransform,
  flattenPrimitiveToRtc,
  multiplyMat4,
  type Mat4,
  type RtcPrimitiveResult,
} from "../rtc";
import { capabilityTarget, loadersBasisFormat } from "./capabilities";
import type {
  CompressedTextureFormat,
  DecodeTileRequest,
  DecodedMaterial,
  DecodedPrimitive,
  DecodedTexture,
  DecodedTileContent,
  DecodeWasmUrls,
  SerializableMaterial,
  SerializableSampler,
} from "./types";

interface BasisMetadata {
  width: number;
  height: number;
  levels: readonly { byteLength: number }[];
  orientation: string;
  srgb: boolean;
}

interface BasisLevel {
  width: number;
  height: number;
  data: Uint8Array;
  compressed: boolean;
}

interface DecodeDiagnosticsAccumulator {
  basisRuntimeInitializationMs: number;
  basisTranscodeMs: number;
  basisTranscodeSamplesMs: number[];
  basisTextures: number;
  basisTarget: import("./types").DecodeTextureTarget | null;
}

const monotonicNow = (): number =>
  globalThis.performance?.now?.() ?? Date.now();

interface BasisRuntimeProviderResult {
  basisEncoder: unknown;
  initializedNow: boolean;
}

export interface DecodeTileOptions {
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
}

interface ExpandedAccessor {
  value: ArrayBufferView;
  count?: number;
  type?: string;
  componentType?: number;
}

interface GltfAccessor {
  bufferView?: number;
  byteOffset?: number;
  componentType: number;
  count: number;
  type: string;
  normalized?: boolean;
  sparse?: unknown;
}

interface GltfBufferView {
  buffer: number;
  byteOffset?: number;
  byteLength: number;
  byteStride?: number;
}

interface GltfBuffer {
  arrayBuffer: ArrayBuffer;
  byteOffset: number;
  byteLength: number;
}

interface GltfTextureInfo {
  index: number;
  texCoord?: number;
}

interface GltfMaterial {
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
}

interface GltfPrimitive {
  attributes: Record<string, number | ExpandedAccessor>;
  indices?: number | ExpandedAccessor;
  material?: number;
  mode?: number;
}

interface GltfMesh {
  primitives: GltfPrimitive[];
}

interface GltfNode {
  mesh?: number;
  children?: number[];
  matrix?: number[];
  translation?: number[];
  rotation?: number[];
  scale?: number[];
}

interface GltfScene {
  nodes?: number[];
}

interface GltfImage {
  uri?: string;
  bufferView?: number;
  mimeType?: string;
}

interface GltfTexture {
  source?: number;
  sampler?: number;
}

interface GltfSampler {
  magFilter?: number;
  minFilter?: number;
  wrapS?: number;
  wrapT?: number;
}

interface GltfJson {
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
}

interface ParsedGltf {
  json: GltfJson;
  buffers: GltfBuffer[];
}

interface TextureResult {
  texture: DecodedTexture;
  orientation: string;
  sourceColorSpace: "srgb" | "linear";
}

interface PendingPrimitive {
  rtc: RtcPrimitiveResult;
  materialIndex?: number;
}

const IDENTITY: Mat4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
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
  if (
    !request.dependencyRootUrl ||
    !value ||
    value.includes("\\") ||
    /%(?:2e|2f|5c)/iu.test(value) ||
    /^[a-z][a-z\d+.-]*:/iu.test(value) ||
    value.startsWith("//") ||
    value.startsWith("/")
  ) {
    throw new Error(
      `external glTF dependency URI is not same-root relative: ${value}`,
    );
  }
  const root = new URL(request.dependencyRootUrl);
  const base = new URL(request.contentUrl);
  const resolved = new URL(value, base);
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

const dependencyAllowlist = (
  request: DecodeTileRequest,
  json: GltfJson,
): ReadonlySet<string> => {
  const urls = new Set<string>();
  const candidates: string[] = [];
  const buffers = (json as GltfJson & { buffers?: { uri?: string }[] }).buffers;
  for (const buffer of buffers ?? []) {
    if (buffer.uri) candidates.push(buffer.uri);
  }
  for (const image of json.images ?? []) {
    if (image.uri) candidates.push(image.uri);
  }
  for (const candidate of candidates) {
    urls.add(dependencyUrl(request, candidate));
  }
  return urls;
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
  const visit = (nodeIndex: number, parent: Mat4): void => {
    if (visiting.has(nodeIndex))
      throw new Error("glTF node graph contains a cycle");
    const node = requireIndex(gltf.json.nodes, nodeIndex, "node");
    visiting.add(nodeIndex);
    const world = multiplyMat4(parent, nodeMatrix(node));
    if (node.mesh !== undefined) {
      const mesh = requireIndex(gltf.json.meshes, node.mesh, "mesh");
      for (const primitive of mesh.primitives) {
        if ((primitive.mode ?? 4) !== 4) {
          throw new Error("only glTF TRIANGLES primitives are supported");
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
        const uvs = accessorFloats(
          primitive.attributes.TEXCOORD_0,
          "VEC2",
          gltf,
        );
        const indices = accessorIndices(primitive.indices, gltf);
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
): Promise<Uint8Array> => {
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
    return new Uint8Array(buffer.arrayBuffer, start, view.byteLength).slice();
  }
  if (image.uri) {
    const url = dependencyUrl(request, image.uri);
    return new Uint8Array(await fetchDependency(url));
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
    magFilter: sampler?.magFilter ?? 9729,
    minFilter: sampler?.minFilter ?? 9987,
    wrapS: sampler?.wrapS ?? 10497,
    wrapT: sampler?.wrapT ?? 10497,
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

const decodeTexture = async (
  textureIndex: number,
  gltf: ParsedGltf,
  request: DecodeTileRequest,
  options: DecodeTileOptions,
  modules: Record<string, unknown>,
  diagnostics: DecodeDiagnosticsAccumulator,
): Promise<TextureResult> => {
  const texture = requireIndex(gltf.json.textures, textureIndex, "texture");
  const image = requireIndex(gltf.json.images, texture.source ?? -1, "image");
  const bytes = await imageBytes(
    gltf,
    image,
    request,
    options.fetchDependency ??
      (async (url) => (await fetch(url)).arrayBuffer()),
  );
  const sampler = samplerFor(gltf.json, texture.sampler);
  if (image.mimeType === "image/ktx2") {
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
    const target = capabilityTarget(request.textureCapabilities);
    const transcode = options.transcodeBasis ?? defaultBasisTranscoder;
    const now = options.now ?? monotonicNow;
    let transcodeModules = modules;
    const runtimeProvider =
      options.basisRuntimeProvider ??
      (options.transcodeBasis ? null : defaultBasisRuntimeProvider);
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
    if (target === "rgba") {
      const base = levels[0]!;
      if (
        base.compressed ||
        base.data.byteLength !== metadata.width * metadata.height * 4
      ) {
        throw new Error(
          "Basis RGBA fallback did not return full RGBA32 pixels",
        );
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
        sourceColorSpace: metadata.srgb ? "srgb" : "linear",
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
        capabilityKey: request.textureCapabilities.capabilityKey,
      },
      orientation: metadata.orientation,
      sourceColorSpace: metadata.srgb ? "srgb" : "linear",
    };
  }

  const decoded = await (options.decodeRasterImage ?? defaultRasterDecoder)(
    bytes,
    image.mimeType ?? "application/octet-stream",
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

const materialFor = async (
  materialIndex: number | undefined,
  primitive: RtcPrimitiveResult,
  gltf: ParsedGltf,
  request: DecodeTileRequest,
  options: DecodeTileOptions,
  modules: Record<string, unknown>,
  textureCache: Map<string, Promise<TextureResult>>,
  diagnostics: DecodeDiagnosticsAccumulator,
): Promise<DecodedMaterial> => {
  const material =
    materialIndex === undefined
      ? undefined
      : requireIndex(gltf.json.materials, materialIndex, "material");
  const pbr = material?.pbrMetallicRoughness;
  const factor = pbr?.baseColorFactor ?? [1, 1, 1, 1];
  if (factor.length !== 4 || factor.some((value) => !Number.isFinite(value))) {
    throw new Error("glTF baseColorFactor must contain four finite numbers");
  }
  const raw: SerializableMaterial = {
    version: 1,
    kind: "gltf-material",
    ...(material?.name ? { name: material.name } : {}),
    alphaMode:
      material?.alphaMode === "MASK" || material?.alphaMode === "BLEND"
        ? material.alphaMode
        : "OPAQUE",
    alphaCutoff: material?.alphaCutoff ?? 0.5,
    doubleSided: material?.doubleSided ?? false,
    metallicFactor: pbr?.metallicFactor ?? 1,
    roughnessFactor: pbr?.roughnessFactor ?? 1,
    emissiveFactor: [
      material?.emissiveFactor?.[0] ?? 0,
      material?.emissiveFactor?.[1] ?? 0,
      material?.emissiveFactor?.[2] ?? 0,
    ],
  };
  const textureInfo = pbr?.baseColorTexture;
  if (!textureInfo) {
    return {
      baseColorFactor: [...factor] as [number, number, number, number],
      raw,
    };
  }
  const gltfTexture = requireIndex(
    gltf.json.textures,
    textureInfo.index,
    "texture",
  );
  const key = `${gltfTexture.source ?? -1}:${gltfTexture.sampler ?? -1}`;
  let decodedPromise = textureCache.get(key);
  if (!decodedPromise) {
    decodedPromise = decodeTexture(
      textureInfo.index,
      gltf,
      request,
      options,
      modules,
      diagnostics,
    );
    textureCache.set(key, decodedPromise);
  }
  const decoded = await decodedPromise;
  raw.baseColorTexture = {
    texture: textureInfo.index,
    texCoord: textureInfo.texCoord ?? 0,
    orientation: decoded.orientation,
    sourceColorSpace: decoded.sourceColorSpace,
  };
  normalizeTextureCoordinates(primitive.uvs, decoded.orientation);
  return {
    baseColorFactor: [...factor] as [number, number, number, number],
    baseColorTexture: decoded.texture,
    raw,
  };
};

const actualBytes = (
  primitives: readonly DecodedPrimitive[],
): DecodedTileContent["byteEstimate"] => {
  const geometry = new Set<ArrayBuffer>();
  const textures = new Set<ArrayBuffer>();
  for (const primitive of primitives) {
    for (const array of [
      primitive.positions,
      primitive.normals,
      primitive.uvs,
      primitive.indices,
    ]) {
      if (array?.buffer instanceof ArrayBuffer) geometry.add(array.buffer);
    }
    const texture = primitive.material.baseColorTexture;
    if (texture?.kind === "rgba") {
      if (texture.rgba.buffer instanceof ArrayBuffer)
        textures.add(texture.rgba.buffer);
    } else if (texture?.kind === "compressed") {
      for (const level of texture.levels) {
        if (level.data.buffer instanceof ArrayBuffer)
          textures.add(level.data.buffer);
      }
    }
  }
  return {
    geometry: [...geometry].reduce((sum, buffer) => sum + buffer.byteLength, 0),
    textures: [...textures].reduce((sum, buffer) => sum + buffer.byteLength, 0),
  };
};

export const decodeTileContent = async (
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
  const originalJson = contentJson(request.content);
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
    const url = new URL(value, request.contentUrl).href;
    if (!allowedDependencies.has(url)) {
      throw new Error(
        `glTF parser requested an unauthorized dependency: ${value}`,
      );
    }
    const data = await (
      options.fetchDependency ??
      (async (target) => (await fetch(target)).arrayBuffer())
    )(url);
    return new Response(data, { status: 200 });
  };
  const parsed = (await parse(request.content.slice(0), GLTFLoader, {
    core: {
      worker: false,
      CDN: null,
      useLocalLibraries: true,
      baseUrl: request.contentUrl,
      fetch: fetchDependency,
    },
    modules,
    gltf: {
      normalize: false,
      loadBuffers: true,
      loadImages: false,
      decompressMeshes: true,
    },
  })) as unknown as ParsedGltf;

  const sceneTransform = composeSceneTransform(
    request.ecefToScene,
    request.accumulatedTransform,
  );
  const pending = applySceneRtc(parsed, sceneTransform);
  const origin = commonOrigin(pending);
  const textureCache = new Map<string, Promise<TextureResult>>();
  const primitives: DecodedPrimitive[] = await Promise.all(
    pending.map(async ({ rtc, materialIndex }) => ({
      positions: rebasePositions(rtc, origin),
      ...(rtc.normals ? { normals: rtc.normals } : {}),
      ...(rtc.uvs ? { uvs: rtc.uvs } : {}),
      ...(rtc.indices ? { indices: rtc.indices } : {}),
      material: await materialFor(
        materialIndex,
        rtc,
        parsed,
        request,
        options,
        modules,
        textureCache,
        diagnostics,
      ),
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

// Keep this exported for focused URI-policy tests without exposing a fetch
// implementation object on the worker protocol.
export const resolveGltfDependencyUrl = dependencyUrl;
