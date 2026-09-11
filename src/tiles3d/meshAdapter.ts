/** vtk.js-only realization of renderer-neutral decoded mesh tiles. */

import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";
import vtkTexture from "@kitware/vtk.js/Rendering/Core/Texture";

import {
  type Submission,
  type SubmissionScheduler,
} from "../submissionScheduler";
import { IDENTITY, translatedMatrix, type Mat16 } from "../camera";
import { safeCall } from "../observers";
import type { Bounds } from "../octree";
import type {
  DecodedPrimitive,
  DecodedTexture,
  DecodedTileContent,
} from "./decode";
import type {
  PickAlphaTexture,
  SubmittedMeshPrimitive,
  SubmittedMeshTile,
} from "./meshPicking";

export type MeshAdapterOptions = {
  readonly renderer: {
    addActor(actor: unknown): void;
    removeActor(actor: unknown): void;
  };
  readonly scheduleRender: () => void;
  readonly submissions: SubmissionScheduler;
  readonly visible?: boolean;
  readonly onError?: (error: unknown) => void;
};

export type MeshAdapterStats = {
  readonly workRevision: number;
  readonly pendingTiles: number;
  readonly pendingJobs: number;
  readonly pendingBytes: number;
  readonly submittedTiles: number;
  /** Current submitted tile ids, bounded by the member residency allowance. */
  readonly submittedTileIds: readonly string[];
  readonly submittedPrimitives: number;
  /** Exact VTK actors currently submitted after primitive chunking. */
  readonly submittedActors: number;
  readonly pooledTiles: number;
  /**
   * Cumulative typed-array bytes successfully accepted by VTK geometry
   * setters since this adapter was created. Actor reuse does not increment it.
   */
  readonly logicalGeometryUploadBytes: number;
  /**
   * Cumulative compressed mip/RGBA bytes successfully accepted by vtkTexture
   * payload setters. This is the logical upload payload, not driver telemetry.
   */
  readonly logicalTextureUploadBytes: number;
  readonly logicalUploadBytes: number;
  readonly residentGeometryBytes: number;
  /** Texture bytes attached to submitted tiles, excluding reusable pool entries. */
  readonly submittedTextureBytes: number;
  /** Texture bytes retained only for reuse by the fallback pool. */
  readonly pooledTextureBytes: number;
  readonly residentTextureBytes: number;
  readonly residentBytes: number;
  readonly resourceCeilingBytes: number;
  readonly drawnTiles: number;
  /** Current drawn tile ids, used by public refinement diagnostics. */
  readonly drawnTileIds: readonly string[];
  readonly drawnPrimitives: number;
  /** Exact visible VTK actor count after primitive chunking. */
  readonly drawnActors: number;
  /** Texture bytes backing the currently drawn tile set. */
  readonly drawnTextureBytes: number;
  readonly drawnTriangles: number;
  readonly visible: boolean;
  readonly textureRepresentation: "none" | "compressed" | "rgba" | "mixed";
  readonly textureFormats: readonly string[];
};

export type MeshTileState = "absent" | "queued" | "submitted" | "failed";
export type MeshSubmitOutcome = "queued" | "budget-blocked" | "failed";

export type MeshAdapter = {
  submitTile(
    id: string,
    content: DecodedTileContent,
    onSubmitted?: () => void,
    replacementIds?: readonly string[],
  ): MeshSubmitOutcome;
  submitTileGroup(
    entries: readonly {
      readonly id: string;
      readonly content: DecodedTileContent;
      readonly onSubmitted?: () => void;
    }[],
    replacementIds: readonly string[],
  ): MeshSubmitOutcome;
  cancelTile(id: string): boolean;
  retireTile(id: string): boolean;
  setDrawnTiles(ids: readonly string[]): void;
  setBaseMatrix(matrix: Mat16 | null): void;
  setVisible(visible: boolean): void;
  /** Returns queued tile ids canceled to honor the smaller reservation cap. */
  setResourceCeilingBytes(bytes: number): readonly string[];
  /** Retire every queued/submitted tile, including currently hidden fallbacks. */
  clearTiles(): void;
  tileState(id: string): MeshTileState;
  /** Restore retained resources without fetching or decoding their payload again. */
  restoreTile(id: string): "restored" | "absent" | "failed";
  /** All admitted resources, including fallbacks not in the drawn frontier. */
  submittedTileIds(): readonly string[];
  /**
   * Changes on every change to the admitted set, and only then.
   *
   * Lets a caller index `submittedTileIds()` once and reuse the index across a
   * batch of admissions instead of rescanning per tile.
   */
  submissionRevision(): number;
  /** Drawn geometry used for picking; excludes hidden admitted resources. */
  submittedTiles(): readonly SubmittedMeshTile[];
  /** Admission and memory counters without constructing detailed tile diagnostics. */
  workState(): Pick<
    MeshAdapterStats,
    "workRevision" | "pendingJobs" | "residentBytes"
  >;
  stats(): MeshAdapterStats;
  dispose(): void;
};

type PrimitiveResources = {
  readonly actor: any;
  readonly mapper: any;
  readonly polyData: any;
  /** Exact charged position/index arrays submitted to VTK and used for picks. */
  readonly primitive: SubmittedMeshPrimitive;
};

type TileResources = {
  readonly id: string;
  readonly origin: readonly [number, number, number];
  readonly bounds?: Bounds;
  readonly primitiveCount: number;
  readonly primitives: PrimitiveResources[];
  readonly textures: Set<any>;
  readonly geometryBytes: number;
  readonly textureBytes: number;
  readonly formats: Set<string>;
  attached: boolean;
};

type PendingTile = {
  readonly id: string;
  readonly jobs: Submission[];
  readonly resources: TileResources;
  readonly onSubmitted?: () => void;
  readonly replacementIds: readonly string[];
  remainingJobs: number;
  pendingBytes: number;
  cancelled: boolean;
  ready: boolean;
  finishBarrier?: () => void;
  groupError?: (error: unknown) => void;
};

type VtkShaderReplacement = {
  readonly shaderType: "Fragment";
  readonly originalValue: string;
  readonly replacementValue: string;
  readonly replaceAll: false;
  readonly replaceFirst: true;
};

const VTK_LIGHT_IMPLEMENTATION = "//VTK::Light::Impl";

const glslFloat = (value: number): string => {
  if (!Number.isFinite(value)) {
    throw new TypeError("glTF alphaCutoff must be finite");
  }
  const literal = String(value);
  return /[.eE]/u.test(literal) ? literal : `${literal}.0`;
};

const alphaShaderReplacement = (
  material: DecodedPrimitive["material"]["raw"],
): VtkShaderReplacement | undefined => {
  let statement: string;
  if (material.alphaMode === "MASK") {
    statement = `if (opacity < ${glslFloat(material.alphaCutoff)}) { discard; }`;
  } else if (material.alphaMode === "OPAQUE") {
    // glTF OPAQUE ignores both base-color factor and texture alpha.
    statement = "opacity = 1.0;";
  } else {
    return undefined;
  }
  return {
    shaderType: "Fragment",
    originalValue: VTK_LIGHT_IMPLEMENTATION,
    replacementValue: `${statement}\n  ${VTK_LIGHT_IMPLEMENTATION}`,
    replaceAll: false,
    replaceFirst: true,
  };
};

const rgbaAlphaIsUniform = (
  texture: Extract<DecodedTexture, { kind: "rgba" }>,
): boolean => {
  const first = texture.rgba[3];
  for (let index = 7; index < texture.rgba.length; index += 4) {
    if (texture.rgba[index] !== first) return false;
  }
  return true;
};

const hasExactCpuAlphaSampler = (
  texture: Extract<DecodedTexture, { kind: "rgba" }>,
): boolean => {
  if (rgbaAlphaIsUniform(texture)) return true;
  const { magFilter, minFilter } = texture.sampler;
  return (minFilter === 9728 || minFilter === 9729) && minFilter === magFilter;
};

const vtkSampler = (sampler: DecodedTexture["sampler"]) => ({
  magFilter: sampler.magFilter === 9728 ? "nearest" : "linear",
  minFilter: {
    9728: "nearest",
    9729: "linear",
    9984: "nearest-mipmap-nearest",
    9985: "linear-mipmap-nearest",
    9986: "nearest-mipmap-linear",
    9987: "linear-mipmap-linear",
  }[sampler.minFilter],
  wrapS: {
    33071: "clamp-to-edge",
    33648: "mirrored-repeat",
    10497: "repeat",
  }[sampler.wrapS],
  wrapT: {
    33071: "clamp-to-edge",
    33648: "mirrored-repeat",
    10497: "repeat",
  }[sampler.wrapT],
});

/**
 * Bytes a primitive costs once submitted, which depends on whether it is
 * split: a chunk is deindexed, so no job retains a whole oversized primitive
 * merely to realize a subset, while a primitive submitted whole keeps the
 * arrays it was decoded with. Both add vtk's `[3, i0, i1, i2]` cell record.
 */
const retainedGeometryBytes = (
  primitive: DecodedPrimitive,
  maxJobBytes: number,
): number => {
  const triangles = trianglesIn(primitive);
  if (splits(primitive, maxJobBytes)) {
    return triangles * bytesPerTriangle(primitive);
  }
  return (
    triangles * 4 * Uint32Array.BYTES_PER_ELEMENT +
    primitive.positions.byteLength +
    (primitive.normals?.byteLength ?? 0) +
    (primitive.uvs?.byteLength ?? 0)
  );
};

const actualGeometryBytes = (
  content: DecodedTileContent,
  maxJobBytes: number,
): number => {
  let bytes = 0;
  for (const primitive of content.primitives) {
    bytes += retainedGeometryBytes(primitive, maxJobBytes);
  }
  const retainedAlpha = new Set<DecodedTexture>();
  for (const primitive of content.primitives) {
    const texture = primitive.material.baseColorTexture;
    if (
      primitive.material.raw.alphaMode === "MASK" &&
      primitive.uvs &&
      texture?.kind === "rgba" &&
      !retainedAlpha.has(texture) &&
      hasExactCpuAlphaSampler(texture)
    ) {
      retainedAlpha.add(texture);
      bytes += texture.width * texture.height;
    }
  }
  return bytes;
};

const textureByteLength = (texture: DecodedTexture): number =>
  texture.kind === "compressed"
    ? texture.levels.reduce((sum, level) => sum + level.data.byteLength, 0)
    : texture.rgba.byteLength;

const actualTextureBytes = (content: DecodedTileContent): number => {
  const seen = new Set<DecodedTexture>();
  let bytes = 0;
  for (const primitive of content.primitives) {
    const texture = primitive.material.baseColorTexture;
    if (texture && !seen.has(texture)) {
      seen.add(texture);
      bytes += textureByteLength(texture);
    }
  }
  return bytes;
};

const boundsOf = (content: DecodedTileContent): Bounds | undefined => {
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  let count = 0;
  for (const primitive of content.primitives) {
    for (let index = 0; index + 2 < primitive.positions.length; index += 3) {
      count += 1;
      for (let axis = 0; axis < 3; axis += 1) {
        const value =
          content.origin[axis]! + primitive.positions[index + axis]!;
        min[axis] = Math.min(min[axis]!, value);
        max[axis] = Math.max(max[axis]!, value);
      }
    }
  }
  return count ? { min, max } : undefined;
};

const trianglesIn = (
  primitive: Pick<DecodedPrimitive, "positions" | "indices">,
): number =>
  Math.floor((primitive.indices?.length ?? primitive.positions.length / 3) / 3);

const cellsFor = (primitive: DecodedPrimitive): Uint32Array => {
  const count = Math.floor(
    (primitive.indices?.length ?? primitive.positions.length / 3) / 3,
  );
  const cells = new Uint32Array(count * 4);
  for (let triangle = 0; triangle < count; triangle += 1) {
    const output = triangle * 4;
    cells[output] = 3;
    for (let corner = 0; corner < 3; corner += 1) {
      cells[output + corner + 1] =
        primitive.indices?.[triangle * 3 + corner] ?? triangle * 3 + corner;
    }
  }
  return cells;
};

/**
 * Whether a primitive has to be cut into more than one submission slice. Its
 * per-triangle cost is the deindexed one either way: that is what a chunk
 * would cost, and a primitive small enough to escape splitting under the
 * larger estimate is small enough under its own.
 */
const splits = (primitive: DecodedPrimitive, maxJobBytes: number): boolean =>
  trianglesIn(primitive) >
  Math.max(1, Math.floor(maxJobBytes / bytesPerTriangle(primitive)));

const bytesPerTriangle = (primitive: DecodedPrimitive): number =>
  9 * Float32Array.BYTES_PER_ELEMENT +
  (primitive.normals ? 9 * Float32Array.BYTES_PER_ELEMENT : 0) +
  (primitive.uvs ? 6 * Float32Array.BYTES_PER_ELEMENT : 0) +
  4 * Uint32Array.BYTES_PER_ELEMENT;

const sourceVertex = (
  primitive: DecodedPrimitive,
  triangle: number,
  corner: number,
): number =>
  primitive.indices?.[triangle * 3 + corner] ?? triangle * 3 + corner;

/** Copy one bounded triangle interval; called only inside its charged job. */
const primitiveChunk = (
  primitive: DecodedPrimitive,
  firstTriangle: number,
  triangleCount: number,
): DecodedPrimitive => {
  const positions = new Float32Array(triangleCount * 9);
  const normals = primitive.normals
    ? new Float32Array(triangleCount * 9)
    : undefined;
  const uvs = primitive.uvs ? new Float32Array(triangleCount * 6) : undefined;
  for (
    let localTriangle = 0;
    localTriangle < triangleCount;
    localTriangle += 1
  ) {
    const sourceTriangle = firstTriangle + localTriangle;
    for (let corner = 0; corner < 3; corner += 1) {
      const vertex = sourceVertex(primitive, sourceTriangle, corner);
      positions.set(
        primitive.positions.subarray(vertex * 3, vertex * 3 + 3),
        (localTriangle * 3 + corner) * 3,
      );
      if (normals && primitive.normals) {
        normals.set(
          primitive.normals.subarray(vertex * 3, vertex * 3 + 3),
          (localTriangle * 3 + corner) * 3,
        );
      }
      if (uvs && primitive.uvs) {
        uvs.set(
          primitive.uvs.subarray(vertex * 2, vertex * 2 + 2),
          (localTriangle * 3 + corner) * 2,
        );
      }
    }
  }
  return {
    positions,
    ...(normals ? { normals } : {}),
    ...(uvs ? { uvs } : {}),
    material: primitive.material,
  };
};

const rgbaImageData = (
  texture: Extract<DecodedTexture, { kind: "rgba" }>,
): ImageData => {
  const pixels = new Uint8ClampedArray(texture.rgba);
  if (typeof ImageData === "function")
    return new ImageData(pixels, texture.width, texture.height);
  return {
    data: pixels,
    width: texture.width,
    height: texture.height,
  } as ImageData;
};

export const createMeshAdapter = (options: MeshAdapterOptions): MeshAdapter => {
  const maxJobBytes = options.submissions.stats().maxBytesPerFrame;
  if (!Number.isFinite(maxJobBytes) || maxJobBytes <= 0) {
    throw new RangeError("maxSubmissionJobBytes must be finite and > 0");
  }
  let visible = options.visible ?? true;
  let baseMatrix: readonly number[] = IDENTITY;
  let resourceCeilingBytes = Number.POSITIVE_INFINITY;
  let disposed = false;
  let workRevision = 0;
  // Running byte totals rather than a walk per question. Trimming asks how
  // many bytes are held once per evicted pool entry, and the pool is largest
  // exactly when the ceiling has just dropped, so re-summing there would make
  // shedding quadratic. Every mutation of the three maps goes through the
  // helpers below; `admit`/`withdraw` also keep `submissionRevision` an exact
  // record of the admitted set changing.
  const pending = new Map<string, PendingTile>();
  const submitted = new Map<string, TileResources>();
  const pooled = new Map<string, TileResources>();
  type ByteTotal = { geometry: number; texture: number };
  const pendingBytes: ByteTotal = { geometry: 0, texture: 0 };
  const submittedBytes: ByteTotal = { geometry: 0, texture: 0 };
  const pooledBytes: ByteTotal = { geometry: 0, texture: 0 };
  let submissionRevision = 0;
  const tally = (total: ByteTotal, tile: TileResources, sign: 1 | -1): void => {
    total.geometry += sign * tile.geometryBytes;
    total.texture += sign * tile.textureBytes;
  };
  const sumOf = (total: ByteTotal): number => total.geometry + total.texture;
  const admit = (id: string, tile: TileResources): void => {
    submitted.set(id, tile);
    tally(submittedBytes, tile, 1);
    submissionRevision += 1;
  };
  const withdraw = (id: string): TileResources | undefined => {
    const tile = submitted.get(id);
    if (!tile) return undefined;
    submitted.delete(id);
    tally(submittedBytes, tile, -1);
    submissionRevision += 1;
    return tile;
  };
  const pool = (id: string, tile: TileResources): void => {
    pooled.set(id, tile);
    tally(pooledBytes, tile, 1);
  };
  const unpool = (id: string, tile: TileResources): void => {
    pooled.delete(id);
    tally(pooledBytes, tile, -1);
  };
  const holdPending = (entry: PendingTile): void => {
    pending.set(entry.id, entry);
    tally(pendingBytes, entry.resources, 1);
  };
  const dropPending = (entry: PendingTile): void => {
    pending.delete(entry.id);
    tally(pendingBytes, entry.resources, -1);
  };
  const failed = new Set<string>();
  const replacementClaims = new Map<string, string>();
  let drawn = new Set<string>();
  let logicalGeometryUploadBytes = 0;
  let logicalTextureUploadBytes = 0;
  const budgetPreapproved = new Set<string>();

  const setActorState = (
    tile: TileResources,
    entry: PrimitiveResources,
  ): void => {
    entry.actor.setUserMatrix(translatedMatrix(baseMatrix, tile.origin));
    entry.actor.setVisibility(visible && drawn.has(tile.id));
  };

  const release = (tile: TileResources): void => {
    for (const entry of tile.primitives) {
      if (tile.attached) options.renderer.removeActor(entry.actor);
      entry.actor.delete?.();
      entry.mapper.delete?.();
      entry.polyData.delete?.();
    }
    for (const texture of tile.textures) texture.delete?.();
    tile.primitives.length = 0;
    tile.textures.clear();
    tile.attached = false;
  };

  const createTexture = (decoded: DecodedTexture): any => {
    const texture = vtkTexture.newInstance();
    try {
      texture.setSampler(vtkSampler(decoded.sampler));
      texture.setFlipY(false);
      if (decoded.kind === "compressed") {
        texture.setCompressedData({
          format: decoded.format,
          width: decoded.width,
          height: decoded.height,
          // Deliberately not `colorSpace === "srgb"`. The RGBA fallback path
          // below uploads raw bytes with no sRGB internal format, and the
          // polydata shaders do no output encoding, so requesting sRGB here
          // would decode the same texture differently depending on whether the
          // context supports compression — the same tileset visibly darker on
          // hardware than in a software-GL run. Both paths stay linear until
          // the fork owns a real sRGB workflow end to end.
          srgb: false,
          levels: decoded.levels,
        });
      } else {
        texture.setJsImageData(rgbaImageData(decoded));
      }
      logicalTextureUploadBytes += textureByteLength(decoded);
      return texture;
    } catch (error) {
      texture.delete?.();
      throw error;
    }
  };

  const createPrimitive = (
    tile: TileResources,
    primitive: DecodedPrimitive,
    textures: Map<DecodedTexture, any>,
    pickAlphaTextures: Map<DecodedTexture, PickAlphaTexture>,
  ): void => {
    let polyData: any;
    let mapper: any;
    let actor: any;
    try {
      polyData = vtkPolyData.newInstance();
      polyData.getPoints().setData(primitive.positions, 3);
      logicalGeometryUploadBytes += primitive.positions.byteLength;
      const cells = cellsFor(primitive);
      polyData.getPolys().setData(cells);
      logicalGeometryUploadBytes += cells.byteLength;
      if (primitive.normals) {
        polyData.getPointData().setNormals(
          vtkDataArray.newInstance({
            name: "Normals",
            values: primitive.normals,
            numberOfComponents: 3,
          }),
        );
        logicalGeometryUploadBytes += primitive.normals.byteLength;
      }
      if (primitive.uvs) {
        polyData.getPointData().setTCoords(
          vtkDataArray.newInstance({
            name: "TCoords",
            values: primitive.uvs,
            numberOfComponents: 2,
          }),
        );
        logicalGeometryUploadBytes += primitive.uvs.byteLength;
      }
      mapper = vtkMapper.newInstance();
      mapper.setInputData(polyData);
      mapper.setStatic?.(true);
      actor = vtkActor.newInstance();
      actor.setMapper(mapper);
      const factor = primitive.material.baseColorFactor;
      const authored = primitive.material.raw;
      actor.getProperty().setLighting(!authored.unlit);
      actor.getProperty().setColor(factor[0], factor[1], factor[2]);
      actor
        .getProperty()
        .setOpacity(authored.alphaMode === "OPAQUE" ? 1 : factor[3]);
      actor.setForceOpaque(authored.alphaMode !== "BLEND");
      actor.setForceTranslucent(authored.alphaMode === "BLEND");
      const replacement = alphaShaderReplacement(authored);
      if (replacement) {
        mapper.setViewSpecificProperties({
          OpenGL: { ShaderReplacements: [replacement] },
        });
      }
      const decodedTexture = primitive.material.baseColorTexture;
      if (decodedTexture) {
        const texture = textures.get(decodedTexture);
        if (!texture) {
          throw new Error("mesh texture admission did not precede geometry");
        }
        actor.addTexture(texture);
      }
      // Keep only the exact geometry arrays submitted to VTK. Retaining the
      // decoded primitive would also retain material.raw and the original
      // texture payload after ContentQueue has evicted its CPU-cache entry.
      const submittedPrimitive: SubmittedMeshPrimitive = {
        positions: primitive.positions,
        ...(primitive.indices ? { indices: primitive.indices } : {}),
        ...(primitive.uvs ? { uvs: primitive.uvs } : {}),
        ...(authored.alphaMode !== "MASK"
          ? {}
          : decodedTexture &&
              primitive.uvs &&
              !pickAlphaTextures.has(decodedTexture)
            ? { alphaMask: { kind: "unknown" as const } }
            : {
                alphaMask: {
                  kind: "known" as const,
                  factorAlpha: factor[3],
                  cutoff: authored.alphaCutoff,
                  ...(decodedTexture && primitive.uvs
                    ? { texture: pickAlphaTextures.get(decodedTexture) }
                    : {}),
                },
              }),
      };
      const resources = {
        actor,
        mapper,
        polyData,
        primitive: submittedPrimitive,
      };
      tile.primitives.push(resources);
      setActorState(tile, resources);
    } catch (error) {
      actor?.delete?.();
      mapper?.delete?.();
      polyData?.delete?.();
      throw error;
    }
  };

  const reservedBytes = (): number =>
    sumOf(submittedBytes) + sumOf(pooledBytes) + sumOf(pendingBytes);

  const cancelPending = (entry: PendingTile): void => {
    if (entry.cancelled) return;
    entry.cancelled = true;
    for (const job of entry.jobs) job.cancel();
    dropPending(entry);
    for (const id of entry.replacementIds) {
      if (replacementClaims.get(id) === entry.id) replacementClaims.delete(id);
    }
    release(entry.resources);
    workRevision += 1;
  };

  const trimPool = (additionalBytes = 0): void => {
    for (const [id, tile] of pooled) {
      if (reservedBytes() + additionalBytes <= resourceCeilingBytes) return;
      unpool(id, tile);
      release(tile);
    }
  };

  const attach = (tile: TileResources): void => {
    const added: any[] = [];
    try {
      for (const primitive of tile.primitives) {
        setActorState(tile, primitive);
        added.push(primitive.actor);
        options.renderer.addActor(primitive.actor);
      }
      tile.attached = true;
    } catch (error) {
      for (const actor of added.reverse()) {
        try {
          options.renderer.removeActor(actor);
        } catch {
          // Keep rolling back the remaining actors even if the renderer is hostile.
        }
      }
      for (const primitive of tile.primitives) {
        try {
          primitive.actor.setVisibility(false);
        } catch {
          // Resource deletion below is the final cleanup boundary.
        }
      }
      tile.attached = false;
      throw error;
    }
  };

  const finish = (entry: PendingTile): void => {
    if (entry.cancelled || disposed || pending.get(entry.id) !== entry) return;
    attach(entry.resources);
    dropPending(entry);
    admit(entry.id, entry.resources);
    workRevision += 1;
    options.scheduleRender();
    try {
      entry.onSubmitted?.();
    } catch (error) {
      safeCall(() => options.onError?.(error));
    } finally {
      for (const id of entry.replacementIds) {
        if (replacementClaims.get(id) === entry.id)
          replacementClaims.delete(id);
      }
    }
  };

  /** Drop every pending, submitted and pooled tile back to an empty adapter. */
  const clearAll = (): void => {
    for (const entry of pending.values()) cancelPending(entry);
    for (const [id, tile] of submitted) {
      release(tile);
      withdraw(id);
    }
    for (const [id, tile] of pooled) {
      release(tile);
      unpool(id, tile);
    }
    drawn.clear();
    failed.clear();
    replacementClaims.clear();
    workRevision += 1;
    options.scheduleRender();
  };

  const api: MeshAdapter = {
    restoreTile(id) {
      if (disposed || failed.has(id)) return "failed";
      const reusable = pooled.get(id);
      if (!reusable) return "absent";
      // Tile ids identify immutable content within this adapter. Source and
      // decode-configuration changes clear it, while placement is reapplied.
      try {
        for (const primitive of reusable.primitives)
          setActorState(reusable, primitive);
      } catch (error) {
        unpool(id, reusable);
        failed.add(id);
        release(reusable);
        safeCall(() => options.onError?.(error));
        return "failed";
      }
      unpool(id, reusable);
      admit(id, reusable);
      workRevision += 1;
      options.scheduleRender();
      return "restored";
    },

    submitTile(id, content, onSubmitted, replacementIds = []) {
      if (disposed || failed.has(id)) return "failed";
      if (pending.has(id) || submitted.has(id)) return "queued";
      if (typeof id !== "string" || id.length === 0)
        throw new TypeError("mesh tile id must be non-empty");
      const restored = api.restoreTile(id);
      if (restored === "failed") return "failed";
      if (restored === "restored") {
        try {
          onSubmitted?.();
        } catch (error) {
          safeCall(() => options.onError?.(error));
        }
        return "queued";
      }
      const geometryBytes = actualGeometryBytes(content, maxJobBytes);
      const textureBytes = actualTextureBytes(content);
      trimPool(geometryBytes + textureBytes);
      const normalizedReplacements = [...new Set(replacementIds)].filter(
        (replacementId) =>
          replacementId !== id &&
          submitted.has(replacementId) &&
          !replacementClaims.has(replacementId),
      );
      const replacementBytes = normalizedReplacements.reduce(
        (sum, replacementId) => {
          const tile = submitted.get(replacementId)!;
          return sum + tile.geometryBytes + tile.textureBytes;
        },
        0,
      );
      if (
        !budgetPreapproved.has(id) &&
        reservedBytes() + geometryBytes + textureBytes - replacementBytes >
          resourceCeilingBytes
      ) {
        return "budget-blocked";
      }
      const formats = new Set<string>();
      for (const primitive of content.primitives) {
        const texture = primitive.material.baseColorTexture;
        if (texture)
          formats.add(texture.kind === "compressed" ? texture.format : "rgba");
      }
      try {
        for (const primitive of content.primitives) {
          if (bytesPerTriangle(primitive) > maxJobBytes) {
            throw new Error(
              `one mesh triangle requires ${bytesPerTriangle(primitive)} bytes, exceeding the ${maxJobBytes}-byte submission cap`,
            );
          }
        }
      } catch (error) {
        failed.add(id);
        safeCall(() => options.onError?.(error));
        return "failed";
      }
      const resources: TileResources = {
        id,
        origin: [...content.origin],
        bounds: boundsOf(content),
        primitiveCount: content.primitives.length,
        primitives: [],
        textures: new Set(),
        geometryBytes,
        textureBytes,
        formats,
        attached: false,
      };
      const entry: PendingTile = {
        id,
        jobs: [],
        resources,
        replacementIds: normalizedReplacements,
        remainingJobs: 0,
        pendingBytes: geometryBytes + textureBytes,
        cancelled: false,
        ready: false,
        ...(onSubmitted ? { onSubmitted } : {}),
      };
      holdPending(entry);
      for (const replacementId of normalizedReplacements)
        replacementClaims.set(replacementId, id);
      const textures = new Map<DecodedTexture, any>();
      const pickAlphaTextures = new Map<DecodedTexture, PickAlphaTexture>();
      const decodedTextures: DecodedTexture[] = [];
      for (const primitive of content.primitives) {
        const texture = primitive.material.baseColorTexture;
        if (texture && !decodedTextures.includes(texture))
          decodedTextures.push(texture);
        if (
          primitive.material.raw.alphaMode === "MASK" &&
          primitive.uvs &&
          texture?.kind === "rgba" &&
          !pickAlphaTextures.has(texture)
        ) {
          const alpha = new Uint8Array(texture.width * texture.height);
          for (let pixel = 0; pixel < alpha.length; pixel += 1)
            alpha[pixel] = texture.rgba[pixel * 4 + 3]!;
          // A pick has one cursor sample, not the rasterizer's pixel
          // derivatives, so it cannot know whether minification or
          // magnification applies or which mip levels WebGL blends. Retain an
          // exact CPU sampler only when both paths are the same non-mip base
          // filter. A uniform alpha plane is exact under every sampler.
          if (hasExactCpuAlphaSampler(texture)) {
            pickAlphaTextures.set(texture, {
              width: texture.width,
              height: texture.height,
              alpha,
              sampler: texture.sampler,
            });
          }
        }
      }
      const enqueue = (
        bytes: number,
        run: () => void,
        atomic = false,
      ): void => {
        entry.remainingJobs += 1;
        const submission = options.submissions.enqueue({
          bytes,
          atomic,
          run: () => {
            if (entry.cancelled || disposed) return;
            run();
            entry.remainingJobs -= 1;
            entry.pendingBytes -= bytes;
            if (entry.remainingJobs === 0) {
              entry.ready = true;
              if (entry.finishBarrier) entry.finishBarrier();
              else finish(entry);
            }
          },
          onError: (error) => {
            if (entry.groupError) {
              entry.groupError(error);
              return;
            }
            cancelPending(entry);
            failed.add(entry.id);
            safeCall(() => options.onError?.(error));
          },
        });
        entry.jobs.push(submission);
      };
      try {
        // Texture ownership is indivisible at vtkTexture's payload setter.
        // The tile already passed the residency budget; a large texture gets
        // its own admission frame instead of becoming a permanent load error.
        for (const decodedTexture of decodedTextures) {
          const bytes = textureByteLength(decodedTexture);
          enqueue(
            bytes,
            () => {
              const texture = createTexture(decodedTexture);
              textures.set(decodedTexture, texture);
              resources.textures.add(texture);
            },
            true,
          );
        }
        content.primitives.forEach((primitive) => {
          const totalTriangles = trianglesIn(primitive);
          const perTriangle = bytesPerTriangle(primitive);
          if (perTriangle > maxJobBytes) {
            throw new Error(
              `one mesh triangle requires ${perTriangle} bytes, exceeding the ${maxJobBytes}-byte submission cap`,
            );
          }
          const trianglesPerChunk = Math.max(
            1,
            Math.floor(maxJobBytes / perTriangle),
          );
          // A primitive that fits one slice is submitted as authored, indices
          // and all. Splitting is what forces triangle soup — a chunk cannot
          // carry the whole index buffer — and expanding a mesh that was never
          // going to be split copies every vertex once per triangle that
          // references it, for geometry identical to what the indices already
          // describe.
          if (!splits(primitive, maxJobBytes)) {
            enqueue(totalTriangles * perTriangle, () =>
              createPrimitive(
                resources,
                primitive,
                textures,
                pickAlphaTextures,
              ),
            );
            return;
          }
          for (
            let first = 0;
            first < totalTriangles;
            first += trianglesPerChunk
          ) {
            const count = Math.min(trianglesPerChunk, totalTriangles - first);
            enqueue(count * perTriangle, () =>
              createPrimitive(
                resources,
                primitiveChunk(primitive, first, count),
                textures,
                pickAlphaTextures,
              ),
            );
          }
        });
      } catch (error) {
        cancelPending(entry);
        failed.add(id);
        safeCall(() => options.onError?.(error));
        return "failed";
      }
      if (entry.remainingJobs === 0) finish(entry);
      workRevision += 1;
      return "queued";
    },

    submitTileGroup(entries, replacementIds) {
      if (disposed || entries.length === 0) return "failed";
      const ids = new Set(entries.map((entry) => entry.id));
      if (
        ids.size !== entries.length ||
        entries.some(
          (entry) =>
            typeof entry.id !== "string" ||
            entry.id.length === 0 ||
            pending.has(entry.id) ||
            submitted.has(entry.id) ||
            failed.has(entry.id),
        )
      ) {
        return "failed";
      }
      const normalizedReplacements = [...new Set(replacementIds)].filter(
        (id) => submitted.has(id) && !replacementClaims.has(id),
      );
      const replacementBytes = normalizedReplacements.reduce((sum, id) => {
        const tile = submitted.get(id)!;
        return sum + tile.geometryBytes + tile.textureBytes;
      }, 0);
      const groupBytes = entries.reduce(
        (sum, entry) =>
          sum +
          actualGeometryBytes(entry.content, maxJobBytes) +
          actualTextureBytes(entry.content),
        0,
      );
      trimPool(groupBytes);
      if (
        reservedBytes() + groupBytes - replacementBytes >
        resourceCeilingBytes
      ) {
        return "budget-blocked";
      }

      for (const { id } of entries) budgetPreapproved.add(id);
      const admitted: string[] = [];
      try {
        for (let index = 0; index < entries.length; index += 1) {
          const entry = entries[index]!;
          const outcome = api.submitTile(
            entry.id,
            entry.content,
            entry.onSubmitted,
            index === entries.length - 1 ? normalizedReplacements : [],
          );
          if (outcome !== "queued") {
            for (const id of admitted) {
              if (!api.cancelTile(id)) api.retireTile(id);
            }
            return outcome;
          }
          admitted.push(entry.id);
        }
        const groupEntries = admitted.map((id) => pending.get(id)!);
        let groupFailed = false;
        const cancelGroup = (error: unknown): void => {
          if (groupFailed) return;
          groupFailed = true;
          for (const entry of groupEntries) cancelPending(entry);
          for (const { id } of entries) failed.add(id);
          safeCall(() => options.onError?.(error));
        };
        const finishGroup = (): void => {
          if (groupFailed || groupEntries.some((entry) => !entry.ready)) return;
          // The replacement remains attached until every child resource is
          // ready. Then the displayed frontier swaps synchronously: no paint
          // can observe the gap, and replacement residency is released before
          // any child becomes submitted.
          for (const id of normalizedReplacements) {
            const tile = withdraw(id);
            if (!tile) continue;
            drawn.delete(id);
            release(tile);
          }
          for (const entry of groupEntries) finish(entry);
        };
        for (const entry of groupEntries) {
          entry.finishBarrier = finishGroup;
          entry.groupError = cancelGroup;
        }
      } finally {
        for (const { id } of entries) budgetPreapproved.delete(id);
      }
      return "queued";
    },

    cancelTile(id) {
      const entry = pending.get(id);
      if (!entry) return false;
      cancelPending(entry);
      options.scheduleRender();
      return true;
    },

    retireTile(id) {
      const queued = pending.get(id);
      if (queued) {
        cancelPending(queued);
        options.scheduleRender();
        return true;
      }
      const tile = withdraw(id);
      if (!tile) return failed.delete(id);
      drawn.delete(id);
      // Keep invisible actors in the renderer while pooled. Removing them
      // destroys vtk render nodes and forces GPU uploads on cache reuse.
      // Pool eviction still releases the actors under the residency ceiling.
      for (const primitive of tile.primitives) setActorState(tile, primitive);
      pool(id, tile);
      trimPool();
      workRevision += 1;
      options.scheduleRender();
      return true;
    },

    setDrawnTiles(ids) {
      if (disposed) return;
      const next = new Set(ids.filter((id) => submitted.has(id)));
      let changed = next.size !== drawn.size;
      if (!changed) for (const id of next) if (!drawn.has(id)) changed = true;
      drawn = next;
      for (const tile of submitted.values()) {
        for (const primitive of tile.primitives) setActorState(tile, primitive);
      }
      if (changed) options.scheduleRender();
    },

    setBaseMatrix(matrix) {
      if (disposed) return;
      const next = matrix === null ? IDENTITY : Array.from(matrix);
      if (next.length !== 16 || next.some((value) => !Number.isFinite(value))) {
        safeCall(() =>
          options.onError?.(
            new TypeError("mesh base matrix must contain 16 finite numbers"),
          ),
        );
        return;
      }
      if (next.every((value, index) => value === baseMatrix[index])) return;
      baseMatrix = next;
      for (const tile of submitted.values()) {
        for (const primitive of tile.primitives) setActorState(tile, primitive);
      }
      for (const tile of pending.values()) {
        for (const primitive of tile.resources.primitives)
          setActorState(tile.resources, primitive);
      }
      options.scheduleRender();
    },

    setVisible(next) {
      if (disposed || visible === next) return;
      visible = next;
      for (const tile of submitted.values()) {
        for (const primitive of tile.primitives) setActorState(tile, primitive);
      }
      options.scheduleRender();
    },

    setResourceCeilingBytes(bytes) {
      if (disposed || !Number.isFinite(bytes) || bytes < 0) return [];
      resourceCeilingBytes = Math.floor(bytes);
      const cancelled: string[] = [];
      // Submitted coverage is never punched out here. The member first admits
      // a shallower replacement and then retires descendants.
      for (const entry of [...pending.values()].reverse()) {
        if (reservedBytes() <= resourceCeilingBytes) break;
        cancelled.push(entry.id);
        cancelPending(entry);
      }
      trimPool();
      return cancelled;
    },

    clearTiles() {
      if (!disposed) clearAll();
    },

    tileState(id) {
      return submitted.has(id)
        ? "submitted"
        : pending.has(id)
          ? "queued"
          : failed.has(id)
            ? "failed"
            : "absent";
    },

    submissionRevision: () => submissionRevision,

    submittedTileIds() {
      return [...submitted.keys()];
    },

    submittedTiles() {
      return [...drawn]
        .map((id) => submitted.get(id))
        .filter((tile): tile is TileResources => tile !== undefined)
        .map((tile) => ({
          id: tile.id,
          origin: tile.origin,
          primitives: tile.primitives.map((entry) => entry.primitive),
          ...(tile.bounds ? { bounds: tile.bounds } : {}),
        }));
    },

    workState() {
      let pendingJobs = 0;
      for (const tile of pending.values()) pendingJobs += tile.remainingJobs;
      return {
        workRevision,
        pendingJobs,
        residentBytes: sumOf(submittedBytes) + sumOf(pooledBytes),
      };
    },

    stats() {
      const submittedResources = [...submitted.values()];
      const resident = [...submittedResources, ...pooled.values()];
      const drawnResources = [...drawn]
        .map((id) => submitted.get(id))
        .filter((tile): tile is TileResources => tile !== undefined);
      const formats = new Set<string>();
      for (const tile of resident)
        for (const format of tile.formats) formats.add(format);
      const hasCompressed = [...formats].some((format) => format !== "rgba");
      const hasRgba = formats.has("rgba");
      return {
        workRevision,
        pendingTiles: pending.size,
        pendingJobs: [...pending.values()].reduce(
          (sum, tile) => sum + tile.remainingJobs,
          0,
        ),
        pendingBytes: [...pending.values()].reduce(
          (sum, tile) => sum + tile.pendingBytes,
          0,
        ),
        submittedTiles: submittedResources.length,
        submittedTileIds: [...submitted.keys()],
        submittedPrimitives: submittedResources.reduce(
          (sum, tile) => sum + tile.primitiveCount,
          0,
        ),
        submittedActors: submittedResources.reduce(
          (sum, tile) => sum + tile.primitives.length,
          0,
        ),
        pooledTiles: pooled.size,
        logicalGeometryUploadBytes,
        logicalTextureUploadBytes,
        logicalUploadBytes:
          logicalGeometryUploadBytes + logicalTextureUploadBytes,
        residentGeometryBytes: submittedBytes.geometry + pooledBytes.geometry,
        submittedTextureBytes: submittedBytes.texture,
        pooledTextureBytes: pooledBytes.texture,
        residentTextureBytes: submittedBytes.texture + pooledBytes.texture,
        residentBytes: sumOf(submittedBytes) + sumOf(pooledBytes),
        resourceCeilingBytes,
        drawnTiles: drawnResources.length,
        drawnTileIds: [...drawn],
        drawnPrimitives: drawnResources.reduce(
          (sum, tile) => sum + tile.primitiveCount,
          0,
        ),
        drawnActors: drawnResources.reduce(
          (sum, tile) => sum + tile.primitives.length,
          0,
        ),
        drawnTextureBytes: drawnResources.reduce(
          (sum, tile) => sum + tile.textureBytes,
          0,
        ),
        drawnTriangles: drawnResources.reduce(
          (sum, tile) =>
            sum +
            tile.primitives.reduce(
              (inner, entry) => inner + trianglesIn(entry.primitive),
              0,
            ),
          0,
        ),
        visible,
        textureRepresentation:
          hasCompressed && hasRgba
            ? "mixed"
            : hasCompressed
              ? "compressed"
              : hasRgba
                ? "rgba"
                : "none",
        textureFormats: [...formats].sort(),
      } satisfies MeshAdapterStats;
    },

    dispose() {
      if (disposed) return;
      disposed = true;
      clearAll();
    },
  };
  return api;
};
