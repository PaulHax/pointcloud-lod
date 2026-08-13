import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";

import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { Document, NodeIO } from "@gltf-transform/core";
import draco3d from "draco3dgltf";

import {
  FIXTURE_CODEC_TILES,
  generateTiles3dFixture,
} from "../../../scripts/generateTiles3dFixture.mjs";

import {
  buildDecodeCacheKey,
  buildTransferList,
  capabilityTarget,
  type DecodeTileRequest,
  type TextureCapabilities,
} from ".";
import {
  decodeTileContent,
  normalizeTextureCoordinates,
  resolveGltfDependencyUrl,
  type DecodeTileOptions,
} from "./decoder";

let fixtureDirectory = "";

beforeAll(async () => {
  fixtureDirectory = await mkdtemp(resolve(tmpdir(), "pointcloud-lod-decode-"));
  await generateTiles3dFixture(fixtureDirectory);
}, 30_000);

afterAll(async () => {
  if (fixtureDirectory)
    await rm(fixtureDirectory, { recursive: true, force: true });
});

const fixture = (name: string) =>
  readFile(resolve(fixtureDirectory, "content", name));

const capabilities = (
  capabilityKey: string,
  compressedFormats: TextureCapabilities["compressedFormats"],
): TextureCapabilities => ({ capabilityKey, compressedFormats });

const request = async (
  name: string,
  textureCapabilities = capabilities("rgba", []),
): Promise<DecodeTileRequest> => ({
  content: Uint8Array.from(await fixture(name)).buffer,
  contentUrl: `https://fixture.invalid/content/${name}`,
  revision: "fixture-revision",
  accumulatedTransform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
  tilesetToScene: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
  textureCapabilities,
});

const raster = async (): Promise<{
  rgba: Uint8Array;
  width: number;
  height: number;
}> => ({ rgba: new Uint8Array(16 * 16 * 4).fill(127), width: 16, height: 16 });

const testOptions: DecodeTileOptions = { decodeRasterImage: raster };

const authoredAlphaGlb = async (): Promise<ArrayBuffer> => {
  const document = new Document();
  const buffer = document.createBuffer("alpha-buffer");
  const positions = document
    .createAccessor("positions")
    .setType("VEC3")
    .setArray(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]))
    .setBuffer(buffer);
  const indices = document
    .createAccessor("indices")
    .setType("SCALAR")
    .setArray(new Uint16Array([0, 1, 2]))
    .setBuffer(buffer);
  const mesh = document.createMesh("alpha-modes");
  for (const [name, alphaMode, alphaCutoff] of [
    ["masked", "MASK", 0.37],
    ["blended", "BLEND", 0.19],
    ["opaque", "OPAQUE", 0.73],
  ] as const) {
    const material = document
      .createMaterial(name)
      .setAlphaMode(alphaMode)
      .setAlphaCutoff(alphaCutoff);
    mesh.addPrimitive(
      document
        .createPrimitive()
        .setName(name)
        .setAttribute("POSITION", positions)
        .setIndices(indices)
        .setMaterial(material),
    );
  }
  document
    .createScene("alpha-scene")
    .addChild(document.createNode("alpha-node").setMesh(mesh));
  const bytes = await new NodeIO().writeBinary(document);
  return bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  ) as ArrayBuffer;
};

const transformedPointGlb = async (): Promise<ArrayBuffer> => {
  const document = new Document();
  const buffer = document.createBuffer("transform-buffer");
  const positions = document
    .createAccessor("positions")
    .setType("VEC3")
    .setArray(new Float32Array([1, 2, 3, 1, 2, 3, 1, 2, 3]))
    .setBuffer(buffer);
  const primitive = document
    .createPrimitive()
    .setAttribute("POSITION", positions);
  const mesh = document.createMesh("mesh").addPrimitive(primitive);
  const node = document
    .createNode("translated")
    .setTranslation([10, 20, 30])
    .setMesh(mesh);
  document.createScene("scene").addChild(node);
  const bytes = await new NodeIO().writeBinary(document);
  return bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  ) as ArrayBuffer;
};

const embeddedGltf = ({
  bufferUri,
  imageUri = "DATA:IMAGE/PNG;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M/wHwAF/gL+X7S6AAAAAElFTkSuQmCC",
  texCoord = 1,
  sampler,
}: {
  bufferUri?: string;
  imageUri?: string;
  texCoord?: number;
  sampler?: Record<string, number>;
} = {}): { content: ArrayBuffer; binary: ArrayBuffer } => {
  const binary = new ArrayBuffer(90);
  new Float32Array(binary, 0, 9).set([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  new Float32Array(binary, 36, 6).set([0, 0, 0.5, 0, 0, 0.5]);
  new Float32Array(binary, 60, 6).set([0.2, 0.3, 0.7, 0.4, 0.4, 0.8]);
  new Uint16Array(binary, 84, 3).set([0, 1, 2]);
  const uri =
    bufferUri ??
    `data:application/octet-stream;base64,${Buffer.from(binary).toString("base64")}`;
  const json = {
    asset: { version: "2.0" },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [
      {
        primitives: [
          {
            attributes: { POSITION: 0, TEXCOORD_0: 1, TEXCOORD_1: 2 },
            indices: 3,
            material: 0,
          },
        ],
      },
    ],
    buffers: [{ uri, byteLength: binary.byteLength }],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: 36 },
      { buffer: 0, byteOffset: 36, byteLength: 24 },
      { buffer: 0, byteOffset: 60, byteLength: 24 },
      { buffer: 0, byteOffset: 84, byteLength: 6 },
    ],
    accessors: [
      { bufferView: 0, componentType: 5126, count: 3, type: "VEC3" },
      { bufferView: 1, componentType: 5126, count: 3, type: "VEC2" },
      { bufferView: 2, componentType: 5126, count: 3, type: "VEC2" },
      { bufferView: 3, componentType: 5123, count: 3, type: "SCALAR" },
    ],
    images: [{ uri: imageUri }],
    textures: [{ source: 0, ...(sampler ? { sampler: 0 } : {}) }],
    ...(sampler ? { samplers: [sampler] } : {}),
    materials: [
      { pbrMetallicRoughness: { baseColorTexture: { index: 0, texCoord } } },
    ],
  };
  const encoded = new TextEncoder().encode(JSON.stringify(json));
  return {
    content: encoded.buffer.slice(
      encoded.byteOffset,
      encoded.byteOffset + encoded.byteLength,
    ),
    binary,
  };
};

describe("decoded tile contract", () => {
  it("rejects glTF sampler enums before they reach a renderer backend", async () => {
    const embedded = embeddedGltf({ sampler: { magFilter: 7 } });
    await expect(
      decodeTileContent(
        {
          ...(await request("level-0-root.glb")),
          content: embedded.content,
          contentUrl: "https://fixture.invalid/content/tile.gltf",
          dependencyRootUrl: "https://fixture.invalid/content/",
        },
        testOptions,
      ),
    ).rejects.toThrow(/invalid magFilter/);
  });

  it("applies glTF nodes, then mandatory Y-up correction, then tile and scene transforms", async () => {
    const result = await decodeTileContent(
      {
        ...(await request("level-0-root.glb")),
        content: await transformedPointGlb(),
        accumulatedTransform: [
          1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 100, 200, 300, 1,
        ],
        tilesetToScene: [
          1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1000, 2000, 3000, 1,
        ],
      },
      testOptions,
    );
    const primitive = result.primitives[0]!;
    const absolute = primitive.positions
      .slice(0, 3)
      .map((value, axis) => value + result.origin[axis]!);
    // [1,2,3] -> node [11,22,33] -> Y-up correction [11,-33,22]
    // -> tile [111,167,322] -> scene [1111,2167,3322].
    expect([...absolute]).toEqual([1111, 2167, 3322]);
  });

  it("chooses only the explicit native-format priority, never auto", () => {
    expect(
      capabilityTarget(
        capabilities("all", ["s3tc-dxt5", "etc2-rgba8", "bc7", "astc-4x4"]),
      ),
    ).toBe("astc-4x4");
    expect(capabilityTarget(capabilities("bc", ["s3tc-dxt5", "bc7"]))).toBe(
      "bc7",
    );
    expect(capabilityTarget(capabilities("etc", ["etc2-rgba8"]))).toBe(
      "etc2-rgba8",
    );
    expect(capabilityTarget(capabilities("s3tc", ["s3tc-dxt5"]))).toBe(
      "s3tc-dxt5",
    );
    expect(capabilityTarget(capabilities("none", []))).toBe("rgba");
  });

  it("normalizes KTX orientation in owned UVs without a pixel-store flip flag", () => {
    const coordinates = new Float32Array([0, 0.25, 1, 0.75]);
    normalizeTextureCoordinates(coordinates, "lu");
    expect([...coordinates]).toEqual([1, 0.75, 0, 0.25]);
    expect(() => normalizeTextureCoordinates(coordinates, "xy")).toThrow(
      /orientation/i,
    );
  });

  it("decodes uncompressed multi-primitive GLB into owned, plain data", async () => {
    const result = await decodeTileContent(
      await request("level-0-root.glb"),
      testOptions,
    );
    expect(result.primitives).toHaveLength(2);
    expect(
      result.primitives.every(
        (primitive) => primitive.positions instanceof Float32Array,
      ),
    ).toBe(true);
    expect(
      result.primitives.every(
        (primitive) => primitive.indices instanceof Uint16Array,
      ),
    ).toBe(true);
    expect(
      result.primitives.map((primitive) => primitive.material.raw),
    ).toEqual([
      expect.objectContaining({ version: 1, kind: "gltf-material" }),
      expect.objectContaining({ version: 1, kind: "gltf-material" }),
    ]);
    const absolutePoints = result.primitives.map((primitive) =>
      Array.from({ length: primitive.positions.length / 3 }, (_, index) =>
        [0, 1, 2].map(
          (axis) =>
            primitive.positions[index * 3 + axis]! + result.origin[axis]!,
        ),
      ),
    );
    expect(absolutePoints[0]?.map((point) => point[2])).toEqual([0, 0, 0, 0]);
    expect(absolutePoints[1]?.map((point) => point[2])).toEqual([3, 3, 3, 3]);
    for (const points of absolutePoints) {
      expect(new Set(points.map((point) => point[0])).size).toBe(2);
      expect(new Set(points.map((point) => point[1])).size).toBe(2);
    }
    for (const primitive of result.primitives) {
      expect(primitive.normals && [...primitive.normals]).toEqual([
        0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
      ]);
    }
    expect(() => structuredClone(result)).not.toThrow();
    expect(JSON.stringify(result.primitives[0]?.material.raw)).not.toContain(
      "loaderData",
    );

    const transfers = buildTransferList(result);
    expect(new Set(transfers).size).toBe(transfers.length);
    const transferred = structuredClone(result, { transfer: transfers });
    expect(transferred.primitives).toHaveLength(2);
    expect(result.primitives[0]?.positions.byteLength).toBe(0);
  });

  it("preserves authored alpha modes and cutoffs as typed material data", async () => {
    const result = await decodeTileContent(
      {
        ...(await request("level-0-root.glb")),
        content: await authoredAlphaGlb(),
        contentUrl: "https://fixture.invalid/content/alpha-modes.glb",
      },
      testOptions,
    );

    expect(result.primitives.map(({ material }) => material.raw)).toEqual([
      expect.objectContaining({
        name: "masked",
        alphaMode: "MASK",
        alphaCutoff: 0.37,
      }),
      expect.objectContaining({
        name: "blended",
        alphaMode: "BLEND",
        alphaCutoff: 0.5,
      }),
      expect.objectContaining({
        name: "opaque",
        alphaMode: "OPAQUE",
        alphaCutoff: 0.5,
      }),
    ]);
  });

  it("decodes Draco geometry without retaining the parsed asset graph", async () => {
    const result = await decodeTileContent(
      await request(`${FIXTURE_CODEC_TILES.draco}.glb`),
      {
        ...testOptions,
        modules: { draco3d },
      },
    );
    expect(result.primitives).toHaveLength(2);
    expect(result.primitives[0]?.positions).toHaveLength(12);
    expect(result.primitives[0]?.indices).toHaveLength(6);
    expect(() => structuredClone(result)).not.toThrow();
  });

  it("fails closed instead of falling back to remote codec defaults", async () => {
    await expect(
      decodeTileContent(
        await request(`${FIXTURE_CODEC_TILES.draco}.glb`),
        testOptions,
      ),
    ).rejects.toThrow(/Draco.*injected/i);
    await expect(
      decodeTileContent(await request(`${FIXTURE_CODEC_TILES.ktx2}.glb`)),
    ).rejects.toThrow(/KTX2.*injected/i);
  });

  it("deduplicates a shared RGBA texture and accounts actual retained bytes", async () => {
    const result = await decodeTileContent(
      await request("level-0-root.glb"),
      testOptions,
    );
    const first = result.primitives[0]?.material.baseColorTexture;
    const second = result.primitives[1]?.material.baseColorTexture;
    expect(first?.kind).toBe("rgba");
    expect(first).toBe(second);
    expect(
      first && first.kind === "rgba" ? [first.width, first.height] : [],
    ).toEqual([16, 16]);
    expect(result.byteEstimate.textures).toBe(16 * 16 * 4);
    const uniqueGeometryBuffers = new Set<ArrayBuffer>();
    for (const primitive of result.primitives) {
      for (const array of [
        primitive.positions,
        primitive.normals,
        primitive.uvs,
        primitive.indices,
      ]) {
        if (array) uniqueGeometryBuffers.add(array.buffer as ArrayBuffer);
      }
    }
    expect(result.byteEstimate.geometry).toBe(
      [...uniqueGeometryBuffers].reduce(
        (sum, buffer) => sum + buffer.byteLength,
        0,
      ),
    );
  });

  it("keys decoded representations by revision, URL, and capability key", () => {
    const base = {
      revision: "r1",
      contentUrl: "https://fixture.invalid/a.glb",
    };
    expect(buildDecodeCacheKey({ ...base, capabilityKey: "astc" })).not.toBe(
      buildDecodeCacheKey({ ...base, capabilityKey: "rgba" }),
    );
    expect(buildDecodeCacheKey({ ...base, capabilityKey: "astc" })).not.toBe(
      buildDecodeCacheKey({ ...base, revision: "r2", capabilityKey: "astc" }),
    );
  });

  it("resolves only injectable same-root external glTF dependencies", () => {
    const base = {
      content: new ArrayBuffer(0),
      contentUrl: "https://fixture.invalid/root/models/model.gltf",
      dependencyRootUrl: "https://fixture.invalid/root/",
      revision: "r1",
      accumulatedTransform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
      tilesetToScene: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
      textureCapabilities: capabilities("rgba", []),
    } satisfies DecodeTileRequest;
    expect(resolveGltfDependencyUrl(base, "buffers/mesh.bin")).toBe(
      "https://fixture.invalid/root/models/buffers/mesh.bin",
    );
    expect(resolveGltfDependencyUrl(base, "mesh%2Ebin")).toBe(
      "https://fixture.invalid/root/models/mesh.bin",
    );
    expect(resolveGltfDependencyUrl(base, "%2e/mesh.bin")).toBe(
      "https://fixture.invalid/root/models/mesh.bin",
    );
    expect(resolveGltfDependencyUrl(base, "%252e/mesh.bin")).toBe(
      "https://fixture.invalid/root/models/mesh.bin",
    );
    expect(resolveGltfDependencyUrl(base, "buffers//%252e/mesh.bin")).toBe(
      "https://fixture.invalid/root/models/buffers/mesh.bin",
    );
    for (const value of [
      "https://other.invalid/file.bin",
      "//other.invalid/file.bin",
      "/absolute.bin",
      "../buffers/mesh.bin",
      "../../escape.bin",
      "%2e%2e/escape.bin",
      "..%2fescape.bin",
      "..\\escape.bin",
      "mesh%252fsecret.bin",
      "%2568%2574%2574%2570%2573%253Aevil.invalid/file.bin",
      "%252e%252e/escape.bin",
      "mesh%00.bin",
      "mesh%ZZ.bin",
    ]) {
      expect(() => resolveGltfDependencyUrl(base, value)).toThrow();
    }
  });

  it("decodes declared data URI buffers and images locally and selects the material UV set", async () => {
    const authored = embeddedGltf();
    const { content } = embeddedGltf({
      bufferUri: `DATA:APPLICATION/OCTET-STREAM;base64,${Buffer.from(
        authored.binary,
      ).toString("base64")}`,
    });
    const fetched: string[] = [];
    let decodedImage: { bytes: Uint8Array; mimeType: string } | undefined;
    const result = await decodeTileContent(
      {
        ...(await request("level-0-root.glb")),
        content,
        contentUrl: "https://fixture.invalid/content/embedded.gltf",
      },
      {
        fetchDependency: async (url) => {
          fetched.push(url);
          throw new Error("embedded dependencies must not be fetched");
        },
        decodeRasterImage: async (bytes, mimeType) => {
          decodedImage = { bytes: bytes.slice(), mimeType };
          return raster();
        },
      },
    );

    expect(fetched).toEqual([]);
    expect(decodedImage?.mimeType).toBe("image/png");
    expect(Array.from(decodedImage!.bytes.slice(0, 8))).toEqual([
      137, 80, 78, 71, 13, 10, 26, 10,
    ]);
    expect(result.primitives[0]?.material.raw.baseColorTexture?.texCoord).toBe(
      1,
    );
    expect(result.primitives[0]?.uvs && [...result.primitives[0].uvs]).toEqual(
      expect.arrayContaining([
        expect.closeTo(0.2),
        expect.closeTo(0.3),
        expect.closeTo(0.7),
        expect.closeTo(0.4),
        expect.closeTo(0.4),
        expect.closeTo(0.8),
      ]),
    );
  });

  it("fetches percent-encoded external dependencies at the validated canonical URL", async () => {
    const { content, binary } = embeddedGltf({
      bufferUri: "%252e/mesh%252Ebin",
    });
    const fetched: string[] = [];
    const result = await decodeTileContent(
      {
        ...(await request("level-0-root.glb")),
        content,
        contentUrl: "https://fixture.invalid/root/models/model.gltf",
        dependencyRootUrl: "https://fixture.invalid/root/",
      },
      {
        ...testOptions,
        fetchDependency: async (url) => {
          fetched.push(url);
          return binary.slice(0);
        },
      },
    );

    expect(result.primitives).toHaveLength(1);
    expect(fetched).toEqual(["https://fixture.invalid/root/models/mesh.bin"]);
  });

  it("rejects malformed embedded dependency data before parsing", async () => {
    const { content } = embeddedGltf({
      bufferUri: "data:application/octet-stream;base64,%%%",
    });
    await expect(
      decodeTileContent(
        {
          ...(await request("level-0-root.glb")),
          content,
          contentUrl: "https://fixture.invalid/content/malformed.gltf",
        },
        testOptions,
      ),
    ).rejects.toThrow(/data URI.*base64/i);
  });

  it.each([
    ["astc-4x4", "astc-4x4", "astc-4x4"],
    ["bc7", "bc7", "bc7-m5"],
    ["etc2", "etc2-rgba8", "etc2"],
    ["s3tc", "s3tc-dxt5", "bc3"],
  ] as const)(
    "keeps all logical KTX2 mips for %s and counts the selected blocks",
    async (key, format, loadersFormat) => {
      const seenFormats: string[] = [];
      const result = await decodeTileContent(
        await request(
          `${FIXTURE_CODEC_TILES.ktx2}.glb`,
          capabilities(key, [format]),
        ),
        {
          transcodeBasis: async (_data, selectedFormat, metadata) => {
            seenFormats.push(selectedFormat);
            return metadata.levels.map((_level, level) => ({
              // Reproduce the loader's compressed-block dimension clamping.
              width: Math.max(4, metadata.width >> level),
              height: Math.max(4, metadata.height >> level),
              data: new Uint8Array(
                Math.max(1, Math.ceil((metadata.width >> level) / 4)) *
                  Math.max(1, Math.ceil((metadata.height >> level) / 4)) *
                  16,
              ),
              compressed: true,
            }));
          },
        },
      );
      expect(seenFormats).toEqual([loadersFormat]);
      const texture = result.primitives[0]?.material.baseColorTexture;
      expect(texture?.kind).toBe("compressed");
      if (texture?.kind !== "compressed")
        throw new Error("expected compressed");
      expect(texture.format).toBe(format);
      expect(
        texture.levels.map(({ width, height }) => [width, height]),
      ).toEqual([
        [16, 16],
        [8, 8],
        [4, 4],
        [2, 2],
        [1, 1],
      ]);
      expect(result.primitives[1]?.material.baseColorTexture).toBe(texture);
      expect(result.byteEstimate.textures).toBe(
        texture.levels.reduce((sum, level) => sum + level.data.byteLength, 0),
      );
      expect("flipY" in texture).toBe(false);
      expect(
        result.primitives[0]?.uvs && [...result.primitives[0].uvs],
      ).toEqual([0, 0, 1, 0, 1, 1, 0, 1]);
    },
  );

  it("uses one full-resolution RGBA representation when compression is unavailable", async () => {
    const result = await decodeTileContent(
      await request(
        `${FIXTURE_CODEC_TILES.ktx2}.glb`,
        capabilities("rgba", []),
      ),
      {
        transcodeBasis: async (_data, selectedFormat, metadata) => {
          expect(selectedFormat).toBe("rgba32");
          return metadata.levels.map((_level, level) => ({
            width: Math.max(4, metadata.width >> level),
            height: Math.max(4, metadata.height >> level),
            data: new Uint8Array(
              Math.max(1, metadata.width >> level) *
                Math.max(1, metadata.height >> level) *
                4,
            ),
            compressed: false,
          }));
        },
      },
    );
    const texture = result.primitives[0]?.material.baseColorTexture;
    expect(texture?.kind).toBe("rgba");
    expect(texture?.kind === "rgba" ? texture.rgba.byteLength : 0).toBe(1024);
    expect(result.byteEstimate.textures).toBe(1024);
  });

  it("separates Basis runtime initialization from per-texture transcode timing", async () => {
    const ticks = [0, 10, 14, 20, 27, 30];
    const result = await decodeTileContent(
      await request(
        `${FIXTURE_CODEC_TILES.ktx2}.glb`,
        capabilities("bc7", ["bc7"]),
      ),
      {
        now: () => ticks.shift() ?? 30,
        basisRuntimeProvider: async () => ({
          basisEncoder: {},
          initializedNow: true,
        }),
        transcodeBasis: async (_data, _format, metadata) =>
          metadata.levels.map((_level, level) => ({
            width: Math.max(4, metadata.width >> level),
            height: Math.max(4, metadata.height >> level),
            data: new Uint8Array(
              Math.max(1, Math.ceil((metadata.width >> level) / 4)) *
                Math.max(1, Math.ceil((metadata.height >> level) / 4)) *
                16,
            ),
            compressed: true,
          })),
      },
    );

    expect(result.diagnostics).toMatchObject({
      totalDecodeMs: 30,
      basisRuntimeInitializationMs: 4,
      basisTranscodeMs: 7,
      basisTranscodeSamplesMs: [7],
      basisTextures: 1,
      basisTarget: "bc7",
    });
  });
});
