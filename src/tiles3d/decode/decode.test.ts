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
  ecefToScene: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
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

describe("decoded tile contract", () => {
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
      ecefToScene: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
      textureCapabilities: capabilities("rgba", []),
    } satisfies DecodeTileRequest;
    expect(resolveGltfDependencyUrl(base, "../buffers/mesh.bin")).toBe(
      "https://fixture.invalid/root/buffers/mesh.bin",
    );
    for (const value of [
      "https://other.invalid/file.bin",
      "//other.invalid/file.bin",
      "/absolute.bin",
      "../../escape.bin",
      "%2e%2e/escape.bin",
      "..%2fescape.bin",
      "..\\escape.bin",
    ]) {
      expect(() => resolveGltfDependencyUrl(base, value)).toThrow();
    }
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
