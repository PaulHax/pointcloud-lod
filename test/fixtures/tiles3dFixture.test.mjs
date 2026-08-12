import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { readdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

import { afterAll, describe, expect, it } from "vitest";

import {
  FIXTURE_CODEC_TILES,
  FIXTURE_SEED,
  createTiles3dFixtureTileset,
  generateTiles3dFixture,
} from "../../scripts/generateTiles3dFixture.mjs";

const execute = promisify(execFile);
const temporaryRoot = resolve(
  tmpdir(),
  `pointcloud-lod-tiles3d-${process.pid}-${Date.now()}`,
);

afterAll(async () => {
  await rm(temporaryRoot, { recursive: true, force: true });
});

async function fileManifest(root, relative = "") {
  const directory = resolve(root, relative);
  const entries = await readdir(directory, { withFileTypes: true });
  const manifest = [];
  for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
    const path = join(relative, entry.name);
    if (entry.isDirectory()) {
      manifest.push(...(await fileManifest(root, path)));
    } else {
      const bytes = await readFile(resolve(root, path));
      manifest.push({
        path: path.replaceAll("\\", "/"),
        bytes: bytes.byteLength,
        sha256: createHash("sha256").update(bytes).digest("hex"),
      });
    }
  }
  return manifest;
}

function walkTiles(tile, depth = 0, parentError = Number.POSITIVE_INFINITY) {
  expect(tile.refine).toBe("REPLACE");
  expect(tile.boundingVolume).toEqual({
    box: expect.arrayContaining([]),
  });
  expect(tile.boundingVolume.box).toHaveLength(12);
  expect(tile.boundingVolume.box.every(Number.isFinite)).toBe(true);
  expect(tile.geometricError).toBeGreaterThanOrEqual(0);
  expect(tile.geometricError).toBeLessThan(parentError);
  expect(tile.transform).toHaveLength(16);
  expect(tile.transform.every(Number.isFinite)).toBe(true);
  expect(tile.content.uri).toMatch(/^content\/[a-z0-9-]+\.glb$/);

  const children = tile.children ?? [];
  return [
    { tile, depth },
    ...children.flatMap((child) =>
      walkTiles(child, depth + 1, tile.geometricError),
    ),
  ];
}

function parseGlb(bytes) {
  expect(bytes.subarray(0, 4).toString("ascii")).toBe("glTF");
  expect(bytes.readUInt32LE(4)).toBe(2);
  expect(bytes.readUInt32LE(8)).toBe(bytes.byteLength);

  let offset = 12;
  let json;
  let binary;
  while (offset < bytes.byteLength) {
    const length = bytes.readUInt32LE(offset);
    const type = bytes.readUInt32LE(offset + 4);
    const data = bytes.subarray(offset + 8, offset + 8 + length);
    if (type === 0x4e4f534a) json = JSON.parse(data.toString("utf8"));
    if (type === 0x004e4942) binary = data;
    offset += 8 + length;
  }
  expect(json).toBeDefined();
  expect(binary).toBeDefined();
  return { json, binary };
}

function embeddedImage(parsed, image) {
  const view = parsed.json.bufferViews[image.bufferView];
  const start = view.byteOffset ?? 0;
  return parsed.binary.subarray(start, start + view.byteLength);
}

function expectCompleteKtx2MipChain(bytes) {
  expect([...bytes.subarray(0, 12)]).toEqual([
    0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a,
  ]);
  const width = bytes.readUInt32LE(20);
  const height = bytes.readUInt32LE(24);
  const levelCount = bytes.readUInt32LE(40);
  expect(width).toBe(16);
  expect(height).toBe(16);
  expect(levelCount).toBe(1 + Math.floor(Math.log2(Math.max(width, height))));

  for (let level = 0; level < levelCount; level += 1) {
    const entry = 80 + level * 24;
    const byteOffset = Number(bytes.readBigUInt64LE(entry));
    const byteLength = Number(bytes.readBigUInt64LE(entry + 8));
    expect(byteLength).toBeGreaterThan(0);
    expect(byteOffset + byteLength).toBeLessThanOrEqual(bytes.byteLength);
  }
}

async function expectStrictTilesetSchema(tilesetFile, reportDirectory) {
  const optionsFile = resolve(reportDirectory, "validator-options.json");
  const reportFile = resolve(reportDirectory, "validator-report.json");
  await writeFile(optionsFile, '{"validateContentData":false}\n', "utf8");

  const validator = fileURLToPath(
    new URL("../../node_modules/.bin/3d-tiles-validator", import.meta.url),
  );
  await execute(
    validator,
    [
      "--tilesetFile",
      tilesetFile,
      "--optionsFile",
      optionsFile,
      "--reportFile",
      reportFile,
    ],
    { cwd: dirname(tilesetFile) },
  );
  const report = JSON.parse(await readFile(reportFile, "utf8"));
  expect(report).toMatchObject({
    numErrors: 0,
    numWarnings: 0,
    numInfos: 0,
  });
  expect(report.issues ?? []).toEqual([]);
}

describe("the generated 3D Tiles 1.1 fixture", () => {
  it("returns a fresh independent hierarchy document", () => {
    const first = createTiles3dFixtureTileset();
    first.root.content.uri = "mutated.glb";
    first.root.children.pop();

    const second = createTiles3dFixtureTileset();
    expect(second.root.content.uri).toBe("content/level-0-root.glb");
    expect(second.root.children).toHaveLength(4);
  });

  it("is deterministic and contains every vertical-slice content variant", async () => {
    const first = await generateTiles3dFixture(resolve(temporaryRoot, "first"));
    const second = await generateTiles3dFixture(
      resolve(temporaryRoot, "second"),
    );
    expect(await fileManifest(second)).toEqual(await fileManifest(first));

    const tilesetFile = resolve(first, "tileset.json");
    await expectStrictTilesetSchema(tilesetFile, temporaryRoot);
    const tileset = JSON.parse(await readFile(tilesetFile, "utf8"));
    expect(tileset.asset).toEqual({
      version: "1.1",
      tilesetVersion: `seed-${FIXTURE_SEED.toString(16)}`,
    });
    expect(tileset.geometricError).toBe(48);

    const walked = walkTiles(tileset.root);
    expect(Math.max(...walked.map(({ depth }) => depth))).toBe(2);
    expect(walked.filter(({ depth }) => depth === 0)).toHaveLength(1);
    expect(walked.filter(({ depth }) => depth === 1)).toHaveLength(4);
    expect(walked.filter(({ depth }) => depth === 2)).toHaveLength(8);
    expect(new Set(walked.map(({ tile }) => tile.content.uri)).size).toBe(
      walked.length,
    );

    const rootTranslation = tileset.root.transform.slice(12, 15);
    expect(Math.hypot(...rootTranslation)).toBeGreaterThan(6_300_000);
    for (const { tile, depth } of walked) {
      if (depth === 0) continue;
      expect(Math.hypot(...tile.transform.slice(12, 15))).toBeGreaterThan(0);
    }

    let dracoTiles = 0;
    let ktx2Tiles = 0;
    let uncompressedTiles = 0;
    for (const { tile } of walked) {
      const parsed = parseGlb(await readFile(resolve(first, tile.content.uri)));
      const extensions = new Set(parsed.json.extensionsUsed ?? []);
      const meshPrimitives = parsed.json.meshes.flatMap(
        (mesh) => mesh.primitives,
      );
      expect(meshPrimitives).toHaveLength(2);
      for (const primitive of meshPrimitives) {
        const material = parsed.json.materials[primitive.material];
        expect(material.pbrMetallicRoughness.baseColorTexture.index).toBeTypeOf(
          "number",
        );
      }

      if (extensions.has("KHR_draco_mesh_compression")) {
        dracoTiles += 1;
        expect(tile.content.uri).toBe(
          `content/${FIXTURE_CODEC_TILES.draco}.glb`,
        );
        expect(parsed.json.extensionsRequired).toContain(
          "KHR_draco_mesh_compression",
        );
        for (const primitive of meshPrimitives) {
          expect(primitive.extensions.KHR_draco_mesh_compression).toBeDefined();
        }
      } else if (extensions.has("KHR_texture_basisu")) {
        ktx2Tiles += 1;
        expect(tile.content.uri).toBe(
          `content/${FIXTURE_CODEC_TILES.ktx2}.glb`,
        );
        expect(parsed.json.extensionsRequired).toContain("KHR_texture_basisu");
        const texture = parsed.json.textures[0];
        const source = texture.extensions.KHR_texture_basisu.source;
        const image = parsed.json.images[source];
        expect(image.mimeType).toBe("image/ktx2");
        expectCompleteKtx2MipChain(embeddedImage(parsed, image));
      } else {
        uncompressedTiles += 1;
      }
    }

    expect(dracoTiles).toBe(1);
    expect(ktx2Tiles).toBe(1);
    expect(uncompressedTiles).toBe(11);
  }, 120_000);
});
