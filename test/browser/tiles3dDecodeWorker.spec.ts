import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

import { chromium, type Browser } from "playwright";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

import {
  FIXTURE_CODEC_TILES,
  generateTiles3dFixture,
} from "../../scripts/generateTiles3dFixture.mjs";
import { startStaticServer, type StaticServer } from "./server";

let browser: Browser;
let server: StaticServer;
let fixtureDirectory = "";

beforeAll(async () => {
  fixtureDirectory = await mkdtemp(
    resolve(tmpdir(), "tiles3d-worker-browser-"),
  );
  await generateTiles3dFixture(fixtureDirectory);
  await writeFile(
    resolve(fixtureDirectory, "index.html"),
    "<!doctype html><title>decode worker test</title>",
    "utf8",
  );
  server = await startStaticServer({
    "/dist": resolve("dist"),
    "/fixture": fixtureDirectory,
    "/draco": resolve("node_modules/@loaders.gl/draco/dist/libs"),
    "/basis": resolve("node_modules/@loaders.gl/textures/dist/libs"),
  });
  browser = await chromium.launch({ headless: true });
}, 60_000);

afterAll(async () => {
  await browser?.close();
  await server?.close();
  if (fixtureDirectory)
    await rm(fixtureDirectory, { recursive: true, force: true });
});

describe("classic 3D Tiles decode worker", () => {
  it.each([
    ["level-0-root.glb", "rgba", [], "rgba"],
    [`${FIXTURE_CODEC_TILES.draco}.glb`, "rgba", [], "rgba"],
    [`${FIXTURE_CODEC_TILES.ktx2}.glb`, "astc", ["astc-4x4"], "astc-4x4"],
    [`${FIXTURE_CODEC_TILES.ktx2}.glb`, "bc7", ["bc7"], "bc7"],
    [`${FIXTURE_CODEC_TILES.ktx2}.glb`, "etc2", ["etc2-rgba8"], "etc2-rgba8"],
    [`${FIXTURE_CODEC_TILES.ktx2}.glb`, "s3tc", ["s3tc-dxt5"], "s3tc-dxt5"],
    [`${FIXTURE_CODEC_TILES.ktx2}.glb`, "fallback", [], "rgba"],
  ] as const)(
    "round-trips generated %s for %s",
    async (file, capabilityKey, compressedFormats, expectedFormat) => {
      const page = await browser.newPage();
      const requestedUrls: string[] = [];
      page.on("request", (request) => requestedUrls.push(request.url()));
      try {
        await page.goto(`${server.origin}/fixture/index.html`);
        const result = await page.evaluate(
          async ({ origin, file, capabilityKey, compressedFormats }) => {
            const response = await fetch(`${origin}/fixture/content/${file}`);
            const content = await response.arrayBuffer();
            const worker = new Worker(
              `${origin}/dist/tiles3dDecodeWorker.classic.js`,
            );
            try {
              const message = await new Promise<unknown>((resolve, reject) => {
                const timeout = setTimeout(
                  () => reject(new Error("decode worker timed out")),
                  30_000,
                );
                worker.onerror = (event) => {
                  clearTimeout(timeout);
                  reject(new Error(event.message));
                };
                worker.onmessage = (event) => {
                  clearTimeout(timeout);
                  resolve(event.data);
                };
                worker.postMessage(
                  {
                    kind: "decode",
                    jobId: 7,
                    generation: 3,
                    request: {
                      content,
                      contentUrl: `${origin}/fixture/content/${file}`,
                      dependencyRootUrl: `${origin}/fixture/`,
                      revision: "browser-fixture",
                      accumulatedTransform: [
                        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                      ],
                      tilesetToScene: [
                        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                      ],
                      textureCapabilities: { capabilityKey, compressedFormats },
                      wasm: {
                        draco: {
                          wrapperUrl: `${origin}/draco/draco_wasm_wrapper.js`,
                          wasmUrl: `${origin}/draco/draco_decoder.wasm`,
                        },
                        basis: {
                          encoderUrl: `${origin}/basis/basis_encoder.js`,
                          wasmUrl: `${origin}/basis/basis_encoder.wasm`,
                        },
                      },
                    },
                  },
                  [content],
                );
              });
              const decoded = message as {
                kind: string;
                jobId: number;
                generation: number;
                error?: string;
                result?: {
                  primitives: Array<{
                    positions: Float32Array;
                    material: {
                      baseColorTexture?: {
                        kind: string;
                        format?: string;
                        levels?: Array<{
                          width: number;
                          height: number;
                          data: Uint8Array;
                        }>;
                        rgba?: Uint8Array;
                        flipY?: boolean;
                      };
                    };
                  }>;
                  origin: number[];
                  byteEstimate: { geometry: number; textures: number };
                };
              };
              if (decoded.kind !== "complete" || !decoded.result) {
                throw new Error(decoded.error ?? "worker returned no result");
              }
              const texture =
                decoded.result.primitives[0]?.material.baseColorTexture;
              return {
                kind: decoded.kind,
                jobId: decoded.jobId,
                generation: decoded.generation,
                primitiveCount: decoded.result.primitives.length,
                positionsAreFloat32: decoded.result.primitives.every(
                  (primitive) => primitive.positions instanceof Float32Array,
                ),
                origin: decoded.result.origin,
                geometryBytes: decoded.result.byteEstimate.geometry,
                textureBytes: decoded.result.byteEstimate.textures,
                textureKind: texture?.kind,
                textureFormat: texture?.format,
                mipDimensions: texture?.levels?.map((level) => [
                  level.width,
                  level.height,
                ]),
                levelByteSum: texture?.levels?.reduce(
                  (sum, level) => sum + level.data.byteLength,
                  0,
                ),
                rgbaBytes: texture?.rgba?.byteLength,
                hasFlipY: texture ? "flipY" in texture : false,
                sharedTexture:
                  texture ===
                  decoded.result.primitives[1]?.material.baseColorTexture,
              };
            } finally {
              worker.terminate();
            }
          },
          { origin: server.origin, file, capabilityKey, compressedFormats },
        );
        expect(result).toMatchObject({
          kind: "complete",
          jobId: 7,
          generation: 3,
          primitiveCount: 2,
          positionsAreFloat32: true,
          origin: [expect.any(Number), expect.any(Number), expect.any(Number)],
        });
        expect(result.geometryBytes).toBeGreaterThan(0);
        expect(result.textureBytes).toBeGreaterThan(0);
        expect(result.sharedTexture).toBe(true);
        expect(result.hasFlipY).toBe(false);
        if (expectedFormat === "rgba") {
          expect(result.textureKind).toBe("rgba");
          expect(result.textureFormat).toBeUndefined();
          expect(result.rgbaBytes).toBe(16 * 16 * 4);
          expect(result.textureBytes).toBe(16 * 16 * 4);
        } else {
          expect(result.textureKind).toBe("compressed");
          expect(result.textureFormat).toBe(expectedFormat);
          expect(result.mipDimensions).toEqual([
            [16, 16],
            [8, 8],
            [4, 4],
            [2, 2],
            [1, 1],
          ]);
          expect(result.levelByteSum).toBe(result.textureBytes);
        }
        expect(
          requestedUrls.every((url) => url.startsWith(`${server.origin}/`)),
          requestedUrls.join("\n"),
        ).toBe(true);
      } finally {
        await page.close();
      }
    },
  );
});
