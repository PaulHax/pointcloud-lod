import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

import { chromium, type Browser } from "playwright";
import sharp from "sharp";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { startStaticServer, type StaticServer } from "./server";

type QuadrantSample = {
  readonly name: string;
  readonly expectedRgb: readonly [number, number, number];
  readonly x: number;
  readonly y: number;
};

type CesiumProof = {
  readonly failures: readonly string[];
  readonly samples: readonly QuadrantSample[];
  readonly commands: number;
  readonly readyTiles: number;
  readonly totalTiles: number;
};

const fixtureRoot = resolve("test/fixtures/tiles3d-implicit");
let browser: Browser;
let server: StaticServer;

beforeAll(async () => {
  server = await startStaticServer({
    "/proof": resolve("test/browser/cesiumImplicitPage"),
    "/fixture": fixtureRoot,
    "/cesium": resolve("node_modules/cesium/Build/CesiumUnminified"),
  });
  browser = await chromium.launch({ headless: true });
}, 60_000);

afterAll(async () => {
  await browser?.close();
  await server?.close();
});

const medianRgb = async (
  screenshot: Buffer,
  x: number,
  y: number,
): Promise<readonly [number, number, number]> => {
  const decoded = await sharp(screenshot)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const channels = [[], [], []] as number[][];
  for (let offsetY = -2; offsetY <= 2; offsetY += 1) {
    for (let offsetX = -2; offsetX <= 2; offsetX += 1) {
      const pixelX = Math.round(x) + offsetX;
      const pixelY = Math.round(y) + offsetY;
      const offset = (pixelY * decoded.info.width + pixelX) * 4;
      for (let channel = 0; channel < 3; channel += 1)
        channels[channel]!.push(decoded.data[offset + channel]!);
    }
  }
  return channels.map(
    (values) => values.sort((left, right) => left - right)[12]!,
  ) as unknown as readonly [number, number, number];
};

describe("the producer implicit profile in independent CesiumJS", () => {
  it("refines visibly and preserves the four asymmetric quadrant identities", async () => {
    JSON.parse(await readFile(resolve(fixtureRoot, "quadrants.json"), "utf8"));
    const page = await browser.newPage({
      viewport: { width: 512, height: 512 },
    });
    const browserErrors: string[] = [];
    const requested = new Set<string>();
    page.on("pageerror", (error) => browserErrors.push(error.message));
    page.on("console", (message) => {
      if (message.type() === "error") browserErrors.push(message.text());
    });
    page.on("request", (request) =>
      requested.add(new URL(request.url()).pathname),
    );
    try {
      await page.goto(`${server.origin}/proof/`);
      await page.locator("html[data-ready]").waitFor({ timeout: 40_000 });
      const proof = await page.evaluate(
        () => globalThis.cesiumImplicitProof as CesiumProof,
      );
      expect(proof.failures, browserErrors.join("\n")).toEqual([]);
      expect(browserErrors).toEqual([]);
      expect(proof.samples).toHaveLength(4);
      expect(proof.readyTiles).toBeGreaterThanOrEqual(5);
      expect(proof.commands).toBeGreaterThanOrEqual(4);
      expect(
        [...requested].filter((path) =>
          /\/content\/1\/\d\/\d\.glb$/.test(path),
        ),
      ).toHaveLength(4);

      const screenshot = await page
        .locator(".cesium-widget canvas")
        .screenshot();
      for (const sample of proof.samples) {
        expect(Number.isFinite(sample.x) && Number.isFinite(sample.y)).toBe(
          true,
        );
        const actual = await medianRgb(screenshot, sample.x, sample.y);
        for (let channel = 0; channel < 3; channel += 1) {
          expect(
            actual[channel],
            `${sample.name} expected ${sample.expectedRgb}, got ${actual}`,
          ).toBeGreaterThanOrEqual(sample.expectedRgb[channel]! - 20);
          expect(actual[channel]).toBeLessThanOrEqual(
            sample.expectedRgb[channel]! + 20,
          );
        }
      }
    } finally {
      await page.close();
    }
  }, 60_000);
});

declare global {
  // Test-only contract installed by cesiumImplicitPage/proof.js.
  var cesiumImplicitProof: CesiumProof;
}
