import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

import { chromium, type Browser } from "playwright";
import sharp from "sharp";
import { build } from "vite";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { startStaticServer, type StaticServer } from "./server";

let browser: Browser;
let server: StaticServer;
let fixtureDirectory = "";

beforeAll(async () => {
  fixtureDirectory = await mkdtemp(resolve(tmpdir(), "mesh-alpha-browser-"));
  await build({
    build: {
      emptyOutDir: true,
      lib: {
        entry: resolve("test/browser/meshAlphaPage.mjs"),
        fileName: "meshAlphaPage",
        formats: ["es"],
      },
      outDir: fixtureDirectory,
    },
    configFile: false,
    logLevel: "silent",
  });
  await writeFile(
    resolve(fixtureDirectory, "index.html"),
    `<!doctype html>
<style>html,body,#view{margin:0;width:96px;height:48px;overflow:hidden}</style>
<div id="view"></div><script type="module" src="meshAlphaPage.js"></script>`,
    "utf8",
  );
  server = await startStaticServer({ "/fixture": fixtureDirectory });
  browser = await chromium.launch({ headless: true });
}, 60_000);

afterAll(async () => {
  await browser?.close();
  await server?.close();
  if (fixtureDirectory)
    await rm(fixtureDirectory, { recursive: true, force: true });
});

const patchMedian = (
  rgba: Uint8Array,
  width: number,
  centerX: number,
  centerY: number,
): [number, number, number] => {
  const channels = [[], [], []] as number[][];
  for (let y = centerY - 2; y <= centerY + 2; y += 1) {
    for (let x = centerX - 2; x <= centerX + 2; x += 1) {
      const offset = (y * width + x) * 4;
      for (let channel = 0; channel < 3; channel += 1)
        channels[channel]!.push(rgba[offset + channel]!);
    }
  }
  return channels.map((values) => values.sort((a, b) => a - b)[12]!) as [
    number,
    number,
    number,
  ];
};

describe("vtk glTF alpha realization", () => {
  it.each(["MASK", "OPAQUE", "BLEND"] as const)(
    "renders %s with its authored pass and fragment semantics",
    async (alphaMode) => {
      const page = await browser.newPage({
        viewport: { width: 96, height: 48 },
      });
      const browserErrors: string[] = [];
      page.on("pageerror", (error) => browserErrors.push(error.message));
      page.on("console", (message) => {
        if (message.type() === "error") browserErrors.push(message.text());
      });
      try {
        await page.goto(
          `${server.origin}/fixture/index.html?mode=${alphaMode}`,
        );
        try {
          await page
            .locator("html[data-ready='true']")
            .waitFor({ timeout: 10_000 });
        } catch (error) {
          throw new Error(
            `VTK alpha fixture did not initialize:\n${browserErrors.join("\n")}`,
            { cause: error },
          );
        }

        const image = await page.locator("canvas").screenshot();
        const stats = JSON.parse(
          (await page.locator("html").getAttribute("data-stats")) ?? "null",
        );
        const adapterErrors = JSON.parse(
          (await page.locator("html").getAttribute("data-adapter-errors")) ??
            "null",
        );
        const decoded = await sharp(image)
          .ensureAlpha()
          .raw()
          .toBuffer({ resolveWithObject: true });
        const left = patchMedian(
          new Uint8Array(decoded.data),
          decoded.info.width,
          24,
          24,
        );
        const right = patchMedian(
          new Uint8Array(decoded.data),
          decoded.info.width,
          72,
          24,
        );

        expect(browserErrors).toEqual([]);
        expect(adapterErrors).toEqual([]);
        expect(stats).toMatchObject({
          pendingTiles: 0,
          submittedTiles: 2,
          drawnTiles: 2,
          drawnActors: 2,
        });

        if (alphaMode === "MASK") {
          expect(left[2]).toBeGreaterThan(220);
          expect(left[0]).toBeLessThan(30);
        } else if (alphaMode === "OPAQUE") {
          expect(left[0]).toBeGreaterThan(220);
          expect(left[2]).toBeLessThan(30);
        } else {
          expect(left[0]).toBeGreaterThan(35);
          expect(left[2]).toBeGreaterThan(100);
        }
        expect(right[0]).toBeGreaterThan(220);
        expect(right[2]).toBeLessThan(30);
      } finally {
        await page.close();
      }
    },
  );

  it("renders KHR_materials_unlit under a light that leaves the equivalent lit primitive dark", async () => {
    const page = await browser.newPage({ viewport: { width: 96, height: 48 } });
    const browserErrors: string[] = [];
    page.on("pageerror", (error) => browserErrors.push(error.message));
    page.on("console", (message) => {
      if (message.type() === "error") browserErrors.push(message.text());
    });
    try {
      await page.goto(`${server.origin}/fixture/index.html?mode=UNLIT`);
      await page
        .locator("html[data-ready='true']")
        .waitFor({ timeout: 10_000 });
      const image = await page.locator("canvas").screenshot();
      const stats = JSON.parse(
        (await page.locator("html").getAttribute("data-stats")) ?? "null",
      );
      const adapterErrors = JSON.parse(
        (await page.locator("html").getAttribute("data-adapter-errors")) ??
          "null",
      );
      const decoded = await sharp(image)
        .ensureAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });
      const left = patchMedian(
        new Uint8Array(decoded.data),
        decoded.info.width,
        24,
        24,
      );
      const right = patchMedian(
        new Uint8Array(decoded.data),
        decoded.info.width,
        72,
        24,
      );

      expect(browserErrors).toEqual([]);
      expect(adapterErrors).toEqual([]);
      expect(stats).toMatchObject({
        pendingTiles: 0,
        submittedTiles: 2,
        drawnTiles: 2,
        drawnActors: 3,
      });
      expect(left[0]).toBeGreaterThan(220);
      expect(left[1]).toBeLessThan(30);
      expect(left[2]).toBeLessThan(30);
      expect(right[0]).toBeLessThan(30);
      expect(right[1]).toBeLessThan(30);
      expect(right[2]).toBeLessThan(30);
    } finally {
      await page.close();
    }
  });
});
