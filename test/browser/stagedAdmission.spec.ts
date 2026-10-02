import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { resolve } from "node:path";
import { chromium } from "playwright";
import { build } from "vite";
import { expect, it } from "vitest";
import { startStaticServer } from "./server";

it("admits real point members in bounded paints with exact GPU prefixes and delayed picking", async () => {
  const directory = await mkdtemp(resolve(tmpdir(), "staged-admission-"));
  let browser;
  let server;
  try {
    await build({
      configFile: false,
      logLevel: "silent",
      build: {
        lib: {
          entry: resolve("test/browser/stagedAdmissionPage.mjs"),
          fileName: "page",
          formats: ["es"],
        },
        outDir: directory,
        emptyOutDir: true,
      },
    });
    await writeFile(
      resolve(directory, "index.html"),
      '<style>html,body,#view{margin:0;width:512px;height:512px;overflow:hidden}</style><div id="view"></div><script type="module" src="page.js"></script>',
    );
    server = await startStaticServer({ "/fixture": directory });
    browser = await chromium.launch({ headless: true });
    const page = await browser.newPage({
      viewport: { width: 512, height: 512 },
    });
    const failures: string[] = [];
    page.on("pageerror", (error) => failures.push(error.message));
    page.on("console", (message) => {
      if (message.type() === "error") failures.push(message.text());
    });
    await page.goto(`${server.origin}/fixture/index.html`);
    await page
      .locator("html[data-ready='true']")
      .waitFor({ timeout: 10_000 })
      .catch((error) => {
        throw new Error(failures.join("\n"), { cause: error });
      });
    const read = (paint = false) =>
      page.evaluate((paint) => {
        const fixture = (window as any).stagedAdmission;
        return paint ? fixture.paint() : fixture.snapshot();
      }, paint);
    const initial = await read();
    expect(initial.drawnPoints).toBe(0);
    expect(initial.pick.status).toBe("miss");
    expect(initial.pending).toBeGreaterThan(0);

    let previous = 0;
    for (const count of [8192, 16384, 20000]) {
      const state = await read(true);
      expect(state.drawnPoints).toBe(count);
      expect(state.controllerPoints).toBe(count);
      expect(state.allocations).toEqual({ positions: 1, colors: 1 });
      expect(state.writes).toEqual([
        {
          kind: "positions",
          offset: previous * 12,
          bytes: (count - previous) * 12,
          mismatches: 0,
        },
        {
          kind: "colors",
          offset: previous * 4,
          bytes: (count - previous) * 4,
          mismatches: 0,
        },
      ]);
      expect(state.mismatches).toBe(0);
      expect(state.submission.lastFrameAdmittedBytes).toBe(
        (count - previous) * 16,
      );
      if (count < 20000) {
        expect(state.pending).toBeGreaterThan(0);
        expect(state.pick.status).toBe("miss");
      } else {
        expect(state.pending).toBe(0);
        expect(state.pick.status).toBe("hit");
        expect(state.pick.scenePoint).toEqual([0, 0, 0]);
        expect(state.pixel).toEqual([255, 0, 0, 255]);
      }
      previous = count;
    }
    expect((await read(true)).writes).toEqual([]);
    expect(failures).toEqual([]);
    await page.evaluate(() => (window as any).stagedAdmission.dispose());
  } finally {
    await browser?.close();
    await server?.close();
    await rm(directory, { recursive: true, force: true });
  }
});
