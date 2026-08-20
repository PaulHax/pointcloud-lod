/**
 * What both browser harnesses need before they diverge.
 *
 * The correctness suite drives the instrumented page and the benchmark drives
 * the explorer, and what they do with a page has almost nothing in common —
 * but where the built example lives, which Chromium to run it in, and what
 * counts as the page having failed are the same question either way, and were
 * being answered twice.
 */

import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  chromium,
  type Browser,
  type ConsoleMessage,
  type Page,
} from "playwright";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "../..");

export const EXAMPLE_DIST = resolve(repoRoot, "examples/vtk/dist");
export const FIXTURES = resolve(repoRoot, "test/fixtures");

/**
 * Which rasteriser a page runs on.
 *
 * `software` is headless SwiftShader: correct pixels, frame times that
 * describe SwiftShader and nothing else. `gpu` is a headed window on the
 * machine's own GPU, which is the only mode a measurement may be taken in.
 */
export type RenderMode = "software" | "gpu";

/** The mode a run asked for, as the environment states it. */
export const requestedRenderMode = (): RenderMode =>
  (process.env.POINTCLOUD_LOD_BROWSER_GPU ?? "") !== "" ? "gpu" : "software";

const browsers = new Map<RenderMode, Browser>();

/**
 * One browser per mode, for the whole file. Launching Chromium costs far more
 * than any check in it, and a fresh page per check already isolates page
 * state.
 */
export const browserFor = async (mode: RenderMode): Promise<Browser> => {
  const running = browsers.get(mode);
  if (running !== undefined) return running;
  const launched = await chromium.launch(
    mode === "gpu"
      ? { headless: false, args: ["--disable-dev-shm-usage"] }
      : {
          args: [
            // Headless Chromium has no GPU; SwiftShader rasterises WebGL in
            // software. Recent builds refuse it for WebGL without the opt-in.
            "--use-gl=angle",
            "--use-angle=swiftshader",
            "--enable-unsafe-swiftshader",
            "--disable-dev-shm-usage",
          ],
        },
  );
  browsers.set(mode, launched);
  return launched;
};

export const closeBrowsers = async (): Promise<void> => {
  const running = [...browsers.values()];
  browsers.clear();
  for (const browser of running) await browser.close();
};

/**
 * Everything the page reported as broken, in the order it said so.
 *
 * Page errors carry their stack: a run long enough to be worth waiting for is
 * long enough that losing the code that threw costs a whole re-run.
 */
export const collectPageFailures = (page: Page): string[] => {
  const failures: string[] = [];
  page.on("pageerror", (error) =>
    failures.push(
      `pageerror: ${error.message}\n${(error.stack ?? "")
        .split("\n")
        .slice(1, 9)
        .join("\n")}`,
    ),
  );
  page.on("console", (message: ConsoleMessage) => {
    if (message.type() === "error") failures.push(`console: ${message.text()}`);
  });
  return failures;
};
