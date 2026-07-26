/**
 * Browser harness for the LOD checks.
 *
 * These run the built example in a real Chromium against a real WebGL context,
 * because the failures this suite exists to catch do not appear in the unit
 * tests: duplicated physical reads, resources that outlive what asked for
 * them, and a budget loop that never settles all look correct against fake
 * sources and a fake clock.
 *
 * WebGL comes from SwiftShader, so frame times measure a software rasteriser
 * and mean nothing in absolute terms. Nothing here asserts on wall-clock
 * speed; where a check needs a frame time it states one through
 * `setSyntheticFrameMs`.
 */

import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { chromium, type Browser, type ConsoleMessage, type Page } from "playwright";

import { startStaticServer, type StaticServer } from "./server";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "../..");

export const EXAMPLE_DIST = resolve(repoRoot, "examples/vtk/dist");
export const FIXTURES = resolve(repoRoot, "test/fixtures");

/** The committed fixture: small, single hierarchy page, always available. */
export const FIXTURE_URL_PATH = "/fixtures/fixture.copc.laz";

/**
 * Extra clouds to run the matrix against, as a path list in
 * `POINTCLOUD_LOD_BROWSER_CLOUDS`. Deliberately not named or defaulted here:
 * the datasets a deployment cares about are not this repository's to know.
 */
export const extraClouds = (): string[] =>
  (process.env.POINTCLOUD_LOD_BROWSER_CLOUDS ?? "")
    .split(":")
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0);

export interface ExampleStats {
  controller: {
    residentTiles: number;
    residentPoints: number;
    residentBytes: number;
    decodedTiles: number;
    decodedBytes: number;
    cachedTiles: number;
    inFlight: number;
    queuedTiles: number;
    physicalTileOperations: number;
    queuedPages: number;
    physicalHierarchyOperations: number;
    pointBudget: number;
    memoryCeilingPoints: number;
    memoryBudgetBytes: number;
    active: boolean;
    selection: {
      generation: number;
      targetTiles: number;
      targetPoints: number;
      readyTerminalFrontier: {
        count: number;
        projectedSpacingCssPx: { p50: number | null; p75: number | null };
      };
    };
  } | null;
  adapter: {
    submittedTiles: number;
    submittedPoints: number;
    submittedBytes: number;
    pooledTiles: number;
    pooledBytes: number;
    gpuResidentTiles: number;
    gpuResidentBytes: number;
    drawnTiles: number;
    drawnPoints: number;
    visible: boolean;
  } | null;
  governor: {
    regime: "interaction" | "stationary";
    targetFrameTimeMs: number;
    trackBudget: number;
    aggregateBudget: number;
    activeConstraint: string;
    memoryCeilingPoints: number | null;
    needsFrame: boolean;
    motion: { source: string | null; settling: boolean };
    lastAdjustment: { reason: string; direction: string } | null;
  } | null;
  lastFrameMs: number;
  source: string | null;
  /** Points the loaded asset holds in total, per its COPC header. */
  sourcePoints: number;
}

export interface ExampleSession {
  readonly page: Page;
  readonly origin: string;
  /** Every uncaught error and rejected promise the page produced. */
  readonly failures: readonly string[];
  stats(): Promise<ExampleStats>;
  /** Load another cloud into the running page; resolves when it has opened. */
  load(url: string): Promise<void>;
  readCamera(): Promise<CameraReading>;
  place(next: {
    position?: readonly number[];
    focalPoint?: readonly number[];
    parallelScale?: number;
  }): Promise<void>;
  azimuth(degrees: number): Promise<void>;
  dolly(factor: number): Promise<void>;
  setProjection(projection: "perspective" | "orthographic"): Promise<void>;
  setBudgetMode(mode: "fixed" | "adaptive"): Promise<void>;
  setVisible(visible: boolean): Promise<void>;
  setActive(active: boolean): Promise<void>;
  setDevicePixelRatio(ratio: number): Promise<void>;
  setSyntheticFrameMs(ms: number | null): Promise<void>;
  render(): Promise<void>;
  dispose(): Promise<void>;
  /** Poll until `predicate(stats)` holds, or fail with the last stats seen. */
  until(
    describe: string,
    predicate: (stats: ExampleStats) => boolean,
    timeoutMs?: number,
  ): Promise<ExampleStats>;
  /** Resolve after the page has painted, so a queued change has landed. */
  frame(): Promise<void>;
  /** Wait for selection and loading to converge and the view to stop asking. */
  settle(timeoutMs?: number): Promise<ExampleStats>;
  close(): Promise<void>;
}

let sharedBrowser: Browser | null = null;

/**
 * One browser for the whole file. Launching Chromium costs far more than any
 * check in it, and a fresh page per check already isolates page state.
 */
export const browser = async (): Promise<Browser> => {
  if (sharedBrowser === null) {
    sharedBrowser = await chromium.launch({
      args: [
        // Headless Chromium has no GPU; SwiftShader rasterises WebGL in
        // software. Recent builds refuse it for WebGL without the opt-in.
        "--use-gl=angle",
        "--use-angle=swiftshader",
        "--enable-unsafe-swiftshader",
        "--disable-dev-shm-usage",
      ],
    });
  }
  return sharedBrowser;
};

export const closeBrowser = async (): Promise<void> => {
  await sharedBrowser?.close();
  sharedBrowser = null;
};

const POLL_INTERVAL_MS = 50;
/** Selection is debounced, so convergence needs a quiet window, not an
 * instant. Comfortably over the controller's 150 ms selection delay. */
const SETTLE_QUIET_MS = 400;

/**
 * Open the built example with a cloud loaded.
 *
 * `cloud` is a URL path this harness serves (the fixture) or an absolute
 * `file:`-free path already exposed by `roots`.
 */
export const openExample = async (
  options: {
    cloud?: string;
    roots?: Record<string, string>;
    query?: Record<string, string>;
  } = {},
): Promise<ExampleSession> => {
  const cloud = options.cloud ?? FIXTURE_URL_PATH;
  const server: StaticServer = await startStaticServer({
    "/fixtures": FIXTURES,
    ...options.roots,
    // The example is the fallback root, so it must be matched last.
    "/": EXAMPLE_DIST,
  });
  const page = await (await browser()).newPage();
  const failures: string[] = [];
  page.on("pageerror", (error) => failures.push(`pageerror: ${error.message}`));
  page.on("console", (message: ConsoleMessage) => {
    if (message.type() === "error") failures.push(`console: ${message.text()}`);
  });

  const query = new URLSearchParams({ url: cloud, ...options.query });
  await page.goto(`${server.origin}/?${query}`, { waitUntil: "load" });
  await page.waitForFunction(
    () => (window as never as ExampleWindow).pointCloudExample !== undefined,
  );

  const stats = (): Promise<ExampleStats> =>
    page.evaluate(
      () => (window as never as ExampleWindow).pointCloudExample.stats(),
    ) as Promise<ExampleStats>;

  const until = async (
    describe: string,
    predicate: (value: ExampleStats) => boolean,
    timeoutMs = 30_000,
  ): Promise<ExampleStats> => {
    const deadline = Date.now() + timeoutMs;
    let last: ExampleStats | null = null;
    while (Date.now() < deadline) {
      last = await stats();
      if (predicate(last)) return last;
      await page.waitForTimeout(POLL_INTERVAL_MS);
    }
    throw new Error(
      `timed out waiting for ${describe}\nlast stats: ${JSON.stringify(last, null, 2)}`,
    );
  };

  const session: ExampleSession = {
    page,
    origin: server.origin,
    failures,
    stats,
    load: (url) =>
      page.evaluate(
        (target) => (window as never as ExampleWindow).pointCloudExample.load(target),
        url,
      ),
    readCamera: () =>
      page.evaluate(
        () => (window as never as ExampleWindow).pointCloudExample.camera.read(),
      ),
    place: (next) =>
      page.evaluate(
        (value) =>
          (window as never as ExampleWindow).pointCloudExample.camera.place(value),
        next as Record<string, unknown>,
      ),
    azimuth: (degrees) =>
      page.evaluate(
        (value) =>
          (window as never as ExampleWindow).pointCloudExample.camera.azimuth(value),
        degrees,
      ),
    dolly: (factor) =>
      page.evaluate(
        (value) =>
          (window as never as ExampleWindow).pointCloudExample.camera.dolly(value),
        factor,
      ),
    setProjection: (projection) =>
      page.evaluate(
        (value) =>
          (window as never as ExampleWindow).pointCloudExample.setProjection(value),
        projection,
      ),
    setBudgetMode: (mode) =>
      page.evaluate(
        (value) =>
          (window as never as ExampleWindow).pointCloudExample.setBudgetMode(value),
        mode,
      ),
    setVisible: (visible) =>
      page.evaluate(
        (value) =>
          (window as never as ExampleWindow).pointCloudExample.setVisible(value),
        visible,
      ),
    setActive: (active) =>
      page.evaluate(
        (value) =>
          (window as never as ExampleWindow).pointCloudExample.setActive(value),
        active,
      ),
    setDevicePixelRatio: (ratio) =>
      page.evaluate(
        (value) =>
          (window as never as ExampleWindow).pointCloudExample.setDevicePixelRatio(
            value,
          ),
        ratio,
      ),
    setSyntheticFrameMs: (ms) =>
      page.evaluate(
        (value) =>
          (window as never as ExampleWindow).pointCloudExample.setSyntheticFrameMs(
            value,
          ),
        ms,
      ),
    render: () =>
      page.evaluate(() =>
        (window as never as ExampleWindow).pointCloudExample.render(),
      ),
    dispose: () =>
      page.evaluate(() =>
        (window as never as ExampleWindow).pointCloudExample.dispose(),
      ),
    until,
    frame: () =>
      page.evaluate(
        () =>
          new Promise<void>((done) =>
            requestAnimationFrame(() => requestAnimationFrame(() => done())),
          ),
      ),
    settle: async (timeoutMs = 30_000) => {
      // A driving call only queues a frame, and the camera does not reach the
      // controller until that frame paints. Polling first would read the state
      // before the change and call it converged.
      await session.frame();
      const deadline = Date.now() + timeoutMs;
      let quietSince: number | null = null;
      let lastGeneration = -1;
      let last: ExampleStats | null = null;
      while (Date.now() < deadline) {
        last = await stats();
        const cloud = last.controller;
        const drained =
          cloud !== null &&
          cloud.physicalTileOperations === 0 &&
          cloud.physicalHierarchyOperations === 0 &&
          cloud.queuedTiles === 0 &&
          cloud.queuedPages === 0 &&
          (last.governor === null || !last.governor.needsFrame);
        const generation = cloud?.selection.generation ?? -1;
        if (!drained || generation !== lastGeneration) {
          lastGeneration = generation;
          quietSince = drained ? Date.now() : null;
        } else if (quietSince === null) {
          quietSince = Date.now();
        }
        // Selection is debounced, so "no work outstanding" is only convergence
        // once it has also stopped producing new selections.
        if (quietSince !== null && Date.now() - quietSince >= SETTLE_QUIET_MS) {
          return last;
        }
        await page.waitForTimeout(POLL_INTERVAL_MS);
      }
      throw new Error(
        `timed out waiting for convergence\nlast stats: ${JSON.stringify(last, null, 2)}`,
      );
    },
    close: async () => {
      await page.close();
      await server.close();
    },
  };

  // The cloud is requested by the query string; wait for it to be readable
  // before handing the session over, so no check has to re-implement that.
  await until(
    "the cloud to open",
    (value) => value.controller !== null && value.source !== null,
  );
  return session;
};

export interface CameraReading {
  position: number[];
  focalPoint: number[];
  parallelScale: number;
  viewAngle: number;
  parallelProjection: boolean;
}

interface ExampleWindow {
  pointCloudExample: {
    stats(): ExampleStats;
    load(url: string): Promise<void>;
    camera: {
      read(): CameraReading;
      place(next: Record<string, unknown>): void;
      azimuth(degrees: number): void;
      dolly(factor: number): void;
    };
    setProjection(projection: "perspective" | "orthographic"): void;
    setBudgetMode(mode: "fixed" | "adaptive"): void;
    setVisible(visible: boolean): void;
    setActive(active: boolean): void;
    setDevicePixelRatio(ratio: number): void;
    setSyntheticFrameMs(ms: number | null): void;
    needsFrame(): boolean;
    render(): void;
    dispose(): void;
  };
}
