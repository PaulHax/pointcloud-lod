/**
 * Opening and driving the scene explorer for the replay benchmark.
 *
 * The existing `harness.ts` drives `complete/`, which is one point cloud with
 * its own governor. The benchmark drives the explorer instead, because that is
 * the page that can hold a point cloud, a 3D Tiles set, or both under one
 * coordinator — the three scenes being measured differ only in the dataset
 * list their URL carries.
 *
 * Convergence here is the panel's own activity light: every dataset settled
 * with nothing loading. Defining it a second way in the harness would let a
 * benchmark call a scene converged that the page is still showing as busy.
 */

import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  chromium,
  type Browser,
  type BrowserContext,
  type ConsoleMessage,
  type Page,
} from "playwright";

import type { TelemetrySummary, TelemetryTrace } from "../../src/telemetry";
import type { RecordedPose } from "../../examples/vtk/scene/inputRecorder";
import {
  createHttpCache,
  type HttpCache,
  type HttpCacheOptions,
} from "./httpCache";
import { startStaticServer, type StaticServer } from "./server";
import type { ViewerBox } from "./inputReplay";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "../..");
export const EXAMPLE_DIST = resolve(repoRoot, "examples/vtk/dist");
export const FIXTURES = resolve(repoRoot, "test/fixtures");

export type DatasetActivity = {
  readonly id: string;
  readonly state: "loading" | "processing" | "rendering" | "settled" | "error";
  readonly label: string;
  readonly detail: string;
};

export type SceneActivity = {
  readonly loading: number;
  readonly datasets: readonly DatasetActivity[];
};

export type SceneStats = {
  readonly coordinator: Record<string, unknown>;
  readonly members: readonly {
    readonly id: string;
    readonly kind: "points" | "tiles";
    readonly label: string;
    readonly stats: Record<string, unknown>;
  }[];
  readonly lastFrameMs: number;
  readonly paints: number;
  readonly loading: number;
  readonly error: string | null;
};

export type SceneSession = {
  readonly page: Page;
  readonly origin: string;
  readonly failures: readonly string[];
  /** Bytes the page was served, by origin, for the network side of a run. */
  readonly transferredBytes: () => number;
  datasets(): Promise<{ id: string; kind: string; label: string }[]>;
  stats(): Promise<SceneStats>;
  activity(): Promise<SceneActivity>;
  viewerBox(): Promise<ViewerBox>;
  readCamera(): Promise<RecordedPose>;
  placeCamera(pose: {
    position?: readonly number[];
    focalPoint?: readonly number[];
    viewUp?: readonly number[];
  }): Promise<void>;
  setQualityTargets(targets: {
    interactionTargetMs?: number;
    stationaryTargetMs?: number;
  }): Promise<void>;
  setScreenSpaceErrorPx(id: string, px: number): Promise<void>;
  setBudgetMode(id: string, mode: "adaptive" | "fixed"): Promise<void>;
  setFixedPointBudget(id: string, points: number): Promise<void>;
  setMaximumPoints(id: string, points: number | null): Promise<void>;
  setPointSize(
    id: string,
    mode: "auto" | "fixed",
    value: number,
  ): Promise<void>;
  startTelemetry(): Promise<void>;
  stopTelemetry(): Promise<void>;
  markTelemetry(label: string): Promise<void>;
  telemetrySummary(): Promise<TelemetrySummary>;
  telemetryTrace(): Promise<TelemetryTrace>;
  gpuTimingSupported(): Promise<boolean>;
  startInputRecorder(): Promise<void>;
  stopInputRecorder(): Promise<void>;
  inputRecording(): Promise<unknown>;
  render(): Promise<void>;
  frame(): Promise<void>;
  /** Resize until the viewer element is exactly this many CSS pixels. */
  resizeViewer(size: { width: number; height: number }): Promise<void>;
  /** Wait until every dataset reports settled and nothing is loading. */
  settle(timeoutMs?: number): Promise<SceneActivity>;
  close(): Promise<void>;
};

type SceneWindow = {
  pointCloudScene: {
    datasets(): { id: string; kind: string; label: string }[];
    stats(): SceneStats;
    activity(): SceneActivity;
    viewport(): { width: number; height: number };
    camera: {
      read(): RecordedPose;
      place(next: Record<string, unknown>): void;
      reset(): void;
    };
    settings: {
      setQualityTargets(targets: Record<string, number>): void;
      setVisible(id: string, visible: boolean): void;
      setScreenSpaceErrorPx(id: string, px: number): void;
      setBudgetMode(id: string, mode: string): void;
      setFixedPointBudget(id: string, points: number): void;
      setMaximumPoints(id: string, points: number | null): void;
      setPointSize(id: string, mode: string, value: number): void;
    };
    telemetry: {
      start(): void;
      stop(): void;
      clear(): void;
      mark(label: string): void;
      summary(): TelemetrySummary;
      trace(): TelemetryTrace;
      gpuTimingSupported(): boolean;
    };
    render(): void;
  };
  pointCloudRecorder: {
    start(): void;
    stop(): void;
    clear(): void;
    recording(): unknown;
  };
};

const POLL_INTERVAL_MS = 100;

let sharedBrowser: Browser | null = null;

/**
 * A headed browser on the real GPU. Unlike the correctness suite this is a
 * measurement, and SwiftShader frame times describe SwiftShader.
 */
export const benchmarkBrowser = async (headless: boolean): Promise<Browser> => {
  if (sharedBrowser === null) {
    sharedBrowser = await chromium.launch(
      headless
        ? {
            args: [
              "--use-gl=angle",
              "--use-angle=swiftshader",
              "--enable-unsafe-swiftshader",
              "--disable-dev-shm-usage",
            ],
          }
        : { headless: false, args: ["--disable-dev-shm-usage"] },
    );
  }
  return sharedBrowser;
};

export const closeBenchmarkBrowser = async (): Promise<void> => {
  await sharedBrowser?.close();
  sharedBrowser = null;
};

/**
 * The path and query of a recorded page, moved onto the server serving the
 * built example. A recording is made against a dev server on whatever port it
 * chose; only the page and its dataset selection are meaningful.
 */
export const replayPath = (href: string): string => {
  const url = new URL(href);
  const path = url.pathname.endsWith("/")
    ? `${url.pathname}index.html`
    : url.pathname;
  const query = new URLSearchParams(url.search);
  // The capture overlay is a recording tool; a replay must not paint it over
  // the view it is measuring.
  query.delete("record");
  query.set("telemetry", "1");
  return `${path}?${query}`;
};

export const openScene = async (options: {
  /** Page path and query, as `replayPath` produces. */
  readonly path: string;
  readonly cache?: HttpCacheOptions;
  readonly deviceScaleFactor?: number;
  readonly headless?: boolean;
  /** Browser window size; `resizeViewer` refines it to the viewer's own size. */
  readonly windowSize?: { width: number; height: number };
}): Promise<SceneSession & { readonly cache: HttpCache | null }> => {
  const server: StaticServer = await startStaticServer({
    "/fixtures": FIXTURES,
    "/": EXAMPLE_DIST,
  });
  const browser = await benchmarkBrowser(options.headless ?? false);
  const context: BrowserContext = await browser.newContext({
    viewport: options.windowSize ?? { width: 1600, height: 1000 },
    deviceScaleFactor: options.deviceScaleFactor ?? 1,
  });
  const page = await context.newPage();

  const cache =
    options.cache === undefined ? null : createHttpCache(options.cache);
  await cache?.install(page);

  const failures: string[] = [];
  page.on("pageerror", (error) =>
    // With the stack: a page error that only reports its message names the
    // failure but not the code, and these runs are long enough that losing
    // that costs a whole re-run to recover.
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
  let transferred = 0;
  page.on("response", (response) => {
    const size = Number(response.headers()["content-length"] ?? 0);
    if (Number.isFinite(size)) transferred += size;
  });

  await page.goto(`${server.origin}${options.path}`, { waitUntil: "load" });
  await page.waitForFunction(
    () => (window as never as SceneWindow).pointCloudScene !== undefined,
    null,
    { timeout: 60_000 },
  );

  const evaluate = <T>(body: () => T): Promise<T> => page.evaluate(body);

  const activity = (): Promise<SceneActivity> =>
    evaluate(() =>
      (window as never as SceneWindow).pointCloudScene.activity(),
    ) as Promise<SceneActivity>;

  const session: SceneSession & { cache: HttpCache | null } = {
    page,
    origin: server.origin,
    failures,
    cache,
    transferredBytes: () => transferred,
    datasets: () =>
      evaluate(() =>
        (window as never as SceneWindow).pointCloudScene.datasets(),
      ),
    stats: () =>
      evaluate(() =>
        (window as never as SceneWindow).pointCloudScene.stats(),
      ) as Promise<SceneStats>,
    activity,
    viewerBox: async () => {
      const box = await page.locator("#viewer").boundingBox();
      if (box === null) throw new Error("the viewer has no box");
      return box;
    },
    readCamera: () =>
      evaluate(() =>
        (window as never as SceneWindow).pointCloudScene.camera.read(),
      ) as Promise<RecordedPose>,
    placeCamera: (pose) =>
      page.evaluate(
        (value) =>
          (window as never as SceneWindow).pointCloudScene.camera.place(value),
        pose as Record<string, unknown>,
      ),
    setQualityTargets: (targets) =>
      page.evaluate(
        (value) =>
          (
            window as never as SceneWindow
          ).pointCloudScene.settings.setQualityTargets(value),
        targets as Record<string, number>,
      ),
    setScreenSpaceErrorPx: (id, px) =>
      page.evaluate(
        ({ id: target, px: value }) =>
          (
            window as never as SceneWindow
          ).pointCloudScene.settings.setScreenSpaceErrorPx(target, value),
        { id, px },
      ),
    setBudgetMode: (id, mode) =>
      page.evaluate(
        ({ id: target, mode: value }) =>
          (
            window as never as SceneWindow
          ).pointCloudScene.settings.setBudgetMode(target, value),
        { id, mode },
      ),
    setFixedPointBudget: (id, points) =>
      page.evaluate(
        ({ id: target, points: value }) =>
          (
            window as never as SceneWindow
          ).pointCloudScene.settings.setFixedPointBudget(target, value),
        { id, points },
      ),
    setMaximumPoints: (id, points) =>
      page.evaluate(
        ({ id: target, points: value }) =>
          (
            window as never as SceneWindow
          ).pointCloudScene.settings.setMaximumPoints(target, value),
        { id, points },
      ),
    setPointSize: (id, mode, value) =>
      page.evaluate(
        ({ id: target, mode: sizeMode, value: size }) =>
          (
            window as never as SceneWindow
          ).pointCloudScene.settings.setPointSize(target, sizeMode, size),
        { id, mode, value },
      ),
    startTelemetry: () =>
      evaluate(() =>
        (window as never as SceneWindow).pointCloudScene.telemetry.start(),
      ),
    stopTelemetry: () =>
      evaluate(() =>
        (window as never as SceneWindow).pointCloudScene.telemetry.stop(),
      ),
    markTelemetry: (label) =>
      page.evaluate(
        (value) =>
          (window as never as SceneWindow).pointCloudScene.telemetry.mark(
            value,
          ),
        label,
      ),
    telemetrySummary: () =>
      evaluate(() =>
        (window as never as SceneWindow).pointCloudScene.telemetry.summary(),
      ) as Promise<TelemetrySummary>,
    telemetryTrace: () =>
      evaluate(() =>
        (window as never as SceneWindow).pointCloudScene.telemetry.trace(),
      ) as Promise<TelemetryTrace>,
    gpuTimingSupported: () =>
      evaluate(() =>
        (
          window as never as SceneWindow
        ).pointCloudScene.telemetry.gpuTimingSupported(),
      ),
    startInputRecorder: () =>
      evaluate(() => {
        const recorder = (window as never as SceneWindow).pointCloudRecorder;
        recorder.clear();
        recorder.start();
      }),
    stopInputRecorder: () =>
      evaluate(() =>
        (window as never as SceneWindow).pointCloudRecorder.stop(),
      ),
    inputRecording: () =>
      evaluate(() =>
        (window as never as SceneWindow).pointCloudRecorder.recording(),
      ),
    render: () =>
      evaluate(() => (window as never as SceneWindow).pointCloudScene.render()),
    frame: () =>
      page.evaluate(
        () =>
          new Promise<void>((done) =>
            requestAnimationFrame(() => requestAnimationFrame(() => done())),
          ),
      ),
    resizeViewer: async (size) => {
      // The panel takes a column out of the window, so the window size and the
      // viewer size differ by an amount that is linear in each axis with a
      // bend where the panel stops scaling. Two probes on one side of that
      // bend land on it exactly.
      let probe = { width: size.width, height: size.height };
      let previous: {
        window: { width: number; height: number };
        seen: { width: number; height: number };
      } | null = null;
      const seenSize = async (): Promise<{ width: number; height: number }> => {
        await page.setViewportSize(probe);
        return evaluate(() =>
          (window as never as SceneWindow).pointCloudScene.viewport(),
        );
      };
      let seen = await seenSize();
      for (let attempt = 0; attempt < 12; attempt += 1) {
        if (seen.width === size.width && seen.height === size.height) break;
        const next = { ...probe };
        for (const axis of ["width", "height"] as const) {
          if (seen[axis] === size[axis]) continue;
          const spread = previous ? probe[axis] - previous.window[axis] : 0;
          const slope =
            spread === 0 ? 1 : (seen[axis] - previous!.seen[axis]) / spread;
          next[axis] = Math.round(
            probe[axis] + (size[axis] - seen[axis]) / (slope || 1),
          );
        }
        previous = { window: probe, seen };
        probe = next;
        seen = await seenSize();
      }
      if (seen.width !== size.width || seen.height !== size.height) {
        throw new Error(
          `no window size gives a ${size.width}x${size.height} viewer: ` +
            `${probe.width}x${probe.height} leaves ${seen.width}x${seen.height}`,
        );
      }
      await session.render();
      await session.frame();
    },
    settle: async (timeoutMs = 300_000) => {
      await session.frame();
      const deadline = Date.now() + timeoutMs;
      let last: SceneActivity | null = null;
      while (Date.now() < deadline) {
        last = await activity();
        const converged =
          last.loading === 0 &&
          last.datasets.length > 0 &&
          last.datasets.every((dataset) => dataset.state === "settled");
        if (converged) return last;
        await page.waitForTimeout(POLL_INTERVAL_MS);
      }
      throw new Error(
        `timed out waiting for the scene to settle\nlast activity: ${JSON.stringify(
          last,
          null,
          2,
        )}`,
      );
    },
    close: async () => {
      await context.close();
      await server.close();
    },
  };

  return session;
};
