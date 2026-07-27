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

import {
  chromium,
  type Browser,
  type ConsoleMessage,
  type Page,
} from "playwright";

// The example hands these three back verbatim, so the harness names the
// library's own types rather than a copy that can drift lossy. `import type`
// is erased under `verbatimModuleSyntax`, so nothing here loads vtk.js in Node.
// `RendererAdapterStats` comes from its own module: index.ts deliberately does
// not re-export it.
import type { LodControllerStats } from "../../src/controller";
import type { RendererAdapterStats } from "../../src/rendererAdapter";
import type { ViewGovernorStats } from "../../src/viewGovernor";
import { startStaticServer, type StaticServer } from "./server";

/** Failure messages in this suite carry the whole sample, pretty-printed. */
export const shown = (value: unknown): string => JSON.stringify(value, null, 2);

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "../..");

export const EXAMPLE_DIST = resolve(repoRoot, "examples/vtk/dist");
export const FIXTURES = resolve(repoRoot, "test/fixtures");

/** The committed fixture: small, single hierarchy page, always available. */
export const FIXTURE_URL_PATH = "/fixtures/fixture.copc.laz";
/**
 * The committed fixture whose hierarchy spans 73 pages over 4 levels. Every
 * real cloud on hand ships exactly one page, so this is the only asset that
 * makes the bounded page scheduler, the page-blocked branch of selection, and
 * multi-level tile traffic run at all.
 */
export const MULTIPAGE_URL_PATH = "/fixtures/multipage.copc.laz";

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

/**
 * A cloud to run a scenario against, named by position rather than by file.
 *
 * Every scenario runs over this list, so the committed fixture proves the
 * mechanism everywhere and a real cloud supplied through the environment
 * proves it against a deep octree — same assertions, no second suite. The
 * label and the URL are positional on purpose: a run's output should not
 * disclose which datasets a deployment pointed it at.
 */
export type CloudUnderTest = {
  readonly name: string;
  readonly urlPath: string;
  readonly files: Record<string, string>;
  /**
   * Holds more detail than any one view selects, so refinement has somewhere
   * to descend and tiles actually stream. The small fixtures do not.
   */
  readonly deep: boolean;
  /** Its hierarchy spans more than one page, so pages must be fetched to select. */
  readonly multipage: boolean;
};

export const FIXTURE_CLOUD: CloudUnderTest = {
  name: "the fixture",
  urlPath: FIXTURE_URL_PATH,
  files: {},
  deep: false,
  multipage: false,
};

export const MULTIPAGE_CLOUD: CloudUnderTest = {
  name: "the multipage fixture",
  urlPath: MULTIPAGE_URL_PATH,
  files: {},
  deep: false,
  multipage: true,
};

export const cloudsUnderTest = (): CloudUnderTest[] => [
  FIXTURE_CLOUD,
  MULTIPAGE_CLOUD,
  ...extraClouds().map((path, index) => ({
    name: `supplied cloud ${index + 1}`,
    urlPath: `/supplied/${index + 1}/cloud.copc.laz`,
    files: { [`/supplied/${index + 1}/cloud.copc.laz`]: path },
    deep: true,
    multipage: false,
  })),
];

/**
 * Two clouds to switch between, preferring a pair that genuinely differ.
 *
 * Supplied clouds first, because they differ in every way. Otherwise the two
 * committed fixtures, which differ in point count, node count and hierarchy
 * shape — enough for a switch to be arithmetically visible, which one file
 * served at two URLs never is.
 */
export const cloudPair = (): [CloudUnderTest, CloudUnderTest] => {
  const supplied = cloudsUnderTest().filter((entry) => entry.deep);
  if (supplied.length >= 2) return [supplied[0]!, supplied[1]!];
  if (supplied.length === 1) return [supplied[0]!, MULTIPAGE_CLOUD];
  return [FIXTURE_CLOUD, MULTIPAGE_CLOUD];
};

export type ExampleStats = {
  controller: LodControllerStats | null;
  adapter: RendererAdapterStats | null;
  governor: ViewGovernorStats | null;
  lastFrameMs: number;
  source: string | null;
  /** Points the loaded asset holds in total, per its COPC header. */
  sourcePoints: number;
};

export type ExampleKeys = {
  controller: { resident: string[]; submitted: string[] } | null;
  adapter: { submitted: string[]; pooled: string[] } | null;
};

export type Viewport = {
  width: number;
  height: number;
};

/** What the renderer holds, read from the scene rather than from the adapter. */
export type SceneReading = {
  actors: number;
  pointSizeDevicePx: number | null;
  mapperScaleFactor: number | null;
};

export type ExampleSession = {
  readonly page: Page;
  readonly origin: string;
  /** Every uncaught error and rejected promise the page produced. */
  readonly failures: readonly string[];
  stats(): Promise<ExampleStats>;
  /** Both sides' key sets, for asserting they agree. */
  keys(): Promise<ExampleKeys>;
  /** The CSS size selection is computed against. */
  viewport(): Promise<Viewport>;
  /** What the renderer is holding — the only view a leaked actor appears in. */
  scene(): Promise<SceneReading>;
  /**
   * A real pointer gesture on the canvas, which is the only way the
   * interactor's animation callbacks fire and the governor sees *explicit*
   * motion. Every other camera handle moves the camera programmatically, so
   * the governor infers motion instead — a different code path.
   */
  drag(steps: { dx: number; dy: number }[], pauseMs?: number): Promise<void>;
  /** Resize the browser viewport, which resizes the canvas under it. */
  resize(size: Viewport): Promise<void>;
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
};

let sharedBrowser: Browser | null = null;

/**
 * True when `POINTCLOUD_LOD_BROWSER_GPU` asks for the machine's real GPU
 * instead of SwiftShader. The stress matrix never needs it — nothing there
 * asserts wall-clock speed — but a measurement run does: frame times taken
 * under a software rasteriser describe SwiftShader, not the library.
 */
export const usingRealGpu = (): boolean =>
  (process.env.POINTCLOUD_LOD_BROWSER_GPU ?? "") !== "";

/**
 * One browser for the whole file. Launching Chromium costs far more than any
 * check in it, and a fresh page per check already isolates page state.
 */
export const browser = async (): Promise<Browser> => {
  if (sharedBrowser === null) {
    sharedBrowser = await chromium.launch(
      usingRealGpu()
        ? {
            headless: false,
            args: ["--disable-dev-shm-usage"],
          }
        : {
            args: [
              // Headless Chromium has no GPU; SwiftShader rasterises WebGL in
              // software. Recent builds refuse it for WebGL without the
              // opt-in.
              "--use-gl=angle",
              "--use-angle=swiftshader",
              "--enable-unsafe-swiftshader",
              "--disable-dev-shm-usage",
            ],
          },
    );
  }
  return sharedBrowser;
};

export const closeBrowser = async (): Promise<void> => {
  await sharedBrowser?.close();
  sharedBrowser = null;
};

/**
 * The render viewport every session starts at: the browser's own default
 * window, which is what these checks were written against back when the canvas
 * filled it.
 */
export const DEFAULT_VIEWPORT: Viewport = { width: 1280, height: 720 };

const POLL_INTERVAL_MS = 50;
/** Selection is debounced, so convergence needs a quiet window, not an
 * instant. Comfortably over the controller's 150 ms selection delay. */
export const SETTLE_QUIET_MS = 400;

/**
 * Open the built example with a cloud loaded.
 *
 * `cloud` is a URL path this harness serves: a committed fixture, or the
 * positional path `cloudsUnderTest()` gives a supplied cloud.
 */
export const openExample = async (
  options: { cloud?: string } = {},
): Promise<ExampleSession> => {
  const cloud = options.cloud ?? FIXTURE_URL_PATH;
  const server: StaticServer = await startStaticServer(
    {
      "/fixtures": FIXTURES,
      // The example is the fallback root, so it must be matched last.
      "/": EXAMPLE_DIST,
    },
    // Every cloud is reachable from every session, so a scenario can switch
    // sources without standing up a second server.
    Object.assign({}, ...cloudsUnderTest().map((entry) => entry.files)),
  );
  const page = await (await browser()).newPage();
  const failures: string[] = [];
  page.on("pageerror", (error) => failures.push(`pageerror: ${error.message}`));
  page.on("console", (message: ConsoleMessage) => {
    if (message.type() === "error") failures.push(`console: ${message.text()}`);
  });

  const query = new URLSearchParams({ url: cloud });
  await page.goto(`${server.origin}/?${query}`, { waitUntil: "load" });
  await page.waitForFunction(
    () => (window as never as ExampleWindow).pointCloudExample !== undefined,
  );

  const stats = (): Promise<ExampleStats> =>
    page.evaluate(() =>
      (window as never as ExampleWindow).pointCloudExample.stats(),
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
      `timed out waiting for ${describe}\nlast stats: ${shown(last)}`,
    );
  };

  const session: ExampleSession = {
    page,
    origin: server.origin,
    failures,
    stats,
    keys: () =>
      page.evaluate(() =>
        (window as never as ExampleWindow).pointCloudExample.keys(),
      ) as Promise<ExampleKeys>,
    viewport: () =>
      page.evaluate(() =>
        (window as never as ExampleWindow).pointCloudExample.viewport(),
      ) as Promise<Viewport>,
    scene: () =>
      page.evaluate(() =>
        (window as never as ExampleWindow).pointCloudExample.scene(),
      ) as Promise<SceneReading>,
    drag: async (steps, pauseMs = 16) => {
      const box = await page.locator("#viewer").boundingBox();
      if (box === null) throw new Error("the viewer has no box to drag in");
      let x = box.x + box.width / 2;
      let y = box.y + box.height / 2;
      await page.mouse.move(x, y);
      await page.mouse.down();
      for (const { dx, dy } of steps) {
        x += dx;
        y += dy;
        await page.mouse.move(x, y);
        // Pace to painted frames, not to the clock. The interactor applies a
        // move on its own animation frame, so on a cloud whose paint takes
        // seconds a fixed delay delivers the whole gesture inside one frame:
        // the camera never moves, and a check guarding against a gesture that
        // missed the interactor fails on a gesture that reached it perfectly
        // well. The delay stays as a floor so the moves remain distinguishable.
        await page.waitForTimeout(pauseMs);
        await session.frame();
      }
      await page.mouse.up();
    },
    resize: async (size) => {
      // `size` is the render viewport, not the browser window: the page's
      // control panel is a column of the layout and takes its width out of the
      // window, so the two differ. Every law under test is about the pixels
      // selection is computed against, and each reads far better stated in
      // those than in a window size carrying a panel-shaped offset.
      //
      // The window-to-viewport relation is linear in each axis, with a bend
      // where the panel stops scaling, so two probes on one side of that bend
      // land on the answer exactly. The first probe assumes no panel at all,
      // which is right on the axis the panel does not touch.
      let probe = { width: size.width, height: size.height };
      let previous: { window: Viewport; seen: Viewport } | null = null;
      let seen = await (async () => {
        await page.setViewportSize(probe);
        return session.viewport();
      })();
      for (let attempt = 0; attempt < 10; attempt += 1) {
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
        await page.setViewportSize(probe);
        seen = await session.viewport();
      }
      if (seen.width !== size.width || seen.height !== size.height) {
        throw new Error(
          `no window size gives a ${size.width}x${size.height} viewport: ` +
            `${probe.width}x${probe.height} leaves ${seen.width}x${seen.height}`,
        );
      }
      // The render window resizes off the window's own resize event, and the
      // controller only learns the new viewport height when a frame is drawn
      // against it.
      await session.render();
      await session.frame();
    },
    load: (url) =>
      page.evaluate(
        (target) =>
          (window as never as ExampleWindow).pointCloudExample.load(target),
        url,
      ),
    readCamera: () =>
      page.evaluate(() =>
        (window as never as ExampleWindow).pointCloudExample.camera.read(),
      ),
    place: (next) =>
      page.evaluate(
        (value) =>
          (window as never as ExampleWindow).pointCloudExample.camera.place(
            value,
          ),
        next as Record<string, unknown>,
      ),
    azimuth: (degrees) =>
      page.evaluate(
        (value) =>
          (window as never as ExampleWindow).pointCloudExample.camera.azimuth(
            value,
          ),
        degrees,
      ),
    dolly: (factor) =>
      page.evaluate(
        (value) =>
          (window as never as ExampleWindow).pointCloudExample.camera.dolly(
            value,
          ),
        factor,
      ),
    setProjection: (projection) =>
      page.evaluate(
        (value) =>
          (window as never as ExampleWindow).pointCloudExample.setProjection(
            value,
          ),
        projection,
      ),
    setBudgetMode: (mode) =>
      page.evaluate(
        (value) =>
          (window as never as ExampleWindow).pointCloudExample.setBudgetMode(
            value,
          ),
        mode,
      ),
    setVisible: (visible) =>
      page.evaluate(
        (value) =>
          (window as never as ExampleWindow).pointCloudExample.setVisible(
            value,
          ),
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
          (
            window as never as ExampleWindow
          ).pointCloudExample.setDevicePixelRatio(value),
        ratio,
      ),
    setSyntheticFrameMs: (ms) =>
      page.evaluate(
        (value) =>
          (
            window as never as ExampleWindow
          ).pointCloudExample.setSyntheticFrameMs(value),
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
        `timed out waiting for convergence\nlast stats: ${shown(last)}`,
      );
    },
    close: async () => {
      await page.close();
      await server.close();
    },
  };

  // Every check is written against one render viewport — how much detail a
  // budget buys, and how far a frame time can push it, are both counted in
  // pixels — so the size is stated here rather than inherited from whatever
  // the browser's default window leaves once the page's panel has taken its
  // column.
  await session.resize(DEFAULT_VIEWPORT);

  // The cloud is requested by the query string; wait for it to be readable
  // before handing the session over, so no check has to re-implement that.
  await until(
    "the cloud to open",
    (value) => value.controller !== null && value.source !== null,
  );
  return session;
};

export type CameraReading = {
  position: number[];
  focalPoint: number[];
  viewUp: number[];
  parallelScale: number;
  viewAngle: number;
  parallelProjection: boolean;
};

export type ExampleWindow = {
  pointCloudExample: {
    stats(): ExampleStats;
    keys(): ExampleKeys;
    viewport(): Viewport;
    scene(): SceneReading;
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
};
