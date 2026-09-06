/**
 * The recorded-gesture benchmark.
 *
 * One recorded hand gesture, replayed against every configuration under test,
 * on the same scene, at the same viewer size and device pixel ratio it was
 * captured at. What comes out per run is a telemetry trace plus the two
 * fidelity numbers that say whether the run is comparable at all: how closely
 * the input kept to its recorded schedule, and how far the camera drifted from
 * the recorded path.
 *
 * Nothing here asserts a speed. A benchmark that failed when a number moved
 * would be a regression test with a much worse signal-to-noise ratio; this
 * writes artifacts and `scripts/analyzeReplay.mjs` compares them. What it does
 * assert is that a run is *evidence*: a real GPU, a scene that converged, no
 * page errors, and — under a replayed network — no request the cache could not
 * answer.
 */

import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { cpus, loadavg } from "node:os";
import { basename, isAbsolute, resolve } from "node:path";

import sharp from "sharp";
import { afterAll, describe, expect, it } from "vitest";

import type {
  InputRecording,
  RecordedPoseSample,
} from "../../examples/vtk/harness/inputRecorder";
import type { CacheMode, HttpCache } from "./httpCache";
import { compareCameraTracks, replayInput } from "./inputReplay";
import {
  closeBenchmarkBrowser,
  openScene,
  replayPath,
  type SceneSession,
} from "./sceneHarness";

const RECORDINGS_DIR = resolve(process.cwd(), "test/recordings");

const environmentPath = (name: string, fallback: string): string => {
  const raw = process.env[name]?.trim();
  if (raw === undefined || raw === "") return resolve(process.cwd(), fallback);
  return isAbsolute(raw) ? raw : resolve(process.cwd(), raw);
};

const environmentNumber = (name: string, fallback: number): number => {
  const raw = process.env[name]?.trim();
  const value = raw === undefined || raw === "" ? fallback : Number(raw);
  if (!Number.isFinite(value)) {
    throw new Error(`${name} must be a finite number`);
  }
  return value;
};

const outputDirectory = environmentPath(
  "POINTCLOUD_LOD_REPLAY_OUT",
  "artifacts/replay",
);
const cacheDirectory = environmentPath(
  "POINTCLOUD_LOD_REPLAY_CACHE_DIR",
  "artifacts/replay/cache",
);
const networkMode = (
  process.env.POINTCLOUD_LOD_REPLAY_NETWORK ?? "replay"
).trim() as CacheMode;
if (!["live", "record", "replay"].includes(networkMode)) {
  throw new Error(
    "POINTCLOUD_LOD_REPLAY_NETWORK must be live, record or replay",
  );
}
const latencyMs = environmentNumber("POINTCLOUD_LOD_REPLAY_LATENCY_MS", 40);
const mbps = environmentNumber("POINTCLOUD_LOD_REPLAY_MBPS", 80);
const repeats = Math.max(
  1,
  Math.floor(environmentNumber("POINTCLOUD_LOD_REPLAY_REPEATS", 1)),
);
/**
 * How long a scene is given to converge before the run is called a failure.
 * A sweep pays this several times per run, so a configuration that cannot
 * converge at all should say so quickly rather than holding a browser open
 * for the default.
 */
const settleMs = environmentNumber("POINTCLOUD_LOD_REPLAY_SETTLE_MS", 120_000);
const usingRealGpu = (process.env.POINTCLOUD_LOD_BROWSER_GPU ?? "") !== "";
/**
 * Runs the whole pipeline against a software rasteriser, for checking that the
 * benchmark still works rather than for measuring anything. Every artifact it
 * writes carries `softwareRenderer: true`, and the analysis prints that in
 * place of a headline number, so such a run cannot be mistaken for evidence.
 */
const allowSoftware =
  (process.env.POINTCLOUD_LOD_REPLAY_ALLOW_SOFTWARE ?? "") !== "";

/**
 * Origins whose traffic the cache owns. Anything else — the local server
 * carrying the example itself — is never cached or shaped, so the page loads
 * at full speed and only the dataset traffic is under the network profile.
 */
const DATASET_ORIGINS = [
  "https://open-lidar-data.s3.eu-central-1.amazonaws.com",
  "https://s3.amazonaws.com",
  "https://data.3dbag.nl",
];

/** One setting sweep. Absent sections leave the page's own defaults alone. */
type BenchmarkConfig = {
  readonly name: string;
  readonly quality?: {
    readonly interactionTargetMs?: number;
    readonly stationaryTargetMs?: number;
  };
  /** Applied to every 3D Tiles dataset in the scene. */
  readonly tiles?: {
    readonly screenSpaceErrorPx?: number;
  };
  /** Applied to every point-cloud dataset in the scene. */
  readonly points?: {
    readonly budgetMode?: "adaptive" | "fixed";
    readonly fixedPointBudget?: number;
    readonly maximumPoints?: number | null;
    readonly pointSizeMode?: "auto" | "fixed";
    readonly pointSize?: number;
  };
};

const DEFAULT_CONFIGS: readonly BenchmarkConfig[] = [{ name: "baseline" }];

const loadConfigs = async (): Promise<readonly BenchmarkConfig[]> => {
  const raw = process.env.POINTCLOUD_LOD_REPLAY_CONFIGS?.trim();
  if (raw === undefined || raw === "") return DEFAULT_CONFIGS;
  const path = isAbsolute(raw) ? raw : resolve(process.cwd(), raw);
  const parsed = JSON.parse(await readFile(path, "utf8")) as {
    configs?: BenchmarkConfig[];
  };
  if (!Array.isArray(parsed.configs) || parsed.configs.length === 0) {
    throw new Error(`${path} must hold a non-empty "configs" array`);
  }
  return parsed.configs;
};

const listRecordings = async (): Promise<string[]> => {
  const named = (process.env.POINTCLOUD_LOD_REPLAY_RECORDINGS ?? "")
    .split(",")
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0);
  if (named.length > 0) {
    return named.map((entry) =>
      isAbsolute(entry) ? entry : resolve(RECORDINGS_DIR, entry),
    );
  }
  const entries = await readdir(RECORDINGS_DIR).catch(() => [] as string[]);
  return entries
    .filter((entry) => entry.endsWith(".json"))
    .sort()
    .map((entry) => resolve(RECORDINGS_DIR, entry));
};

const applyConfig = async (
  session: SceneSession,
  config: BenchmarkConfig,
): Promise<void> => {
  if (config.quality) await session.setQualityTargets(config.quality);
  const datasets = await session.datasets();
  for (const dataset of datasets) {
    if (
      dataset.kind === "tiles" &&
      config.tiles?.screenSpaceErrorPx !== undefined
    ) {
      await session.setScreenSpaceErrorPx(
        dataset.id,
        config.tiles.screenSpaceErrorPx,
      );
    }
    if (dataset.kind !== "points") continue;
    const points = config.points;
    if (points === undefined) continue;
    if (points.budgetMode !== undefined) {
      await session.setBudgetMode(dataset.id, points.budgetMode);
    }
    if (points.fixedPointBudget !== undefined) {
      await session.setFixedPointBudget(dataset.id, points.fixedPointBudget);
    }
    if (points.maximumPoints !== undefined) {
      await session.setMaximumPoints(dataset.id, points.maximumPoints);
    }
    if (points.pointSizeMode !== undefined || points.pointSize !== undefined) {
      await session.setPointSize(
        dataset.id,
        points.pointSizeMode ?? "auto",
        points.pointSize ?? 0.5,
      );
    }
  }
};

/**
 * What else the machine was doing, sampled either side of a run.
 *
 * A benchmark is supposed to have the machine to itself, and a run that did
 * not is not comparable with one that did — but the run still happened and its
 * artifact is still worth keeping, so this records rather than refuses.
 * `perCore` is the one to read: load is a queue length, so it only means
 * something against the number of cores available to drain it.
 *
 * Two samples, because contention that arrives mid-run is exactly the case a
 * single reading at the start would miss. On a platform with no load average —
 * Windows reports zeroes — the numbers are null rather than a flattering zero.
 */
const machineLoad = (): {
  readonly loadAvg1: number | null;
  readonly cores: number;
  readonly perCore: number | null;
} => {
  const cores = Math.max(1, cpus().length);
  const loadAvg1 = loadavg()[0] ?? 0;
  const usable = Number.isFinite(loadAvg1) && loadAvg1 > 0 ? loadAvg1 : null;
  return {
    loadAvg1: usable,
    cores,
    perCore: usable === null ? null : usable / cores,
  };
};

/** Load either side of a run, as "before->after per-core", for one line. */
const loadSummary = (
  before: ReturnType<typeof machineLoad>,
  after: ReturnType<typeof machineLoad>,
): string =>
  before.perCore === null || after.perCore === null
    ? "n/a"
    : `${before.perCore.toFixed(2)}->${after.perCore.toFixed(2)}/core`;

const artifactName = (
  recordingPath: string,
  config: BenchmarkConfig,
  repeat: number,
): string =>
  `${basename(recordingPath, ".json")}__${config.name}__${repeat + 1}.json`;

type FrameEvidence = {
  readonly path: string;
  readonly width: number;
  readonly height: number;
  readonly backgroundRgb: readonly [number, number, number];
  readonly visiblePixels: number;
  readonly visibleFraction: number;
  readonly maximumBackgroundDelta: number;
};

const captureFrameEvidence = async (
  session: SceneSession,
  path: string,
): Promise<FrameEvidence> => {
  const png = await session.page.locator("#viewer canvas").first().screenshot();
  await writeFile(path, png);
  const decoded = await sharp(png)
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const { width, height, channels } = decoded.info;
  const cornerOffsets = [
    0,
    (width - 1) * channels,
    (height - 1) * width * channels,
    (height * width - 1) * channels,
  ];
  const backgroundRgb = [0, 1, 2].map((channel) => {
    const values = cornerOffsets
      .map((offset) => decoded.data[offset + channel]!)
      .sort((left, right) => left - right);
    return Math.round((values[1]! + values[2]!) / 2);
  }) as [number, number, number];
  let visiblePixels = 0;
  let maximumBackgroundDelta = 0;
  for (let offset = 0; offset < decoded.data.length; offset += channels) {
    const delta = Math.max(
      Math.abs(decoded.data[offset]! - backgroundRgb[0]),
      Math.abs(decoded.data[offset + 1]! - backgroundRgb[1]),
      Math.abs(decoded.data[offset + 2]! - backgroundRgb[2]),
    );
    maximumBackgroundDelta = Math.max(maximumBackgroundDelta, delta);
    if (delta > 8) visiblePixels += 1;
  }
  return {
    path,
    width,
    height,
    backgroundRgb,
    visiblePixels,
    visibleFraction: visiblePixels / (width * height),
    maximumBackgroundDelta,
  };
};

/**
 * One test per recording, configuration and repeat, rather than one test that
 * loops over all of them. A sweep is the point of this file, and a single test
 * around the whole sweep gives every combination one shared timeout: adding a
 * fourth configuration would push a passing sweep over a limit that describes
 * nothing about any run in it, and the artifacts already written would be
 * reported as a failure. Split, each run is bounded by its own gesture's
 * length and the ones that finished are kept.
 */
const recordingPaths = await listRecordings();
if (recordingPaths.length === 0) {
  throw new Error(
    `no recordings in ${RECORDINGS_DIR}. Capture one by opening the ` +
      `example with ?record=1 and downloading the JSON.`,
  );
}
const benchmarkConfigs = await loadConfigs();

/** Settles a run waits out: two to configure, two to frame, one after. */
const SETTLES_PER_RUN = 5;
/**
 * Everything a run spends outside settling and the gesture: starting a
 * browser, loading the page, applying a configuration, and writing an
 * artifact that carries a frame-by-frame trace.
 */
const RUN_OVERHEAD_MS = 120_000;

/**
 * A run is given what it can actually need, rather than a fixed ceiling that
 * silently stops fitting. At the default settle budget five settles alone
 * exhaust a 600 s timeout, so a scene converging slowly failed as "timed out"
 * — indistinguishable from a hang, and the artifact that would have said
 * which was never written.
 */
const runTimeoutMs = (gestureMs: number): number =>
  SETTLES_PER_RUN * settleMs + gestureMs + RUN_OVERHEAD_MS;

const gestureDurations = new Map<string, number>(
  await Promise.all(
    recordingPaths.map(
      async (path): Promise<[string, number]> => [
        path,
        (JSON.parse(await readFile(path, "utf8")) as InputRecording).durationMs,
      ],
    ),
  ),
);

describe("recorded-gesture replay benchmark", { tags: ["perf"] }, () => {
  afterAll(async () => {
    await closeBenchmarkBrowser();
  });

  for (const recordingPath of recordingPaths) {
    for (const config of benchmarkConfigs) {
      for (let repeat = 0; repeat < repeats; repeat += 1) {
        it(
          `${basename(recordingPath, ".json")} · ${config.name} · run ${
            repeat + 1
          }`,
          async () => {
            if (!usingRealGpu && !allowSoftware) {
              throw new Error(
                "the replay benchmark requires POINTCLOUD_LOD_BROWSER_GPU=1: " +
                  "frame times under a software rasterizer describe the " +
                  "rasterizer. Set POINTCLOUD_LOD_REPLAY_ALLOW_SOFTWARE=1 to " +
                  "exercise the pipeline without measuring anything.",
              );
            }
            await mkdir(outputDirectory, { recursive: true });
            const path = resolve(
              outputDirectory,
              artifactName(recordingPath, config, repeat),
            );
            const framePath = path.replace(/\.json$/, "__frame.png");
            const loadBefore = machineLoad();
            const recording = JSON.parse(
              await readFile(recordingPath, "utf8"),
            ) as InputRecording;
            if (recording.schemaVersion !== 1) {
              throw new Error(`${recordingPath} is not a version 1 recording`);
            }

            const settle = async (
              session: SceneSession & { readonly cache: HttpCache | null },
            ) => {
              try {
                return await session.settle(settleMs);
              } catch (error) {
                // A replayed network that cannot answer a request aborts it, and
                // an aborted tile is one the scene waits on forever. Reported as
                // "did not settle" alone that reads as a streaming bug, so the
                // requests nothing recorded are named here instead.
                const missing = session.cache?.stats().unrecorded ?? [];
                if (missing.length === 0) throw error;
                throw new Error(
                  `${(error as Error).message}\n\n` +
                    `${missing.length} request(s) the ${networkMode} cache could ` +
                    `not answer, which is why it never converged:\n` +
                    `${missing.slice(0, 10).join("\n")}`,
                );
              }
            };

            const session = await openScene({
              path: replayPath(recording.href),
              deviceScaleFactor: recording.viewer.devicePixelRatio,
              headless: !usingRealGpu,
              cache: {
                mode: networkMode,
                directory: cacheDirectory,
                origins: DATASET_ORIGINS,
                // Shaped whenever the cache is answering, so a cached byte
                // costs what a networked one would. A read-through fetch pays
                // the real network instead and is counted separately.
                ...(networkMode === "live"
                  ? {}
                  : {
                      shape: {
                        latencyMs,
                        bytesPerSecond: (mbps * 1_000_000) / 8,
                      },
                    }),
              },
            });
            try {
              await session.resizeViewer({
                width: recording.viewer.widthCssPx,
                height: recording.viewer.heightCssPx,
              });
              await settle(session);
              await applyConfig(session, config);
              await settle(session);

              // The gesture is only reproducible from the camera it was recorded
              // from. Placing it before telemetry starts keeps the reframing
              // work out of the measurement.
              await session.placeCamera({
                position: recording.startPose.position,
                focalPoint: recording.startPose.focalPoint,
                viewUp: recording.startPose.viewUp,
              });
              await settle(session);
              await session.render();
              await session.frame();
              await settle(session);

              // A renderer can keep reporting frames and valid timings while
              // drawing only its background. Capture the actual vtk canvas
              // before telemetry and reject such a run as benchmark evidence.
              const frame = await captureFrameEvidence(session, framePath);
              if (
                frame.visiblePixels <= 256 ||
                frame.visibleFraction <= 0.00005
              ) {
                throw new Error(
                  `vtk canvas is blank (${frame.visiblePixels} visible pixels, ` +
                    `${(frame.visibleFraction * 100).toFixed(4)}%):\n` +
                    session.failures.join("\n"),
                );
              }

              await session.startTelemetry();
              await session.startInputRecorder();
              await session.markTelemetry("replay-start");

              const replay = await replayInput({
                page: session.page,
                recording,
                viewer: await session.viewerBox(),
                onMarker: (label) => session.markTelemetry(label),
              });

              await session.markTelemetry("replay-ended");
              await session.frame();
              const settledAfter = await settle(session);
              await session.markTelemetry("settled-after-replay");
              await session.stopInputRecorder();
              await session.stopTelemetry();

              const replayed =
                (await session.inputRecording()) as InputRecording;
              const drift = compareCameraTracks(
                recording.poses,
                replayed.poses as readonly RecordedPoseSample[],
              );
              const trace = await session.telemetryTrace();
              const cacheStats = session.cache?.stats() ?? null;

              expect(trace.environment.webgl.softwareRenderer).toBe(
                !usingRealGpu && allowSoftware,
              );
              expect(trace.summary.frames).toBeGreaterThan(0);
              expect(session.failures).toEqual([]);
              expect(cacheStats?.unrecorded ?? []).toEqual([]);
              // A dataset that gave up on some of its content has stopped
              // working, which is a run that finished. What it is not is a run
              // that drew everything, so the state travels with the artifact and
              // is printed beside it rather than passing as an ordinary result.
              expect(
                settledAfter.datasets.every(
                  (dataset) =>
                    dataset.state === "settled" || dataset.state === "error",
                ),
              ).toBe(true);
              const failedDatasets = settledAfter.datasets.filter(
                (dataset) => dataset.state === "error",
              );

              const artifact = {
                schemaVersion: 1 as const,
                recording: {
                  path: recordingPath,
                  name: basename(recordingPath, ".json"),
                  recordedAt: recording.recordedAt,
                  href: recording.href,
                  label: recording.label,
                  viewer: recording.viewer,
                  durationMs: recording.durationMs,
                  events: recording.events.length,
                  markers: recording.markers,
                  environment: recording.environment,
                },
                config,
                repeat: repeat + 1,
                network: {
                  mode: networkMode,
                  ...(networkMode === "live"
                    ? {}
                    : { latencyMs, mbpsPerRequest: mbps }),
                  cache: cacheStats,
                  transferredBytes: session.transferredBytes(),
                },
                fidelity: { dispatch: replay.dispatch, drift },
                machine: { loadBefore, loadAfter: machineLoad() },
                activity: settledAfter,
                datasets: await session.datasets(),
                finalStats: await session.stats(),
                frame,
                trace,
                replayedPoses: replayed.poses,
              };
              await writeFile(
                path,
                `${JSON.stringify(artifact, null, 2)}\n`,
                "utf8",
              );
              process.stdout.write(
                `REPLAY_ARTIFACT ${path}\n` +
                  `  frames=${trace.summary.frames} clean=${trace.summary.cleanFrames}` +
                  ` longTasks=${trace.summary.longTasks}` +
                  ` visible=${frame.visiblePixels}/${(frame.visibleFraction * 100).toFixed(2)}%` +
                  ` lateness=${replay.dispatch.meanLatenessMs.toFixed(1)}/${replay.dispatch.maxLatenessMs.toFixed(1)} ms` +
                  ` drift=${drift ? `${(drift.meanRelativeError * 100).toFixed(2)}%/${(drift.maxRelativeError * 100).toFixed(2)}%` : "n/a"}` +
                  ` load=${loadSummary(loadBefore, machineLoad())}\n` +
                  failedDatasets
                    .map(
                      (dataset) =>
                        `  INCOMPLETE ${dataset.id}: ${dataset.detail}\n`,
                    )
                    .join(""),
              );
            } finally {
              await session.close();
            }
          },
          runTimeoutMs(gestureDurations.get(recordingPath) ?? 0),
        );
      }
    }
  }
});
