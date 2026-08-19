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
import { basename, isAbsolute, resolve } from "node:path";

import { afterAll, describe, expect, it } from "vitest";

import type {
  InputRecording,
  RecordedPoseSample,
} from "../../examples/vtk/scene/inputRecorder";
import type { CacheMode } from "./httpCache";
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

const artifactName = (
  recordingPath: string,
  config: BenchmarkConfig,
  repeat: number,
): string =>
  `${basename(recordingPath, ".json")}__${config.name}__${repeat + 1}.json`;

describe("recorded-gesture replay benchmark", { tags: ["perf"] }, () => {
  afterAll(async () => {
    await closeBenchmarkBrowser();
  });

  it("replays every recording against every configuration", async () => {
    const recordings = await listRecordings();
    if (recordings.length === 0) {
      throw new Error(
        `no recordings in ${RECORDINGS_DIR}. Capture one by opening the ` +
          `example with ?record=1 and downloading the JSON.`,
      );
    }
    if (!usingRealGpu && !allowSoftware) {
      throw new Error(
        "the replay benchmark requires POINTCLOUD_LOD_BROWSER_GPU=1: frame " +
          "times under a software rasterizer describe the rasterizer. Set " +
          "POINTCLOUD_LOD_REPLAY_ALLOW_SOFTWARE=1 to exercise the pipeline " +
          "without measuring anything.",
      );
    }
    const configs = await loadConfigs();
    await mkdir(outputDirectory, { recursive: true });

    for (const recordingPath of recordings) {
      const recording = JSON.parse(
        await readFile(recordingPath, "utf8"),
      ) as InputRecording;
      if (recording.schemaVersion !== 1) {
        throw new Error(`${recordingPath} is not a version 1 recording`);
      }

      for (const config of configs) {
        for (let repeat = 0; repeat < repeats; repeat += 1) {
          const session = await openScene({
            path: replayPath(recording.href),
            deviceScaleFactor: recording.viewer.devicePixelRatio,
            headless: !usingRealGpu,
            cache: {
              mode: networkMode,
              directory: cacheDirectory,
              origins: DATASET_ORIGINS,
              ...(networkMode === "replay"
                ? {
                    shape: {
                      latencyMs,
                      bytesPerSecond: (mbps * 1_000_000) / 8,
                    },
                  }
                : {}),
            },
          });
          try {
            await session.resizeViewer({
              width: recording.viewer.widthCssPx,
              height: recording.viewer.heightCssPx,
            });
            await session.settle();
            await applyConfig(session, config);
            await session.settle();

            // The gesture is only reproducible from the camera it was recorded
            // from. Placing it before telemetry starts keeps the reframing
            // work out of the measurement.
            await session.placeCamera({
              position: recording.startPose.position,
              focalPoint: recording.startPose.focalPoint,
              viewUp: recording.startPose.viewUp,
            });
            await session.settle();
            await session.render();
            await session.frame();
            await session.settle();

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
            const settledAfter = await session.settle();
            await session.markTelemetry("settled-after-replay");
            await session.stopInputRecorder();
            await session.stopTelemetry();

            const replayed = (await session.inputRecording()) as InputRecording;
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
            expect(
              settledAfter.datasets.every(
                (dataset) => dataset.state === "settled",
              ),
            ).toBe(true);

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
                ...(networkMode === "replay"
                  ? { latencyMs, mbpsPerRequest: mbps }
                  : {}),
                cache: cacheStats,
                transferredBytes: session.transferredBytes(),
              },
              fidelity: { dispatch: replay.dispatch, drift },
              datasets: await session.datasets(),
              finalStats: await session.stats(),
              trace,
              replayedPoses: replayed.poses,
            };
            const path = resolve(
              outputDirectory,
              artifactName(recordingPath, config, repeat),
            );
            await writeFile(
              path,
              `${JSON.stringify(artifact, null, 2)}\n`,
              "utf8",
            );
            process.stdout.write(
              `REPLAY_ARTIFACT ${path}\n` +
                `  frames=${trace.summary.frames} clean=${trace.summary.cleanFrames}` +
                ` longTasks=${trace.summary.longTasks}` +
                ` lateness=${replay.dispatch.meanLatenessMs.toFixed(1)}/${replay.dispatch.maxLatenessMs.toFixed(1)} ms` +
                ` drift=${drift ? `${(drift.meanRelativeError * 100).toFixed(2)}%/${(drift.maxRelativeError * 100).toFixed(2)}%` : "n/a"}\n`,
            );
          } finally {
            await session.close();
          }
        }
      }
    }
  });
});
