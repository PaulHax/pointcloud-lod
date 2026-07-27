/**
 * Record what streaming a cloud actually costs.
 *
 * This is not a test — nothing here passes or fails. The stress matrix proves
 * the invariants hold; this answers the questions a number has to answer:
 * how long until something is on screen, what the budget settled at, how the
 * two regimes differ, how much memory it took to get there. Those belong in a
 * report next to the plan, not in an assertion.
 *
 * Run it against whatever clouds the machine has:
 *
 *     npm run example:build
 *     POINTCLOUD_LOD_BROWSER_CLOUDS=/path/a.copc.laz:/path/b.copc.laz \
 *       npx vite-node scripts/measureClouds.mts -- /somewhere/outside/this/repo
 *
 * The output directory defaults to a scratch path and is deliberately not
 * inside the repository: a report carries point counts and screenshots of
 * whatever it was pointed at, and those belong to the deployment that owns
 * the data, not to this package.
 */

import { promises as fs } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import {
  closeBrowser,
  cloudsUnderTest,
  openExample,
  usingRealGpu,
  type ExampleSession,
  type ExampleStats,
} from "../test/browser/harness";

const OUT = resolve(
  process.argv[2] ?? join(tmpdir(), "pointcloud-lod-measurements"),
);

/**
 * What the frame-time columns actually measured. Under the default
 * SwiftShader launch they describe a software rasteriser and say nothing
 * about the library on real hardware; set `POINTCLOUD_LOD_BROWSER_GPU=1` for
 * numbers a report can stand behind. Recorded into the report so a table can
 * never be quoted without its provenance.
 */
const GL_MODE = usingRealGpu() ? "real-gpu" : "swiftshader-software";

/** How long the moving regime is held before the camera stops. */
const MOVING_MS = 20_000;

/** Wall-clock, from the caller's side of the page boundary. */
const since = (start: number): number => Date.now() - start;

const percentile = (values: number[], fraction: number): number | null => {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(
    sorted.length - 1,
    Math.max(0, Math.ceil(fraction * sorted.length) - 1),
  );
  return sorted[index]!;
};

interface Track {
  frameMs: number[];
  /** Last `lastFrameMs` value taken, so polling cannot re-count one paint. */
  lastSeenFrameMs: number;
  budgets: { atMs: number; regime: string; budget: number }[];
  peaks: {
    physicalTileOperations: number;
    physicalHierarchyOperations: number;
    residentPoints: number;
    decodedBytes: number;
    gpuResidentBytes: number;
    cachedBytes: number;
  };
}

const emptyTrack = (): Track => ({
  frameMs: [],
  lastSeenFrameMs: 0,
  budgets: [],
  peaks: {
    physicalTileOperations: 0,
    physicalHierarchyOperations: 0,
    residentPoints: 0,
    decodedBytes: 0,
    gpuResidentBytes: 0,
    cachedBytes: 0,
  },
});

const observe = (track: Track, stats: ExampleStats, atMs: number): void => {
  const cloud = stats.controller;
  if (cloud === null) return;
  // `lastFrameMs` is a gauge: it holds one paint's duration until the next
  // paint replaces it. The poll runs every 50 ms, so pushing it every poll
  // would weight each frame by how long it lasted — a "p90" over that series
  // describes wall-clock occupancy, not frames. One sample per new value.
  if (stats.lastFrameMs > 0 && stats.lastFrameMs !== track.lastSeenFrameMs) {
    track.lastSeenFrameMs = stats.lastFrameMs;
    track.frameMs.push(stats.lastFrameMs);
  }
  if (stats.governor !== null) {
    track.budgets.push({
      atMs,
      regime: stats.governor.regime,
      budget: stats.governor.trackBudget,
    });
  }
  const peaks = track.peaks;
  peaks.physicalTileOperations = Math.max(
    peaks.physicalTileOperations,
    cloud.physicalTileOperations,
  );
  peaks.physicalHierarchyOperations = Math.max(
    peaks.physicalHierarchyOperations,
    cloud.physicalHierarchyOperations,
  );
  peaks.residentPoints = Math.max(peaks.residentPoints, cloud.residentPoints);
  peaks.decodedBytes = Math.max(peaks.decodedBytes, cloud.decodedBytes);
  peaks.cachedBytes = Math.max(peaks.cachedBytes, cloud.cachedBytes);
  if (stats.adapter !== null) {
    peaks.gpuResidentBytes = Math.max(
      peaks.gpuResidentBytes,
      stats.adapter.gpuResidentBytes,
    );
  }
};

/** Poll until `done`, feeding every sample to the track. */
const sampleUntil = async (
  session: ExampleSession,
  track: Track,
  start: number,
  label: string,
  done: (stats: ExampleStats) => boolean,
  timeoutMs: number,
): Promise<ExampleStats> => {
  const deadline = Date.now() + timeoutMs;
  let last = await session.stats();
  while (Date.now() < deadline) {
    last = await session.stats();
    observe(track, last, since(start));
    if (done(last)) return last;
    await session.page.waitForTimeout(50);
  }
  // A run that never converged must not be written into the report as if it
  // had: every column downstream of this sample would silently describe an
  // arbitrary moment mid-stream.
  throw new Error(
    `timed out after ${timeoutMs} ms waiting for ${label}\n${JSON.stringify(last, null, 2)}`,
  );
};

const measure = async (
  cloud: ReturnType<typeof cloudsUnderTest>[number],
  index: number,
) => {
  const start = Date.now();
  const session = await openExample({ cloud: cloud.urlPath });
  const load = emptyTrack();
  const moving = emptyTrack();
  const stationary = emptyTrack();

  try {
    await session.setBudgetMode("adaptive");

    // Time to first visible points: the first moment the renderer is actually
    // drawing something, not the first moment a request was issued. Measured
    // from before the page is opened, so it includes navigation and the
    // example's own startup — the number a user waits through, not the
    // library's share of it.
    const firstDrawn = await sampleUntil(
      session,
      load,
      start,
      "the first drawn points",
      (stats) => (stats.adapter?.drawnPoints ?? 0) > 0,
      120_000,
    );
    const timeToFirstPointsMs = since(start);

    const converged = await sampleUntil(
      session,
      load,
      start,
      "load convergence",
      (stats) =>
        stats.controller !== null &&
        stats.controller.physicalTileOperations === 0 &&
        stats.controller.queuedTiles === 0 &&
        stats.controller.physicalHierarchyOperations === 0,
      180_000,
    );
    const timeToConvergedMs = since(start);
    await session.page.screenshot({
      path: `${OUT}/cloud-${index + 1}-loaded.png`,
    });

    // Moving: orbit continuously, which is the regime a video-driven camera
    // and a user drag both land in. Bounded by time rather than by a step
    // count, because a step is a painted frame and a painted frame on a real
    // cloud is seconds — a fixed count would run for a quarter of an hour on
    // the largest asset and finish in moments on the fixture, measuring two
    // different things under one name.
    const movingStart = Date.now();
    let movingFrames = 0;
    while (Date.now() - movingStart < MOVING_MS) {
      await session.azimuth(1.5);
      await session.frame();
      movingFrames += 1;
      observe(moving, await session.stats(), since(start));
    }
    const movingEnd = Date.now();
    await session.page.screenshot({
      path: `${OUT}/cloud-${index + 1}-moving.png`,
    });

    // Stationary: stop, and time how long refinement takes to settle.
    const stopped = Date.now();
    const settled = await sampleUntil(
      session,
      stationary,
      start,
      "stationary convergence",
      (stats) =>
        stats.governor !== null &&
        stats.governor.regime === "stationary" &&
        !stats.governor.needsFrame &&
        stats.controller !== null &&
        stats.controller.physicalTileOperations === 0 &&
        stats.controller.queuedTiles === 0,
      180_000,
    );
    const stopToStationaryMs = Date.now() - stopped;
    await session.page.screenshot({
      path: `${OUT}/cloud-${index + 1}-settled.png`,
    });

    return {
      cloud: cloud.name,
      sourcePoints: settled.sourcePoints,
      timeToFirstPointsMs,
      timeToConvergedMs,
      movingDurationMs: movingEnd - movingStart,
      movingFrames,
      stopToStationaryMs,
      firstDrawnPoints: firstDrawn.adapter?.drawnPoints ?? 0,
      atConvergence: {
        selectedPoints: converged.controller?.selection.targetPoints ?? null,
        residentPoints: converged.controller?.residentPoints ?? null,
        cachedTiles: converged.controller?.cachedTiles ?? null,
        gpuResidentPoints: converged.adapter?.gpuResidentPoints ?? null,
      },
      atSettled: {
        selectedPoints: settled.controller?.selection.targetPoints ?? null,
        residentPoints: settled.controller?.residentPoints ?? null,
        budget: settled.governor?.trackBudget ?? null,
        regime: settled.governor?.regime ?? null,
        memoryCeilingPoints: settled.governor?.memoryCeilingPoints ?? null,
      },
      frameMs: {
        loadP90: percentile(load.frameMs, 0.9),
        movingP90: percentile(moving.frameMs, 0.9),
        stationaryP90: percentile(stationary.frameMs, 0.9),
      },
      budgets: {
        moving: moving.budgets.at(-1)?.budget ?? null,
        stationary: stationary.budgets.at(-1)?.budget ?? null,
        movingSeries: moving.budgets,
        stationarySeries: stationary.budgets,
      },
      peaks: {
        load: load.peaks,
        moving: moving.peaks,
        stationary: stationary.peaks,
      },
      browserErrors: [...session.failures],
    };
  } finally {
    await session.close();
  }
};

const main = async (): Promise<void> => {
  await fs.mkdir(OUT, { recursive: true });
  const clouds = cloudsUnderTest();
  const results = [];
  for (const [index, cloud] of clouds.entries()) {
    process.stdout.write(`measuring ${cloud.name}…\n`);
    results.push(await measure(cloud, index));
  }
  await closeBrowser();

  const report = { measuredAt: new Date().toISOString(), gl: GL_MODE, results };
  await fs.writeFile(
    `${OUT}/measurements.json`,
    `${JSON.stringify(report, null, 2)}\n`,
  );

  const table = [
    "| Cloud | Source points | To first points | To converged | Stop→stationary | Moving budget | Stationary budget | Peak decoded | Peak GPU | Errors |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ...results.map((r) =>
      [
        r.cloud,
        r.sourcePoints.toLocaleString(),
        `${r.timeToFirstPointsMs} ms`,
        `${r.timeToConvergedMs} ms`,
        `${r.stopToStationaryMs} ms`,
        r.budgets.moving?.toLocaleString() ?? "—",
        r.budgets.stationary?.toLocaleString() ?? "—",
        `${(Math.max(r.peaks.load.decodedBytes, r.peaks.moving.decodedBytes, r.peaks.stationary.decodedBytes) / 1024 ** 2).toFixed(1)} MB`,
        `${(Math.max(r.peaks.load.gpuResidentBytes, r.peaks.moving.gpuResidentBytes, r.peaks.stationary.gpuResidentBytes) / 1024 ** 2).toFixed(1)} MB`,
        String(r.browserErrors.length),
      ].join(" | "),
    ),
  ].join("\n");
  const glNote =
    GL_MODE === "real-gpu"
      ? "Frame-time columns measured on the machine's real GPU."
      : "Frame-time columns measured under SwiftShader (software rasteriser) " +
        "and describe it, not the library on real hardware; set " +
        "POINTCLOUD_LOD_BROWSER_GPU=1 for reportable numbers.";
  await fs.writeFile(`${OUT}/measurements.md`, `${table}\n\n${glNote}\n`);
  process.stdout.write(`\n${table}\n\n${glNote}\n\nwrote ${OUT}\n`);
};

await main();
