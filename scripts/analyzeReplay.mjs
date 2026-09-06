/**
 * Compares replay benchmark artifacts.
 *
 * A run's headline number is not its average frame time. A configuration can
 * buy a smooth gesture by drawing almost nothing, or a detailed view by
 * stuttering through it, and an average hides both. So every run is reported
 * as a pair: what the gesture cost while it was moving, and what detail the
 * view had once it settled — plus how long it took to get there.
 *
 * Frames are split by the coordinator's own motion regime rather than by a
 * timer, because that is the classification the adaptive loop acted on. A
 * frame the governor thought was part of a gesture belongs with the gesture
 * even if the hand had already stopped.
 *
 * Usage:
 *   node scripts/analyzeReplay.mjs [artifact-directory] [--json out.json]
 */

import { readdir, readFile, writeFile } from "node:fs/promises";
import { basename, resolve } from "node:path";

import {
  churnMetrics,
  detailOf,
  framesOf,
  integer,
  isMoving,
  markerTimes,
  number,
  percent,
  percentile,
  phaseMetrics,
  table,
} from "./traceMetrics.mjs";

/**
 * The same run repeated is not the same numbers, and a tuning change is only
 * worth reporting if it moved a metric further than repetition does. Each
 * configuration is therefore summarised as a median and the spread its own
 * repeats covered, so the two can be read against each other; a difference
 * inside the spread is a difference this benchmark cannot see.
 */
const across = (rows, read) => {
  const values = rows.map(read).filter((value) => Number.isFinite(value));
  if (values.length === 0) return { median: null, spread: null, runs: 0 };
  const sorted = [...values].sort((left, right) => left - right);
  return {
    median: percentile(sorted, 0.5),
    spread: sorted[sorted.length - 1] - sorted[0],
    runs: sorted.length,
  };
};

const withSpread = (summary, format) =>
  summary.median === null
    ? "—"
    : summary.runs < 2
      ? format(summary.median)
      : `${format(summary.median)} ±${format(summary.spread / 2)}`;

const analyzeArtifact = (artifact) => {
  const trace = artifact.trace;
  const frames = framesOf(trace);
  const marks = markerTimes(trace);
  const replayStart = marks.get("replay-start") ?? 0;
  const replayEnd = marks.get("replay-ended") ?? Number.POSITIVE_INFINITY;
  const settledAfter = marks.get("settled-after-replay") ?? null;

  const duringReplay = frames.filter(
    (frame) => frame.atMs >= replayStart && frame.atMs <= replayEnd,
  );
  const moving = duringReplay.filter(isMoving);
  const afterReplay = frames.filter(
    (frame) =>
      frame.atMs > replayEnd &&
      (settledAfter === null || frame.atMs <= settledAfter),
  );
  const settledFrames = frames.filter(
    (frame) => settledAfter !== null && frame.atMs >= settledAfter,
  );
  // The last frame of the run describes the converged view even when no frame
  // was recorded after the settle marker, which happens on a view that had
  // nothing left to draw.
  const settledDetail = detailOf(
    settledFrames[settledFrames.length - 1] ??
      frames[frames.length - 1] ?? { state: {} },
  );

  const longTasks = trace.events.filter((event) => event.type === "long-task");
  const longTaskMs = longTasks.reduce(
    (sum, event) => sum + event.durationMs,
    0,
  );

  return {
    name: artifact.recording.name,
    config: artifact.config.name,
    repeat: artifact.repeat,
    renderer:
      trace.environment.webgl.unmaskedRenderer ??
      trace.environment.webgl.renderer,
    softwareRenderer: trace.environment.webgl.softwareRenderer,
    network: {
      mode: artifact.network.mode,
      transferredMb: artifact.network.transferredBytes / 1e6,
      cacheHits: artifact.network.cache?.hits ?? null,
      cacheMisses: artifact.network.cache?.misses ?? null,
      fetched: artifact.network.cache?.fetched ?? null,
    },
    fidelity: {
      meanLatenessMs: artifact.fidelity.dispatch.meanLatenessMs,
      maxLatenessMs: artifact.fidelity.dispatch.maxLatenessMs,
      lateEvents: artifact.fidelity.dispatch.lateEvents,
      timedDrift: artifact.fidelity.drift?.meanRelativeError ?? null,
      pathDrift: artifact.fidelity.drift?.meanRelativePathError ?? null,
      maxPathDrift: artifact.fidelity.drift?.maxRelativePathError ?? null,
    },
    gesture: phaseMetrics(duringReplay),
    motion: phaseMetrics(moving),
    // Churn is reported per regime and never as one number over the run.
    // Refinement after a settle is wanted change; counting it against a
    // configuration would reward one that simply refuses to refine.
    movingChurn: churnMetrics(moving),
    settledChurn: churnMetrics(settledFrames),
    convergence: {
      // How long the view took to finish what the gesture asked of it. The
      // number a user experiences as "it catches up quickly".
      msAfterGesture:
        settledAfter === null || replayEnd === Number.POSITIVE_INFINITY
          ? null
          : settledAfter - replayEnd,
      framesAfterGesture: afterReplay.length,
    },
    settled: {
      // The phase means describe whatever frames landed after the settle
      // marker, which on a converged view is often none. The detail of the
      // last frame drawn is what the view actually ended up showing, so it
      // wins over an average of nothing.
      ...phaseMetrics(settledFrames),
      ...settledDetail,
    },
    longTasks: { count: longTasks.length, totalMs: longTaskMs },
  };
};

const COLUMNS = [
  { title: "config", value: (row) => row.config },
  { title: "run", value: (row) => String(row.repeat) },
  { title: "moving fps", value: (row) => number(row.motion.fps) },
  { title: "moving p50 ms", value: (row) => number(row.motion.intervalMs.p50) },
  { title: "moving p95 ms", value: (row) => number(row.motion.intervalMs.p95) },
  { title: "moving gpu p95", value: (row) => number(row.motion.gpuMs.p95) },
  { title: "moving quality", value: (row) => percent(row.motion.quality) },
  { title: "moving pts", value: (row) => integer(row.motion.drawnPoints) },
  { title: "moving tris", value: (row) => integer(row.motion.drawnTriangles) },
  {
    title: "settle ms",
    value: (row) => integer(row.convergence.msAfterGesture),
  },
  { title: "settled pts", value: (row) => integer(row.settled.drawnPoints) },
  {
    title: "settled tris",
    value: (row) => integer(row.settled.drawnTriangles),
  },
  { title: "settled sse", value: (row) => number(row.settled.worstSse) },
  { title: "long tasks", value: (row) => integer(row.longTasks.count) },
  { title: "MB", value: (row) => number(row.network.transferredMb) },
  {
    // Requests that went to the origin rather than the cache. A run with more
    // than a handful paid a real network for part of its measurement.
    title: "live req",
    value: (row) => integer(row.network.fetched),
  },
  { title: "path drift", value: (row) => percent(row.fidelity.pathDrift) },
];

/**
 * What moved, rather than what it cost.
 *
 * Reversals are listed beside changes because they are the half that reads as
 * noise: a run that refines steadily upward has many changes and no reversals
 * and is behaving correctly, while one that trades the same detail back and
 * forth has few changes and many reversals and is the reported defect.
 *
 * `within-hyst` is here because it says whether the adaptive loop was able to
 * hold still at all. A near-zero count on a run with many adjustments means
 * the loop never converged, which is a different fault from converging badly.
 */
const CHURN_COLUMNS = [
  { title: "config", value: (row) => row.config },
  { title: "run", value: (row) => String(row.repeat) },
  {
    title: "moving detail chg",
    value: (row) => integer(row.movingChurn.detail.changes),
  },
  {
    title: "moving detail rev",
    value: (row) => integer(row.movingChurn.detail.reversals),
  },
  {
    title: "moving turnover",
    value: (row) => number(row.movingChurn.detail.turnover),
  },
  {
    title: "rev/s",
    value: (row) => number(row.movingChurn.detailReversalsPerSecond, 2),
  },
  {
    title: "gov rev",
    value: (row) => integer(row.movingChurn.governor.reversals),
  },
  {
    title: "cuts",
    value: (row) => integer(row.movingChurn.governorMoves.emergencyCuts),
  },
  {
    title: "restores",
    value: (row) => integer(row.movingChurn.governorMoves.emergencyRestores),
  },
  {
    title: "within-hyst",
    value: (row) =>
      integer(
        row.movingChurn.governorMoves.reasons.find(
          ([reason]) => reason === "within-hysteresis",
        )?.[1] ?? 0,
      ),
  },
  {
    title: "tile +/-",
    value: (row) =>
      `${integer(row.movingChurn.tiles.adds)}/${integer(row.movingChurn.tiles.removes)}`,
  },
  {
    title: "settled chg",
    value: (row) => integer(row.settledChurn.detail.changes),
  },
];

const main = async () => {
  const args = process.argv.slice(2);
  const jsonFlag = args.indexOf("--json");
  const jsonOut = jsonFlag >= 0 ? args[jsonFlag + 1] : null;
  const positional = args.filter(
    (entry, index) =>
      !entry.startsWith("--") && !(jsonFlag >= 0 && index === jsonFlag + 1),
  );
  const directory = resolve(positional[0] ?? "artifacts/replay");

  const files = (await readdir(directory).catch(() => null))
    ?.filter((entry) => entry.endsWith(".json"))
    .sort();
  if (files === undefined || files === null || files.length === 0) {
    process.stderr.write(`no artifacts in ${directory}\n`);
    process.exitCode = 1;
    return;
  }

  const analyses = [];
  for (const file of files) {
    const artifact = JSON.parse(
      await readFile(resolve(directory, file), "utf8"),
    );
    if (artifact.schemaVersion !== 1) {
      process.stderr.write(`skipping ${file}: unknown artifact version\n`);
      continue;
    }
    analyses.push({ file: basename(file), ...analyzeArtifact(artifact) });
  }

  const byRecording = new Map();
  for (const analysis of analyses) {
    const group = byRecording.get(analysis.name) ?? [];
    group.push(analysis);
    byRecording.set(analysis.name, group);
  }

  const lines = [];
  for (const [name, group] of byRecording) {
    const first = group[0];
    lines.push(`## ${name}`);
    lines.push("");
    lines.push(
      `renderer: ${first.renderer}${first.softwareRenderer ? " **(software — not evidence)**" : ""}  `,
    );
    lines.push(`network: ${first.network.mode}`);
    lines.push("");
    lines.push(table(group, COLUMNS));
    lines.push("");
    lines.push("Visual churn — how much the view changed, split by regime:");
    lines.push("");
    lines.push(table(group, CHURN_COLUMNS));
    lines.push("");

    const configs = [...new Set(group.map((row) => row.config))];
    if (group.length > configs.length) {
      const summaries = configs.map((config) => {
        const rows = group.filter((row) => row.config === config);
        return {
          config,
          runs: rows.length,
          fps: across(rows, (row) => row.motion.fps),
          p95: across(rows, (row) => row.motion.intervalMs.p95),
          gpu: across(rows, (row) => row.motion.gpuMs.p95),
          quality: across(rows, (row) => row.motion.quality),
          points: across(rows, (row) => row.motion.drawnPoints),
          triangles: across(rows, (row) => row.motion.drawnTriangles),
          settle: across(rows, (row) => row.convergence.msAfterGesture),
          turnover: across(rows, (row) => row.movingChurn.detail.turnover),
          reversals: across(rows, (row) => row.movingChurn.detail.reversals),
        };
      });
      lines.push(
        "Median across repeats, ± half the spread those repeats covered:",
      );
      lines.push("");
      lines.push(
        table(summaries, [
          { title: "config", value: (row) => row.config },
          { title: "runs", value: (row) => String(row.runs) },
          { title: "moving fps", value: (row) => withSpread(row.fps, number) },
          {
            title: "moving p95 ms",
            value: (row) => withSpread(row.p95, number),
          },
          {
            title: "moving gpu p95",
            value: (row) => withSpread(row.gpu, number),
          },
          {
            title: "moving quality",
            value: (row) => withSpread(row.quality, percent),
          },
          {
            title: "moving pts",
            value: (row) => withSpread(row.points, integer),
          },
          {
            title: "moving tris",
            value: (row) => withSpread(row.triangles, integer),
          },
          {
            title: "settle ms",
            value: (row) => withSpread(row.settle, integer),
          },
          {
            title: "turnover",
            value: (row) => withSpread(row.turnover, number),
          },
          {
            title: "detail rev",
            value: (row) => withSpread(row.reversals, integer),
          },
        ]),
      );
      lines.push("");
    }
    const drifted = group.filter(
      (row) => (row.fidelity.maxPathDrift ?? 0) > 0.1,
    );
    const late = group.filter((row) => row.fidelity.maxLatenessMs > 50);
    if (drifted.length > 0) {
      lines.push(
        "> Off the recorded path: " +
          drifted
            .map(
              (row) =>
                `${row.config}#${row.repeat} (${percent(row.fidelity.maxPathDrift)})`,
            )
            .join(", ") +
          ". These runs travelled somewhere the recording did not, so they" +
          " drew different scenery and their numbers do not belong beside" +
          " the others.",
      );
      lines.push("");
    }
    if (late.length > 0) {
      lines.push(
        "> Input fell behind: " +
          late
            .map(
              (row) =>
                `${row.config}#${row.repeat} (worst ${number(row.fidelity.maxLatenessMs)} ms)`,
            )
            .join(", ") +
          ". The gesture was delivered slower than it was performed, so the" +
          " view had longer to catch up than a hand would have given it.",
      );
      lines.push("");
    }
  }

  process.stdout.write(`${lines.join("\n")}\n`);
  if (jsonOut) {
    await writeFile(
      resolve(jsonOut),
      `${JSON.stringify(analyses, null, 2)}\n`,
      "utf8",
    );
    process.stderr.write(`wrote ${resolve(jsonOut)}\n`);
  }
};

await main();
