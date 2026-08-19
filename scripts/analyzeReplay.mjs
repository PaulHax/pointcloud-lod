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

const percentile = (sorted, fraction) => {
  if (sorted.length === 0) return null;
  const rank = (sorted.length - 1) * fraction;
  const low = Math.floor(rank);
  const high = Math.ceil(rank);
  if (low === high) return sorted[low];
  return sorted[low] + (sorted[high] - sorted[low]) * (rank - low);
};

const summarize = (values) => {
  const sorted = [...values].sort((left, right) => left - right);
  return {
    count: sorted.length,
    p50: percentile(sorted, 0.5),
    p95: percentile(sorted, 0.95),
    p99: percentile(sorted, 0.99),
    max: sorted.length === 0 ? null : sorted[sorted.length - 1],
    mean:
      sorted.length === 0
        ? null
        : sorted.reduce((sum, value) => sum + value, 0) / sorted.length,
  };
};

const markerTimes = (trace) => {
  const marks = new Map();
  for (const event of trace.events) {
    if (event.type !== "state") continue;
    if (event.reason.startsWith("marker:")) {
      marks.set(event.reason.slice("marker:".length), event.atMs);
    }
  }
  return marks;
};

/**
 * The regime the coordinator was in when a frame was recorded. The governor
 * names the two "interaction" and "stationary".
 */
const regimeOf = (frame) => frame.state?.coordinator?.governor?.regime ?? null;
const isMoving = (frame) => regimeOf(frame) === "interaction";

const membersOf = (frame) => frame.state?.members ?? [];

/**
 * Detail the view was showing: points and triangles actually drawn, and the
 * screen-space error the mesh settled at. Summed across members, because a
 * combined scene's cost is the whole view's, not any one dataset's.
 */
const detailOf = (frame) => {
  let drawnPoints = 0;
  let drawnTriangles = 0;
  let drawnTiles = 0;
  let residentBytes = 0;
  let worstSse = null;
  for (const member of membersOf(frame)) {
    if (member.kind === "pointCloud") {
      drawnPoints += member.drawnPoints ?? 0;
      drawnTiles += member.drawnTiles ?? 0;
      residentBytes += member.gpuResidentBytes ?? 0;
    } else {
      drawnTriangles += member.drawnTriangles ?? 0;
      drawnTiles += member.drawnTiles ?? 0;
      residentBytes +=
        (member.residentGeometryBytes ?? 0) +
        (member.residentTextureBytes ?? 0);
      const sse = member.effectiveScreenSpaceErrorPx;
      if (typeof sse === "number") {
        worstSse = worstSse === null ? sse : Math.max(worstSse, sse);
      }
    }
  }
  return { drawnPoints, drawnTriangles, drawnTiles, residentBytes, worstSse };
};

const meanOf = (values) =>
  values.length === 0
    ? null
    : values.reduce((sum, value) => sum + value, 0) / values.length;

const phaseMetrics = (frames) => {
  const intervals = frames
    .map((frame) => frame.rafIntervalMs)
    .filter((value) => Number.isFinite(value) && value > 0);
  const cpu = frames
    .map((frame) => frame.vtkCpuMs)
    .filter((value) => Number.isFinite(value));
  const gpu = frames
    .filter((frame) => frame.gpuStatus === "valid" && frame.gpuMs !== null)
    .map((frame) => frame.gpuMs);
  const details = frames.map(detailOf);
  const spanMs =
    frames.length < 2 ? 0 : frames[frames.length - 1].atMs - frames[0].atMs;
  return {
    frames: frames.length,
    spanMs,
    // Frames the display actually showed over the span, which is the only
    // frame rate a hand can feel. Reciprocal mean frame time flatters a run
    // that dropped frames entirely.
    fps: spanMs > 0 ? ((frames.length - 1) / spanMs) * 1000 : null,
    intervalMs: summarize(intervals),
    cpuMs: summarize(cpu),
    gpuMs: summarize(gpu),
    cleanShare:
      frames.length === 0
        ? null
        : frames.filter((frame) => frame.clean).length / frames.length,
    drawnPoints: meanOf(details.map((detail) => detail.drawnPoints)),
    drawnTriangles: meanOf(details.map((detail) => detail.drawnTriangles)),
    residentBytes: meanOf(details.map((detail) => detail.residentBytes)),
    worstSse: meanOf(
      details
        .map((detail) => detail.worstSse)
        .filter((value) => value !== null),
    ),
  };
};

const analyzeArtifact = (artifact) => {
  const trace = artifact.trace;
  const frames = trace.events.filter((event) => event.type === "frame");
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

const number = (value, digits = 1) =>
  value === null || value === undefined || !Number.isFinite(value)
    ? "—"
    : value.toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      });

const integer = (value) =>
  value === null || value === undefined || !Number.isFinite(value)
    ? "—"
    : Math.round(value).toLocaleString("en-US");

const percent = (value) =>
  value === null || value === undefined || !Number.isFinite(value)
    ? "—"
    : `${(value * 100).toFixed(1)}%`;

const table = (rows, columns) => {
  const header = `| ${columns.map((column) => column.title).join(" | ")} |`;
  const rule = `| ${columns.map(() => "---").join(" | ")} |`;
  const body = rows.map(
    (row) => `| ${columns.map((column) => column.value(row)).join(" | ")} |`,
  );
  return [header, rule, ...body].join("\n");
};

const COLUMNS = [
  { title: "config", value: (row) => row.config },
  { title: "run", value: (row) => String(row.repeat) },
  { title: "moving fps", value: (row) => number(row.motion.fps) },
  { title: "moving p50 ms", value: (row) => number(row.motion.intervalMs.p50) },
  { title: "moving p95 ms", value: (row) => number(row.motion.intervalMs.p95) },
  { title: "moving gpu p95", value: (row) => number(row.motion.gpuMs.p95) },
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
  { title: "path drift", value: (row) => percent(row.fidelity.pathDrift) },
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
