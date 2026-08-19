/**
 * Reads telemetry traces downloaded from the example.
 *
 * The replay benchmark compares configurations against one another; this
 * describes a single session on its own terms, which is what a trace captured
 * by hand can support. It answers three questions: what the gesture cost, what
 * detail it bought, and — when those two disagree with each other — what the
 * view was busy with instead of drawing.
 *
 * Usage:
 *   node scripts/analyzeTrace.mjs <trace.json...> [--json out.json]
 */

import { readFile, writeFile } from "node:fs/promises";
import { basename, resolve } from "node:path";

import {
  contaminationCounts,
  detailOf,
  framesOf,
  integer,
  isMoving,
  markerTimes,
  number,
  percent,
  phaseMetrics,
  qualityOf,
  table,
} from "./traceMetrics.mjs";

const analyze = (trace, label) => {
  const frames = framesOf(trace);
  const moving = frames.filter(isMoving);
  const still = frames.filter((frame) => !isMoving(frame));
  const marks = markerTimes(trace);
  const qualities = frames.map(qualityOf).filter((value) => value !== null);

  return {
    label,
    renderer:
      trace.environment.webgl.unmaskedRenderer ??
      trace.environment.webgl.renderer,
    softwareRenderer: trace.environment.webgl.softwareRenderer,
    viewport: trace.environment.viewport,
    stopped: trace.stoppedAt !== null,
    durationMs: trace.summary.durationMs,
    markers: [...marks.keys()],
    all: phaseMetrics(frames),
    moving: phaseMetrics(moving),
    still: phaseMetrics(still),
    movingShare: frames.length === 0 ? null : moving.length / frames.length,
    quality: {
      min: qualities.length === 0 ? null : Math.min(...qualities),
      max: qualities.length === 0 ? null : Math.max(...qualities),
      everReduced: qualities.some((value) => value < 1),
    },
    finalDetail: detailOf(frames[frames.length - 1] ?? { state: {} }),
    longTasks: {
      count: trace.summary.longTasks,
      totalMs: trace.events
        .filter((event) => event.type === "long-task")
        .reduce((sum, event) => sum + event.durationMs, 0),
      worstMs: trace.events
        .filter((event) => event.type === "long-task")
        .reduce((worst, event) => Math.max(worst, event.durationMs), 0),
    },
    contamination: contaminationCounts(frames).slice(0, 8),
    frames: frames.length,
    cleanFrames: trace.summary.cleanFrames,
    gpuValidFrames: frames.filter((frame) => frame.gpuStatus === "valid")
      .length,
  };
};

const PHASES = [
  { title: "phase", value: (row) => row.phase },
  { title: "frames", value: (row) => integer(row.metrics.frames) },
  { title: "fps", value: (row) => number(row.metrics.fps) },
  { title: "p50 ms", value: (row) => number(row.metrics.intervalMs.p50) },
  { title: "p95 ms", value: (row) => number(row.metrics.intervalMs.p95) },
  { title: "p99 ms", value: (row) => number(row.metrics.intervalMs.p99) },
  { title: "cpu p50", value: (row) => number(row.metrics.cpuMs.p50) },
  { title: "cpu p95", value: (row) => number(row.metrics.cpuMs.p95) },
  { title: "gpu p50", value: (row) => number(row.metrics.gpuMs.p50) },
  { title: "gpu p95", value: (row) => number(row.metrics.gpuMs.p95) },
  { title: "points", value: (row) => integer(row.metrics.drawnPoints) },
  { title: "triangles", value: (row) => integer(row.metrics.drawnTriangles) },
  { title: "quality", value: (row) => percent(row.metrics.quality) },
  { title: "clean", value: (row) => percent(row.metrics.cleanShare) },
];

const report = (analysis) => {
  const lines = [];
  lines.push(`## ${analysis.label}`);
  lines.push("");
  lines.push(
    `renderer: ${analysis.renderer}${
      analysis.softwareRenderer ? " **(software — not evidence)**" : ""
    }  `,
  );
  lines.push(
    `viewport: ${analysis.viewport.widthCssPx}×${analysis.viewport.heightCssPx} css px ` +
      `at dpr ${analysis.viewport.devicePixelRatio.toFixed(2)} ` +
      `(${Math.round(
        analysis.viewport.widthCssPx *
          analysis.viewport.devicePixelRatio *
          analysis.viewport.heightCssPx *
          analysis.viewport.devicePixelRatio,
      ).toLocaleString("en-US")} device px)  `,
  );
  lines.push(
    `${integer(analysis.frames)} frames over ${number(
      analysis.durationMs / 1000,
    )} s · ${percent(analysis.movingShare)} of frames classed as interaction · ` +
      `GPU timed on ${integer(analysis.gpuValidFrames)}`,
  );
  if (!analysis.stopped) {
    lines.push("");
    lines.push(
      "> The recording was never stopped, so the trace ends wherever the" +
        " download caught it.",
    );
  }
  lines.push("");
  lines.push(
    table(
      [
        { phase: "all", metrics: analysis.all },
        { phase: "interaction", metrics: analysis.moving },
        { phase: "stationary", metrics: analysis.still },
      ],
      PHASES,
    ),
  );
  lines.push("");
  lines.push(
    `quality fraction: ${number(analysis.quality.min, 3)} – ${number(
      analysis.quality.max,
      3,
    )}${
      analysis.quality.everReduced
        ? ""
        : " — **the governor never reduced quality**"
    }`,
  );
  lines.push(
    `long tasks: ${integer(analysis.longTasks.count)}, ` +
      `${integer(analysis.longTasks.totalMs)} ms total, ` +
      `worst ${integer(analysis.longTasks.worstMs)} ms`,
  );
  lines.push("");
  lines.push(
    `frames not describing steady-state cost: ${percent(
      1 - analysis.cleanFrames / Math.max(1, analysis.frames),
    )}`,
  );
  lines.push("");
  lines.push("| busy with | frames |");
  lines.push("| --- | --- |");
  for (const [reason, count] of analysis.contamination) {
    lines.push(`| ${reason} | ${integer(count)} |`);
  }
  lines.push("");
  return lines.join("\n");
};

const main = async () => {
  const args = process.argv.slice(2);
  const jsonFlag = args.indexOf("--json");
  const jsonOut = jsonFlag >= 0 ? args[jsonFlag + 1] : null;
  const files = args.filter(
    (entry, index) =>
      !entry.startsWith("--") && !(jsonFlag >= 0 && index === jsonFlag + 1),
  );
  if (files.length === 0) {
    process.stderr.write("usage: analyzeTrace.mjs <trace.json...>\n");
    process.exitCode = 1;
    return;
  }

  const analyses = [];
  for (const file of files) {
    const trace = JSON.parse(await readFile(resolve(file), "utf8"));
    if (trace.schemaVersion !== 1) {
      process.stderr.write(`skipping ${file}: unknown trace version\n`);
      continue;
    }
    analyses.push(analyze(trace, basename(file, ".json")));
  }

  process.stdout.write(`${analyses.map(report).join("\n")}\n`);
  if (jsonOut) {
    await writeFile(
      resolve(jsonOut),
      `${JSON.stringify(analyses, null, 2)}\n`,
      "utf8",
    );
  }
};

await main();
