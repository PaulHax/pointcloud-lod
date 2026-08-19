/**
 * Reading a telemetry trace.
 *
 * Shared by the replay comparison and by the analysis of a single trace
 * downloaded from the example, so a number means the same thing whichever one
 * printed it.
 *
 * The two things a reader has to keep apart are what a frame *cost* and what
 * it *showed*. A configuration can buy a smooth gesture by drawing almost
 * nothing, so cost is only ever reported next to the detail it bought.
 */

export const percentile = (sorted, fraction) => {
  if (sorted.length === 0) return null;
  const rank = (sorted.length - 1) * fraction;
  const low = Math.floor(rank);
  const high = Math.ceil(rank);
  if (low === high) return sorted[low];
  return sorted[low] + (sorted[high] - sorted[low]) * (rank - low);
};

export const summarize = (values) => {
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

export const meanOf = (values) =>
  values.length === 0
    ? null
    : values.reduce((sum, value) => sum + value, 0) / values.length;

export const framesOf = (trace) =>
  trace.events.filter((event) => event.type === "frame");

export const markerTimes = (trace) => {
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
 * Two pages write these traces and they are shaped differently on purpose.
 * The explorer drives a scene coordinator holding a list of members; the
 * instrumented page drives a controller, adapter and governor directly, which
 * is the low-level API it exists to demonstrate. Neither shape is a mistake,
 * so every reader below takes the coordinator's answer first and the bare
 * governor's second.
 */

/**
 * The regime the governor was in when a frame was recorded. It names the two
 * "interaction" and "stationary".
 */
export const regimeOf = (frame) =>
  frame.state?.coordinator?.governor?.regime ??
  frame.state?.governor?.regime ??
  null;

export const isMoving = (frame) => regimeOf(frame) === "interaction";

/** The one member an instrumented-page trace has, in coordinator shape. */
export const membersOf = (frame) => {
  const state = frame.state;
  if (state === null || state === undefined) return [];
  if (Array.isArray(state.members)) return state.members;
  if (state.controller || state.adapter) {
    return [
      {
        kind: "pointCloud",
        id: "cloud",
        label: state.source ?? "point cloud",
        drawnPoints: state.adapter?.drawnPoints ?? 0,
        drawnTiles: state.adapter?.drawnTiles ?? 0,
        gpuResidentBytes: state.adapter?.gpuResidentBytes ?? 0,
        densityFraction: state.controller?.densityFraction ?? null,
        pointBudget: state.controller?.pointBudget ?? null,
      },
    ];
  }
  return [];
};

/**
 * Detail the view was showing: points and triangles actually drawn, and the
 * screen-space error the mesh settled at. Summed across members, because a
 * combined scene's cost is the whole view's, not any one dataset's.
 */
export const detailOf = (frame) => {
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

/** The quality fraction the governor had settled on for this frame. */
export const qualityOf = (frame) =>
  frame.state?.coordinator?.viewQualityFraction ??
  frame.state?.governor?.viewQualityFraction ??
  null;

export const phaseMetrics = (frames) => {
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
    quality: meanOf(frames.map(qualityOf).filter((value) => value !== null)),
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

/** Every contamination reason seen, with how many frames carried it. */
export const contaminationCounts = (frames) => {
  const counts = new Map();
  for (const frame of frames) {
    for (const reason of frame.contamination ?? []) {
      counts.set(reason, (counts.get(reason) ?? 0) + 1);
    }
  }
  return [...counts.entries()].sort((left, right) => right[1] - left[1]);
};

export const number = (value, digits = 1) =>
  value === null || value === undefined || !Number.isFinite(value)
    ? "—"
    : value.toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      });

export const integer = (value) =>
  value === null || value === undefined || !Number.isFinite(value)
    ? "—"
    : Math.round(value).toLocaleString("en-US");

export const percent = (value) =>
  value === null || value === undefined || !Number.isFinite(value)
    ? "—"
    : `${(value * 100).toFixed(1)}%`;

export const table = (rows, columns) => {
  const header = `| ${columns.map((column) => column.title).join(" | ")} |`;
  const rule = `| ${columns.map(() => "---").join(" | ")} |`;
  const body = rows.map(
    (row) => `| ${columns.map((column) => column.value(row)).join(" | ")} |`,
  );
  return [header, rule, ...body].join("\n");
};
