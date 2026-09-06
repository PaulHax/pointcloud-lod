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

/**
 * The detail knob a member actually shows, normalized to "fraction of full".
 *
 * A point cloud thins to a density fraction; a mesh trades screen-space error.
 * They are different axes, so a scene-wide average of them would mean nothing.
 * Each member is therefore read on its own knob and reported separately.
 */
export const detailKnobOf = (member) =>
  member.kind === "pointCloud"
    ? (member.densityFraction ?? null)
    : (member.sseMultiplier ?? null);

/**
 * Movement in one scalar over a frame sequence.
 *
 * `changes` counts frames where the value moved at all; `reversals` counts how
 * often it changed direction. The two are kept apart because they answer
 * different complaints: refinement after a settle is many changes and no
 * reversals, which is wanted, while an oscillation is few changes and many
 * reversals, which is the defect.
 *
 * `turnover` is the summed relative movement — how much of what was on screen
 * was replaced — and is the scale-free companion to a raw change count.
 */
export const movementOf = (values) => {
  let changes = 0;
  let reversals = 0;
  let increases = 0;
  let decreases = 0;
  let turnover = 0;
  let previousDirection = 0;
  let previous = null;
  const distinct = new Set();
  for (const value of values) {
    if (value === null || value === undefined || !Number.isFinite(value)) {
      continue;
    }
    distinct.add(value.toFixed(8));
    if (previous !== null && value !== previous) {
      changes += 1;
      const direction = value > previous ? 1 : -1;
      if (direction > 0) increases += 1;
      else decreases += 1;
      if (previousDirection !== 0 && direction !== previousDirection) {
        reversals += 1;
      }
      previousDirection = direction;
      const scale = Math.max(Math.abs(value), Math.abs(previous));
      if (scale > 0) turnover += Math.abs(value - previous) / scale;
    }
    previous = value;
  }
  return {
    changes,
    reversals,
    increases,
    decreases,
    distinct: distinct.size,
    turnover,
  };
};

/**
 * How long a scalar took to stop moving, measured from the first frame given.
 *
 * Null when it never settled inside the window, which is itself the answer:
 * a regime that never reaches a steady state has no time-to-steady-state.
 */
export const settleOf = (frames, read) => {
  let lastChangeIndex = -1;
  let previous = null;
  for (const [index, frame] of frames.entries()) {
    const value = read(frame);
    if (value === null || value === undefined) continue;
    if (previous !== null && value !== previous) lastChangeIndex = index;
    previous = value;
  }
  if (frames.length === 0) return { ms: null, frames: null, settled: false };
  if (lastChangeIndex < 0) return { ms: 0, frames: 0, settled: true };
  return {
    ms: frames[lastChangeIndex].atMs - frames[0].atMs,
    frames: lastChangeIndex,
    settled: lastChangeIndex < frames.length - 1,
  };
};

/**
 * Every distinct governor adjustment the trace saw, in order.
 *
 * `lastAdjustment` is a snapshot repeated on every frame until the next one,
 * so the same adjustment appears in hundreds of frames. Identity is the
 * timestamp plus what it did, which is what de-duplicates it without assuming
 * adjustments are spaced further apart than a frame.
 */
export const adjustmentsOf = (frames) => {
  const seen = new Set();
  const adjustments = [];
  for (const frame of frames) {
    const adjustment =
      frame.state?.coordinator?.governor?.lastAdjustment ??
      frame.state?.governor?.lastAdjustment ??
      null;
    if (adjustment === null) continue;
    const identity = `${adjustment.atMs}|${adjustment.reason}|${adjustment.fromFraction}|${adjustment.toFraction}`;
    if (seen.has(identity)) continue;
    seen.add(identity);
    adjustments.push({ ...adjustment, regime: regimeOf(frame) });
  }
  return adjustments;
};

/**
 * Visual churn over a frame sequence, decomposed by where it enters.
 *
 * Three levels, because a change at one is not a change at the next and the
 * difference is the attribution. The governor's own fraction is what the
 * adaptive loop decided; the allocated fraction is that after demand-capped
 * water filling, which moves with the camera even when the governor holds; the
 * detail knob is what a member finally drew, which is a ratio of two separately
 * clamped budgets and so can hold still while both inputs move, or move while
 * both hold.
 *
 * Point turnover is read from the drawn totals rather than per-tile prefixes:
 * a trace carries aggregates by design, and every selected tile is thinned to
 * the same fraction, so the drawn total moves with the prefix that produced it.
 * What aggregates cannot see is an add and a remove that cancel, which is why
 * tile adds and removes are counted separately rather than folded in.
 */
export const churnMetrics = (frames) => {
  const spanMs =
    frames.length < 2 ? 0 : frames[frames.length - 1].atMs - frames[0].atMs;
  const perSecond = (count) => (spanMs > 0 ? (count * 1000) / spanMs : null);

  const governor = movementOf(
    frames.map(
      (frame) =>
        frame.state?.coordinator?.governor?.viewQualityFraction ??
        frame.state?.governor?.viewQualityFraction ??
        null,
    ),
  );
  const allocated = movementOf(
    frames.map((frame) => {
      const members = membersOf(frame);
      return members.length === 0
        ? null
        : (members[0].allocation?.qualityFraction ?? null);
    }),
  );
  const detail = movementOf(
    frames.map((frame) => {
      const members = membersOf(frame);
      return members.length === 0 ? null : detailKnobOf(members[0]);
    }),
  );

  const drawnPoints = movementOf(
    frames.map((frame) => detailOf(frame).drawnPoints),
  );
  let tileAdds = 0;
  let tileRemoves = 0;
  let previousTiles = null;
  for (const frame of frames) {
    const tiles = detailOf(frame).drawnTiles;
    if (previousTiles !== null) {
      if (tiles > previousTiles) tileAdds += tiles - previousTiles;
      else if (tiles < previousTiles) tileRemoves += previousTiles - tiles;
    }
    previousTiles = tiles;
  }

  const adjustments = adjustmentsOf(frames);
  const reasons = new Map();
  for (const adjustment of adjustments) {
    reasons.set(adjustment.reason, (reasons.get(adjustment.reason) ?? 0) + 1);
  }
  const moves = adjustments.filter(
    (adjustment) => adjustment.direction !== "none",
  );
  let moveReversals = 0;
  let previousDirection = null;
  for (const move of moves) {
    if (previousDirection !== null && move.direction !== previousDirection) {
      moveReversals += 1;
    }
    previousDirection = move.direction;
  }

  return {
    frames: frames.length,
    spanMs,
    governor,
    allocated,
    detail,
    drawnPoints,
    tiles: {
      adds: tileAdds,
      removes: tileRemoves,
      addsPerSecond: perSecond(tileAdds),
      removesPerSecond: perSecond(tileRemoves),
    },
    detailReversalsPerSecond: perSecond(detail.reversals),
    settle: {
      detail: settleOf(frames, (frame) => {
        const members = membersOf(frame);
        return members.length === 0 ? null : detailKnobOf(members[0]);
      }),
    },
    governorMoves: {
      count: moves.length,
      reversals: moveReversals,
      emergencyCuts: reasons.get("emergency-cut") ?? 0,
      emergencyRestores: reasons.get("emergency-restore") ?? 0,
      reasons: [...reasons.entries()].sort((left, right) => right[1] - left[1]),
    },
  };
};
