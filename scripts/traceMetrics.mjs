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
 * The detail knob a member shows, normalized to "fraction of full".
 *
 * A point cloud thins to a density fraction. A mesh trades screen-space error,
 * and its `sseMultiplier` is `1 / max(allocation.qualityFraction, 0.05)` — a
 * pure restatement of the fraction it was allocated, not an independent
 * reading. So for a mesh this level says nothing the allocated level did not,
 * except when the fraction saturates at the 0.05 floor, where the clamp makes
 * it under-report movement the governor really made. What actually changes on
 * screen for a mesh is which tiles are drawn, which is counted separately.
 */
export const detailKnobOf = (member) =>
  member.kind === "pointCloud"
    ? (member.densityFraction ?? null)
    : (member.sseMultiplier ?? null);

/**
 * Resolution below which two values are the same number.
 *
 * Quality fractions arrive through repeated multiplication, so a value can
 * differ from the previous one in the last bits without anything having moved.
 * `distinct` has always quantized; comparing raw would let that noise into the
 * change and reversal counts, which are the numbers the churn table calls the
 * defect.
 */
const SAME_VALUE_EPSILON = 1e-8;

const same = (left, right) => Math.abs(left - right) <= SAME_VALUE_EPSILON;

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
 * was replaced — and is the scale-free companion to a raw change count. It is
 * a sum over the sequence, so it grows with how many frames were drawn and is
 * only comparable between runs of similar length.
 */
export const movementOf = (values) => {
  let changes = 0;
  let reversals = 0;
  let increases = 0;
  let decreases = 0;
  let turnover = 0;
  let previousDirection = 0;
  let previous = null;
  let read = 0;
  const distinct = new Set();
  for (const value of values) {
    if (value === null || value === undefined || !Number.isFinite(value)) {
      continue;
    }
    read += 1;
    distinct.add(value.toFixed(8));
    if (previous !== null && !same(value, previous)) {
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
    /** Frames that carried a readable value. Zero means no data, not no churn. */
    read,
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
 * `settled` is false both when the value was still moving at the end and when
 * there was nothing to read, because neither is evidence that it came to rest.
 * The two are told apart by `read`.
 */
export const settleOf = (frames, read) => {
  let lastChangeIndex = -1;
  let previous = null;
  let readCount = 0;
  for (const [index, frame] of frames.entries()) {
    const value = read(frame);
    if (value === null || value === undefined || !Number.isFinite(value)) {
      continue;
    }
    readCount += 1;
    if (previous !== null && !same(value, previous)) lastChangeIndex = index;
    previous = value;
  }
  if (readCount === 0) {
    return { ms: null, frames: null, read: 0, settled: false };
  }
  if (lastChangeIndex < 0) {
    return { ms: 0, frames: 0, read: readCount, settled: true };
  }
  return {
    ms: frames[lastChangeIndex].atMs - frames[0].atMs,
    frames: lastChangeIndex,
    read: readCount,
    settled: lastChangeIndex < frames.length - 1,
  };
};

/**
 * Every distinct governor adjustment record the trace saw, in order.
 *
 * `lastAdjustment` is a snapshot repeated on every frame until the next one,
 * so identity is the timestamp plus what it did. That collapses the repeats of
 * a real move, but *not* the no-op outcomes: the loop re-records
 * `within-hysteresis`, `cooldown`, `insufficient-samples` and `clamped` on
 * every evaluation with a fresh timestamp, so counts of those are counts of
 * evaluations, not of adjustments. Only entries with a direction other than
 * "none" are adjustments in the sense of having moved anything.
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
 * Why the loop held still, which is not one reason but several.
 *
 * `within-hysteresis` means it judged the view good enough. `clamped` means it
 * wanted to move and could not, which at the ceiling is contentment and at the
 * floor is the opposite — the view pinned at minimum quality. Reading either
 * as convergence without separating them turns a bottomed-out run into a
 * healthy one.
 */
export const holdReasonsOf = (adjustments) => {
  let withinHysteresis = 0;
  let clampedAtFloor = 0;
  let clampedAtCeiling = 0;
  let clampedInterior = 0;
  let cooldown = 0;
  let insufficientSamples = 0;
  for (const adjustment of adjustments) {
    if (adjustment.direction !== "none") continue;
    if (adjustment.reason === "within-hysteresis") withinHysteresis += 1;
    else if (adjustment.reason === "cooldown") cooldown += 1;
    else if (adjustment.reason === "insufficient-samples") {
      insufficientSamples += 1;
    } else if (adjustment.reason === "clamped") {
      if (adjustment.fromFraction <= 0.0501) clampedAtFloor += 1;
      else if (adjustment.fromFraction >= 0.9999) clampedAtCeiling += 1;
      else clampedInterior += 1;
    }
  }
  return {
    withinHysteresis,
    clampedAtFloor,
    clampedAtCeiling,
    clampedInterior,
    cooldown,
    insufficientSamples,
  };
};

/** Members seen anywhere in the sequence, in first-seen order, keyed by id. */
const memberIdsOf = (frames) => {
  const ids = [];
  const seen = new Set();
  for (const frame of frames) {
    for (const member of membersOf(frame)) {
      const id = member.id ?? "";
      if (seen.has(id)) continue;
      seen.add(id);
      ids.push(id);
    }
  }
  return ids;
};

/**
 * Visual churn over a frame sequence, decomposed by where it enters.
 *
 * Members are followed by id rather than by position, because a scene can gain
 * or lose a dataset mid-run and comparing one member's knob against another's
 * would read the swap as an enormous change. Each member is measured on its
 * own axis and reported separately; the summed `detail` is what a viewer saw
 * move anywhere in the scene, and is the only figure that does not quietly
 * describe one dataset while ignoring the rest.
 *
 * Point turnover is read from drawn totals rather than per-tile prefixes: a
 * trace carries aggregates by design, and every selected tile is thinned to the
 * same fraction, so the drawn total moves with the prefix that produced it.
 * Tiles are counted per member, but an add and a remove *within one member on
 * one frame* still cancel and are invisible — the aggregate cannot see them.
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

  const members = memberIdsOf(frames).map((id) => {
    const seriesOf = (read) =>
      frames.map((frame) => {
        const member = membersOf(frame).find(
          (candidate) => (candidate.id ?? "") === id,
        );
        return member === undefined ? null : read(member);
      });
    const kind =
      frames
        .flatMap((frame) => membersOf(frame))
        .find((member) => (member.id ?? "") === id)?.kind ?? null;
    let adds = 0;
    let removes = 0;
    let previousTiles = null;
    for (const tiles of seriesOf((member) => member.drawnTiles ?? null)) {
      if (tiles === null) continue;
      if (previousTiles !== null) {
        if (tiles > previousTiles) adds += tiles - previousTiles;
        else if (tiles < previousTiles) removes += previousTiles - tiles;
      }
      previousTiles = tiles;
    }
    return {
      id,
      kind,
      detail: movementOf(seriesOf(detailKnobOf)),
      allocated: movementOf(
        seriesOf((member) => member.allocation?.qualityFraction ?? null),
      ),
      drawnPoints: movementOf(seriesOf((member) => member.drawnPoints ?? null)),
      tiles: { adds, removes },
    };
  });

  const sum = (read) =>
    members.reduce((total, member) => total + read(member), 0);
  const detail = {
    read: sum((member) => member.detail.read),
    changes: sum((member) => member.detail.changes),
    reversals: sum((member) => member.detail.reversals),
    increases: sum((member) => member.detail.increases),
    decreases: sum((member) => member.detail.decreases),
    distinct: sum((member) => member.detail.distinct),
    turnover: sum((member) => member.detail.turnover),
  };
  const tileAdds = sum((member) => member.tiles.adds);
  const tileRemoves = sum((member) => member.tiles.removes);

  const adjustments = adjustmentsOf(frames);
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
  const reasons = new Map();
  for (const adjustment of adjustments) {
    reasons.set(adjustment.reason, (reasons.get(adjustment.reason) ?? 0) + 1);
  }

  return {
    frames: frames.length,
    spanMs,
    governor,
    members,
    detail,
    tiles: {
      adds: tileAdds,
      removes: tileRemoves,
      addsPerSecond: perSecond(tileAdds),
      removesPerSecond: perSecond(tileRemoves),
    },
    detailReversalsPerSecond: perSecond(detail.reversals),
    settle: {
      detail: settleOf(frames, (frame) => {
        const scene = membersOf(frame);
        return scene.length === 0 ? null : detailKnobOf(scene[0]);
      }),
    },
    held: holdReasonsOf(adjustments),
    governorMoves: {
      count: moves.length,
      reversals: moveReversals,
      emergencyCuts: moves.filter((move) => move.reason === "emergency-cut")
        .length,
      emergencyRestores: moves.filter(
        (move) => move.reason === "emergency-restore",
      ).length,
      reasons: [...reasons.entries()].sort((left, right) => right[1] - left[1]),
    },
  };
};

/**
 * How busy the machine was while a run was measured.
 *
 * A benchmark is supposed to have the machine to itself. Load is a queue
 * length, so it only means anything divided by the cores available to drain
 * it, and the peak of the samples either side of the run is what matters —
 * contention arriving halfway through still spoils the timings.
 *
 * Runs recorded before the bench started sampling have no reading at all,
 * which is reported as unknown rather than quiet.
 */
export const BUSY_LOAD_PER_CORE = 0.5;

export const machineLoadOf = (artifact) => {
  const machine = artifact?.machine;
  if (machine === null || machine === undefined) {
    return { perCore: null, cores: null, busy: false, known: false };
  }
  const readings = [machine.loadBefore, machine.loadAfter]
    .map((sample) => sample?.perCore)
    .filter((value) => typeof value === "number" && Number.isFinite(value));
  if (readings.length === 0) {
    return {
      perCore: null,
      cores: machine.loadBefore?.cores ?? null,
      busy: false,
      known: false,
    };
  }
  const perCore = Math.max(...readings);
  const browsers = [machine.loadBefore, machine.loadAfter]
    .map((sample) => sample?.browserProcesses)
    .filter((value) => typeof value === "number" && Number.isFinite(value));
  return {
    perCore,
    browserProcesses: browsers.length === 0 ? null : Math.max(...browsers),
    cores: machine.loadBefore?.cores ?? machine.loadAfter?.cores ?? null,
    busy: perCore > BUSY_LOAD_PER_CORE,
    known: true,
  };
};
