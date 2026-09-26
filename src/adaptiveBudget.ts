/**
 * Adaptive normalized view quality.
 *
 * Two independent tracks learn the quality fraction sustainable while the
 * view is moving and while it is stationary. Fractions are format-neutral and
 * always remain in [0.05, 1]; members alone translate them into points,
 * screen-space error, concurrency, or another format-specific quality axis.
 *
 * Samples are presentation intervals, and those arrive in whole display
 * refreshes: a frame made its refresh or waited for a later one. A track
 * therefore does not steer a statistic towards a continuous target. At
 * 60 Hz a percentile reads 16.7 ms or 33.3 ms and nothing between, so a band
 * around any target either falls between them and is never reached, or
 * swallows both and never moves. Each track instead grants a budget of whole
 * refreshes and counts the frames that missed it. Too many misses lowers
 * quality. A full window with none says only that headroom may exist, so the
 * track probes one bounded step up after holding its level for a while, and
 * remembers a level that proved too expensive, closing half the distance to
 * it rather than returning to it. The one headroom a quantized interval can
 * show is a frame presenting in fewer refreshes than its budget allows: a
 * still view drawn every refresh under a two-refresh budget costs at most half
 * of it, and a probe spends most of that headroom in one step. Everything
 * between holds, which is what keeps a steady view steady.
 */

import {
  finiteAbove,
  finiteAtLeast,
  finiteWithin,
  percentileOrNull,
  wholeAtLeast,
} from "./numeric";
import {
  MAX_VIEW_QUALITY_FRACTION,
  MIN_VIEW_QUALITY_FRACTION,
} from "./viewBudget";

export type QualityRegime = "stationary" | "interaction";
export type QualityAdjustmentDirection = "increase" | "decrease" | "none";
export type QualityAdjustmentReason =
  | "below-target"
  | "above-target"
  | "within-hysteresis"
  | "cooldown"
  | "insufficient-samples"
  | "clamped"
  | "emergency-cut"
  | "emergency-restore"
  | "seeded";

export type QualityAdjustment = {
  readonly atMs: number;
  readonly direction: QualityAdjustmentDirection;
  readonly reason: QualityAdjustmentReason;
  readonly fromFraction: number;
  readonly toFraction: number;
  /** Median interval of the window judged, when a window was judged. */
  readonly estimateMs: number | null;
  /** Share of that window that missed its refresh budget. */
  readonly lateFraction: number | null;
};

export type AdaptiveQualityOptions = {
  /** Starting fraction for both tracks. Default 1. */
  readonly initialFraction?: number;
  readonly stationaryTargetMs?: number;
  readonly interactionTargetMs?: number;
  /** Presentation intervals judged together. */
  readonly windowSize?: number;
  /** Share of a moving window that may miss its budget before quality falls. */
  readonly interactionLateFrameTolerance?: number;
  /**
   * The same for a still view. A late frame at rest only delays refinement,
   * while a step back is a visible change with nothing moving to hide it, so
   * a still view gives up detail only once most of its frames miss.
   */
  readonly stationaryLateFrameTolerance?: number;
  readonly maxIncreaseStep?: number;
  readonly maxDecreaseStep?: number;
  /** Shortest time between two ordinary adjustments of one track. */
  readonly cooldownMs?: number;
  /**
   * How long the interaction track holds a level before probing above it.
   * The stationary track probes after `cooldownMs`: refining a still view is
   * wanted change, while every probe under a moving hand is a visible one.
   */
  readonly interactionProbeDwellMs?: number;
  /** How long a level that proved too expensive caps later probes. */
  readonly failedLevelMemoryMs?: number;
  readonly minSamples?: number;
};

export type AdaptiveQualityTrackStats = {
  readonly fraction: number;
  /**
   * Fraction an emergency cut took this track down from, while it still owes
   * that quality back; null when nothing is outstanding.
   */
  readonly emergencyCeiling: number | null;
  readonly samples: number;
  /** Median interval in the current window. */
  readonly estimateMs: number | null;
  /** Share of the current window that missed the budget. */
  readonly lateFraction: number | null;
  /** The configured target, as given. */
  readonly targetMs: number;
  /** The interval a frame may take and still be on time. */
  readonly effectiveTargetMs: number;
  /** Refreshes a frame may take, once the display quantum is known. */
  readonly budgetRefreshes: number | null;
  /**
   * The last fraction found too expensive. It caps probes for
   * `failedLevelMemoryMs` after it was found.
   */
  readonly failedLevel: number | null;
  readonly lastAdjustment: QualityAdjustment | null;
};

export type AdaptiveQualityStats = {
  readonly minimumFraction: typeof MIN_VIEW_QUALITY_FRACTION;
  readonly maximumFraction: typeof MAX_VIEW_QUALITY_FRACTION;
  readonly cooldownMs: number;
  /** Shortest frame interval the display has been seen to present, if known. */
  readonly displayQuantumMs: number | null;
  readonly stationary: AdaptiveQualityTrackStats;
  readonly interaction: AdaptiveQualityTrackStats;
};

/**
 * The shortest frame interval the display has been seen to present, or `null`
 * while that is not yet known. Whoever measures the display owns this number;
 * the tracks read it whenever they need a budget, so it is a cheap accessor,
 * not a computation.
 */
export type DisplayQuantumSupplier = () => number | null;

export type AdaptiveQuality = {
  recordFrame(
    durationMs: number,
    options: { readonly interacting: boolean; readonly now: number },
  ): number;
  fraction(interacting: boolean): number;
  /** The interval a frame may take and still count as on time. */
  onTimeMs(interacting: boolean): number;
  /** The interval above which a frame counts as late. */
  lateThresholdMs(interacting: boolean): number;
  restartAt(interacting: boolean, fraction: number, now: number): number;
  /** Discard costs from an older frontier without changing quality or memory. */
  clearSamples(interacting: boolean, now: number): void;
  /** A new workload forgets both the window and any too-expensive level. */
  invalidateCapacity(interacting: boolean, now: number): void;
  /** Cut by `factor`, clamped to [0.5, 1], outside the sampling loop. */
  reduceNow(interacting: boolean, now: number, factor?: number): number;
  /**
   * Give back emergency cuts: one halving, or with `all` everything back to
   * where the first outstanding cut started. Never past that.
   */
  restoreNow(interacting: boolean, now: number, all?: boolean): number;
  /** True while an emergency cut is owed back. */
  owesRestore(interacting: boolean): boolean;
  /**
   * The track's fraction when enough of its current window was measured and
   * every frame in it presented within `limitMs`; otherwise null.
   */
  provenWithin(interacting: boolean, limitMs: number): number | null;
  lastAdjustment(interacting: boolean): QualityAdjustment | null;
  stats(): AdaptiveQualityStats;
};

export const ADAPTIVE_QUALITY_DEFAULTS = {
  /**
   * Floor an adaptive point budget may not be configured below.
   *
   * It lives here, in the vtk-free module the package root re-exports, so a
   * host can read it without importing the renderer entry point. Hosts mirror
   * this number in their own validators and check it against this export;
   * leaving it somewhere unreachable is what silently unhooked that check.
   */
  minBudget: 200_000,
  initialFraction: 1,
  stationaryTargetMs: 33,
  interactionTargetMs: 16,
  windowSize: 30,
  interactionLateFrameTolerance: 0.2,
  stationaryLateFrameTolerance: 0.5,
  maxIncreaseStep: 0.15,
  maxDecreaseStep: 0.5,
  cooldownMs: 400,
  interactionProbeDwellMs: 1500,
  failedLevelMemoryMs: 15_000,
  // Fewer than ten samples would let two isolated hitches outvote a window.
  minSamples: 10,
} as const;

/** Largest single emergency cut. */
const EMERGENCY_CUT = 0.5;
/** An ordinary decrease when misses exceed tolerance without dominating. */
const ORDINARY_DECREASE = 0.85;
/** Above this share of late frames the median says how far over the view is. */
const MOSTLY_LATE = 0.5;
/**
 * Share of measured headroom one probe spends. Cost does not scale exactly
 * with quality, and the slowest frame in a window is only a sample.
 */
const HEADROOM_SPENT = 0.85;
/** A probe smaller than this is not worth the visible change it makes. */
const MIN_PROBE_STEP = 0.03;
/** How far over its target a frame may run and still fit the refresh count. */
const TARGET_TOLERANCE = 1.2;
/** The late threshold as a multiple of the target before the quantum is known. */
const LATE_WITHOUT_QUANTUM = 1.5;

type Track = {
  fraction: number;
  readonly samples: number[];
  readonly targetMs: number;
  readonly probeDwellMs: number;
  readonly lateFrameTolerance: number;
  /** Last ordinary or emergency change; gates the next ordinary one. */
  lastChangeAt: number;
  /** Since when the current level has held; gates probes. */
  heldSince: number;
  lastAdjustment: QualityAdjustment | null;
  /**
   * What an emergency cut took this track down from, until it is given back.
   * The restore can undo a cut and no more, so it can never stand in for the
   * probe that actually measures capacity.
   */
  emergencyCeiling: number | null;
  failedLevel: { readonly fraction: number; readonly atMs: number } | null;
  /** The level a probe left, until the probe's first window judges it. */
  probedFrom: number | null;
};

export const createAdaptiveQuality = (
  options: AdaptiveQualityOptions = {},
  displayQuantum: DisplayQuantumSupplier = () => null,
): AdaptiveQuality => {
  const defaults = ADAPTIVE_QUALITY_DEFAULTS;
  const initialFraction = finiteWithin(
    "initialFraction",
    options.initialFraction ?? defaults.initialFraction,
    MIN_VIEW_QUALITY_FRACTION,
    MAX_VIEW_QUALITY_FRACTION,
  );
  const stationaryTargetMs = finiteAbove(
    "stationaryTargetMs",
    options.stationaryTargetMs ?? defaults.stationaryTargetMs,
    0,
  );
  const interactionTargetMs = finiteAbove(
    "interactionTargetMs",
    options.interactionTargetMs ?? defaults.interactionTargetMs,
    0,
  );
  const windowSize = wholeAtLeast(
    "windowSize",
    options.windowSize ?? defaults.windowSize,
    1,
  );
  const interactionLateFrameTolerance = finiteWithin(
    "interactionLateFrameTolerance",
    options.interactionLateFrameTolerance ??
      defaults.interactionLateFrameTolerance,
    0,
    1,
  );
  const stationaryLateFrameTolerance = finiteWithin(
    "stationaryLateFrameTolerance",
    options.stationaryLateFrameTolerance ??
      defaults.stationaryLateFrameTolerance,
    0,
    1,
  );
  const maxIncreaseStep = finiteAtLeast(
    "maxIncreaseStep",
    options.maxIncreaseStep ?? defaults.maxIncreaseStep,
    0,
  );
  const maxDecreaseStep = finiteWithin(
    "maxDecreaseStep",
    options.maxDecreaseStep ?? defaults.maxDecreaseStep,
    0,
    1,
  );
  const cooldownMs = finiteAtLeast(
    "cooldownMs",
    options.cooldownMs ?? defaults.cooldownMs,
    0,
  );
  const interactionProbeDwellMs = finiteAtLeast(
    "interactionProbeDwellMs",
    options.interactionProbeDwellMs ?? defaults.interactionProbeDwellMs,
    0,
  );
  const failedLevelMemoryMs = finiteAtLeast(
    "failedLevelMemoryMs",
    options.failedLevelMemoryMs ?? defaults.failedLevelMemoryMs,
    0,
  );
  const minSamples = Math.min(
    windowSize,
    wholeAtLeast("minSamples", options.minSamples ?? defaults.minSamples, 1),
  );

  const clamp = (fraction: number): number =>
    Math.min(
      MAX_VIEW_QUALITY_FRACTION,
      Math.max(MIN_VIEW_QUALITY_FRACTION, fraction),
    );
  const makeTrack = (
    targetMs: number,
    probeDwellMs: number,
    lateFrameTolerance: number,
  ): Track => ({
    fraction: initialFraction,
    samples: [],
    targetMs,
    probeDwellMs,
    lateFrameTolerance,
    lastChangeAt: Number.NEGATIVE_INFINITY,
    heldSince: Number.NEGATIVE_INFINITY,
    lastAdjustment: null,
    emergencyCeiling: null,
    failedLevel: null,
    probedFrom: null,
  });
  const stationary = makeTrack(
    stationaryTargetMs,
    cooldownMs,
    stationaryLateFrameTolerance,
  );
  const interaction = makeTrack(
    interactionTargetMs,
    interactionProbeDwellMs,
    interactionLateFrameTolerance,
  );
  const trackFor = (interacting: boolean): Track =>
    interacting ? interaction : stationary;

  const budgetRefreshes = (track: Track): number | null => {
    const quantumMs = displayQuantum();
    return quantumMs === null
      ? null
      : Math.max(
          1,
          Math.floor((track.targetMs * TARGET_TOLERANCE) / quantumMs),
        );
  };
  const onTimeMs = (track: Track): number => {
    const refreshes = budgetRefreshes(track);
    return refreshes === null ? track.targetMs : refreshes * displayQuantum()!;
  };
  // Halfway to the next refresh: presentation jitter moves an interval a
  // little either side of its refresh multiple, never half a refresh.
  const lateThresholdMs = (track: Track): number => {
    const refreshes = budgetRefreshes(track);
    return refreshes === null
      ? track.targetMs * LATE_WITHOUT_QUANTUM
      : (refreshes + 0.5) * displayQuantum()!;
  };
  const lateFractionOf = (track: Track): number | null => {
    if (track.samples.length === 0) return null;
    const threshold = lateThresholdMs(track);
    let late = 0;
    for (const sample of track.samples) if (sample > threshold) late += 1;
    return late / track.samples.length;
  };
  /**
   * The cut a mostly-late window argues for. A median of k refreshes against a
   * budget of n says only that the cost lies between k - 1 and k refreshes, so
   * the cut that restores the budget lies between n / k and n / (k - 1); this
   * takes the geometric middle rather than assuming the worst end of it.
   */
  const overloadCut = (track: Track, medianMs: number): number => {
    const refreshes = budgetRefreshes(track);
    if (refreshes === null) return onTimeMs(track) / medianMs;
    const presented = Math.max(
      refreshes + 1,
      Math.round(medianMs / displayQuantum()!),
    );
    return refreshes / Math.sqrt(presented * (presented - 1));
  };
  const failedLevelOf = (track: Track, now: number): number | null =>
    track.failedLevel !== null &&
    now - track.failedLevel.atMs < failedLevelMemoryMs
      ? track.failedLevel.fraction
      : null;

  const record = (
    track: Track,
    atMs: number,
    direction: QualityAdjustmentDirection,
    reason: QualityAdjustmentReason,
    fromFraction: number,
    estimateMs: number | null = null,
    lateFraction: number | null = null,
  ): void => {
    track.lastAdjustment = {
      atMs,
      direction,
      reason,
      fromFraction,
      toFraction: track.fraction,
      estimateMs,
      lateFraction,
    };
  };

  const change = (track: Track, next: number, now: number): void => {
    track.fraction = next;
    track.lastChangeAt = now;
    track.heldSince = now;
    track.samples.length = 0;
    track.probedFrom = null;
  };

  const adjust = (track: Track, now: number): void => {
    const from = track.fraction;
    if (track.samples.length < minSamples) {
      record(track, now, "none", "insufficient-samples", from);
      return;
    }
    const estimate = percentileOrNull(track.samples, 0.5);
    const late = lateFractionOf(track)!;
    const hold = (reason: QualityAdjustmentReason): void =>
      record(track, now, "none", reason, from, estimate, late);

    if (late > track.lateFrameTolerance) {
      if (now - track.lastChangeAt < cooldownMs) return hold("cooldown");
      // A probe that failed its first window goes back to the level a full
      // clean window had just proved, not to wherever an overload cut lands.
      const factor =
        track.probedFrom !== null
          ? track.probedFrom / from
          : late > MOSTLY_LATE
            ? Math.min(ORDINARY_DECREASE, overloadCut(track, estimate!))
            : ORDINARY_DECREASE;
      const next = clamp(from * Math.max(1 - maxDecreaseStep, factor));
      if (next === from) return hold("clamped");
      track.failedLevel = { fraction: from, atMs: now };
      track.emergencyCeiling = null;
      change(track, next, now);
      record(track, now, "decrease", "above-target", from, estimate, late);
      return;
    }
    if (track.samples.length >= windowSize) track.probedFrom = null;
    // A late frame inside tolerance is the equilibrium, not a reason to move.
    if (late > 0) return hold("within-hysteresis");
    if (from >= MAX_VIEW_QUALITY_FRACTION) return hold("clamped");
    // Not converged: a still view renders only while this asks for frames,
    // and a probe needs the whole window.
    if (track.samples.length < windowSize) {
      return hold("insufficient-samples");
    }
    // A probe risks a visible step back only near a level known to fail. With
    // none remembered it is most likely refinement, so it comes sooner and
    // reaches further.
    const failed = failedLevelOf(track, now);
    const dwellMs =
      failed === null ? track.probeDwellMs / 2 : track.probeDwellMs;
    if (now - track.heldSince < dwellMs) return hold("cooldown");
    const headroom = onTimeMs(track) / Math.max(...track.samples);
    const step = Math.max(
      failed === null ? 2 * maxIncreaseStep : maxIncreaseStep,
      headroom * HEADROOM_SPENT - 1,
    );
    const next = clamp(
      Math.min(
        from * (1 + step),
        failed === null ? MAX_VIEW_QUALITY_FRACTION : (from + failed) / 2,
      ),
    );
    if (next <= from * (1 + MIN_PROBE_STEP)) return hold("within-hysteresis");
    change(track, next, now);
    track.probedFrom = from;
    record(track, now, "increase", "below-target", from, estimate, late);
  };

  const safeNow = (track: Track, now: number): number =>
    Number.isFinite(now) ? now : (track.lastAdjustment?.atMs ?? 0);

  return {
    recordFrame(durationMs, { interacting, now }) {
      const track = trackFor(interacting);
      if (
        Number.isFinite(durationMs) &&
        durationMs >= 0 &&
        Number.isFinite(now)
      ) {
        track.samples.push(durationMs);
        if (track.samples.length > windowSize) track.samples.shift();
        adjust(track, now);
      }
      return track.fraction;
    },

    fraction: (interacting) => trackFor(interacting).fraction,
    onTimeMs: (interacting) => onTimeMs(trackFor(interacting)),
    lateThresholdMs: (interacting) => lateThresholdMs(trackFor(interacting)),

    restartAt(interacting, fraction, now) {
      const track = trackFor(interacting);
      if (!Number.isFinite(fraction) || !Number.isFinite(now)) {
        return track.fraction;
      }
      const from = track.fraction;
      const next = clamp(fraction);
      // Emergency debt outlives a reseed that does not already repay it, so a
      // cut can still be given back once calm returns in a later gesture.
      if (track.emergencyCeiling !== null && next >= track.emergencyCeiling) {
        track.emergencyCeiling = null;
      }
      track.lastChangeAt = Number.NEGATIVE_INFINITY;
      // Reseeding to the level already held keeps its evidence: gestures are
      // often shorter than a window plus a dwell, and discarding both at every
      // gesture would leave a track that can only ever fall.
      if (next !== from) {
        track.fraction = next;
        track.samples.length = 0;
        track.heldSince = now;
      }
      record(track, now, "none", "seeded", from);
      return track.fraction;
    },

    clearSamples(interacting, now) {
      const track = trackFor(interacting);
      track.samples.length = 0;
      record(track, now, "none", "insufficient-samples", track.fraction);
    },

    invalidateCapacity(interacting, now) {
      const track = trackFor(interacting);
      track.samples.length = 0;
      track.failedLevel = null;
      record(track, now, "none", "insufficient-samples", track.fraction);
    },

    reduceNow(interacting, now, factor = EMERGENCY_CUT) {
      const track = trackFor(interacting);
      const at = safeNow(track, now);
      const from = track.fraction;
      const cut = Number.isFinite(factor)
        ? Math.min(1, Math.max(EMERGENCY_CUT, factor))
        : EMERGENCY_CUT;
      const next = clamp(from * cut);
      if (next !== from) {
        // Successive cuts owe back to the first one's starting point.
        track.emergencyCeiling = Math.max(track.emergencyCeiling ?? 0, from);
        change(track, next, at);
      }
      record(
        track,
        at,
        next < from ? "decrease" : "none",
        next < from ? "emergency-cut" : "clamped",
        from,
      );
      return track.fraction;
    },

    restoreNow(interacting, now, all = false) {
      const track = trackFor(interacting);
      const ceiling = track.emergencyCeiling;
      const from = track.fraction;
      if (ceiling === null) return from;
      const at = safeNow(track, now);
      const next = all
        ? ceiling
        : Math.min(ceiling, clamp(from / EMERGENCY_CUT));
      // The samples behind the cut described a slower view than the one
      // measuring now; keeping them would argue the fraction straight back
      // down on the next adjustment.
      if (next > from) change(track, next, at);
      if (next >= ceiling) track.emergencyCeiling = null;
      record(
        track,
        at,
        next > from ? "increase" : "none",
        next > from ? "emergency-restore" : "clamped",
        from,
      );
      return track.fraction;
    },

    owesRestore: (interacting) =>
      trackFor(interacting).emergencyCeiling !== null,

    provenWithin(interacting, limitMs) {
      const track = trackFor(interacting);
      return track.samples.length >= minSamples &&
        track.samples.every((sample) => sample <= limitMs)
        ? track.fraction
        : null;
    },

    lastAdjustment: (interacting) => trackFor(interacting).lastAdjustment,

    stats() {
      const stats = (track: Track): AdaptiveQualityTrackStats => ({
        fraction: track.fraction,
        emergencyCeiling: track.emergencyCeiling,
        samples: track.samples.length,
        estimateMs: percentileOrNull(track.samples, 0.5),
        lateFraction: lateFractionOf(track),
        targetMs: track.targetMs,
        effectiveTargetMs: onTimeMs(track),
        budgetRefreshes: budgetRefreshes(track),
        failedLevel: track.failedLevel?.fraction ?? null,
        lastAdjustment: track.lastAdjustment,
      });
      return {
        minimumFraction: MIN_VIEW_QUALITY_FRACTION,
        maximumFraction: MAX_VIEW_QUALITY_FRACTION,
        cooldownMs,
        displayQuantumMs: displayQuantum(),
        stationary: stats(stationary),
        interaction: stats(interaction),
      };
    },
  };
};
