/**
 * Adaptive normalized view quality.
 *
 * Two independent tracks learn the quality fraction sustainable while the
 * view is moving and stationary. Fractions are format-neutral and always
 * remain in [0.05, 1]; members alone translate them into points, screen-space
 * error, concurrency, or another format-specific quality axis.
 */

import {
  finiteAbove,
  finiteAtLeast,
  finiteWithin,
  percentile,
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
  readonly estimateMs: number | null;
};

export type AdaptiveQualityOptions = {
  /** Starting fraction for both tracks. Default 1. */
  readonly initialFraction?: number;
  readonly stationaryTargetMs?: number;
  readonly interactionTargetMs?: number;
  readonly windowSize?: number;
  readonly percentile?: number;
  readonly hysteresis?: number;
  readonly maxIncreaseStep?: number;
  readonly maxDecreaseStep?: number;
  readonly cooldownMs?: number;
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
  readonly estimateMs: number | null;
  /** The configured target, as given. */
  readonly targetMs: number;
  /** The target actually steered to, once the display quantum is known. */
  readonly effectiveTargetMs: number;
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
 * while that is not yet known.
 *
 * Frame samples are measured between presentations, so no sample can fall
 * below the display's refresh period however little was drawn. A target near
 * that period has an increase threshold — `target * (1 - hysteresis)` — that
 * no measurement can reach, so quality can only ever fall. The quantum raises
 * each track's effective target far enough that the threshold sits above it
 * and both directions are reachable again.
 *
 * Whoever measures the display owns this number; the tracks read it whenever
 * they need a target, so it is a cheap accessor, not a computation.
 */
export type DisplayQuantumSupplier = () => number | null;

export type AdaptiveQuality = {
  recordFrame(
    durationMs: number,
    options: { readonly interacting: boolean; readonly now: number },
  ): number;
  fraction(interacting: boolean): number;
  target(interacting: boolean): number;
  restartAt(interacting: boolean, fraction: number, now: number): number;
  reduceNow(interacting: boolean, now: number): number;
  /** Gives back one emergency cut, never past what the cut took away. */
  restoreNow(interacting: boolean, now: number): number;
  /** The estimate below which `recordFrame` would argue for more quality. */
  increaseThresholdMs(interacting: boolean): number;
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
  percentile: 0.9,
  hysteresis: 0.2,
  maxIncreaseStep: 0.25,
  maxDecreaseStep: 0.5,
  cooldownMs: 400,
  minSamples: 8,
} as const;

const EMERGENCY_CUT = 0.5;

/**
 * How far the increase threshold must clear the display quantum.
 *
 * At exactly the quantum the branch is still unreachable — the estimate is a
 * p90 of intervals that jitter a little above the refresh period, never below
 * it. This is the margin that turns "the display kept up" into evidence of
 * headroom. On a 60 Hz display with the 0.2 default hysteresis it puts the
 * effective interaction target at 24 ms.
 */
const INCREASE_HEADROOM = 1.15;

type Track = {
  fraction: number;
  readonly samples: number[];
  readonly targetMs: number;
  lastAdjust: number;
  lastAdjustment: QualityAdjustment | null;
  /**
   * What an emergency cut took this track down from, until it is given back.
   *
   * A cut answers a frame the sampling loop is not allowed to answer: it needs
   * no eligible capacity sample, so it fires while tiles stream. The restore
   * has to be reachable under those same conditions or quality only ratchets
   * down for as long as a gesture keeps work pending. This ceiling is what
   * keeps the restore honest — it can undo a cut and no more, so it can never
   * stand in for the eligible increase that actually measures capacity.
   */
  emergencyCeiling: number | null;
};

export const createAdaptiveQuality = (
  options: AdaptiveQualityOptions = {},
  displayQuantum: DisplayQuantumSupplier = () => null,
): AdaptiveQuality => {
  const initialFraction = finiteWithin(
    "initialFraction",
    options.initialFraction ?? ADAPTIVE_QUALITY_DEFAULTS.initialFraction,
    MIN_VIEW_QUALITY_FRACTION,
    MAX_VIEW_QUALITY_FRACTION,
  );
  const stationaryTargetMs = finiteAbove(
    "stationaryTargetMs",
    options.stationaryTargetMs ?? ADAPTIVE_QUALITY_DEFAULTS.stationaryTargetMs,
    0,
  );
  const interactionTargetMs = finiteAbove(
    "interactionTargetMs",
    options.interactionTargetMs ??
      ADAPTIVE_QUALITY_DEFAULTS.interactionTargetMs,
    0,
  );
  const windowSize = wholeAtLeast(
    "windowSize",
    options.windowSize ?? ADAPTIVE_QUALITY_DEFAULTS.windowSize,
    1,
  );
  const percentileP = finiteWithin(
    "percentile",
    options.percentile ?? ADAPTIVE_QUALITY_DEFAULTS.percentile,
    0,
    1,
  );
  const hysteresis = finiteWithin(
    "hysteresis",
    options.hysteresis ?? ADAPTIVE_QUALITY_DEFAULTS.hysteresis,
    0,
    1,
  );
  const maxIncreaseStep = finiteAtLeast(
    "maxIncreaseStep",
    options.maxIncreaseStep ?? ADAPTIVE_QUALITY_DEFAULTS.maxIncreaseStep,
    0,
  );
  const maxDecreaseStep = finiteWithin(
    "maxDecreaseStep",
    options.maxDecreaseStep ?? ADAPTIVE_QUALITY_DEFAULTS.maxDecreaseStep,
    0,
    1,
  );
  const cooldownMs = finiteAtLeast(
    "cooldownMs",
    options.cooldownMs ?? ADAPTIVE_QUALITY_DEFAULTS.cooldownMs,
    0,
  );
  const minSamples = wholeAtLeast(
    "minSamples",
    options.minSamples ?? ADAPTIVE_QUALITY_DEFAULTS.minSamples,
    1,
  );
  const effectiveMinSamples = Math.min(windowSize, minSamples);

  const clamp = (fraction: number): number =>
    Math.min(
      MAX_VIEW_QUALITY_FRACTION,
      Math.max(MIN_VIEW_QUALITY_FRACTION, fraction),
    );
  const makeTrack = (targetMs: number): Track => ({
    fraction: initialFraction,
    samples: [],
    targetMs,
    lastAdjust: Number.NEGATIVE_INFINITY,
    lastAdjustment: null,
    emergencyCeiling: null,
  });
  const stationary = makeTrack(stationaryTargetMs);
  const interaction = makeTrack(interactionTargetMs);
  const trackFor = (interacting: boolean): Track =>
    interacting ? interaction : stationary;

  /** The lowest target whose increase threshold a real sample can reach. */
  const reachableTargetMs = (): number => {
    const quantumMs = displayQuantum();
    return quantumMs === null
      ? 0
      : (quantumMs * INCREASE_HEADROOM) / Math.max(0.05, 1 - hysteresis);
  };
  const effectiveTargetMs = (track: Track): number =>
    Math.max(track.targetMs, reachableTargetMs());

  const record = (
    track: Track,
    atMs: number,
    direction: QualityAdjustmentDirection,
    reason: QualityAdjustmentReason,
    fromFraction: number,
    estimateMs: number | null,
  ): void => {
    track.lastAdjustment = {
      atMs,
      direction,
      reason,
      fromFraction,
      toFraction: track.fraction,
      estimateMs,
    };
  };

  const adjust = (track: Track, now: number): void => {
    const from = track.fraction;
    if (track.samples.length < effectiveMinSamples) {
      record(track, now, "none", "insufficient-samples", from, null);
      return;
    }
    const estimate = percentile(track.samples, percentileP);
    if (now - track.lastAdjust < cooldownMs) {
      record(track, now, "none", "cooldown", from, estimate);
      return;
    }
    const targetMs = effectiveTargetMs(track);
    const slowLimit = targetMs * (1 + hysteresis);
    const fastLimit = targetMs * (1 - hysteresis);
    let factor: number;
    let reason: QualityAdjustmentReason;
    if (estimate > slowLimit) {
      factor = Math.max(targetMs / estimate, 1 - maxDecreaseStep);
      reason = "above-target";
    } else if (estimate < fastLimit) {
      factor = Math.min(targetMs / estimate, 1 + maxIncreaseStep);
      reason = "below-target";
    } else {
      record(track, now, "none", "within-hysteresis", from, estimate);
      return;
    }
    const next = clamp(track.fraction * factor);
    if (next === from) {
      record(track, now, "none", "clamped", from, estimate);
      return;
    }
    track.fraction = next;
    track.lastAdjust = now;
    track.emergencyCeiling = null;
    record(
      track,
      now,
      next > from ? "increase" : "decrease",
      reason,
      from,
      estimate,
    );
    track.samples.length = 0;
  };

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
    target: (interacting) => effectiveTargetMs(trackFor(interacting)),

    restartAt(interacting, fraction, now) {
      const track = trackFor(interacting);
      if (!Number.isFinite(fraction) || !Number.isFinite(now)) {
        return track.fraction;
      }
      const from = track.fraction;
      track.fraction = clamp(fraction);
      track.samples.length = 0;
      track.lastAdjust = Number.NEGATIVE_INFINITY;
      track.emergencyCeiling = null;
      record(track, now, "none", "seeded", from, null);
      return track.fraction;
    },

    reduceNow(interacting, now) {
      const track = trackFor(interacting);
      const from = track.fraction;
      const next = clamp(from * EMERGENCY_CUT);
      if (next !== from) {
        track.fraction = next;
        track.samples.length = 0;
        // Successive cuts owe back to the first one's starting point, not to
        // the step before them.
        track.emergencyCeiling = Math.max(track.emergencyCeiling ?? 0, from);
      }
      record(
        track,
        Number.isFinite(now) ? now : (track.lastAdjustment?.atMs ?? 0),
        next < from ? "decrease" : "none",
        next < from ? "emergency-cut" : "clamped",
        from,
        null,
      );
      return track.fraction;
    },

    restoreNow(interacting, now) {
      const track = trackFor(interacting);
      const ceiling = track.emergencyCeiling;
      const from = track.fraction;
      if (ceiling === null) return from;
      const next = Math.min(ceiling, clamp(from / EMERGENCY_CUT));
      if (next > from) {
        track.fraction = next;
        // The samples behind the cut described a slower machine than the one
        // measuring now; keeping them would argue the fraction straight back
        // down on the next eligible adjustment.
        track.samples.length = 0;
      }
      if (next >= ceiling) track.emergencyCeiling = null;
      record(
        track,
        Number.isFinite(now) ? now : (track.lastAdjustment?.atMs ?? 0),
        next > from ? "increase" : "none",
        next > from ? "emergency-restore" : "clamped",
        from,
        null,
      );
      return track.fraction;
    },

    increaseThresholdMs: (interacting) =>
      effectiveTargetMs(trackFor(interacting)) * (1 - hysteresis),

    stats() {
      const stats = (track: Track): AdaptiveQualityTrackStats => ({
        fraction: track.fraction,
        emergencyCeiling: track.emergencyCeiling,
        samples: track.samples.length,
        estimateMs: percentileOrNull(track.samples, percentileP),
        targetMs: track.targetMs,
        effectiveTargetMs: effectiveTargetMs(track),
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
