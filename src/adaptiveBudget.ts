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
  readonly samples: number;
  readonly estimateMs: number | null;
  readonly targetMs: number;
  readonly lastAdjustment: QualityAdjustment | null;
};

export type AdaptiveQualityStats = {
  readonly minimumFraction: typeof MIN_VIEW_QUALITY_FRACTION;
  readonly maximumFraction: typeof MAX_VIEW_QUALITY_FRACTION;
  readonly cooldownMs: number;
  readonly stationary: AdaptiveQualityTrackStats;
  readonly interaction: AdaptiveQualityTrackStats;
};

export type AdaptiveQuality = {
  recordFrame(
    durationMs: number,
    options: { readonly interacting: boolean; readonly now: number },
  ): number;
  fraction(interacting: boolean): number;
  target(interacting: boolean): number;
  restartAt(interacting: boolean, fraction: number, now: number): number;
  reduceNow(interacting: boolean, now: number): number;
  stats(): AdaptiveQualityStats;
};

export const ADAPTIVE_QUALITY_DEFAULTS = {
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

type Track = {
  fraction: number;
  readonly samples: number[];
  readonly targetMs: number;
  lastAdjust: number;
  lastAdjustment: QualityAdjustment | null;
};

export const createAdaptiveQuality = (
  options: AdaptiveQualityOptions = {},
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
  });
  const stationary = makeTrack(stationaryTargetMs);
  const interaction = makeTrack(interactionTargetMs);
  const trackFor = (interacting: boolean): Track =>
    interacting ? interaction : stationary;

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
    const slowLimit = track.targetMs * (1 + hysteresis);
    const fastLimit = track.targetMs * (1 - hysteresis);
    let factor: number;
    let reason: QualityAdjustmentReason;
    if (estimate > slowLimit) {
      factor = Math.max(track.targetMs / estimate, 1 - maxDecreaseStep);
      reason = "above-target";
    } else if (estimate < fastLimit) {
      factor = Math.min(track.targetMs / estimate, 1 + maxIncreaseStep);
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
    target: (interacting) => trackFor(interacting).targetMs,

    restartAt(interacting, fraction, now) {
      const track = trackFor(interacting);
      if (!Number.isFinite(fraction) || !Number.isFinite(now)) {
        return track.fraction;
      }
      const from = track.fraction;
      track.fraction = clamp(fraction);
      track.samples.length = 0;
      track.lastAdjust = Number.NEGATIVE_INFINITY;
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

    stats() {
      const stats = (track: Track): AdaptiveQualityTrackStats => ({
        fraction: track.fraction,
        samples: track.samples.length,
        estimateMs: percentileOrNull(track.samples, percentileP),
        targetMs: track.targetMs,
        lastAdjustment: track.lastAdjustment,
      });
      return {
        minimumFraction: MIN_VIEW_QUALITY_FRACTION,
        maximumFraction: MAX_VIEW_QUALITY_FRACTION,
        cooldownMs,
        stationary: stats(stationary),
        interaction: stats(interaction),
      };
    },
  };
};
