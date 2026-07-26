/**
 * Adaptive visible-point budget.
 *
 * Turns a stream of measured render durations into a visible-point budget that
 * tracks a target frame time. The device, the cloud, and the camera all vary,
 * so a fixed budget is either too heavy (janky) or too light (blurry); this
 * closes the loop instead.
 *
 * Two independent tracks — one for a moving camera, one for a settled camera —
 * because the two regimes have opposite goals. A moving camera is what the
 * user is steering, so it gets the tighter 16 ms target: responsiveness beats
 * detail while the view is still changing. A settled camera is what the user
 * is reading, so it gets the looser 33 ms target and spends the extra frame
 * time on points. Each track keeps a bounded window of recent frame times,
 * estimates load from a high percentile of that window, and nudges its budget
 * toward the track's target frame time.
 *
 * Anti-oscillation is three guards working together:
 *   - hysteresis: a dead-band around the target where nothing changes, so
 *     frame-time noise never drives a change;
 *   - rate limiting: each adjustment has bounded directional steps, with
 *     reductions allowed to react faster than increases;
 *   - cooldown + window reset: after any change a track waits `cooldownMs` and
 *     discards its now-stale samples, so it measures the new budget's cost
 *     before deciding again.
 *
 * Every decision — including the decision to change nothing — is recorded as
 * the track's `lastAdjustment`, which is both the diagnostic answer to "why is
 * this cloud drawing N points" and the signal a caller uses to tell a track
 * that is still refining from one that has converged.
 *
 * Pure and clock-free: every method that depends on time takes an explicit
 * `now`, so behavior is fully deterministic under test. The library caller
 * (`viewGovernor.ts`) supplies `Date.now()`.
 */

import {
  finiteAbove,
  finiteAtLeast,
  finiteWithin,
  wholeAtLeast,
} from "./numeric";

export type BudgetRegime = "stationary" | "interaction";

export type BudgetAdjustmentDirection = "increase" | "decrease" | "none";

export type BudgetAdjustmentReason =
  /** Estimate under the dead-band: bought more points with the headroom. */
  | "below-target"
  /** Estimate over the dead-band: gave points back to hit the target. */
  | "above-target"
  /** Estimate inside the dead-band — the track has converged. */
  | "within-hysteresis"
  /** A change is due but the previous one is still being measured. */
  | "cooldown"
  /** Not enough frames measured under the current budget to decide. */
  | "insufficient-samples"
  /** A change was due but the budget is pinned at `minBudget`/`maxBudget`. */
  | "clamped"
  /** A gesture missed badly enough to cut immediately (`reduceNow`). */
  | "emergency-cut"
  /** The track was restarted at an observed budget (`restartAt`). */
  | "seeded";

/** The most recent decision a track made, whether or not it moved anything. */
export interface BudgetAdjustment {
  readonly atMs: number;
  readonly direction: BudgetAdjustmentDirection;
  readonly reason: BudgetAdjustmentReason;
  readonly fromBudget: number;
  readonly toBudget: number;
  /** Percentile estimate the decision used; null when it had none. */
  readonly estimateMs: number | null;
}

export interface AdaptiveBudgetOptions {
  /**
   * Starting budget for both tracks, points, clamped into
   * [`minBudget`, `maxBudget`]. Default 1,000,000 — the moving regime's
   * initial budget. The stationary track has no separate initial value: every
   * settle seeds it from the budget the moving regime just sustained, and
   * before the first settle it starts here too.
   */
  initialBudget?: number;
  /** Hard floor; the loop never drops a budget below this. Default 200_000. */
  minBudget?: number;
  /**
   * Optional configured maximum, points. Omitted means the loop's own
   * measurements and the caller's memory ceiling are the only upper bounds.
   * Configured here rather than clamped by the caller so the loop cannot wind
   * up above a ceiling it can never spend and then jump when the ceiling rises.
   */
  maxBudget?: number;
  /** Target frame time while the camera is settled, ms. Default 33 (~30 fps). */
  stationaryTargetMs?: number;
  /** Target frame time while the camera moves, ms. Default 16 (~60 fps). */
  interactionTargetMs?: number;
  /** Frame-time samples retained per track for the percentile. Default 30. */
  windowSize?: number;
  /** Percentile (0..1) of the window taken as the load estimate. Default 0.9. */
  percentile?: number;
  /**
   * Half-width of the no-change dead-band around the target, as a fraction of
   * the target (e.g. 0.2 → no change while the estimate is within ±20% of the
   * target). Default 0.2, which puts the no-change bands at 12.8-19.2 ms while
   * moving and 26.4-39.6 ms while settled.
   */
  hysteresis?: number;
  /** Largest fractional increase per adjustment. Default 0.25. */
  maxIncreaseStep?: number;
  /** Largest fractional decrease per adjustment. Default 0.5. */
  maxDecreaseStep?: number;
  /** Minimum time between adjustments on a track, ms. Default 400. */
  cooldownMs?: number;
  /** Samples a track needs before it will adjust at all. Default 8. */
  minSamples?: number;
}

export interface AdaptiveBudgetTrackStats {
  readonly budget: number;
  readonly samples: number;
  /** Percentile estimate of the current window, or null if empty. */
  readonly estimateMs: number | null;
  readonly targetMs: number;
  /** Null until the track has made its first decision. */
  readonly lastAdjustment: BudgetAdjustment | null;
}

export interface AdaptiveBudgetStats {
  readonly minBudget: number;
  /** Configured maximum, or null when none was configured. */
  readonly maxBudget: number | null;
  readonly stationary: AdaptiveBudgetTrackStats;
  readonly interaction: AdaptiveBudgetTrackStats;
}

export interface RecordFrameOptions {
  /** Whether the camera was moving when this frame was rendered. */
  readonly interacting: boolean;
  /** Monotonic-ish timestamp in ms (the caller passes `Date.now()`). */
  readonly now: number;
}

export interface AdaptiveBudget {
  /**
   * Record one rendered frame's duration and return the (possibly updated)
   * budget for that frame's regime. Non-finite or negative durations, and a
   * non-finite `now`, are ignored and return the current budget unchanged.
   */
  recordFrame(durationMs: number, options: RecordFrameOptions): number;
  /** Current budget for a regime, without recording a sample. */
  budget(interacting: boolean): number;
  /** Target frame time for a regime, ms. */
  target(interacting: boolean): number;
  /**
   * Start a fresh measurement run at an observed budget.
   * Always clears timing samples from the preceding run. A non-finite budget
   * or timestamp is ignored and leaves the track untouched.
   */
  restartAt(interacting: boolean, points: number, now: number): number;
  /** Immediately reduce one track, bypassing sample and cooldown thresholds. */
  reduceNow(interacting: boolean, now: number, factor?: number): number;
  stats(): AdaptiveBudgetStats;
}

/** Shared so the governor's thresholds cannot drift from the loop's. */
export const DEFAULTS = {
  initialBudget: 1_000_000,
  minBudget: 200_000,
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

/**
 * Nearest-rank percentile of `values` (0..1). Does not mutate the input.
 * `percentile(xs, 0.9)` of ten values returns the 9th-smallest.
 */
export const percentile = (values: readonly number[], p: number): number => {
  if (values.length === 0) return Number.NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const clampedP = Math.min(Math.max(p, 0), 1);
  const rank = Math.ceil(clampedP * sorted.length);
  const index = Math.min(Math.max(rank - 1, 0), sorted.length - 1);
  return sorted[index]!;
};

interface Track {
  budget: number;
  readonly samples: number[];
  readonly targetMs: number;
  lastAdjust: number;
  lastAdjustment: BudgetAdjustment | null;
}

export const createAdaptiveBudget = (
  options: AdaptiveBudgetOptions = {},
): AdaptiveBudget => {
  const minBudget = wholeAtLeast(
    "minBudget",
    options.minBudget ?? DEFAULTS.minBudget,
    1,
  );
  const maxBudget =
    options.maxBudget === undefined
      ? Number.POSITIVE_INFINITY
      : wholeAtLeast("maxBudget", options.maxBudget, minBudget);
  const stationaryTargetMs = finiteAbove(
    "stationaryTargetMs",
    options.stationaryTargetMs ?? DEFAULTS.stationaryTargetMs,
    0,
  );
  const interactionTargetMs = finiteAbove(
    "interactionTargetMs",
    options.interactionTargetMs ?? DEFAULTS.interactionTargetMs,
    0,
  );
  const windowSize = wholeAtLeast(
    "windowSize",
    options.windowSize ?? DEFAULTS.windowSize,
    1,
  );
  const percentileP = finiteWithin(
    "percentile",
    options.percentile ?? DEFAULTS.percentile,
    0,
    1,
  );
  const hysteresis = finiteWithin(
    "hysteresis",
    options.hysteresis ?? DEFAULTS.hysteresis,
    0,
    1,
  );
  const maxIncreaseStep = finiteAtLeast(
    "maxIncreaseStep",
    options.maxIncreaseStep ?? DEFAULTS.maxIncreaseStep,
    0,
  );
  const maxDecreaseStep = finiteWithin(
    "maxDecreaseStep",
    options.maxDecreaseStep ?? DEFAULTS.maxDecreaseStep,
    0,
    1,
  );
  const cooldownMs = finiteAtLeast(
    "cooldownMs",
    options.cooldownMs ?? DEFAULTS.cooldownMs,
    0,
  );
  const minSamples = wholeAtLeast(
    "minSamples",
    options.minSamples ?? DEFAULTS.minSamples,
    1,
  );
  // A window smaller than minSamples could never reach the threshold, which
  // would silently freeze the loop; cap the requirement at the window size.
  const effectiveMinSamples = Math.min(minSamples, windowSize);

  const clamp = (points: number): number =>
    Math.round(Math.min(Math.max(points, minBudget), maxBudget));

  const initialBudget = clamp(
    wholeAtLeast("initialBudget", options.initialBudget ?? DEFAULTS.initialBudget, 1),
  );

  const newTrack = (targetMs: number): Track => ({
    budget: initialBudget,
    samples: [],
    targetMs,
    lastAdjust: Number.NEGATIVE_INFINITY,
    lastAdjustment: null,
  });
  const stationary = newTrack(stationaryTargetMs);
  const interaction = newTrack(interactionTargetMs);

  const trackFor = (interacting: boolean): Track =>
    interacting ? interaction : stationary;

  const record = (
    track: Track,
    atMs: number,
    direction: BudgetAdjustmentDirection,
    reason: BudgetAdjustmentReason,
    fromBudget: number,
    estimateMs: number | null,
  ): void => {
    track.lastAdjustment = {
      atMs,
      direction,
      reason,
      fromBudget,
      toBudget: track.budget,
      estimateMs,
    };
  };

  const adjust = (track: Track, now: number): void => {
    const from = track.budget;
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

    // recordFrame already rejected non-finite/negative durations, so estimate
    // is finite and >= 0. A 0 ms estimate is legitimate (very fast frames) and
    // must be allowed to grow the budget — only the grow branch sees it, where
    // target/0 clamps to the +maxIncreaseStep cap.
    let factor: number;
    let reason: BudgetAdjustmentReason;
    if (estimate > slowLimit) {
      // Too slow: shrink toward the target, but by at most one step.
      factor = Math.max(track.targetMs / estimate, 1 - maxDecreaseStep);
      reason = "above-target";
    } else if (estimate < fastLimit) {
      // Headroom: grow toward the target, but by at most one step.
      factor = Math.min(track.targetMs / estimate, 1 + maxIncreaseStep);
      reason = "below-target";
    } else {
      // Inside the dead-band: leave the budget alone (anti-oscillation).
      record(track, now, "none", "within-hysteresis", from, estimate);
      return;
    }

    const next = clamp(track.budget * factor);
    if (next === from) {
      // Already pinned at a clamp bound; nothing to do (and don't reset the
      // window — that would keep re-estimating with no effect).
      record(track, now, "none", "clamped", from, estimate);
      return;
    }
    track.budget = next;
    track.lastAdjust = now;
    record(
      track,
      now,
      next > from ? "increase" : "decrease",
      reason,
      from,
      estimate,
    );
    // The old samples measured the previous budget's cost; discard them so the
    // next adjustment waits for frames rendered under the new budget.
    track.samples.length = 0;
  };

  return {
    recordFrame(durationMs, { interacting, now }) {
      const track = trackFor(interacting);
      if (Number.isFinite(durationMs) && durationMs >= 0 && Number.isFinite(now)) {
        track.samples.push(durationMs);
        if (track.samples.length > windowSize) track.samples.shift();
        adjust(track, now);
      }
      return track.budget;
    },

    budget(interacting) {
      return trackFor(interacting).budget;
    },

    target(interacting) {
      return trackFor(interacting).targetMs;
    },

    restartAt(interacting, points, now) {
      const track = trackFor(interacting);
      if (!Number.isFinite(points) || !Number.isFinite(now)) return track.budget;
      const from = track.budget;
      track.budget = clamp(points);
      track.samples.length = 0;
      track.lastAdjust = now;
      record(track, now, "none", "seeded", from, null);
      return track.budget;
    },

    reduceNow(interacting, now, factor = 0.5) {
      const track = trackFor(interacting);
      const safeFactor = Number.isFinite(factor)
        ? Math.min(Math.max(factor, 0), 1)
        : 0.5;
      const from = track.budget;
      const next = clamp(track.budget * safeFactor);
      if (next !== from) {
        track.budget = next;
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
      return track.budget;
    },

    stats() {
      const trackStats = (track: Track): AdaptiveBudgetTrackStats => ({
        budget: track.budget,
        samples: track.samples.length,
        estimateMs:
          track.samples.length > 0 ? percentile(track.samples, percentileP) : null,
        targetMs: track.targetMs,
        lastAdjustment: track.lastAdjustment,
      });
      return {
        minBudget,
        maxBudget: Number.isFinite(maxBudget) ? maxBudget : null,
        stationary: trackStats(stationary),
        interaction: trackStats(interaction),
      };
    },
  };
};
