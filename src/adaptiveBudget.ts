/**
 * Adaptive visible-point budget (Phase 5).
 *
 * Turns a stream of measured render durations into a visible-point budget that
 * tracks a target frame time. The device, the cloud, and the camera all vary,
 * so a fixed budget is either too heavy (janky) or too light (blurry); this
 * closes the loop instead.
 *
 * Two independent tracks — one for a moving camera, one for a settled camera —
 * because the two regimes have different cost (a moving camera streams and
 * re-selects) and different goals (responsiveness while moving, detail while
 * still). Each track keeps a bounded window of recent frame times, estimates
 * load from a high percentile of that window, and nudges its budget toward the
 * track's target frame time.
 *
 * Anti-oscillation, as PLAN.md Phase 5 requires, is three guards working
 * together:
 *   - hysteresis: a dead-band around the target where nothing changes, so
 *     frame-time noise never drives a change;
 *   - rate limiting: each adjustment moves the budget by at most `maxStep`, so
 *     one slow frame cannot collapse the budget;
 *   - cooldown + window reset: after any change a track waits `cooldownMs` and
 *     discards its now-stale samples, so it measures the new budget's cost
 *     before deciding again.
 *
 * Pure and clock-free: every method that depends on time takes an explicit
 * `now`, so behavior is fully deterministic under test. The library caller
 * (`controller.ts`) supplies `Date.now()`.
 */

export type BudgetRegime = 'stationary' | 'interaction';

export interface AdaptiveBudgetOptions {
  /** Starting budget for both tracks, points. Clamped to the range below. */
  initialBudget?: number;
  /** Hard floor; the loop never drops a budget below this. Default 200_000. */
  minBudget?: number;
  /**
   * Hard ceiling. Default unbounded: frame time is the governor, and the LOD
   * controller supplies a memory-derived ceiling via `setMaxBudget` — the
   * loop cannot sense GPU memory, and frame time stays healthy right up
   * until an allocation fails, so the ceiling must come from outside.
   */
  maxBudget?: number;
  /** Target frame time while the camera is settled, ms. Default 16 (~60 fps). */
  stationaryTargetMs?: number;
  /** Target frame time while the camera moves, ms. Default 33 (~30 fps). */
  interactionTargetMs?: number;
  /** Frame-time samples retained per track for the percentile. Default 30. */
  windowSize?: number;
  /** Percentile (0..1) of the window taken as the load estimate. Default 0.9. */
  percentile?: number;
  /**
   * Half-width of the no-change dead-band around the target, as a fraction of
   * the target (e.g. 0.2 → no change while the estimate is within ±20% of the
   * target). Default 0.2.
   */
  hysteresis?: number;
  /** Largest fractional budget change per adjustment. Default 0.25. */
  maxStep?: number;
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
}

export interface AdaptiveBudgetStats {
  readonly minBudget: number;
  readonly maxBudget: number;
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
   * budget for that frame's regime. Non-finite or negative durations are
   * ignored and return the current budget unchanged.
   */
  recordFrame(durationMs: number, options: RecordFrameOptions): number;
  /** Current budget for a regime, without recording a sample. */
  budget(interacting: boolean): number;
  /**
   * Raise or lower the ceiling (e.g. when the memory-derived cap moves).
   * Budgets already above the new ceiling drop to it immediately.
   */
  setMaxBudget(points: number): void;
  /** Reset both tracks to the initial budget and clear their windows. */
  reset(): void;
  stats(): AdaptiveBudgetStats;
}

const DEFAULTS = {
  initialBudget: 2_000_000,
  minBudget: 200_000,
  maxBudget: Number.POSITIVE_INFINITY,
  stationaryTargetMs: 16,
  interactionTargetMs: 33,
  windowSize: 30,
  percentile: 0.9,
  hysteresis: 0.2,
  maxStep: 0.25,
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
  lastAdjust: number;
}

export const createAdaptiveBudget = (
  options: AdaptiveBudgetOptions = {},
): AdaptiveBudget => {
  const minBudget = Math.max(1, options.minBudget ?? DEFAULTS.minBudget);
  let maxBudget = Math.max(minBudget, options.maxBudget ?? DEFAULTS.maxBudget);
  const stationaryTargetMs =
    options.stationaryTargetMs ?? DEFAULTS.stationaryTargetMs;
  const interactionTargetMs =
    options.interactionTargetMs ?? DEFAULTS.interactionTargetMs;
  const windowSize = Math.max(1, Math.floor(options.windowSize ?? DEFAULTS.windowSize));
  const percentileP = options.percentile ?? DEFAULTS.percentile;
  const hysteresis = Math.max(0, options.hysteresis ?? DEFAULTS.hysteresis);
  const maxStep = Math.max(0, options.maxStep ?? DEFAULTS.maxStep);
  const cooldownMs = Math.max(0, options.cooldownMs ?? DEFAULTS.cooldownMs);
  const minSamples = Math.max(1, Math.floor(options.minSamples ?? DEFAULTS.minSamples));
  // A window smaller than minSamples could never reach the threshold, which
  // would silently freeze the loop; cap the requirement at the window size.
  const effectiveMinSamples = Math.min(minSamples, windowSize);

  const clamp = (points: number): number => {
    // The ceiling wins if it was lowered below the floor (a low quality
    // setting): honor the requested ceiling exactly rather than the floor.
    const lower = Math.min(minBudget, maxBudget);
    return Math.round(Math.min(Math.max(points, lower), maxBudget));
  };

  const initialBudget = clamp(options.initialBudget ?? DEFAULTS.initialBudget);

  const stationary: Track = { budget: initialBudget, samples: [], lastAdjust: Number.NEGATIVE_INFINITY };
  const interaction: Track = { budget: initialBudget, samples: [], lastAdjust: Number.NEGATIVE_INFINITY };

  const trackFor = (interacting: boolean): Track =>
    interacting ? interaction : stationary;
  const targetFor = (interacting: boolean): number =>
    interacting ? interactionTargetMs : stationaryTargetMs;

  const adjust = (track: Track, targetMs: number, now: number): void => {
    if (track.samples.length < effectiveMinSamples) return;
    if (now - track.lastAdjust < cooldownMs) return;

    const estimate = percentile(track.samples, percentileP);
    // recordFrame already rejected non-finite/negative durations, so estimate
    // is finite and >= 0. A 0 ms estimate is legitimate (very fast frames) and
    // must be allowed to grow the budget — only the grow branch sees it, where
    // target/0 clamps to the +maxStep cap — so bail only on non-finite here.
    if (!Number.isFinite(estimate)) return;

    const slowLimit = targetMs * (1 + hysteresis);
    const fastLimit = targetMs * (1 - hysteresis);

    let factor: number;
    if (estimate > slowLimit) {
      // Too slow: shrink toward the target, but by at most one step.
      factor = Math.max(targetMs / estimate, 1 - maxStep);
    } else if (estimate < fastLimit) {
      // Headroom: grow toward the target, but by at most one step.
      factor = Math.min(targetMs / estimate, 1 + maxStep);
    } else {
      // Inside the dead-band: leave the budget alone (anti-oscillation).
      return;
    }

    const next = clamp(track.budget * factor);
    if (next === track.budget) {
      // Already pinned at a clamp bound; nothing to do (and don't reset the
      // window — that would keep re-estimating with no effect).
      return;
    }
    track.budget = next;
    track.lastAdjust = now;
    // The old samples measured the previous budget's cost; discard them so the
    // next adjustment waits for frames rendered under the new budget.
    track.samples.length = 0;
  };

  return {
    recordFrame(durationMs, { interacting, now }) {
      const track = trackFor(interacting);
      if (Number.isFinite(durationMs) && durationMs >= 0) {
        track.samples.push(durationMs);
        if (track.samples.length > windowSize) track.samples.shift();
        adjust(track, targetFor(interacting), now);
      }
      return track.budget;
    },

    budget(interacting) {
      return trackFor(interacting).budget;
    },

    setMaxBudget(points) {
      // Don't pin the ceiling up to the floor — a low ceiling must win so the
      // effective budget never exceeds the user's requested quality.
      maxBudget = Math.max(1, Math.round(points));
      for (const track of [stationary, interaction]) {
        const next = clamp(track.budget);
        if (next !== track.budget) {
          track.budget = next;
          // Same invariant as adjust(): the retained samples measured the old
          // budget's cost; discard them so the next adjustment waits for
          // frames rendered under the new one.
          track.samples.length = 0;
        }
      }
    },

    reset() {
      // Re-clamp: the ceiling may have been lowered (setMaxBudget) since
      // construction, and the restored budget must never exceed it.
      stationary.budget = clamp(initialBudget);
      interaction.budget = clamp(initialBudget);
      stationary.samples.length = 0;
      interaction.samples.length = 0;
      stationary.lastAdjust = Number.NEGATIVE_INFINITY;
      interaction.lastAdjust = Number.NEGATIVE_INFINITY;
    },

    stats() {
      const trackStats = (track: Track): AdaptiveBudgetTrackStats => ({
        budget: track.budget,
        samples: track.samples.length,
        estimateMs:
          track.samples.length > 0 ? percentile(track.samples, percentileP) : null,
      });
      return {
        // Report the effective floor: a ceiling lowered below it wins.
        minBudget: Math.min(minBudget, maxBudget),
        maxBudget,
        stationary: trackStats(stationary),
        interaction: trackStats(interaction),
      };
    },
  };
};
