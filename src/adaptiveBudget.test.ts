import { describe, expect, it } from "vitest";

import {
  createAdaptiveQuality,
  type AdaptiveQuality,
  type AdaptiveQualityOptions,
} from "./adaptiveBudget";
import { MIN_VIEW_QUALITY_FRACTION } from "./viewBudget";

const REFRESH_MS = 1000 / 60;

const options: AdaptiveQualityOptions = {
  initialFraction: 0.5,
  minSamples: 5,
  windowSize: 10,
  cooldownMs: 0,
  interactionProbeDwellMs: 0,
};

/** Record `count` frames of one interval, a refresh apart, from `start`. */
const frames = (
  quality: AdaptiveQuality,
  durationMs: number,
  interacting: boolean,
  count: number,
  start = 0,
): number => {
  let now = start;
  for (let index = 0; index < count; index += 1) {
    now += REFRESH_MS;
    quality.recordFrame(durationMs, { interacting, now });
  }
  return now;
};

describe("createAdaptiveQuality", () => {
  it("governs normalized fractions and never point counts", () => {
    const quality = createAdaptiveQuality();
    expect(quality.fraction(false)).toBe(1);
    expect(quality.stats()).toMatchObject({
      minimumFraction: 0.05,
      maximumFraction: 1,
      stationary: { fraction: 1 },
      interaction: { fraction: 1 },
    });
    expect(Object.keys(quality.stats()).join(" ")).not.toMatch(
      /points|budget/i,
    );
  });

  it("grants each track whole refreshes of the display", () => {
    let quantumMs: number | null = null;
    const quality = createAdaptiveQuality({}, () => quantumMs);
    // Before the display is measured the configured targets stand as given.
    expect(quality.onTimeMs(true)).toBe(16);
    expect(quality.onTimeMs(false)).toBe(33);

    quantumMs = REFRESH_MS;
    expect(quality.stats().interaction.budgetRefreshes).toBe(1);
    expect(quality.stats().stationary.budgetRefreshes).toBe(2);
    expect(quality.onTimeMs(true)).toBeCloseTo(REFRESH_MS);
    expect(quality.lateThresholdMs(true)).toBeCloseTo(1.5 * REFRESH_MS);
    expect(quality.lateThresholdMs(false)).toBeCloseTo(2.5 * REFRESH_MS);

    // A 120 Hz display grants two of its refreshes to the same 16 ms target.
    quantumMs = REFRESH_MS / 2;
    expect(quality.stats().interaction.budgetRefreshes).toBe(2);
    expect(quality.onTimeMs(true)).toBeCloseTo(REFRESH_MS);
  });

  it("keeps measuring a view whose frames make their refresh", () => {
    const quality = createAdaptiveQuality(options, () => REFRESH_MS);
    frames(quality, REFRESH_MS, true, 9);
    expect(quality.fraction(true)).toBe(0.5);
    // Not converged: a probe needs the whole window first.
    expect(quality.stats().interaction.lastAdjustment).toMatchObject({
      direction: "none",
      reason: "insufficient-samples",
      lateFraction: 0,
    });
  });

  it("settles where a few misses stay inside tolerance", () => {
    const quality = createAdaptiveQuality(options, () => REFRESH_MS);
    const now = frames(quality, REFRESH_MS, true, 9);
    frames(quality, 2 * REFRESH_MS, true, 1, now);
    expect(quality.fraction(true)).toBe(0.5);
    expect(quality.stats().interaction.lastAdjustment).toMatchObject({
      reason: "within-hysteresis",
      lateFraction: 0.1,
    });
  });

  it("holds through an isolated hitch", () => {
    const quality = createAdaptiveQuality(options, () => REFRESH_MS);
    let now = frames(quality, REFRESH_MS, true, 5);
    now = frames(quality, 3 * REFRESH_MS, true, 1, now);
    frames(quality, REFRESH_MS, true, 3, now);
    expect(quality.fraction(true)).toBe(0.5);
    expect(quality.stats().interaction.lastAdjustment?.reason).toBe(
      "within-hysteresis",
    );
  });

  it("lowers quality gently when misses exceed the tolerance", () => {
    const quality = createAdaptiveQuality(options, () => REFRESH_MS);
    let now = frames(quality, REFRESH_MS, true, 3);
    frames(quality, 2 * REFRESH_MS, true, 2, now);
    // Two of five missed: over the default tolerance, not most of them.
    expect(quality.fraction(true)).toBeCloseTo(0.425);
    expect(quality.stats().interaction.lastAdjustment).toMatchObject({
      direction: "decrease",
      reason: "above-target",
      lateFraction: 0.4,
    });
  });

  it("holds a still view through misses a moving one would cut for", () => {
    const quality = createAdaptiveQuality(
      { ...options, initialFraction: 0.8 },
      () => REFRESH_MS,
    );
    // Four of ten frames take a third refresh against a two-refresh budget.
    let now = frames(quality, 2 * REFRESH_MS, false, 6);
    now = frames(quality, 3 * REFRESH_MS, false, 4, now);
    expect(quality.fraction(false)).toBe(0.8);
    expect(quality.stats().stationary.lastAdjustment?.reason).toBe(
      "within-hysteresis",
    );
    // Most frames missing still gives detail up.
    frames(quality, 3 * REFRESH_MS, false, 3, now);
    expect(quality.fraction(false)).toBeLessThan(0.8);
  });

  it("cuts as far as the median says when most frames are late", () => {
    const quality = createAdaptiveQuality(
      { ...options, initialFraction: 0.8 },
      () => REFRESH_MS,
    );
    // Three refreshes against a stationary budget of two: the cost lies
    // between two and three refreshes, so the cut lies between 2/3 and 1.
    frames(quality, 3 * REFRESH_MS, false, 5);
    expect(quality.fraction(false)).toBeCloseTo((0.8 * 2) / Math.sqrt(6));
    expect(quality.stats().stationary.lastAdjustment).toMatchObject({
      reason: "above-target",
      estimateMs: 3 * REFRESH_MS,
    });
  });

  it("probes only after a full clean window and the dwell", () => {
    const quality = createAdaptiveQuality(
      { ...options, interactionProbeDwellMs: 2000 },
      () => REFRESH_MS,
    );
    quality.restartAt(true, 0.6, 0);
    let now = frames(quality, REFRESH_MS, true, 10);
    expect(quality.fraction(true)).toBe(0.6);
    expect(quality.stats().interaction.lastAdjustment?.reason).toBe("cooldown");
    // With no level remembered as too expensive, the probe is most likely
    // refinement: half the dwell and twice the step.
    while (quality.fraction(true) === 0.6) {
      now = frames(quality, REFRESH_MS, true, 1, now);
    }
    expect(now).toBeGreaterThan(1000);
    expect(now).toBeLessThan(1100);
    expect(quality.fraction(true)).toBeCloseTo(0.78);
    expect(quality.stats().interaction.lastAdjustment).toMatchObject({
      direction: "increase",
      reason: "below-target",
    });
  });

  it("goes back to the level a failed probe left", () => {
    const quality = createAdaptiveQuality(options, () => REFRESH_MS);
    quality.restartAt(true, 0.6, 0);
    let now = frames(quality, REFRESH_MS, true, 10);
    expect(quality.fraction(true)).toBeCloseTo(0.78);
    now = frames(quality, 2 * REFRESH_MS, true, 5, now);
    expect(quality.fraction(true)).toBeCloseTo(0.6);
    expect(quality.stats().interaction.failedLevel).toBeCloseTo(0.78);
    // Near a level known to fail, probes are patient and small.
    frames(quality, REFRESH_MS, true, 10, now);
    expect(quality.fraction(true)).toBeCloseTo(0.69);
  });

  it("spends the headroom a faster refresh count shows in one step", () => {
    const quality = createAdaptiveQuality(
      { ...options, initialFraction: 0.3 },
      () => REFRESH_MS,
    );
    // Every frame presents in one refresh under a two-refresh budget.
    frames(quality, REFRESH_MS, false, 10);
    expect(quality.fraction(false)).toBeCloseTo(0.3 * 2 * 0.85);
  });

  it("closes half the distance to a level that proved too expensive", () => {
    const quality = createAdaptiveQuality(
      { ...options, initialFraction: 0.8, failedLevelMemoryMs: 5000 },
      () => REFRESH_MS,
    );
    let now = frames(quality, 2 * REFRESH_MS, true, 5);
    expect(quality.fraction(true)).toBeCloseTo(0.8 / Math.SQRT2);
    expect(quality.stats().interaction.failedLevel).toBe(0.8);

    now = frames(quality, REFRESH_MS, true, 10, now);
    expect(quality.fraction(true)).toBeCloseTo((0.8 / Math.SQRT2) * 1.15);
    now = frames(quality, REFRESH_MS, true, 10, now);
    now = frames(quality, REFRESH_MS, true, 10, now);
    now = frames(quality, REFRESH_MS, true, 10, now);
    now = frames(quality, REFRESH_MS, true, 10, now);
    // Each probe stops short of the level that failed.
    expect(quality.fraction(true)).toBeLessThan(0.8);

    // Once the memory lapses, probes step freely again.
    const held = quality.fraction(true);
    frames(quality, REFRESH_MS, true, 10, now + 5000);
    expect(quality.fraction(true)).toBeCloseTo(held * 1.3);
  });

  it("stops probing when the next step is too small to matter", () => {
    const quality = createAdaptiveQuality(
      { ...options, initialFraction: 0.8, failedLevelMemoryMs: 60_000 },
      () => REFRESH_MS,
    );
    let now = frames(quality, 2 * REFRESH_MS, true, 5);
    for (let window = 0; window < 12; window += 1) {
      now = frames(quality, REFRESH_MS, true, 10, now);
    }
    expect(quality.stats().interaction.lastAdjustment).toMatchObject({
      direction: "none",
      reason: "within-hysteresis",
    });
    expect(quality.fraction(true)).toBeGreaterThan(0.75);
  });

  it("keeps interaction and stationary learning independent", () => {
    const quality = createAdaptiveQuality(options, () => REFRESH_MS);
    frames(quality, 3 * REFRESH_MS, true, 5);
    expect(quality.fraction(true)).toBeCloseTo(0.25);
    expect(quality.fraction(false)).toBe(0.5);
    quality.restartAt(false, 0.6, 5);
    expect(quality.fraction(false)).toBe(0.6);
    expect(quality.fraction(true)).toBeCloseTo(0.25);
  });

  it("keeps its evidence when reseeded to the level it holds", () => {
    const quality = createAdaptiveQuality(options, () => REFRESH_MS);
    const now = frames(quality, REFRESH_MS, true, 6);
    quality.restartAt(true, 0.5, now);
    expect(quality.stats().interaction.samples).toBe(6);
    quality.restartAt(true, 0.7, now);
    expect(quality.stats().interaction.samples).toBe(0);
  });

  it("clamps restarts and emergency cuts to [0.05, 1]", () => {
    const quality = createAdaptiveQuality({
      ...options,
      initialFraction: MIN_VIEW_QUALITY_FRACTION,
    });
    expect(quality.reduceNow(false, 1)).toBe(MIN_VIEW_QUALITY_FRACTION);
    expect(quality.stats().stationary.lastAdjustment?.reason).toBe("clamped");
    expect(quality.restartAt(false, 100, 2)).toBe(1);
    expect(quality.restartAt(false, 0, 3)).toBe(MIN_VIEW_QUALITY_FRACTION);
    // An emergency cut never takes more than half in one step.
    quality.restartAt(false, 1, 4);
    expect(quality.reduceNow(false, 5, 0.1)).toBe(0.5);
  });

  it("gives back emergency cuts one at a time or all at once", () => {
    const quality = createAdaptiveQuality({ initialFraction: 1 });
    quality.reduceNow(true, 0);
    quality.reduceNow(true, 1);
    quality.clearSamples(true, 2);
    expect(quality.stats().interaction).toMatchObject({
      fraction: 0.25,
      emergencyCeiling: 1,
      samples: 0,
    });
    expect(quality.restoreNow(true, 3)).toBe(0.5);
    expect(quality.owesRestore(true)).toBe(true);
    expect(quality.restoreNow(true, 4, true)).toBe(1);
    expect(quality.owesRestore(true)).toBe(false);
    expect(quality.restoreNow(true, 5)).toBe(1);
  });

  it("keeps emergency debt through a reseed that does not repay it", () => {
    const quality = createAdaptiveQuality({ initialFraction: 1 });
    quality.reduceNow(true, 0);
    quality.restartAt(true, 0.5, 1);
    expect(quality.owesRestore(true)).toBe(true);
    quality.restartAt(true, 1, 2);
    expect(quality.owesRestore(true)).toBe(false);
  });

  it.each([
    [{ initialFraction: 0.049 }, "initialFraction"],
    [{ initialFraction: 1.01 }, "initialFraction"],
    [{ stationaryTargetMs: 0 }, "stationaryTargetMs"],
    [{ interactionTargetMs: Number.NaN }, "interactionTargetMs"],
    [{ interactionLateFrameTolerance: 2 }, "interactionLateFrameTolerance"],
    [{ stationaryLateFrameTolerance: -1 }, "stationaryLateFrameTolerance"],
    [{ interactionProbeDwellMs: -1 }, "interactionProbeDwellMs"],
    [{ minSamples: 0 }, "minSamples"],
  ] as const)("rejects invalid option %s", (bad, name) => {
    expect(() => createAdaptiveQuality(bad)).toThrow(name);
  });
});
