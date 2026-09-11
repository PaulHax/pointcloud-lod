import { describe, expect, it } from "vitest";

import {
  createAdaptiveQuality,
  type AdaptiveQualityOptions,
} from "./adaptiveBudget";
import { MIN_VIEW_QUALITY_FRACTION } from "./viewBudget";

const options: AdaptiveQualityOptions = {
  initialFraction: 0.8,
  interactionTargetMs: 16,
  stationaryTargetMs: 32,
  minSamples: 2,
  windowSize: 4,
  cooldownMs: 0,
  hysteresis: 0.1,
  maxIncreaseStep: 0.25,
  maxDecreaseStep: 0.5,
};

const frames = (
  quality: ReturnType<typeof createAdaptiveQuality>,
  durationMs: number,
  interacting: boolean,
  start = 0,
): void => {
  quality.recordFrame(durationMs, { interacting, now: start });
  quality.recordFrame(durationMs, { interacting, now: start + 1 });
};

describe("createAdaptiveQuality", () => {
  it("does not turn one isolated presentation hitch into an initial p90 cut", () => {
    const quality = createAdaptiveQuality();
    // A warmed-view trace: one hitch among otherwise on-target presentations.
    const intervals = [
      33.4, 16.7, 16.7, 16.6, 50.1, 16.6, 16.7, 16.7, 16.7, 16.7,
    ];
    intervals.forEach((duration, index) => {
      quality.recordFrame(duration, { interacting: false, now: index * 33.4 });
      expect(quality.fraction(false)).toBe(1);
    });
    expect(quality.stats().stationary.lastAdjustment).toMatchObject({
      reason: "within-hysteresis",
      estimateMs: 33.4,
    });
  });

  it.each([2, 10])(
    "still reduces when %i of ten presentations miss the target",
    (slowFrames) => {
      const quality = createAdaptiveQuality();
      for (let index = 0; index < 10; index += 1) {
        quality.recordFrame(index < slowFrames ? 50 : 33.3, {
          interacting: false,
          now: index * 50,
        });
      }
      expect(quality.fraction(false)).toBeCloseTo(0.66);
      expect(quality.stats().stationary.lastAdjustment).toMatchObject({
        direction: "decrease",
        reason: "above-target",
        estimateMs: 50,
      });
    },
  );

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

  it("reduces slow tracks and grows fast tracks by bounded steps", () => {
    const quality = createAdaptiveQuality(options);
    frames(quality, 64, false);
    expect(quality.fraction(false)).toBeCloseTo(0.4);
    expect(quality.stats().stationary.lastAdjustment).toMatchObject({
      direction: "decrease",
      reason: "above-target",
      fromFraction: 0.8,
      toFraction: 0.4,
    });
    frames(quality, 8, false, 2);
    expect(quality.fraction(false)).toBeCloseTo(0.5);
  });

  it("keeps interaction and stationary learning independent", () => {
    const quality = createAdaptiveQuality(options);
    frames(quality, 64, true);
    expect(quality.fraction(true)).toBeCloseTo(0.4);
    expect(quality.fraction(false)).toBe(0.8);
    quality.restartAt(false, 0.6, 5);
    expect(quality.fraction(false)).toBe(0.6);
    expect(quality.fraction(true)).toBeCloseTo(0.4);
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
  });

  it("preserves emergency recovery when completed work invalidates samples", () => {
    const quality = createAdaptiveQuality({ initialFraction: 1 });
    quality.reduceNow(true, 0);
    quality.clearSamples(true, 1);
    expect(quality.stats().interaction).toMatchObject({
      fraction: 0.5,
      emergencyCeiling: 1,
      samples: 0,
    });
    expect(quality.restoreNow(true, 2)).toBe(1);
  });

  it("collects a full baseline before probing inside hysteresis", () => {
    const quality = createAdaptiveQuality(options);
    frames(quality, 32, false);
    expect(quality.fraction(false)).toBe(0.8);
    expect(quality.stats().stationary.lastAdjustment?.reason).toBe(
      "trial-warming",
    );
  });

  it.each([
    [{ initialFraction: 0.049 }, "initialFraction"],
    [{ initialFraction: 1.01 }, "initialFraction"],
    [{ stationaryTargetMs: 0 }, "stationaryTargetMs"],
    [{ interactionTargetMs: Number.NaN }, "interactionTargetMs"],
    [{ percentile: 2 }, "percentile"],
    [{ minSamples: 0 }, "minSamples"],
  ] as const)("rejects invalid option %s", (bad, name) => {
    expect(() => createAdaptiveQuality(bad)).toThrow(name);
  });

  it("raises a target the display can never beat", () => {
    let quantumMs: number | null = null;
    const quality = createAdaptiveQuality(
      { ...options, hysteresis: 0.2 },
      () => quantumMs,
    );
    // 60 Hz: a frame that draws nothing still waits for the refresh, so 16.7
    // is both the floor of the measurement and, at a 16 ms target, inside the
    // hysteresis band — the interaction track can never be told it has room.
    frames(quality, 16.7, true);
    expect(quality.stats().interaction).toMatchObject({
      fraction: 0.8,
      lastAdjustment: { reason: "within-hysteresis" },
    });

    quantumMs = 16.7;
    expect(quality.stats().interaction.targetMs).toBe(16);
    expect(quality.stats().interaction.effectiveTargetMs).toBeCloseTo(24, 1);
    expect(quality.target(true)).toBeCloseTo(24, 1);

    quality.recordFrame(16.7, { interacting: true, now: 2 });
    expect(quality.stats().interaction).toMatchObject({
      lastAdjustment: { direction: "increase", reason: "below-target" },
    });
    expect(quality.fraction(true)).toBeGreaterThan(0.8);
  });

  it("leaves a target the display can already beat alone", () => {
    const quality = createAdaptiveQuality(
      { ...options, hysteresis: 0.2 },
      () => 16.7,
    );
    // 32 ms is nearly two refreshes away: its increase threshold at 25.6 is
    // reachable as it stands, so the quantum has nothing to say about it.
    expect(quality.stats().stationary.effectiveTargetMs).toBe(32);

    // A frame slower than the raised interaction target is still slow.
    frames(quality, 40, true);
    expect(quality.stats().interaction).toMatchObject({
      lastAdjustment: { direction: "decrease", reason: "above-target" },
    });
  });

  it("steers to the configured targets while the quantum is unknown", () => {
    // Tracks are built before the first frame is measured, so the supplier
    // has nothing to say yet and must not distort a target on the way past.
    const quality = createAdaptiveQuality(options, () => null);
    expect(quality.stats().displayQuantumMs).toBeNull();
    expect(quality.stats().interaction.effectiveTargetMs).toBe(16);
    expect(quality.target(true)).toBe(16);
  });
});
