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

  it("records convergence inside hysteresis without changing quality", () => {
    const quality = createAdaptiveQuality(options);
    frames(quality, 32, false);
    expect(quality.fraction(false)).toBe(0.8);
    expect(quality.stats().stationary.lastAdjustment?.reason).toBe(
      "within-hysteresis",
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
});
