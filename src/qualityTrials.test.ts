import { describe, expect, it } from "vitest";

import { createAdaptiveQuality } from "./adaptiveBudget";
import { createViewGovernor } from "./viewGovernor";

const options = {
  initialFraction: 0.8,
  minSamples: 2,
  windowSize: 4,
  cooldownMs: 0,
};

describe("stationary quality trials", () => {
  it("respects the configured decrease bound when a trial is reverted", () => {
    const quality = createAdaptiveQuality({ ...options, maxDecreaseStep: 0.1 });
    for (let now = 0; now < 4; now += 1)
      quality.recordFrame(33.4, { interacting: false, now });
    const candidate = quality.fraction(false);
    expect(candidate).toBeCloseTo(0.8 / 0.9);
    for (let now = 4; now < 6; now += 1)
      quality.recordFrame(50, { interacting: false, now });
    expect(quality.fraction(false)).toBeCloseTo(candidate * 0.9);
  });

  it("accepts higher detail only after a complete on-target window", () => {
    const governor = createViewGovernor(options);
    for (let now = 0; now < 3; now += 1) {
      governor.recordHostFrame({ hostFrameMs: 33.4, now });
      expect(governor.qualityFraction()).toBe(0.8);
      expect(governor.needsFrame()).toBe(true);
    }
    governor.recordHostFrame({ hostFrameMs: 33.4, now: 3 });
    expect(governor.qualityFraction()).toBe(1);
    expect(governor.stats().trial).toMatchObject({
      fromFraction: 0.8,
      toFraction: 1,
    });
    for (let now = 4; now < 7; now += 1) {
      governor.recordHostFrame({ hostFrameMs: 33.4, now });
      expect(governor.stats().trial).not.toBeNull();
      expect(governor.needsFrame()).toBe(true);
    }
    governor.recordHostFrame({ hostFrameMs: 33.4, now: 7 });
    expect(governor.stats().lastAdjustment?.reason).toBe("trial-accepted");
    expect(governor.stats().trial).toBeNull();
    expect(governor.needsFrame()).toBe(false);
    governor.dispose();
  });

  it("reverts an overloaded trial and stops retrying the failed increase", () => {
    const governor = createViewGovernor(options);
    for (let now = 0; now < 4; now += 1)
      governor.recordHostFrame({ hostFrameMs: 33.4, now });
    for (let now = 4; now < 6; now += 1)
      governor.recordHostFrame({ hostFrameMs: 50, now });
    expect(governor.qualityFraction()).toBe(0.8);
    expect(governor.stats()).toMatchObject({
      increaseCeiling: 0.8,
      trialAttempts: 1,
    });
    expect(governor.stats().lastAdjustment?.reason).toBe("trial-rejected");
    for (let now = 6; now < 100; now += 1)
      governor.recordHostFrame({ hostFrameMs: 16.7, now });
    expect(governor.qualityFraction()).toBe(0.8);
    expect(governor.needsFrame()).toBe(false);
    governor.invalidateCapacity();
    for (let now = 100; now < 108; now += 1)
      governor.recordHostFrame({ hostFrameMs: 33.4, now });
    expect(governor.qualityFraction()).toBe(1);
    expect(governor.needsFrame()).toBe(false);
    governor.dispose();
  });

  it("excludes streaming frames and starts a fresh trial window after work", () => {
    const governor = createViewGovernor(options);
    for (let now = 0; now < 4; now += 1)
      governor.recordHostFrame({ hostFrameMs: 33.4, now });
    governor.recordHostFrame({ hostFrameMs: 33.4, now: 4 });
    governor.setWorkState({
      workPending: true,
      physicalTileOperations: 1,
      physicalHierarchyOperations: 0,
    });
    for (let now = 5; now < 20; now += 1)
      governor.recordHostFrame({ hostFrameMs: 100, now });
    expect(governor.stats().trial).not.toBeNull();
    expect(governor.qualityFraction()).toBe(1);
    governor.setWorkState({
      workPending: false,
      physicalTileOperations: 0,
      physicalHierarchyOperations: 0,
    });
    for (let now = 20; now < 23; now += 1)
      governor.recordHostFrame({ hostFrameMs: 33.4, now });
    expect(governor.stats().trial).not.toBeNull();
    governor.recordHostFrame({ hostFrameMs: 33.4, now: 23 });
    expect(governor.stats().trial).toBeNull();
    expect(governor.needsFrame()).toBe(false);
    governor.dispose();
  });

  it.each(["motion", "configuration"])(
    "cancels stale comparisons after %s changes",
    (change) => {
      const governor = createViewGovernor(options);
      for (let now = 0; now < 4; now += 1)
        governor.recordHostFrame({ hostFrameMs: 33.4, now });
      const motion =
        change === "motion" ? governor.beginMotion("explicit") : null;
      if (!motion) governor.invalidateCapacity();
      motion?.release();
      expect(governor.stats().trial).toBeNull();
      expect(governor.qualityFraction()).toBe(0.8);
      expect(governor.stats().samples).toBe(0);
      governor.dispose();
    },
  );

  it("does not probe the on-target interaction track", () => {
    const quality = createAdaptiveQuality(options);
    for (let now = 0; now < 100; now += 1)
      quality.recordFrame(16, { interacting: true, now });
    expect(quality.fraction(true)).toBe(0.8);
    expect(quality.stats().interaction.trialAttempts).toBe(0);
  });

  it("bounds attempts even with an arbitrarily small increase step", () => {
    const quality = createAdaptiveQuality({
      ...options,
      maxIncreaseStep: 0.001,
    });
    for (let now = 0; now < 1000; now += 1)
      quality.recordFrame(33.4, { interacting: false, now });
    expect(quality.stats().stationary.trialAttempts).toBe(16);
    expect(quality.stats().stationary.trial).toBeNull();
    expect(quality.stats().stationary.lastAdjustment?.reason).toBe(
      "within-hysteresis",
    );
  });
});
