import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createViewGovernor } from "./viewGovernor";

describe("createViewGovernor", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("distributes one aggregate budget by projected importance", () => {
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      interactionInitialBudget: 300_000,
    });
    const a = vi.fn();
    const b = vi.fn();
    governor.register({ setPointBudget: a, projectedImportance: 1 });
    governor.register({ setPointBudget: b, projectedImportance: 3 });
    expect(a).toHaveBeenLastCalledWith(250_000);
    expect(b).toHaveBeenLastCalledWith(750_000);
  });

  it("uses the interaction budget through nested interaction and release", () => {
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      interactionInitialBudget: 300_000,
      interactionSettleMs: 100,
    });
    const setBudget = vi.fn();
    governor.register({ setPointBudget: setBudget });
    governor.beginInteraction();
    governor.beginInteraction();
    expect(setBudget).toHaveBeenLastCalledWith(300_000);
    governor.endInteraction();
    vi.advanceTimersByTime(200);
    expect(governor.stats().interacting).toBe(true);
    governor.endInteraction();
    vi.advanceTimersByTime(99);
    expect(governor.stats().interacting).toBe(true);
    vi.advanceTimersByTime(1);
    expect(setBudget).toHaveBeenLastCalledWith(300_000);
  });

  it("carries learned interaction detail into the stationary handoff", () => {
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      interactionInitialBudget: 1_000_000,
      interactionSettleMs: 100,
      minSamples: 1,
      cooldownMs: 0,
      maxIncreaseStep: 1,
    });
    const setBudget = vi.fn();
    governor.register({ setPointBudget: setBudget });
    governor.beginInteraction();
    governor.recordHostFrame({ hostFrameMs: 1, now: 0 });
    expect(setBudget).toHaveBeenLastCalledWith(2_000_000);

    governor.endInteraction();
    vi.advanceTimersByTime(100);

    expect(governor.stats().interacting).toBe(false);
    expect(governor.stats().stationaryLocked).toBe(true);
    expect(governor.stats().aggregateBudget).toBe(2_000_000);
    expect(setBudget).toHaveBeenLastCalledWith(2_000_000);
    expect(governor.stats().adaptive.stationary.samples).toBe(0);
  });

  it("preserves released density until the next interaction", () => {
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      interactionInitialBudget: 1_000_000,
      interactionSettleMs: 100,
      minSamples: 8,
      cooldownMs: 0,
    });
    governor.register({ setPointBudget: vi.fn() });

    for (let index = 0; index < 7; index += 1) {
      governor.recordHostFrame({ hostFrameMs: 40, now: index });
    }
    expect(governor.stats().adaptive.stationary.samples).toBe(7);

    governor.beginInteraction();
    governor.endInteraction();
    vi.advanceTimersByTime(100);
    expect(governor.stats().adaptive.stationary.samples).toBe(0);
    const handoffNow = Date.now();

    governor.recordHostFrame({
      hostFrameMs: 100,
      now: handoffNow + 1,
    });
    expect(governor.stats().aggregateBudget).toBe(1_000_000);
    expect(governor.stats().adaptive.stationary.samples).toBe(0);

    for (let index = 0; index < 16; index += 1) {
      governor.recordHostFrame({
        hostFrameMs: index < 8 ? 40 : 1,
        now: handoffNow + 2 + index,
      });
    }
    expect(governor.stats().aggregateBudget).toBe(1_000_000);
    expect(governor.stats().adaptive.stationary.samples).toBe(0);

    governor.beginInteraction();
    expect(governor.stats().stationaryLocked).toBe(false);
  });

  it("reduces immediately on a severely missed host frame during interaction", () => {
    const governor = createViewGovernor({
      interactionInitialBudget: 1_000_000,
    });
    const setBudget = vi.fn();
    governor.register({ setPointBudget: setBudget });
    governor.beginInteraction();
    governor.recordHostFrame({ hostFrameMs: 80, now: 1 });
    expect(setBudget).toHaveBeenLastCalledWith(500_000);
  });

  it("halves the budget on severe input delay during interaction", () => {
    const governor = createViewGovernor({
      interactionInitialBudget: 1_000_000,
    });
    const setBudget = vi.fn();
    governor.register({ setPointBudget: setBudget });
    governor.beginInteraction();
    // 5ms host frame is well under target*2, so only the input-delay signal
    // can trigger the cut — this isolates the severe-input branch.
    governor.recordHostFrame({ hostFrameMs: 5, inputDelayMs: 80, now: 1 });
    expect(setBudget).toHaveBeenLastCalledWith(500_000);
  });

  it("halves the budget on a severe long task during interaction", () => {
    const governor = createViewGovernor({
      interactionInitialBudget: 1_000_000,
    });
    const setBudget = vi.fn();
    governor.register({ setPointBudget: setBudget });
    governor.beginInteraction();
    governor.recordHostFrame({ hostFrameMs: 5, longTaskMs: 120, now: 1 });
    expect(setBudget).toHaveBeenLastCalledWith(500_000);
  });

  it("feeds a severely slow stationary frame through the damped path", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const setBudget = vi.fn();
    governor.register({ setPointBudget: setBudget });

    governor.recordHostFrame({ hostFrameMs: 40, longTaskMs: 120, now: 1 });
    expect(governor.stats().aggregateBudget).toBe(1_000_000);

    for (let index = 1; index < 8; index += 1) {
      governor.recordHostFrame({ hostFrameMs: 40, now: index * 100 });
    }
    expect(governor.stats().aggregateBudget).toBe(500_000);
  });

  it("holds an emergency cut through its recovery cooldown", () => {
    const governor = createViewGovernor({
      interactionInitialBudget: 1_000_000,
      cooldownMs: 400,
      minSamples: 1,
    });
    const setBudget = vi.fn();
    governor.register({ setPointBudget: setBudget });
    governor.beginInteraction();

    governor.recordHostFrame({ hostFrameMs: 80, now: 0 });
    expect(governor.stats().aggregateBudget).toBe(500_000);

    governor.recordHostFrame({ hostFrameMs: 1, now: 399 });
    expect(governor.stats().aggregateBudget).toBe(500_000);

    governor.recordHostFrame({ hostFrameMs: 1, now: 400 });
    expect(governor.stats().aggregateBudget).toBe(625_000);
  });

  it("does not turn stationary long-task spikes into a budget sawtooth", () => {
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      maxBudget: 2_000_000,
      cooldownMs: 0,
    });
    governor.register({ setPointBudget: vi.fn() });

    for (let index = 0; index < 80; index += 1) {
      governor.recordHostFrame({
        hostFrameMs: 5,
        longTaskMs: index % 10 === 0 ? 120 : undefined,
        now: index * 50,
      });
      expect(governor.stats().aggregateBudget).toBeGreaterThanOrEqual(
        1_000_000,
      );
    }
  });

  it("does not cut the budget on a fast frame with mild input delay", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const setBudget = vi.fn();
    governor.register({ setPointBudget: setBudget });
    governor.recordHostFrame({ hostFrameMs: 5, inputDelayMs: 20, now: 1 });
    expect(setBudget).not.toHaveBeenCalledWith(500_000);
  });

  it("accounts for VTK's configured share of the complete host frame", () => {
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      vtkFrameFraction: 0.5,
      minSamples: 1,
      cooldownMs: 0,
      maxDecreaseStep: 0.25,
    });
    const setBudget = vi.fn();
    governor.register({ setPointBudget: setBudget });
    governor.recordHostFrame({ hostFrameMs: 10, vtkFrameMs: 12, now: 1 });
    expect(setBudget).toHaveBeenLastCalledWith(750_000);
  });
});
