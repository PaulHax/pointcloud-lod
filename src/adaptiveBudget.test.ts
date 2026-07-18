import { describe, expect, it } from 'vitest';

import {
  createAdaptiveBudget,
  percentile,
  type AdaptiveBudgetOptions,
} from './adaptiveBudget';

/** Feed `count` identical frames at `t0, t0+dt, ...`; return the last budget. */
const feed = (
  budget: ReturnType<typeof createAdaptiveBudget>,
  durationMs: number,
  interacting: boolean,
  count: number,
  t0 = 0,
  dt = 1000,
): number => {
  let last = budget.budget(interacting);
  for (let i = 0; i < count; i += 1) {
    last = budget.recordFrame(durationMs, { interacting, now: t0 + i * dt });
  }
  return last;
};

describe('percentile', () => {
  it('nearest-rank: p0.9 of ten values is the 9th-smallest', () => {
    const values = [10, 1, 9, 2, 8, 3, 7, 4, 6, 5];
    expect(percentile(values, 0.9)).toBe(9);
    expect(percentile(values, 1)).toBe(10);
    expect(percentile(values, 0)).toBe(1);
    expect(percentile(values, 0.5)).toBe(5);
  });

  it('does not mutate its input', () => {
    const values = [3, 1, 2];
    percentile(values, 0.5);
    expect(values).toEqual([3, 1, 2]);
  });

  it('returns NaN for an empty window', () => {
    expect(Number.isNaN(percentile([], 0.9))).toBe(true);
  });
});

const OPTS: AdaptiveBudgetOptions = {
  initialBudget: 2_000_000,
  minBudget: 200_000,
  maxBudget: 3_000_000,
  stationaryTargetMs: 16,
  interactionTargetMs: 33,
  windowSize: 30,
  percentile: 0.9,
  hysteresis: 0.2,
  maxStep: 0.25,
  cooldownMs: 400,
  minSamples: 8,
};

describe('createAdaptiveBudget', () => {
  it('shrinks the budget when frames are slower than target', () => {
    const budget = createAdaptiveBudget(OPTS);
    // 40ms >> 16ms*(1.2) stationary target: expect a shrink, capped at 25%.
    const result = feed(budget, 40, false, 8);
    expect(result).toBeLessThan(2_000_000);
    expect(result).toBeGreaterThanOrEqual(2_000_000 * (1 - 0.25));
    expect(result).toBe(2_000_000 * 0.75); // one full step down
  });

  it('grows the budget when frames are faster than target', () => {
    const budget = createAdaptiveBudget({ ...OPTS, initialBudget: 1_000_000 });
    // 4ms << 16ms*(0.8): grow, capped at +25%.
    const result = feed(budget, 4, false, 8);
    expect(result).toBe(1_000_000 * 1.25);
  });

  it('leaves the budget alone inside the hysteresis dead-band', () => {
    const budget = createAdaptiveBudget(OPTS);
    // 16ms is exactly the target; ±20% dead-band covers [12.8, 19.2].
    expect(feed(budget, 16, false, 20)).toBe(2_000_000);
    expect(feed(budget, 18, false, 20)).toBe(2_000_000);
    expect(feed(budget, 13, false, 20)).toBe(2_000_000);
  });

  it('rate-limits: a single catastrophic frame cannot collapse the budget', () => {
    const budget = createAdaptiveBudget(OPTS);
    // 10 seconds/frame — a huge overshoot; still only one 25% step.
    const result = feed(budget, 10_000, false, 8);
    expect(result).toBe(2_000_000 * 0.75);
  });

  it('honors the cooldown: no second adjustment before cooldownMs elapses', () => {
    const budget = createAdaptiveBudget(OPTS);
    // First 8 frames (t=0..7000, dt=1000) trigger one shrink at t=7000.
    feed(budget, 40, false, 8, 0, 1000);
    const afterFirst = budget.budget(false);
    expect(afterFirst).toBe(2_000_000 * 0.75);
    // Immediately pile on more slow frames within the 400ms cooldown window.
    let last = afterFirst;
    for (let i = 0; i < 8; i += 1) {
      last = budget.recordFrame(40, { interacting: false, now: 7000 + i }); // dt=1ms
    }
    expect(last).toBe(afterFirst); // cooldown blocks the second step
  });

  it('resets its window after an adjustment (measures the new budget next)', () => {
    const budget = createAdaptiveBudget(OPTS);
    feed(budget, 40, false, 8, 0, 1000); // shrink at t=7000
    expect(budget.stats().stationary.samples).toBe(0);
    // After cooldown, it needs a fresh full window before adjusting again.
    const afterOne = budget.budget(false);
    feed(budget, 40, false, 7, 8000, 1000); // only 7 fresh samples
    expect(budget.budget(false)).toBe(afterOne); // still below minSamples
    budget.recordFrame(40, { interacting: false, now: 15000 }); // 8th fresh sample
    expect(budget.budget(false)).toBe(afterOne * 0.75); // second step
  });

  it('never drops below minBudget', () => {
    const budget = createAdaptiveBudget({ ...OPTS, minBudget: 500_000 });
    // Many rounds of slow frames, each past cooldown.
    for (let round = 0; round < 40; round += 1) {
      feed(budget, 40, false, 8, round * 10_000, 1000);
    }
    expect(budget.budget(false)).toBe(500_000);
  });

  it('never rises above maxBudget', () => {
    const budget = createAdaptiveBudget({ ...OPTS, initialBudget: 2_900_000 });
    for (let round = 0; round < 40; round += 1) {
      feed(budget, 2, false, 8, round * 10_000, 1000);
    }
    expect(budget.budget(false)).toBe(3_000_000);
  });

  it('tracks interaction and stationary budgets independently', () => {
    const budget = createAdaptiveBudget(OPTS);
    // Slow while interacting, fast while stationary — the two diverge.
    feed(budget, 80, true, 8, 0, 1000); // interaction shrinks (target 33ms)
    feed(budget, 4, false, 8, 100_000, 1000); // stationary grows (target 16ms)
    expect(budget.budget(true)).toBe(2_000_000 * 0.75); // one step down
    expect(budget.budget(false)).toBe(2_000_000 * 1.25); // one step up, below ceiling
  });

  it('interaction target tolerates a frame time the stationary target would cut', () => {
    const budget = createAdaptiveBudget(OPTS);
    // 33ms is fine for interaction (dead-band [26.4, 39.6]) but slow for
    // stationary (>19.2) — same frame time, opposite verdicts.
    expect(feed(budget, 33, true, 12)).toBe(2_000_000); // no change interacting
    const still = createAdaptiveBudget(OPTS);
    expect(feed(still, 33, false, 12)).toBeLessThan(2_000_000); // shrinks stationary
  });

  it('setMaxBudget lowers a budget that now exceeds the ceiling', () => {
    const budget = createAdaptiveBudget(OPTS); // budgets start at 2M
    budget.setMaxBudget(1_000_000);
    expect(budget.budget(false)).toBe(1_000_000);
    expect(budget.budget(true)).toBe(1_000_000);
    // And it caps future growth.
    expect(feed(budget, 2, false, 40, 0, 1000)).toBe(1_000_000);
  });

  it('setMaxBudget raising the ceiling lets the loop grow again', () => {
    const budget = createAdaptiveBudget({ ...OPTS, maxBudget: 1_000_000 });
    feed(budget, 2, false, 40, 0, 1000);
    expect(budget.budget(false)).toBe(1_000_000); // pinned at old ceiling
    budget.setMaxBudget(3_000_000);
    const grown = feed(budget, 2, false, 8, 100_000, 1000);
    expect(grown).toBeGreaterThan(1_000_000);
  });

  it('a ceiling lowered below the floor wins (budget never exceeds it)', () => {
    const budget = createAdaptiveBudget(OPTS); // floor 200k, start 2M
    budget.setMaxBudget(100_000); // below the floor
    expect(budget.budget(false)).toBe(100_000);
    expect(budget.budget(true)).toBe(100_000);
    expect(budget.stats().maxBudget).toBe(100_000);
    expect(budget.stats().minBudget).toBe(100_000); // effective floor yields
    // Growth stays capped at the low ceiling.
    expect(feed(budget, 2, false, 40, 0, 1000)).toBe(100_000);
  });

  it('reset respects a ceiling lowered after construction', () => {
    // setSource → reset() must not resurrect an initial budget above a
    // ceiling the user lowered in the meantime (the quality slider).
    const budget = createAdaptiveBudget(OPTS); // initial 2M
    budget.setMaxBudget(500_000);
    budget.reset();
    expect(budget.budget(false)).toBe(500_000);
    expect(budget.budget(true)).toBe(500_000);
  });

  it('setMaxBudget discards samples measured under the old budget', () => {
    const budget = createAdaptiveBudget(OPTS); // minSamples 8
    // 7 slow frames — one short of adjusting.
    feed(budget, 40, false, 7, 0, 1000);
    expect(budget.stats().stationary.samples).toBe(7);
    // Lowering the ceiling changes the budget; the retained samples measured
    // the old (higher) budget's cost and must be discarded, or the very next
    // frame would trigger a bogus extra shrink below the new ceiling.
    budget.setMaxBudget(1_500_000);
    expect(budget.stats().stationary.samples).toBe(0);
    budget.recordFrame(40, { interacting: false, now: 8000 });
    expect(budget.budget(false)).toBe(1_500_000); // no adjust on 1 fresh sample
    // A ceiling change that leaves a budget untouched keeps its window.
    feed(budget, 40, false, 3, 9000, 1000);
    budget.setMaxBudget(1_500_000);
    expect(budget.stats().stationary.samples).toBe(4);
  });

  it('grows on 0 ms frames instead of freezing', () => {
    // An integer-ms host reads sub-millisecond frames as 0; those are valid
    // samples and must still let the budget grow toward the ceiling.
    const budget = createAdaptiveBudget({ ...OPTS, initialBudget: 1_000_000 });
    expect(feed(budget, 0, false, 8)).toBe(1_000_000 * 1.25); // one step up
  });

  it('adapts when windowSize is below minSamples (no dead zone)', () => {
    const budget = createAdaptiveBudget({ ...OPTS, windowSize: 5, minSamples: 8 });
    // The window caps at 5 (< minSamples 8); effectiveMinSamples drops to 5 so
    // the loop still acts instead of freezing.
    expect(feed(budget, 40, false, 6)).toBe(2_000_000 * 0.75);
  });

  it('ignores non-finite and negative frame durations', () => {
    const budget = createAdaptiveBudget(OPTS);
    for (let i = 0; i < 20; i += 1) {
      budget.recordFrame(Number.NaN, { interacting: false, now: i * 1000 });
      budget.recordFrame(-5, { interacting: false, now: i * 1000 });
      budget.recordFrame(Number.POSITIVE_INFINITY, { interacting: false, now: i * 1000 });
    }
    expect(budget.budget(false)).toBe(2_000_000);
    expect(budget.stats().stationary.samples).toBe(0);
  });

  it('reset restores the initial budget and clears the windows', () => {
    const budget = createAdaptiveBudget(OPTS);
    feed(budget, 40, false, 8, 0, 1000);
    expect(budget.budget(false)).toBeLessThan(2_000_000);
    budget.reset();
    expect(budget.budget(false)).toBe(2_000_000);
    expect(budget.stats().stationary.samples).toBe(0);
  });

  it('converges toward the target and then holds (no oscillation)', () => {
    // A device where cost is ~ budget: frameMs = budget / 125_000 (so 2M→16ms).
    // Drive stationary frames; the loop should settle inside the dead-band.
    const budget = createAdaptiveBudget({ ...OPTS, initialBudget: 3_000_000 });
    let t = 0;
    const frameMsFor = (points: number): number => points / 125_000;
    for (let round = 0; round < 30; round += 1) {
      for (let i = 0; i < 8; i += 1) {
        budget.recordFrame(frameMsFor(budget.budget(false)), {
          interacting: false,
          now: t,
        });
        t += 1000;
      }
    }
    const settled = budget.budget(false);
    // Target 16ms ⇒ ~2M points; dead-band ±20% ⇒ frame [12.8,19.2] ⇒
    // budget in [1.6M, 2.4M]. It must land in that band and stay bounded.
    expect(settled).toBeGreaterThanOrEqual(1_600_000);
    expect(settled).toBeLessThanOrEqual(2_400_000);
  });
});
