import { describe, expect, it } from 'vitest';

import {
  createAdaptiveBudget,
  DEFAULTS,
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
  stationaryTargetMs: 33,
  interactionTargetMs: 16,
  windowSize: 30,
  percentile: 0.9,
  hysteresis: 0.2,
  maxIncreaseStep: 0.25,
  maxDecreaseStep: 0.25,
  cooldownMs: 400,
  minSamples: 8,
};

describe('createAdaptiveBudget defaults', () => {
  it('targets 16 ms while moving and 33 ms while settled', () => {
    // Responsiveness is what a moving camera owes the user; detail is what a
    // settled one owes. The defaults are not interchangeable.
    expect(DEFAULTS.interactionTargetMs).toBe(16);
    expect(DEFAULTS.stationaryTargetMs).toBe(33);
    const budget = createAdaptiveBudget();
    expect(budget.target(true)).toBe(16);
    expect(budget.target(false)).toBe(33);
  });

  it('starts both tracks at one initial budget with a 200k floor', () => {
    // There is no separate stationary starting budget: the stationary track
    // seeds from the moving budget at every settle, so before the first settle
    // it can only start where the moving track starts.
    const budget = createAdaptiveBudget();
    expect(budget.budget(true)).toBe(1_000_000);
    expect(budget.budget(false)).toBe(1_000_000);
    expect(budget.stats().minBudget).toBe(200_000);
    expect(budget.stats().maxBudget).toBeNull();
  });

  it('puts the no-change bands at 12.8-19.2 and 26.4-39.6 ms', () => {
    const budget = createAdaptiveBudget({ minSamples: 1, cooldownMs: 0 });
    const held = (durationMs: number, interacting: boolean): boolean => {
      const fresh = createAdaptiveBudget({ minSamples: 1, cooldownMs: 0 });
      return feed(fresh, durationMs, interacting, 1) === 1_000_000;
    };
    expect(held(12.9, true)).toBe(true);
    expect(held(19.1, true)).toBe(true);
    expect(held(12.7, true)).toBe(false);
    expect(held(19.3, true)).toBe(false);
    expect(held(26.5, false)).toBe(true);
    expect(held(39.5, false)).toBe(true);
    expect(held(26.3, false)).toBe(false);
    expect(held(39.7, false)).toBe(false);
    expect(budget.budget(false)).toBe(1_000_000);
  });
});

describe('createAdaptiveBudget', () => {
  it('shrinks the budget when frames are slower than target', () => {
    const budget = createAdaptiveBudget(OPTS);
    // 60ms >> 33ms*(1.2) stationary target: expect a shrink, capped at 25%.
    const result = feed(budget, 60, false, 8);
    expect(result).toBeLessThan(2_000_000);
    expect(result).toBeGreaterThanOrEqual(2_000_000 * (1 - 0.25));
    expect(result).toBe(2_000_000 * 0.75); // one full step down
  });

  it('grows the budget when frames are faster than target', () => {
    const budget = createAdaptiveBudget({ ...OPTS, initialBudget: 1_000_000 });
    // 4ms << 33ms*(0.8): grow, capped at +25%.
    const result = feed(budget, 4, false, 8);
    expect(result).toBe(1_000_000 * 1.25);
  });

  it('leaves the budget alone inside the hysteresis dead-band', () => {
    const budget = createAdaptiveBudget(OPTS);
    // 33ms is exactly the target; ±20% dead-band covers [26.4, 39.6].
    expect(feed(budget, 33, false, 20)).toBe(2_000_000);
    expect(feed(budget, 39, false, 20)).toBe(2_000_000);
    expect(feed(budget, 27, false, 20)).toBe(2_000_000);
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
    feed(budget, 60, false, 8, 0, 1000);
    const afterFirst = budget.budget(false);
    expect(afterFirst).toBe(2_000_000 * 0.75);
    // Immediately pile on more slow frames within the 400ms cooldown window.
    let last = afterFirst;
    for (let i = 0; i < 8; i += 1) {
      last = budget.recordFrame(60, { interacting: false, now: 7000 + i }); // dt=1ms
    }
    expect(last).toBe(afterFirst); // cooldown blocks the second step
    expect(budget.stats().stationary.lastAdjustment?.reason).toBe('cooldown');
  });

  it('resets its window after an adjustment (measures the new budget next)', () => {
    const budget = createAdaptiveBudget(OPTS);
    feed(budget, 60, false, 8, 0, 1000); // shrink at t=7000
    expect(budget.stats().stationary.samples).toBe(0);
    // After cooldown, it needs a fresh full window before adjusting again.
    const afterOne = budget.budget(false);
    feed(budget, 60, false, 7, 8000, 1000); // only 7 fresh samples
    expect(budget.budget(false)).toBe(afterOne); // still below minSamples
    budget.recordFrame(60, { interacting: false, now: 15000 }); // 8th fresh sample
    expect(budget.budget(false)).toBe(afterOne * 0.75); // second step
  });

  it('never drops below minBudget', () => {
    const budget = createAdaptiveBudget({ ...OPTS, minBudget: 500_000 });
    // Many rounds of slow frames, each past cooldown.
    for (let round = 0; round < 40; round += 1) {
      feed(budget, 60, false, 8, round * 10_000, 1000);
    }
    expect(budget.budget(false)).toBe(500_000);
  });

  it('never grows past a configured maximum', () => {
    const budget = createAdaptiveBudget({
      ...OPTS,
      initialBudget: 1_000_000,
      maxBudget: 1_200_000,
    });
    for (let round = 0; round < 10; round += 1) {
      feed(budget, 1, false, 8, round * 10_000, 1000);
    }
    expect(budget.budget(false)).toBe(1_200_000);
    // Pinned at the bound, so the loop reports that it has nowhere to go
    // rather than pretending the target was met.
    expect(budget.stats().stationary.lastAdjustment?.reason).toBe('clamped');
    expect(budget.stats().maxBudget).toBe(1_200_000);
  });

  it('tracks interaction and stationary budgets independently', () => {
    const budget = createAdaptiveBudget(OPTS);
    // Slow while interacting, fast while stationary — the two diverge.
    feed(budget, 80, true, 8, 0, 1000); // interaction shrinks (target 16ms)
    feed(budget, 4, false, 8, 100_000, 1000); // stationary grows (target 33ms)
    expect(budget.budget(true)).toBe(2_000_000 * 0.75); // one step down
    expect(budget.budget(false)).toBe(2_000_000 * 1.25); // one step up
  });

  it('restarts a stale track at an observed budget and discards stale samples', () => {
    const budget = createAdaptiveBudget({ ...OPTS, initialBudget: 1_000_000 });
    budget.recordFrame(60, { interacting: false, now: 0 });
    expect(budget.stats().stationary.samples).toBe(1);

    expect(budget.restartAt(false, 2_000_000, 100)).toBe(2_000_000);
    expect(budget.stats().stationary.samples).toBe(0);
    expect(budget.stats().stationary.lastAdjustment).toMatchObject({
      atMs: 100,
      direction: 'none',
      reason: 'seeded',
      fromBudget: 1_000_000,
      toBudget: 2_000_000,
    });
    budget.recordFrame(60, { interacting: false, now: 150 });
    expect(budget.stats().stationary.samples).toBe(1);
    expect(budget.restartAt(false, 1_000_000, 200)).toBe(1_000_000);
    expect(budget.stats().stationary.samples).toBe(0);
  });

  it('ignores a restart at a non-finite budget or timestamp', () => {
    const budget = createAdaptiveBudget({ ...OPTS, initialBudget: 1_000_000 });
    expect(budget.restartAt(false, Number.NaN, 0)).toBe(1_000_000);
    expect(budget.restartAt(false, Number.POSITIVE_INFINITY, 0)).toBe(1_000_000);
    expect(budget.restartAt(false, 2_000_000, Number.NaN)).toBe(1_000_000);
    expect(budget.stats().stationary.lastAdjustment).toBeNull();
  });

  it('decreases faster than it increases by default', () => {
    const shrinking = createAdaptiveBudget({ ...OPTS, maxIncreaseStep: undefined, maxDecreaseStep: undefined });
    expect(feed(shrinking, 10_000, false, 8)).toBe(1_000_000);
    const growing = createAdaptiveBudget({
      ...OPTS,
      initialBudget: 1_000_000,
      maxIncreaseStep: undefined,
      maxDecreaseStep: undefined,
    });
    expect(feed(growing, 1, false, 8)).toBe(1_250_000);
  });

  it('supports an immediate emergency reduction', () => {
    const budget = createAdaptiveBudget(OPTS);
    expect(budget.reduceNow(false, 10)).toBe(1_000_000);
    expect(budget.reduceNow(true, 10, 0.25)).toBe(500_000);
    expect(budget.stats().stationary.samples).toBe(0);
    expect(budget.stats().interaction.lastAdjustment).toMatchObject({
      atMs: 10,
      direction: 'decrease',
      reason: 'emergency-cut',
      fromBudget: 2_000_000,
      toBudget: 500_000,
    });
  });

  it('the stationary target tolerates a frame time the moving target would cut', () => {
    const budget = createAdaptiveBudget(OPTS);
    // 33ms is the stationary target itself (dead-band [26.4, 39.6]) and well
    // past the moving dead-band's 19.2ms ceiling — same frame time, opposite
    // verdicts, and this way round: moving is the strict one.
    expect(feed(budget, 33, false, 12)).toBe(2_000_000); // no change settled
    const moving = createAdaptiveBudget(OPTS);
    expect(feed(moving, 33, true, 12)).toBeLessThan(2_000_000); // shrinks moving
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
    expect(feed(budget, 60, false, 6)).toBe(2_000_000 * 0.75);
  });

  it('ignores non-finite and negative frame durations and timestamps', () => {
    const budget = createAdaptiveBudget(OPTS);
    for (let i = 0; i < 20; i += 1) {
      budget.recordFrame(Number.NaN, { interacting: false, now: i * 1000 });
      budget.recordFrame(-5, { interacting: false, now: i * 1000 });
      budget.recordFrame(Number.POSITIVE_INFINITY, { interacting: false, now: i * 1000 });
      budget.recordFrame(60, { interacting: false, now: Number.NaN });
    }
    expect(budget.budget(false)).toBe(2_000_000);
    expect(budget.budget(true)).toBe(2_000_000);
    expect(budget.stats().stationary.samples).toBe(0);
  });

  it('converges toward the target and then holds (no oscillation)', () => {
    // A device where cost is ~ budget: frameMs = budget / 125_000.
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
    // Target 33ms ⇒ ~4.125M points; dead-band ±20% ⇒ frame [26.4,39.6] ⇒
    // budget in [3.3M, 4.95M]. It must land in that band and stay bounded.
    expect(settled).toBeGreaterThanOrEqual(3_300_000);
    expect(settled).toBeLessThanOrEqual(4_950_000);
    expect(budget.stats().stationary.lastAdjustment?.reason).toBe(
      'within-hysteresis',
    );
  });
});

describe('createAdaptiveBudget adjustment records', () => {
  it('names the reason a budget did or did not move', () => {
    const budget = createAdaptiveBudget({ ...OPTS, initialBudget: 1_000_000 });
    budget.recordFrame(60, { interacting: false, now: 0 });
    expect(budget.stats().stationary.lastAdjustment).toMatchObject({
      atMs: 0,
      direction: 'none',
      reason: 'insufficient-samples',
      estimateMs: null,
    });

    feed(budget, 60, false, 7, 1000, 1000);
    expect(budget.stats().stationary.lastAdjustment).toMatchObject({
      direction: 'decrease',
      reason: 'above-target',
      fromBudget: 1_000_000,
      toBudget: 750_000,
      estimateMs: 60,
    });

    feed(budget, 33, false, 8, 100_000, 1000);
    expect(budget.stats().stationary.lastAdjustment).toMatchObject({
      direction: 'none',
      reason: 'within-hysteresis',
      estimateMs: 33,
    });

    // Runs until the growth lands: the percentile answers for the whole
    // window, so the settled frames have to age out of it first.
    for (let i = 0; i < 40 && budget.budget(false) === 750_000; i += 1) {
      budget.recordFrame(1, { interacting: false, now: 200_000 + i * 1000 });
    }
    expect(budget.stats().stationary.lastAdjustment).toMatchObject({
      direction: 'increase',
      reason: 'below-target',
      fromBudget: 750_000,
      toBudget: 937_500,
    });
  });

  it("keeps each track's record independent", () => {
    const budget = createAdaptiveBudget({ ...OPTS, minSamples: 1, cooldownMs: 0 });
    budget.recordFrame(60, { interacting: true, now: 0 });
    expect(budget.stats().interaction.lastAdjustment?.direction).toBe('decrease');
    expect(budget.stats().stationary.lastAdjustment).toBeNull();
  });
});

/**
 * Construction options are programmer errors, so every one of them fails fast
 * naming itself and the offending value. Nothing non-finite may survive into a
 * budget, a comparison, or a statistic.
 */
describe('createAdaptiveBudget numeric configuration', () => {
  const NON_FINITE = [
    Number.NaN,
    Number.POSITIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
  ];
  const REJECTED: ReadonlyArray<
    readonly [keyof AdaptiveBudgetOptions, readonly number[]]
  > = [
    ['initialBudget', [...NON_FINITE, 0, -1]],
    ['minBudget', [...NON_FINITE, 0, -1]],
    // 100 is finite and positive but below minBudget: an inverted range.
    ['maxBudget', [...NON_FINITE, 0, -1, 100]],
    ['stationaryTargetMs', [...NON_FINITE, 0, -16]],
    ['interactionTargetMs', [...NON_FINITE, 0, -16]],
    ['windowSize', [...NON_FINITE, 0, -1]],
    ['percentile', [...NON_FINITE, -0.1, 1.1]],
    ['hysteresis', [...NON_FINITE, -0.1, 1.1]],
    ['maxIncreaseStep', [...NON_FINITE, -0.1]],
    ['maxDecreaseStep', [...NON_FINITE, -0.1, 1.1]],
    ['cooldownMs', [...NON_FINITE, -1]],
    ['minSamples', [...NON_FINITE, 0, -1]],
  ];

  for (const [option, values] of REJECTED) {
    for (const value of values) {
      it(`rejects ${option} = ${value}`, () => {
        expect(() =>
          createAdaptiveBudget({ ...OPTS, [option]: value }),
        ).toThrow(new RegExp(`^${option} must be`));
      });
    }
  }

  it('accepts every valid boundary value', () => {
    const budget = createAdaptiveBudget({
      minBudget: 1,
      maxBudget: 1,
      initialBudget: 1,
      percentile: 0,
      hysteresis: 0,
      maxIncreaseStep: 0,
      maxDecreaseStep: 1,
      cooldownMs: 0,
      windowSize: 1,
      minSamples: 1,
      stationaryTargetMs: Number.MIN_VALUE,
      interactionTargetMs: Number.MIN_VALUE,
    });
    expect(budget.budget(false)).toBe(1);
    expect(createAdaptiveBudget({ percentile: 1, hysteresis: 1 }).budget(true)).toBe(
      1_000_000,
    );
  });

  it('truncates fractional counts instead of rejecting them', () => {
    const budget = createAdaptiveBudget({
      ...OPTS,
      initialBudget: 1_000_000.75,
      minBudget: 200_000.75,
      maxBudget: 3_000_000.75,
      windowSize: 30.9,
      minSamples: 8.9,
    });
    expect(budget.budget(false)).toBe(1_000_000);
    expect(budget.stats().minBudget).toBe(200_000);
    expect(budget.stats().maxBudget).toBe(3_000_000);
  });

  it('clamps an initial budget outside the configured range', () => {
    expect(
      createAdaptiveBudget({ ...OPTS, initialBudget: 1 }).budget(false),
    ).toBe(200_000);
    expect(
      createAdaptiveBudget({
        ...OPTS,
        initialBudget: 9_000_000,
        maxBudget: 800_000,
      }).budget(false),
    ).toBe(800_000);
  });

  it('ignores a non-finite ceiling instead of lifting the memory bound', () => {
    const budget = createAdaptiveBudget({
      ...OPTS,
      initialBudget: 1_000_000,
      cooldownMs: 0,
    });
    budget.setCeiling(400_000);
    expect(feed(budget, 1, false, 12)).toBe(400_000);

    // An invalid value is broken arithmetic upstream, never permission to
    // spend more memory: it leaves the bound exactly where it was.
    for (const value of NON_FINITE) {
      budget.setCeiling(value);
      expect(feed(budget, 1, false, 40)).toBe(400_000);
    }

    // `null` is the deliberate "no ceiling" signal and still clears it.
    budget.setCeiling(null);
    expect(feed(budget, 1, false, 40)).toBeGreaterThan(400_000);
  });

  it('stops growing at the largest representable point count', () => {
    // Nothing configures a maximum and nothing reports a memory ceiling: on a
    // scene where extra points cost no frame time the loop would otherwise
    // integrate for ever, hand the host budgets past exact integer
    // representation, and never report itself pinned.
    const budget = createAdaptiveBudget({
      ...OPTS,
      initialBudget: 1_000_000,
      cooldownMs: 0,
    });
    feed(budget, 1, false, 4000);
    const stats = budget.stats().stationary;
    expect(stats.budget).toBe(Number.MAX_SAFE_INTEGER);
    expect(Number.isSafeInteger(stats.budget)).toBe(true);
    expect(stats.lastAdjustment?.reason).toBe('clamped');
  });

  it('ignores a non-finite emergency factor rather than poisoning the budget', () => {
    const budget = createAdaptiveBudget({ ...OPTS, initialBudget: 1_000_000 });
    expect(budget.reduceNow(false, 0, Number.NaN)).toBe(500_000);
    expect(budget.reduceNow(false, Number.NaN, 0.5)).toBe(250_000);
    expect(Number.isFinite(budget.stats().stationary.lastAdjustment?.atMs)).toBe(
      true,
    );
  });
});
