import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createViewGovernor, type ViewGovernorOptions } from "./viewGovernor";

const VSYNC_MS = 1000 / 60;
type Cost = { fixedMs?: number; spanMs?: number };
type Governor = ReturnType<typeof createViewGovernor>;

/** Present on the next refresh boundary, even when rendering is cheaper. */
const presentedMs = (
  fraction: number,
  { fixedMs = 4, spanMs = 30 }: Cost = {},
): number =>
  Math.max(1, Math.ceil((fixedMs + fraction * spanMs) / VSYNC_MS)) * VSYNC_MS;

const withGesture = <T>(
  options: ViewGovernorOptions,
  run: (governor: Governor, step: (cost?: Cost) => number) => T,
  warmDisplay = false,
): T => {
  const governor = createViewGovernor(options);
  if (warmDisplay) {
    for (let frame = 0; frame < 3; frame += 1) {
      vi.advanceTimersByTime(VSYNC_MS);
      governor.recordTransientFrame({ hostFrameMs: VSYNC_MS });
    }
  }
  const motion = governor.beginMotion("explicit");
  try {
    return run(governor, (cost) => {
      const interval = presentedMs(governor.qualityFraction(), cost);
      vi.advanceTimersByTime(interval);
      governor.recordHostFrame({ hostFrameMs: interval });
      return interval;
    });
  } finally {
    motion.release();
    governor.dispose();
  }
};

const runGesture = (options: ViewGovernorOptions, cost: Cost = {}) =>
  withGesture(options, (governor, step) =>
    Array.from({ length: 240 }, () => ({
      presentedMs: step(cost),
      fraction: governor.qualityFraction(),
      reason: governor.stats().lastAdjustment?.reason,
    })),
  );

const reversalsOf = (samples: readonly { fraction: number }[]): number => {
  let reversals = 0;
  let previousDirection = 0;
  for (let index = 1; index < samples.length; index += 1) {
    const direction = Math.sign(
      samples[index]!.fraction - samples[index - 1]!.fraction,
    );
    if (direction === 0) continue;
    if (previousDirection !== 0 && direction !== previousDirection)
      reversals += 1;
    previousDirection = direction;
  }
  return reversals;
};

describe("interaction quality under a quantized display", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it.each([
    { minSamples: 8, maxMs: 1100, maxSlowFrames: 32 },
    // Ten samples make p90 robust to one isolated hitch, at the cost of
    // additional observations before a sustained-overload adjustment.
    { minSamples: undefined, maxMs: 1200, maxSlowFrames: 35 },
  ])("recovers promptly with minimum sample count $minSamples", ({ minSamples, maxMs, maxSlowFrames }) => {
    withGesture(
      { initialFraction: 0.7, minSamples },
      (_governor, step) => {
        const start = Date.now();
        let slowFrames = 0;
        while (Date.now() - start < 5000) {
          if (step({ spanMs: 40 }) === VSYNC_MS) break;
          slowFrames += 1;
        }
        expect(Date.now() - start).toBeLessThanOrEqual(maxMs);
        expect(slowFrames).toBeLessThanOrEqual(maxSlowFrames);
      },
      true,
    );
  });

  it("discovers newly available capacity during the same gesture", () => {
    withGesture(
      { initialFraction: 0.7 },
      (governor, step) => {
        const start = Date.now();
        while (Date.now() - start < 10000) step();
        const changedAt = Date.now();
        while (
          governor.qualityFraction() < 1 &&
          Date.now() - changedAt < 5000
        ) {
          step({ spanMs: 5 });
        }
        expect(governor.qualityFraction()).toBe(1);
        expect(Date.now() - changedAt).toBeLessThanOrEqual(1600);
      },
      true,
    );
  });

  it("oscillates at the default target on a quantized display", () => {
    const samples = runGesture({});
    // The effective target is about 24 ms. Neither one nor two refreshes
    // falls inside its hysteresis band.
    const band = { low: 24 * 0.8, high: 24 * 1.2 };
    for (const { presentedMs: interval } of samples) {
      expect(interval > band.low && interval < band.high).toBe(false);
    }
    expect(samples.some(({ reason }) => reason === "within-hysteresis")).toBe(
      false,
    );
    expect(reversalsOf(samples)).toBeGreaterThan(4);
  });

  it("converges once a reachable interval lands inside the band", () => {
    // 33 ms admits the 33.3 ms two-refresh interval: band (26.4, 39.6).
    const samples = runGesture({ interactionTargetMs: 33 });
    expect(samples.some(({ reason }) => reason === "within-hysteresis")).toBe(
      true,
    );
    expect(reversalsOf(samples)).toBeLessThan(reversalsOf(runGesture({})));
  });

  it("holds full quality when even the full view fits in one refresh", () => {
    const samples = runGesture(
      { interactionTargetMs: 17 },
      { fixedMs: 1, spanMs: 2 },
    );
    expect(samples.every(({ fraction }) => fraction === 1)).toBe(true);
  });
});
