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
    Array.from({ length: 1200 }, () => ({
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
  ])(
    "recovers promptly with minimum sample count $minSamples",
    ({ minSamples, maxMs, maxSlowFrames }) => {
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
    },
  );

  it("discovers newly available capacity during the same gesture", () => {
    withGesture(
      { initialFraction: 0.7 },
      (governor, step) => {
        const start = Date.now();
        while (Date.now() - start < 10000) step();
        const changedAt = Date.now();
        const fractions: number[] = [];
        while (
          governor.qualityFraction() < 1 &&
          Date.now() - changedAt < 30000
        ) {
          step({ spanMs: 5 });
          fractions.push(governor.qualityFraction());
        }
        expect(governor.qualityFraction()).toBe(1);
        // A moving view at one refresh per frame cannot see its headroom, so
        // it climbs by patient probes, and never back down while it does.
        expect(Date.now() - changedAt).toBeLessThanOrEqual(15000);
        expect(reversalsOf(fractions.map((fraction) => ({ fraction })))).toBe(
          0,
        );
      },
      true,
    );
  });

  it("settles a steady view at the default target on a quantized display", () => {
    // Every interval is one or two refreshes. Probes narrow in on the most
    // detail that makes its refresh, then stop: the last level found too
    // expensive caps them, and a step too small to matter is not taken.
    const samples = runGesture({});
    expect(reversalsOf(samples)).toBeLessThanOrEqual(4);
    const settled = samples.slice(-300);
    expect(new Set(settled.map(({ fraction }) => fraction)).size).toBe(1);
    expect(settled.every(({ presentedMs }) => presentedMs <= VSYNC_MS)).toBe(
      true,
    );
    expect(settled.at(-1)?.reason).toBe("within-hysteresis");
  });

  it("grants a 33 ms interaction target two refreshes", () => {
    const samples = runGesture({ interactionTargetMs: 33 });
    const afterFirstCut = samples.slice(
      samples.findIndex(({ fraction }) => fraction < 1),
    );
    // Only the rare probe of the level that failed pays a third refresh.
    const late = afterFirstCut.filter(
      ({ presentedMs }) => presentedMs > 2.5 * VSYNC_MS,
    );
    expect(late.length / afterFirstCut.length).toBeLessThan(0.03);
    // One look above the ceiling each time the memory of it lapses.
    expect(reversalsOf(samples)).toBeLessThanOrEqual(4);
  });

  it("holds full quality when even the full view fits in one refresh", () => {
    const samples = runGesture(
      { interactionTargetMs: 17 },
      { fixedMs: 1, spanMs: 2 },
    );
    expect(samples.every(({ fraction }) => fraction === 1)).toBe(true);
  });
});
