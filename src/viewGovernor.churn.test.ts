import { afterEach, describe, expect, it, vi } from "vitest";

import { createViewGovernor, type ViewGovernorOptions } from "./viewGovernor";

/**
 * A display that can only present on a refresh boundary.
 *
 * This is the whole point of the simulation. Frame samples are presentation
 * intervals, so however cheap a frame is it still costs a whole refresh
 * period, and however slightly a frame overruns it costs two. The adaptive
 * loop compares that quantized number against a continuous target, and whether
 * it can ever hold still depends on whether a multiple of the period lands
 * inside the hysteresis band.
 */
const VSYNC_MS = 1000 / 60;

/**
 * Frame cost as a function of drawn quality, quantized to the display.
 *
 * `fixedMs` is what a frame costs before anything is drawn and `spanMs` is
 * what a full-quality view adds. With the defaults below the view stops
 * holding 60 Hz at a quality fraction of about 0.42, which puts the
 * interesting boundary in the middle of the range rather than at an end.
 */
const presentedMs = (
  fraction: number,
  { fixedMs = 4, spanMs = 30 }: { fixedMs?: number; spanMs?: number } = {},
): number =>
  Math.max(1, Math.ceil((fixedMs + fraction * spanMs) / VSYNC_MS)) * VSYNC_MS;

type Sample = {
  readonly atMs: number;
  readonly fraction: number;
  readonly presentedMs: number;
  readonly reason: string | undefined;
};

/**
 * Drive a real governor through a gesture and record what it asked for.
 *
 * Time is faked, the cost model is closed-form and no measurement is sampled
 * from the machine, so a run is bit-for-bit repeatable: the noise band of
 * every number below is exactly zero.
 */
const runGesture = (
  options: ViewGovernorOptions,
  {
    frames = 240,
    cost = {},
  }: { frames?: number; cost?: { fixedMs?: number; spanMs?: number } } = {},
): readonly Sample[] => {
  const governor = createViewGovernor(options);
  governor.setWorkState({
    workPending: false,
    physicalTileOperations: 0,
    physicalHierarchyOperations: 0,
  });
  const motion = governor.beginMotion("explicit");
  const samples: Sample[] = [];
  for (let index = 0; index < frames; index += 1) {
    const fraction = governor.qualityFraction();
    const interval = presentedMs(fraction, cost);
    vi.advanceTimersByTime(interval);
    governor.recordHostFrame({ hostFrameMs: interval });
    samples.push({
      atMs: Date.now(),
      fraction: governor.qualityFraction(),
      presentedMs: interval,
      reason: governor.stats().lastAdjustment?.reason,
    });
  }
  motion.release();
  governor.dispose();
  return samples;
};

const reversalsOf = (samples: readonly Sample[]): number => {
  let reversals = 0;
  let previousDirection = 0;
  let previous: number | null = null;
  for (const { fraction } of samples) {
    if (previous !== null && fraction !== previous) {
      const direction = fraction > previous ? 1 : -1;
      if (previousDirection !== 0 && direction !== previousDirection) {
        reversals += 1;
      }
      previousDirection = direction;
    }
    previous = fraction;
  }
  return reversals;
};

const countReason = (samples: readonly Sample[], reason: string): number =>
  samples.filter((sample) => sample.reason === reason).length;

/** Distinct quality fractions the gesture put on screen. */
const distinctOf = (samples: readonly Sample[]): number =>
  new Set(samples.map((sample) => sample.fraction.toFixed(6))).size;

describe("interaction quality under a quantized display", () => {
  afterEach(() => vi.useRealTimers());

  it("cannot converge at the default target, because the dead band is empty", () => {
    vi.useFakeTimers();
    const samples = runGesture({});

    // The display quantum raises the 16 ms interaction target to about 24 ms,
    // so the band the loop would hold still inside is (19.2, 28.8). Every
    // presentation interval is a multiple of 16.67, and none lands in it.
    const band = { low: 24 * 0.8, high: 24 * 1.2 };
    const presented = new Set(samples.map((sample) => sample.presentedMs));
    for (const interval of presented) {
      expect(interval > band.low && interval < band.high).toBe(false);
    }

    // So the loop never reports the one reason that means "this is fine",
    // and it swings for as long as the gesture lasts.
    expect(countReason(samples, "within-hysteresis")).toBe(0);
    expect(reversalsOf(samples)).toBeGreaterThan(4);
  });

  it("converges once a reachable interval lands inside the band", () => {
    vi.useFakeTimers();
    // 33 ms admits the 33.3 ms two-refresh interval: band (26.4, 39.6).
    const samples = runGesture({ interactionTargetMs: 33 });

    expect(countReason(samples, "within-hysteresis")).toBeGreaterThan(0);
    expect(reversalsOf(samples)).toBeLessThan(reversalsOf(runGesture({})));
  });

  it("holds perfectly still when the estimate is not quantized away from the target", () => {
    vi.useFakeTimers();
    // A view cheap enough to always present in one refresh, with a target the
    // refresh period itself satisfies, is the case the loop is built for.
    const samples = runGesture(
      { interactionTargetMs: 17 },
      { cost: { fixedMs: 1, spanMs: 2 } },
    );
    expect(reversalsOf(samples)).toBe(0);
    expect(distinctOf(samples)).toBe(1);
  });
});
