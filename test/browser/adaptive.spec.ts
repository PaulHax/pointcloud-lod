/**
 * The adaptive quality loop, driven at frame times this machine cannot produce.
 *
 * The loop's arithmetic is unit-tested against a fake clock. What that cannot
 * show is the assembled chain: the page paints, times the paint, reports it,
 * asks whether another frame is owed, and the governor's answer is what decides
 * whether the renderer keeps spinning. Both defects this file exists for lived
 * in that chain — a budget that integrated upward for ever because nothing
 * bounded it, and a view that kept asking for frames after everything had
 * converged.
 *
 * Every frame time here is stated through `setSyntheticFrameMs`, never
 * measured: SwiftShader's real durations say nothing about a target, and a
 * stated one also means the instrumentation below cannot perturb the input the
 * loop is deciding from.
 */

import { afterAll, describe, expect, it } from "vitest";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

import { ADAPTIVE_QUALITY_DEFAULTS } from "../../src/adaptiveBudget";
import { DEFAULT_MIN_POINT_BUDGET } from "../../src/pointCloudMember";
import { MIN_VIEW_QUALITY_FRACTION } from "../../src/viewBudget";
import {
  ADAPTIVE_CLOUD,
  closeBrowser,
  FIXTURES,
  openExample,
  type ExampleSession,
  type ExampleStats,
} from "./harness";
import { settleAndAssert, watching } from "./invariants";

/**
 * Stated host frame times. The governor takes the worst of `hostFrameMs` and
 * `vtkFrameMs / vtkFrameFraction`, and this page configures a fraction of 1 —
 * it paints nothing but the cloud, so VTK's allowance is the whole frame — so
 * each of these reaches the loop as exactly what it says. Both are far enough
 * either side of both targets (16 ms moving, 33 ms settled, ±20% dead-bands)
 * that no arithmetic here lands in a dead-band by accident.
 */
const SLOW_FRAME_MS = 200;
const FAST_FRAME_MS = 1;
/** The settled target itself, the loop's own thresholds rather than a copy. */
const SETTLED_TARGET_MS = ADAPTIVE_QUALITY_DEFAULTS.stationaryTargetMs;
/**
 * The frame time a run starts from, stated so that the starting budget is not
 * decided by how loaded this machine happened to be.
 *
 * Below the settled dead-band (26.4 ms), and deliberately so: framing the
 * cloud at page load is camera motion, which measures these same frames
 * against the 16 ms moving target and cuts the budget hard, and the settled
 * track is then seeded from wherever that left it. A frame time inside the
 * settled dead-band would leave the seed exactly there — on the floor — and a
 * run that starts on its floor cannot show a fall.
 */
const NEUTRAL_FRAME_MS = 23;

/**
 * How long convergence may take.
 *
 * The loop will not call itself converged until a whole window of frames has
 * been measured under the current budget, and a frame here is a real software
 * paint however small the number reported for it. On a multi-million-point
 * cloud that is seconds each, so a 30 s settle expired with six of the eight
 * samples in hand — a slow machine, not a stuck loop. This bounds a hang.
 */
const SETTLE_MS = 300_000;

/** Comfortably over the loop's 400 ms cooldown for the final no-paint check. */
const QUIET_MS = 1_500;
/** Long enough to cover several cooldowns' worth of further slow frames. */
const HOLD_MS = 2_500;
/** And enough frames for those cooldowns to have had something to measure. */
const HOLD_FRAMES = 12;
const SAMPLE_INTERVAL_MS = 25;

/** One distinct state of the budget loop, as the page's own frame clock saw it. */
type QualityState = {
  readonly viewQualityFraction: number;
  readonly needsFrame: boolean;
  readonly reason: string;
  readonly direction: string;
  readonly fromFraction: number;
  readonly toFraction: number;
};

type QualityTrace = {
  readonly states: QualityState[];
  /** Frames the loop measured, counted from the stamp it puts on each decision. */
  readonly measuredFrames: number;
};

/** The page-side shape the recorder reads; the example's own public handles. */
type RecorderWindow = {
  pointCloudExample: {
    stats(): {
      governor: {
        viewQualityFraction: number;
        needsFrame: boolean;
        lastAdjustment: {
          atMs: number;
          reason: string;
          direction: string;
          fromFraction: number;
          toFraction: number;
        } | null;
      } | null;
    };
    needsFrame(): boolean;
  };
  __budgetTrace?: {
    states: QualityState[];
    measuredFrames: number;
    previousKey: string;
    previousAtMs: number | null;
  };
};

/**
 * Record every distinct state of the budget loop, on the page's frame clock.
 *
 * The loop records a decision on every frame it measures, so the frames that
 * actually move the budget are one in tens: polling `lastAdjustment` from Node
 * samples the frames between the decisions and reports "nothing ever changed"
 * about a loop that changed the budget six times. It also cannot prove a
 * negative — an oscillation that completes between two polls is invisible —
 * and "it does not oscillate" is half of what scenario 8 has to show. A reader
 * on the page's own frame clock misses neither.
 */
const recordBudgetStates = (session: ExampleSession): Promise<void> =>
  session.page.evaluate(() => {
    const view = window as unknown as RecorderWindow;
    if (view.__budgetTrace !== undefined) return;
    const trace = {
      states: [] as QualityState[],
      measuredFrames: 0,
      previousKey: "",
      previousAtMs: null as number | null,
    };
    view.__budgetTrace = trace;
    const read = (): void => {
      const governor = view.pointCloudExample.stats().governor;
      if (governor !== null) {
        const adjustment = governor.lastAdjustment;
        // Every measured frame is stamped, including the ones that decide to
        // change nothing, so this counts frames the loop was fed.
        if (adjustment !== null && adjustment.atMs !== trace.previousAtMs) {
          trace.previousAtMs = adjustment.atMs;
          trace.measuredFrames += 1;
        }
        const state: QualityState = {
          viewQualityFraction: governor.viewQualityFraction,
          needsFrame: governor.needsFrame,
          reason: adjustment?.reason ?? "none",
          direction: adjustment?.direction ?? "none",
          fromFraction:
            adjustment?.fromFraction ?? governor.viewQualityFraction,
          toFraction: adjustment?.toFraction ?? governor.viewQualityFraction,
        };
        const key = JSON.stringify(state);
        if (key !== trace.previousKey) {
          trace.previousKey = key;
          trace.states.push(state);
        }
      }
      requestAnimationFrame(read);
    };
    requestAnimationFrame(read);
  });

/**
 * Take everything recorded since the last drain, so each phase reads its own
 * trace. The frame stamp survives a drain — a phase that must show no frames
 * were painted cannot have its counter reset into counting one.
 */
const drainBudgetStates = (session: ExampleSession): Promise<QualityTrace> =>
  session.page.evaluate(() => {
    const trace = (window as unknown as RecorderWindow).__budgetTrace;
    if (trace === undefined) return { states: [], measuredFrames: 0 };
    const taken = {
      states: trace.states,
      measuredFrames: trace.measuredFrames,
    };
    trace.states = [];
    trace.measuredFrames = 0;
    trace.previousKey = "";
    return taken;
  }) as Promise<QualityTrace>;

const pageNeedsFrame = (session: ExampleSession): Promise<boolean> =>
  session.page.evaluate(() =>
    (window as unknown as RecorderWindow).pointCloudExample.needsFrame(),
  );

const governorOf = (stats: ExampleStats) => {
  expect(
    stats.governor,
    "the example is not running an adaptive governor",
  ).not.toBeNull();
  return stats.governor!;
};

type Sample = {
  readonly at: number;
  readonly stats: ExampleStats;
};

/**
 * Request frames until `done` holds.
 *
 * The page repaints itself while the governor still wants frames, so most of
 * these requests coalesce into the loop's own. They matter once it stops
 * asking, which is exactly the state scenario 8's second half has to keep
 * feeding frames into.
 */
const drive = async (
  session: ExampleSession,
  what: string,
  done: (trail: readonly Sample[]) => boolean,
  timeoutMs = SETTLE_MS,
): Promise<Sample[]> => {
  const trail: Sample[] = [];
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    await session.render();
    await session.frame();
    trail.push({ at: Date.now(), stats: await session.stats() });
    if (done(trail)) return trail;
  }
  throw new Error(
    `timed out driving frames until ${what}\nlast stats: ${JSON.stringify(
      trail[trail.length - 1]?.stats,
      null,
      2,
    )}`,
  );
};

/** The governor has reached a bound and has stopped requesting measurements. */
const qualityClamped =
  (acceptFraction: (fraction: number) => boolean) =>
  (trail: readonly Sample[]): boolean => {
    const view = governorOf(trail[trail.length - 1]!.stats);
    return (
      acceptFraction(view.viewQualityFraction) &&
      view.needsFrame === false &&
      view.lastAdjustment?.reason === "clamped"
    );
  };

/** Held for a duration and for frames, for the same reason. */
const heldFor =
  (durationMs: number, frames: number) =>
  (trail: readonly Sample[]): boolean =>
    trail[trail.length - 1]!.at - trail[0]!.at >= durationMs &&
    trail.length >= frames;

const shown = (trace: QualityTrace): string => JSON.stringify(trace, null, 2);

/**
 * What is true of the budget at every frame of every scenario, whichever way
 * it is moving. The upper bound is the one with history: an unbounded track
 * budget once integrated to 9.46e34 points, which is finite, above every
 * ceiling, and invisible to any check that only asked whether the number was a
 * number.
 */
const assertQualityBounded = (trace: QualityTrace): void => {
  expect(
    trace.states.length,
    `nothing was recorded\n${shown(trace)}`,
  ).toBeGreaterThan(0);
  for (const state of trace.states) {
    expect(
      Number.isFinite(state.viewQualityFraction),
      `the quality fraction stopped being finite\n${shown(trace)}`,
    ).toBe(true);
    expect(
      state.viewQualityFraction,
      `quality cut below its normalized floor\n${shown(trace)}`,
    ).toBeGreaterThanOrEqual(MIN_VIEW_QUALITY_FRACTION);
    expect(
      state.viewQualityFraction,
      `quality grew past its normalized maximum\n${shown(trace)}`,
    ).toBeLessThanOrEqual(1);
  }
};

/** Every decision that actually moved the budget. */
const changesIn = (trace: QualityTrace): QualityState[] =>
  trace.states.filter((state) => state.direction !== "none");

/**
 * Adaptive mode, a settled scene, and a starting budget that a stated frame
 * time chose.
 */
const openAdaptive = async (cloud: string): Promise<ExampleSession> => {
  const session = await openExample({ cloud });
  await session.setBudgetMode("adaptive");
  await session.setSyntheticFrameMs(NEUTRAL_FRAME_MS);
  await recordBudgetStates(session);
  return session;
};

const describeAdaptive = existsSync(resolve(FIXTURES, "adaptive.copc.laz"))
  ? describe
  : describe.skip;

describeAdaptive("the adaptive budget loop at stated frame times", () => {
  afterAll(async () => {
    await closeBrowser();
  });

  for (const cloud of [ADAPTIVE_CLOUD]) {
    it(`gives points back when host frames run long, then holds: ${cloud.name}`, async () => {
      const session = await openAdaptive(cloud.urlPath);
      try {
        const settled = await settleAndAssert(session, SETTLE_MS);
        const before = governorOf(settled).viewQualityFraction;
        expect(
          before,
          "settled quality was already on the floor, so no fall could show",
        ).toBeGreaterThan(MIN_VIEW_QUALITY_FRACTION);

        // The settle's own decisions belong to the previous frame time.
        await drainBudgetStates(session);
        await session.setSyntheticFrameMs(SLOW_FRAME_MS);
        const { result: falling } = await watching(
          session,
          SAMPLE_INTERVAL_MS,
          () =>
            drive(
              session,
              "quality stops falling",
              qualityClamped(
                (fraction) => fraction === MIN_VIEW_QUALITY_FRACTION,
              ),
            ),
        );
        const fall = await drainBudgetStates(session);
        const after = governorOf(falling[falling.length - 1]!.stats);

        expect(
          after.viewQualityFraction,
          `frames far over target left quality where it was\n${shown(fall)}`,
        ).toBeLessThan(before);
        assertQualityBounded(fall);

        // Frame time is what the fall must be attributed to. A cut from the
        // gesture path (`emergency-cut`) or a reseed would move the same number
        // for a reason nothing in this scenario supplies.
        const cuts = changesIn(fall);
        expect(
          cuts.length,
          `the loop never moved the budget\n${shown(fall)}`,
        ).toBeGreaterThan(0);
        for (const cut of cuts) {
          expect(
            cut.direction,
            `a fall step went the wrong way\n${shown(fall)}`,
          ).toBe("decrease");
          expect(
            cut.reason,
            `the fall was not attributed to frame time\n${shown(fall)}`,
          ).toBe("above-target");
          expect(
            cut.toFraction,
            `a decrease that did not decrease\n${shown(fall)}`,
          ).toBeLessThan(cut.fromFraction);
        }

        // The normalized decision has to reach point-specific allocation, not
        // remain a governor diagnostic.
        expect((await session.stats()).controller!.pointBudget).toBeLessThan(
          settled.controller!.pointBudget,
        );

        // Monotonic is not the same as stable: keep the slow frames coming and
        // the loop must sit still rather than integrate downward or oscillate.
        const landed = after.viewQualityFraction;
        await drainBudgetStates(session);
        await watching(session, SAMPLE_INTERVAL_MS, () =>
          drive(
            session,
            "the slow frames have kept coming",
            heldFor(HOLD_MS, HOLD_FRAMES),
          ),
        );
        const hold = await drainBudgetStates(session);

        expect(
          hold.measuredFrames,
          `the hold measured no frames, so it proved nothing\n${shown(hold)}`,
        ).toBeGreaterThan(10);
        assertQualityBounded(hold);
        expect(
          changesIn(hold),
          `the loop kept adjusting after it had come down\n${shown(hold)}`,
        ).toEqual([]);
        expect(
          [...new Set(hold.states.map((state) => state.viewQualityFraction))],
          `quality moved while nothing about the frames did\n${shown(hold)}`,
        ).toEqual([landed]);

        await settleAndAssert(session, SETTLE_MS);
        expect(session.failures).toEqual([]);
      } finally {
        await session.close();
      }
    });

    it(`leaves an honest on-target frame alone: ${cloud.name}`, async () => {
      // The example configures `vtkFrameFraction: 1`, because it paints
      // nothing but the point cloud. The library's 0.7 default is right for a
      // view compositing a basemap and video underneath, and applying it here
      // would normalise every honest 33 ms frame up to 47 ms — past the top of
      // the settled dead-band — and walk the budget down to its floor while
      // nothing was ever late. A frame reported at exactly the settled target
      // is the one input that tells the two apart.
      const session = await openAdaptive(cloud.urlPath);
      try {
        const settled = await settleAndAssert(session, SETTLE_MS);
        const view = governorOf(settled);
        expect(
          view.targetFrameTimeMs,
          "the settled target is not the one these frames are stated against",
        ).toBe(SETTLED_TARGET_MS);
        const before = view.viewQualityFraction;
        expect(
          before,
          "quality was already on its floor, so a fall could not show",
        ).toBeGreaterThan(MIN_VIEW_QUALITY_FRACTION);

        await drainBudgetStates(session);
        await session.setSyntheticFrameMs(SETTLED_TARGET_MS);
        await watching(session, SAMPLE_INTERVAL_MS, () =>
          drive(
            session,
            "on-target frames have kept coming",
            heldFor(HOLD_MS, HOLD_FRAMES),
          ),
        );
        const trace = await drainBudgetStates(session);

        expect(
          trace.measuredFrames,
          `no frames were measured, so nothing was proved\n${shown(trace)}`,
        ).toBeGreaterThan(10);
        expect(
          changesIn(trace),
          `frames at the target moved the budget\n${shown(trace)}`,
        ).toEqual([]);
        expect(
          governorOf(await session.stats()).viewQualityFraction,
          `frames at the target reduced quality\n${shown(trace)}`,
        ).toBe(before);
        expect(
          trace.states.map((state) => state.reason),
          `an on-target frame was not read as on target\n${shown(trace)}`,
        ).not.toContain("above-target");

        expect(session.failures).toEqual([]);
      } finally {
        await session.close();
      }
    });

    it(`spends settled headroom on points, then stops asking: ${cloud.name}`, async () => {
      const session = await openAdaptive(cloud.urlPath);
      try {
        await settleAndAssert(session, SETTLE_MS);

        // Growth is only observable from below, and the honest way down is the
        // loop's own: slow frames until it is pinned on its floor.
        await session.setSyntheticFrameMs(SLOW_FRAME_MS);
        await watching(session, SAMPLE_INTERVAL_MS, () =>
          drive(
            session,
            "quality reaches its floor",
            qualityClamped(
              (fraction) => fraction === MIN_VIEW_QUALITY_FRACTION,
            ),
          ),
        );
        const low = governorOf(await session.stats()).viewQualityFraction;
        expect(low, "slow frames did not pin quality to its floor").toBe(
          MIN_VIEW_QUALITY_FRACTION,
        );

        assertQualityBounded(await drainBudgetStates(session));
        await session.setSyntheticFrameMs(FAST_FRAME_MS);
        const { result: rising } = await watching(
          session,
          SAMPLE_INTERVAL_MS,
          () =>
            drive(
              session,
              "quality stops growing",
              qualityClamped(
                (fraction) => fraction > MIN_VIEW_QUALITY_FRACTION,
              ),
            ),
        );
        const growth = await drainBudgetStates(session);
        const grown = governorOf(rising[rising.length - 1]!.stats);

        expect(
          grown.viewQualityFraction,
          `settled frames with headroom bought no quality\n${shown(growth)}`,
        ).toBeGreaterThan(low);
        assertQualityBounded(growth);

        const gains = changesIn(growth);
        expect(
          gains.length,
          `the loop never moved the budget\n${shown(growth)}`,
        ).toBeGreaterThan(0);
        for (const gain of gains) {
          expect(
            gain.direction,
            `a growth step went the wrong way\n${shown(growth)}`,
          ).toBe("increase");
          expect(
            gain.reason,
            `growth was not attributed to frame time\n${shown(growth)}`,
          ).toBe("below-target");
          expect(
            gain.toFraction,
            `an increase that did not increase\n${shown(growth)}`,
          ).toBeGreaterThan(gain.fromFraction);
        }
        expect(
          (await session.stats()).controller!.pointBudget,
          "the normalized quality gain never reached point allocation",
        ).toBeGreaterThan(DEFAULT_MIN_POINT_BUDGET);

        // The defect that shipped: a view that keeps asking for frames once
        // nothing can change repaints identical pixels for ever.
        // Reaching the normalized ceiling can start point-member selection:
        // the format-neutral governor is correctly clamped before that member
        // has finished its final fetch/submission frames. Prove whole-scene
        // convergence before interpreting no governor demand as no painting.
        const idle = await settleAndAssert(session, SETTLE_MS);
        expect(idle.governor?.needsFrame).toBe(false);
        expect(
          await pageNeedsFrame(session),
          `the view still wants frames\n${shown(growth)}`,
        ).toBe(false);

        // Asking is one thing; painting is what costs. Nothing here requests a
        // frame, so a page that is still measuring them is still spinning.
        await drainBudgetStates(session);
        await session.page.waitForTimeout(QUIET_MS);
        const quiet = await drainBudgetStates(session);
        expect(
          quiet.measuredFrames,
          `the renderer kept painting after the loop said it was done\n${shown(quiet)}`,
        ).toBe(0);
        expect(
          quiet.states.filter((state) => state.needsFrame),
          `the view started asking for frames again while idle\n${shown(quiet)}`,
        ).toEqual([]);
        assertQualityBounded(quiet);
        expect(
          governorOf(idle).viewQualityFraction,
          "quality moved on after growth was supposed to have stopped",
        ).toBe(grown.viewQualityFraction);

        await settleAndAssert(session, SETTLE_MS);
        expect(session.failures).toEqual([]);
      } finally {
        await session.close();
      }
    });
  }
});
