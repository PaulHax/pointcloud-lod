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

import {
  closeBrowser,
  cloudsUnderTest,
  openExample,
  type ExampleSession,
  type ExampleStats,
} from "./harness";
import { settleAndAssert, watching } from "./invariants";

/** The floor the example configures (`ADAPTIVE_MIN_BUDGET`, examples/vtk/main.ts). */
const FLOOR_POINTS = 200_000;

/**
 * Stated host frame times. The governor takes the worst of `hostFrameMs` and
 * `vtkFrameMs / vtkFrameFraction`, so each of these reaches the loop as about
 * 1.43x what it says — far enough either side of both targets (16 ms moving,
 * 33 ms settled, ±20% dead-bands) that no arithmetic here lands in a dead-band
 * by accident.
 */
const SLOW_FRAME_MS = 200;
const FAST_FRAME_MS = 1;
/**
 * Inside the settled dead-band (23 / 0.7 = 32.9 ms against a 33 ms target), so
 * the budget a run starts from is decided by a stated frame time too. Letting
 * the software rasteriser decide it would leave the question of whether a fall
 * is even observable to how loaded this machine happened to be.
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

/** Comfortably over the loop's 400 ms cooldown, so "stopped" is not "waiting". */
const QUIET_MS = 1_500;
/** Long enough to cover several cooldowns' worth of further slow frames. */
const HOLD_MS = 2_500;
const SAMPLE_INTERVAL_MS = 25;

/** One distinct state of the budget loop, as the page's own frame clock saw it. */
interface BudgetState {
  readonly trackBudget: number;
  readonly memoryCeilingPoints: number | null;
  readonly needsFrame: boolean;
  readonly reason: string;
  readonly direction: string;
  readonly fromBudget: number;
  readonly toBudget: number;
}

interface BudgetTrace {
  readonly states: BudgetState[];
  /** Frames the loop measured, counted from the stamp it puts on each decision. */
  readonly measuredFrames: number;
}

/** The page-side shape the recorder reads; the example's own public handles. */
interface RecorderWindow {
  pointCloudExample: {
    stats(): {
      governor: {
        trackBudget: number;
        memoryCeilingPoints: number | null;
        needsFrame: boolean;
        lastAdjustment: {
          atMs: number;
          reason: string;
          direction: string;
          fromBudget: number;
          toBudget: number;
        } | null;
      } | null;
    };
    needsFrame(): boolean;
  };
  __budgetTrace?: {
    states: BudgetState[];
    measuredFrames: number;
    previousKey: string;
    previousAtMs: number | null;
  };
}

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
      states: [] as BudgetState[],
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
        const state: BudgetState = {
          trackBudget: governor.trackBudget,
          memoryCeilingPoints: governor.memoryCeilingPoints,
          needsFrame: governor.needsFrame,
          reason: adjustment?.reason ?? "none",
          direction: adjustment?.direction ?? "none",
          fromBudget: adjustment?.fromBudget ?? governor.trackBudget,
          toBudget: adjustment?.toBudget ?? governor.trackBudget,
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
const drainBudgetStates = (session: ExampleSession): Promise<BudgetTrace> =>
  session.page.evaluate(() => {
    const trace = (window as unknown as RecorderWindow).__budgetTrace;
    if (trace === undefined) return { states: [], measuredFrames: 0 };
    const taken = { states: trace.states, measuredFrames: trace.measuredFrames };
    trace.states = [];
    trace.measuredFrames = 0;
    trace.previousKey = "";
    return taken;
  }) as Promise<BudgetTrace>;

const pageNeedsFrame = (session: ExampleSession): Promise<boolean> =>
  session.page.evaluate(() =>
    (window as unknown as RecorderWindow).pointCloudExample.needsFrame(),
  );

const governorOf = (stats: ExampleStats) => {
  expect(stats.governor, "the example is not running an adaptive governor").not.toBeNull();
  return stats.governor!;
};

interface Sample {
  readonly at: number;
  readonly stats: ExampleStats;
}

const budgetOf = (sample: Sample): number => governorOf(sample.stats).trackBudget;

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
  timeoutMs = 60_000,
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

/** The budget moved, and has then held still for longer than a cooldown. */
const budgetSteady =
  (quietMs: number) =>
  (trail: readonly Sample[]): boolean => {
    const last = trail[trail.length - 1]!;
    const first = trail[0]!;
    if (trail.every((sample) => budgetOf(sample) === budgetOf(first))) return false;
    const moved = trail.filter((sample) => budgetOf(sample) !== budgetOf(last));
    const movedAt = moved[moved.length - 1]?.at ?? first.at;
    return last.at - movedAt >= quietMs;
  };

const heldFor =
  (durationMs: number) =>
  (trail: readonly Sample[]): boolean =>
    trail[trail.length - 1]!.at - trail[0]!.at >= durationMs;

const shown = (trace: BudgetTrace): string => JSON.stringify(trace, null, 2);

/**
 * What is true of the budget at every frame of every scenario, whichever way
 * it is moving. The upper bound is the one with history: an unbounded track
 * budget once integrated to 9.46e34 points, which is finite, above every
 * ceiling, and invisible to any check that only asked whether the number was a
 * number.
 */
const assertBudgetBounded = (trace: BudgetTrace): void => {
  expect(trace.states.length, `nothing was recorded\n${shown(trace)}`).toBeGreaterThan(0);
  for (const state of trace.states) {
    expect(
      Number.isFinite(state.trackBudget),
      `the budget stopped being a finite number\n${shown(trace)}`,
    ).toBe(true);
    expect(
      state.trackBudget,
      `the budget cut below its floor\n${shown(trace)}`,
    ).toBeGreaterThanOrEqual(FLOOR_POINTS);
    if (state.memoryCeilingPoints !== null) {
      expect(
        state.trackBudget,
        `the budget grew past the memory ceiling\n${shown(trace)}`,
      ).toBeLessThanOrEqual(state.memoryCeilingPoints);
    }
  }
};

/** Every decision that actually moved the budget. */
const changesIn = (trace: BudgetTrace): BudgetState[] =>
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

describe("the adaptive budget loop at stated frame times", () => {
  afterAll(async () => {
    await closeBrowser();
  });

  for (const cloud of cloudsUnderTest()) {
    it(`gives points back when host frames run long, then holds: ${cloud.name}`, async () => {
      const session = await openAdaptive(cloud.urlPath);
      try {
        const settled = await settleAndAssert(session, SETTLE_MS);
        const before = governorOf(settled).trackBudget;
        expect(
          before,
          "the settled budget was already on the floor, so no fall could show",
        ).toBeGreaterThan(FLOOR_POINTS);

        // The settle's own decisions belong to the previous frame time.
        await drainBudgetStates(session);
        await session.setSyntheticFrameMs(SLOW_FRAME_MS);
        const { result: falling } = await watching(session, SAMPLE_INTERVAL_MS, () =>
          drive(session, "the budget stops falling", budgetSteady(QUIET_MS)),
        );
        const fall = await drainBudgetStates(session);
        const after = governorOf(falling[falling.length - 1]!.stats);

        expect(
          after.trackBudget,
          `frames far over target left the budget where it was\n${shown(fall)}`,
        ).toBeLessThan(before);
        assertBudgetBounded(fall);

        // Frame time is what the fall must be attributed to. A cut from the
        // gesture path (`emergency-cut`) or a reseed would move the same number
        // for a reason nothing in this scenario supplies.
        const cuts = changesIn(fall);
        expect(cuts.length, `the loop never moved the budget\n${shown(fall)}`).toBeGreaterThan(0);
        for (const cut of cuts) {
          expect(cut.direction, `a fall step went the wrong way\n${shown(fall)}`).toBe("decrease");
          expect(cut.reason, `the fall was not attributed to frame time\n${shown(fall)}`).toBe(
            "above-target",
          );
          expect(cut.toBudget, `a decrease that did not decrease\n${shown(fall)}`).toBeLessThan(
            cut.fromBudget,
          );
        }

        // The decision has to reach the cloud, not just the governor's stats.
        expect(
          governorOf(await session.stats()).aggregateBudget,
          "the governor's budget never reached the controller",
        ).toBe((await session.stats()).controller!.pointBudget);

        // Monotonic is not the same as stable: keep the slow frames coming and
        // the loop must sit still rather than integrate downward or oscillate.
        const landed = after.trackBudget;
        await drainBudgetStates(session);
        await watching(session, SAMPLE_INTERVAL_MS, () =>
          drive(session, "the slow frames have kept coming", heldFor(HOLD_MS)),
        );
        const hold = await drainBudgetStates(session);

        expect(
          hold.measuredFrames,
          `the hold measured no frames, so it proved nothing\n${shown(hold)}`,
        ).toBeGreaterThan(10);
        assertBudgetBounded(hold);
        expect(
          changesIn(hold),
          `the loop kept adjusting after it had come down\n${shown(hold)}`,
        ).toEqual([]);
        expect(
          [...new Set(hold.states.map((state) => state.trackBudget))],
          `the budget moved while nothing about the frames did\n${shown(hold)}`,
        ).toEqual([landed]);

        await settleAndAssert(session, SETTLE_MS);
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
          drive(session, "the budget reaches its floor", budgetSteady(QUIET_MS)),
        );
        const low = governorOf(await session.stats()).trackBudget;
        expect(low, "slow frames did not pin the budget to its floor").toBe(FLOOR_POINTS);

        assertBudgetBounded(await drainBudgetStates(session));
        await session.setSyntheticFrameMs(FAST_FRAME_MS);
        const { result: rising } = await watching(session, SAMPLE_INTERVAL_MS, () =>
          drive(session, "the budget stops growing", budgetSteady(QUIET_MS)),
        );
        const growth = await drainBudgetStates(session);
        const grown = governorOf(rising[rising.length - 1]!.stats);

        expect(
          grown.trackBudget,
          `settled frames with headroom bought no points\n${shown(growth)}`,
        ).toBeGreaterThan(low);
        assertBudgetBounded(growth);
        expect(
          grown.memoryCeilingPoints,
          "the governor reported no memory ceiling to bound growth",
        ).not.toBeNull();
        expect(
          grown.trackBudget,
          `growth ended above the ceiling the governor reports\n${shown(growth)}`,
        ).toBeLessThanOrEqual(grown.memoryCeilingPoints!);

        const gains = changesIn(growth);
        expect(gains.length, `the loop never moved the budget\n${shown(growth)}`).toBeGreaterThan(0);
        for (const gain of gains) {
          expect(gain.direction, `a growth step went the wrong way\n${shown(growth)}`).toBe(
            "increase",
          );
          expect(gain.reason, `growth was not attributed to frame time\n${shown(growth)}`).toBe(
            "below-target",
          );
          expect(gain.toBudget, `an increase that did not increase\n${shown(growth)}`).toBeGreaterThan(
            gain.fromBudget,
          );
        }
        expect(
          governorOf(await session.stats()).aggregateBudget,
          "the governor's budget never reached the controller",
        ).toBe((await session.stats()).controller!.pointBudget);

        // The defect that shipped: a view that keeps asking for frames once
        // nothing can change repaints identical pixels for ever.
        const idle = await session.until(
          "the governor to stop asking for frames",
          (stats) => stats.governor?.needsFrame === false,
          30_000,
        );
        expect(await pageNeedsFrame(session), `the view still wants frames\n${shown(growth)}`).toBe(
          false,
        );

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
        assertBudgetBounded(quiet);
        expect(
          governorOf(idle).trackBudget,
          "the budget moved on after growth was supposed to have stopped",
        ).toBe(grown.trackBudget);

        await settleAndAssert(session, SETTLE_MS);
        expect(session.failures).toEqual([]);
      } finally {
        await session.close();
      }
    });
  }
});
