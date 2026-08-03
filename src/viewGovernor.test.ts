import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createViewGovernor, type ViewGovernor } from "./viewGovernor";

const memberOptions = (
  setPointBudget: (points: number) => void,
  setDensityFraction: (densityFraction: number) => void = vi.fn(),
) => ({ setPointBudget, setDensityFraction });

/**
 * The clock frames are reported with. It tracks the fake timers so advancing
 * them also advances the reported timeline — that is how a test lets a
 * cooldown elapse — and it only ever moves forward, since a frame stamped
 * before the last adjustment would read as a cooldown that never elapses.
 */
let clock = 0;
const tick = (): number => {
  clock = Math.max(clock + 1, Date.now());
  return clock;
};

/** Report `count` frames of the same duration, one tick apart. */
const frames = (
  governor: ViewGovernor,
  hostFrameMs: number,
  count: number,
): void => {
  for (let index = 0; index < count; index += 1) {
    governor.recordHostFrame({ hostFrameMs, now: tick() });
  }
};

/** Take and immediately drop a motion reference, then let the view settle. */
const moveAndSettle = (governor: ViewGovernor, settleMs: number): void => {
  const motion = governor.beginMotion("explicit");
  governor.recordCameraChange();
  motion.release();
  vi.advanceTimersByTime(settleMs);
};

/** Settles in 100 ms, adjusts on every 8-frame window, no recovery cooldown. */
const FAST_ADAPT = {
  initialBudget: 1_000_000,
  interactionSettleMs: 100,
  minSamples: 8,
  cooldownMs: 0,
} as const;

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(0);
  clock = 0;
});
afterEach(() => vi.useRealTimers());

describe("createViewGovernor", () => {
  it("distributes one aggregate budget by projected importance", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const a = vi.fn();
    const b = vi.fn();
    governor.register(memberOptions(a)).update({ projectedImportance: 1 });
    governor.register(memberOptions(b)).update({ projectedImportance: 3 });
    expect(a).toHaveBeenLastCalledWith(250_000);
    expect(b).toHaveBeenLastCalledWith(750_000);
  });

  it("does not hand a culled view more than the one showing something", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const showing = vi.fn();
    const culled = vi.fn();
    // Importance is screen-space error in CSS px, so a distant cloud reports
    // well under 1 while a fully culled one reports exactly 0.
    governor
      .register(memberOptions(showing))
      .update({ projectedImportance: 0.04 });
    governor.register(memberOptions(culled)).update({ projectedImportance: 0 });

    const showingBudget: number = showing.mock.calls.at(-1)![0];
    const culledBudget: number = culled.mock.calls.at(-1)![0];
    expect(showingBudget).toBeGreaterThan(culledBudget);
    expect(showingBudget + culledBudget).toBeLessThanOrEqual(1_000_000);
  });

  it("splits evenly while no active member has reported anything", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const a = vi.fn();
    const b = vi.fn();
    governor.register(memberOptions(a));
    governor.register(memberOptions(b));
    expect(a).toHaveBeenLastCalledWith(500_000);
    expect(b).toHaveBeenLastCalledWith(500_000);
  });

  it("never starves an active view when another's importance dominates", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const quiet = vi.fn();
    const dominant = vi.fn();
    governor.register(memberOptions(quiet)).update({ projectedImportance: 1 });
    governor
      .register(memberOptions(dominant))
      .update({ projectedImportance: 1_000_000 });

    const quietBudget: number = quiet.mock.calls.at(-1)![0];
    const dominantBudget: number = dominant.mock.calls.at(-1)![0];
    // A strict proportional split would hand the quiet view ~1 point. It keeps
    // at least a quarter of the 500k even split, so it stays visible.
    expect(quietBudget).toBeGreaterThanOrEqual(125_000);
    expect(dominantBudget).toBeGreaterThan(quietBudget);
    expect(quietBudget + dominantBudget).toBeLessThanOrEqual(1_000_000);
  });
});

describe("createViewGovernor motion references", () => {
  it("holds the moving regime until the last source releases", () => {
    const governor = createViewGovernor({ interactionSettleMs: 100 });
    governor.register(memberOptions(vi.fn()));

    const gesture = governor.beginMotion("explicit");
    const playback = governor.beginMotion("inferred");
    governor.recordCameraChange();
    expect(governor.stats().regime).toBe("interaction");
    expect(governor.stats().motion).toMatchObject({
      explicitReferences: 1,
      inferredReferences: 1,
      source: "both",
      settling: false,
    });

    gesture.release();
    expect(governor.stats().regime).toBe("interaction");
    expect(governor.stats().motion.source).toBe("inferred");

    playback.release();
    // The settle debounce still holds the regime, but nothing is moving.
    expect(governor.stats().regime).toBe("interaction");
    expect(governor.stats().motion).toMatchObject({
      source: null,
      settling: true,
    });

    vi.advanceTimersByTime(99);
    expect(governor.stats().regime).toBe("interaction");
    vi.advanceTimersByTime(1);
    expect(governor.stats().regime).toBe("stationary");
  });

  it("ignores a double release rather than unbalancing the count", () => {
    const governor = createViewGovernor({ interactionSettleMs: 100 });
    const held = governor.beginMotion("inferred");
    const other = governor.beginMotion("inferred");
    governor.recordCameraChange();
    held.release();
    held.release();
    expect(governor.stats().motion.inferredReferences).toBe(1);
    expect(governor.stats().regime).toBe("interaction");
    other.release();
    vi.advanceTimersByTime(100);
    expect(governor.stats().regime).toBe("stationary");
  });

  it("reports inferred motion as motion without calling it explicit", () => {
    const governor = createViewGovernor();
    const inferred = governor.beginMotion("inferred");
    const stats = governor.stats();
    expect(stats.regime).toBe("interaction");
    expect(stats.motion.source).toBe("inferred");
    expect(stats.targetFrameTimeMs).toBe(16);
    inferred.release();
  });
});

describe("createViewGovernor stationary refinement", () => {
  it("starts stationary at the last moving budget without a jump", () => {
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      interactionSettleMs: 100,
      minSamples: 1,
      cooldownMs: 0,
      maxIncreaseStep: 1,
    });
    const setBudget = vi.fn();
    governor.register(memberOptions(setBudget));

    const gesture = governor.beginMotion("explicit");
    governor.recordCameraChange();
    frames(governor, 1, 1); // fast moving frame: the moving track doubles
    expect(setBudget).toHaveBeenLastCalledWith(2_000_000);

    gesture.release();
    vi.advanceTimersByTime(100);

    const stats = governor.stats();
    expect(stats.regime).toBe("stationary");
    expect(stats.aggregateBudget).toBe(2_000_000);
    expect(setBudget).toHaveBeenLastCalledWith(2_000_000);
    // Seeded, not inherited: the moving regime's frames measured a different
    // target and must not decide the first stationary step.
    expect(stats.samples).toBe(0);
    expect(stats.lastAdjustment).toMatchObject({ reason: "seeded" });
  });

  it("jumps stationary refinement back to the density it last sustained", () => {
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      interactionSettleMs: 100,
    });
    const setBudget = vi.fn();
    governor.register(memberOptions(setBudget));

    // Two consecutive missed frames crash the moving budget mid-gesture.
    const gesture = governor.beginMotion("explicit");
    governor.recordCameraChange();
    governor.recordHostFrame({ hostFrameMs: 80, now: tick() });
    governor.recordHostFrame({ hostFrameMs: 80, now: tick() });
    expect(governor.stats().trackBudget).toBe(500_000);

    gesture.release();
    vi.advanceTimersByTime(100);

    // Settling re-seeds from the stationary track's own budget — capacity
    // this machine already proved at the stationary target — not from the
    // crashed moving budget it would otherwise spend seconds climbing out of.
    const stats = governor.stats();
    expect(stats.regime).toBe("stationary");
    expect(stats.trackBudget).toBe(1_000_000);
    expect(stats.lastAdjustment).toMatchObject({ reason: "seeded" });
  });

  it("recovers a crashed moving budget from the settled track", () => {
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      interactionSettleMs: 100,
      minSamples: 1,
      cooldownMs: 0,
      maxIncreaseStep: 1,
    });
    const setBudget = vi.fn();
    governor.register(memberOptions(setBudget));

    // The moving track crashes: the first miss halves through the damped
    // path (minSamples 1), the second through the emergency cut.
    const gesture = governor.beginMotion("explicit");
    governor.recordCameraChange();
    governor.recordHostFrame({ hostFrameMs: 80, now: tick() });
    governor.recordHostFrame({ hostFrameMs: 80, now: tick() });
    expect(governor.stats().trackBudget).toBe(250_000);

    // The settled track then proves out a far higher density. The moving
    // track's own frames can never vote it back up — a healthy gesture on a
    // vsync-floored display reads exactly the display cadence, inside the
    // moving dead-band — so this is the only way it recovers.
    gesture.release();
    vi.advanceTimersByTime(100);
    frames(governor, 1, 3);
    expect(governor.stats().trackBudget).toBe(8_000_000);

    // A fresh gesture inherits a floor from that recovery instead of drawing
    // the crashed density for the rest of the session.
    governor.beginMotion("explicit");
    const stats = governor.stats();
    expect(stats.regime).toBe("interaction");
    expect(stats.trackBudget).toBe(2_000_000);
    expect(stats.lastAdjustment).toMatchObject({ reason: "seeded" });
  });

  it("adapts on a host clock that is not the wall clock", () => {
    // `HostFrameMetrics.now` may be any epoch. A host that stamps frames with
    // performance.now() — the clock it already measures frame time with —
    // starts near 0 while the wall clock reads 1.7e12, so a governor that
    // stamped its own decisions with Date.now() would compare the two and
    // leave both tracks in a cooldown that never elapses.
    vi.setSystemTime(1_700_000_000_000);
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      interactionSettleMs: 750,
    });
    governor.register(memberOptions(vi.fn())).update({
      projectedImportance: 1,
    });

    let hostNow = 0;
    const hostFrames = (count: number): void => {
      for (let index = 0; index < count; index += 1) {
        governor.recordHostFrame({ hostFrameMs: 2, now: hostNow });
        hostNow += 16;
      }
    };

    const gesture = governor.beginMotion("explicit");
    governor.recordCameraChange();
    hostFrames(200);
    expect(governor.stats().regime).toBe("interaction");
    // Wedged in a foreign-epoch cooldown, this budget never moves at all.
    expect(governor.stats().trackBudget).toBeGreaterThan(1_000_000);

    gesture.release();
    vi.advanceTimersByTime(750);
    const seeded = governor.stats().trackBudget;
    hostFrames(400);
    const stats = governor.stats();
    expect(stats.regime).toBe("stationary");
    expect(stats.trackBudget).toBeGreaterThan(seeded);
    governor.dispose();
  });

  it("keeps adapting when the host stops stamping its frames", () => {
    // A host that reports `now` on some frames and not others — a metric
    // collected on a sampling interval, a frame assembled by a path that lost
    // it — hands the governor its epoch once and then goes quiet. Holding the
    // last stamp as the current time freezes the clock there: the emergency
    // cooldown armed by the slow frame below can never elapse, so no later
    // frame is ever measured and the budget stays halved for the session.
    vi.setSystemTime(1_700_000_000_000);
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      interactionSettleMs: 750,
    });
    governor.register(memberOptions(vi.fn())).update({
      projectedImportance: 1,
    });

    // The only stamped frames, slow enough in a row to force an emergency cut
    // while the camera moves: that is what arms the cooldown.
    const gesture = governor.beginMotion("explicit");
    governor.recordCameraChange();
    governor.recordHostFrame({ hostFrameMs: 400, now: 0 });
    governor.recordHostFrame({ hostFrameMs: 400, now: 16 });
    const cut = governor.stats().trackBudget;
    expect(cut).toBeLessThan(1_000_000);

    // Every frame from here carries no stamp at all, and each takes 16 ms of
    // real time, so the cooldown is long past by the end of the first burst.
    const unstampedFrames = (count: number): void => {
      for (let index = 0; index < count; index += 1) {
        governor.recordHostFrame({ hostFrameMs: 2 });
        vi.advanceTimersByTime(16);
      }
    };

    unstampedFrames(400);
    expect(governor.stats().trackBudget).toBeGreaterThan(cut);

    gesture.release();
    vi.advanceTimersByTime(750);
    const seeded = governor.stats().trackBudget;
    unstampedFrames(600);
    const stats = governor.stats();
    expect(stats.regime).toBe("stationary");
    expect(stats.trackBudget).toBeGreaterThan(seeded);
    governor.dispose();
  });

  it("raises quality gradually while stationary frames are fast", () => {
    const governor = createViewGovernor(FAST_ADAPT);
    const setBudget = vi.fn();
    governor.register(memberOptions(setBudget));
    moveAndSettle(governor, 100);

    frames(governor, 1, 8);
    expect(governor.stats().aggregateBudget).toBe(1_250_000);
    frames(governor, 1, 8);
    expect(governor.stats().aggregateBudget).toBe(1_562_500);
    // Bounded steps, not a jump to whatever the estimate implies.
    expect(governor.stats().lastAdjustment).toMatchObject({
      direction: "increase",
      reason: "below-target",
    });
  });

  it("lowers quality while stationary frames are slow", () => {
    const governor = createViewGovernor(FAST_ADAPT);
    governor.register(memberOptions(vi.fn()));
    moveAndSettle(governor, 100);

    // 60ms is past the 39.6ms stationary dead-band but is not a gesture, so it
    // goes through the sampled controller rather than an emergency cut, and
    // gives back only what the 33ms target asks for (33/60).
    frames(governor, 60, 8);
    expect(governor.stats().aggregateBudget).toBe(550_000);
    expect(governor.stats().lastAdjustment).toMatchObject({
      direction: "decrease",
      reason: "above-target",
    });
  });

  it("converges inside hysteresis and then stops asking for frames", () => {
    // A device where cost is ~ budget: 33ms at 2M points.
    const governor = createViewGovernor(FAST_ADAPT);
    governor.register(memberOptions(vi.fn()));
    moveAndSettle(governor, 100);
    const frameMsFor = (points: number): number => points / 60_606;

    expect(governor.needsFrame()).toBe(true);
    for (let round = 0; round < 20; round += 1) {
      for (let index = 0; index < 8; index += 1) {
        governor.recordHostFrame({
          hostFrameMs: frameMsFor(governor.stats().aggregateBudget),
          now: tick(),
        });
      }
    }
    const settled = governor.stats().aggregateBudget;
    // Dead-band [26.4, 39.6] ms ⇒ [1.6M, 2.4M] points on this device.
    expect(settled).toBeGreaterThanOrEqual(1_600_000);
    expect(settled).toBeLessThanOrEqual(2_400_000);
    expect(governor.stats().lastAdjustment).toMatchObject({
      direction: "none",
      reason: "within-hysteresis",
    });
    // Converged with no outstanding work: another frame would draw the same
    // pixels, so the governor stops asking the host for one.
    expect(governor.needsFrame()).toBe(false);

    frames(governor, frameMsFor(settled), 16);
    expect(governor.stats().aggregateBudget).toBe(settled);
  });

  it("does not ask for unchanged frames solely because work is pending", () => {
    const governor = createViewGovernor(FAST_ADAPT);
    const member = governor.register(memberOptions(vi.fn()));
    moveAndSettle(governor, 100);
    frames(governor, 33, 8); // straight into the dead-band
    expect(governor.needsFrame()).toBe(false);

    member.update({ physicalTileOperations: 2 });
    expect(governor.needsFrame()).toBe(false);
    member.update({
      physicalTileOperations: 0,
      physicalHierarchyOperations: 1,
    });
    expect(governor.needsFrame()).toBe(false);
    member.update({ physicalHierarchyOperations: 0 });
    expect(governor.needsFrame()).toBe(false);
  });

  it("resumes an unconverged track when required work drains", () => {
    const governor = createViewGovernor(FAST_ADAPT);
    const member = governor.register(memberOptions(vi.fn()));
    moveAndSettle(governor, 100);

    expect(governor.needsFrame()).toBe(true);
    member.update({ workPending: true });
    expect(governor.needsFrame()).toBe(false);

    governor.recordHostFrame({ hostFrameMs: 1, now: tick() });
    expect(governor.stats().capacitySamples).toMatchObject({
      eligible: 0,
      rejected: 1,
    });
    expect(governor.needsFrame()).toBe(false);

    member.update({ workPending: false });
    expect(governor.needsFrame()).toBe(true);
  });

  it("returns to the moving track the moment motion resumes", () => {
    const governor = createViewGovernor(FAST_ADAPT);
    const setBudget = vi.fn();
    const setDensity = vi.fn();
    governor.register(memberOptions(setBudget, setDensity));
    moveAndSettle(governor, 100);
    frames(governor, 1, 16); // stationary refines up to 1.5625M
    expect(governor.stats().aggregateBudget).toBe(1_562_500);

    governor.beginMotion("explicit");
    const stats = governor.stats();
    expect(stats.regime).toBe("interaction");
    expect(stats.targetFrameTimeMs).toBe(16);
    // The moving track kept its own learned draw budget, but the refined
    // stationary selection remains resident and is thinned in place.
    expect(stats.aggregateBudget).toBe(1_000_000);
    expect(stats.selectionBudget).toBe(1_562_500);
    expect(setBudget).toHaveBeenLastCalledWith(1_562_500);
    expect(setDensity).toHaveBeenLastCalledWith(0.64);
    expect(governor.needsFrame()).toBe(true);
  });

  it("decides each regime only from frames measured in that regime", () => {
    const governor = createViewGovernor(FAST_ADAPT);
    governor.register(memberOptions(vi.fn()));

    const gesture = governor.beginMotion("explicit");
    governor.recordCameraChange();
    // 30ms is slow for the 16ms moving target and squarely inside the settled
    // dead-band: if these frames leaked, the stationary track would hold.
    frames(governor, 30, 7);
    expect(governor.stats().samples).toBe(7);

    gesture.release();
    vi.advanceTimersByTime(100);
    expect(governor.stats().samples).toBe(0);

    frames(governor, 1, 8);
    expect(governor.stats().aggregateBudget).toBe(1_250_000);

    // And back: a fresh gesture starts measuring from zero as well.
    governor.beginMotion("explicit");
    expect(governor.stats().samples).toBe(0);
  });
});

describe("createViewGovernor settle window", () => {
  it("does not let cheap settle frames grow the moving budget", () => {
    const governor = createViewGovernor(FAST_ADAPT);
    governor.register(memberOptions(vi.fn()));

    const gesture = governor.beginMotion("explicit");
    governor.recordCameraChange();
    // 16 ms is the moving target, squarely inside the dead-band: the gesture
    // itself asks for no change, so anything that moves the budget below came
    // from the settle window.
    frames(governor, 16, 8);
    const sustained = governor.stats().trackBudget;
    expect(sustained).toBe(1_000_000);

    gesture.release();
    // The camera has stopped but the debounce still reports the interaction
    // regime. These frames are cheap precisely because nothing is moving.
    frames(governor, 1, 40);
    expect(governor.stats().regime).toBe("interaction");
    expect(governor.stats().motion).toMatchObject({
      source: null,
      settling: true,
    });
    expect(governor.stats().trackBudget).toBe(sustained);
  });

  it("keeps the moving budget stable across gesture and hover cycles", () => {
    const governor = createViewGovernor(FAST_ADAPT);
    governor.register(memberOptions(vi.fn()));

    // What each gesture starts from. Cheap hover frames counted as moving
    // frames would ratchet this up every cycle, and the user would feel it as
    // a gesture that opens too heavy and then halves.
    const opening: number[] = [];
    for (let cycle = 0; cycle < 5; cycle += 1) {
      const gesture = governor.beginMotion("explicit");
      governor.recordCameraChange();
      opening.push(governor.stats().trackBudget);
      frames(governor, 16, 8); // sustained exactly on the moving target
      gesture.release();
      frames(governor, 1, 45); // hover, inside the settle window
      vi.advanceTimersByTime(100);
      frames(governor, 33, 8); // reading the scene, on the settled target
    }

    expect(opening[0]).toBe(1_000_000);
    expect(new Set(opening).size).toBe(1);
  });

  it("resumes measuring the settled track once the window closes", () => {
    const governor = createViewGovernor(FAST_ADAPT);
    governor.register(memberOptions(vi.fn()));

    const motion = governor.beginMotion("explicit");
    governor.recordCameraChange();
    motion.release();
    frames(governor, 1, 40); // discarded: still inside the settle window
    vi.advanceTimersByTime(100);
    expect(governor.stats().regime).toBe("stationary");
    expect(governor.stats().samples).toBe(0);

    // Discarding the window's frames suspends measurement, it does not end it.
    frames(governor, 1, 8);
    expect(governor.stats().aggregateBudget).toBe(1_250_000);
    expect(governor.stats().lastAdjustment).toMatchObject({
      direction: "increase",
      reason: "below-target",
    });
  });
});

describe("createViewGovernor emergency response", () => {
  it("cuts and restores draw density without changing the selected budget", () => {
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      interactionSettleMs: 100,
    });
    const setBudget = vi.fn();
    const setDensity = vi.fn();
    governor.register(memberOptions(setBudget, setDensity));
    setBudget.mockClear();
    setDensity.mockClear();

    const motion = governor.beginMotion("explicit");
    governor.recordCameraChange();
    governor.recordHostFrame({ hostFrameMs: 80, now: tick() });
    governor.recordHostFrame({ hostFrameMs: 80, now: tick() });
    expect(setBudget).not.toHaveBeenCalled();
    expect(setDensity).toHaveBeenLastCalledWith(0.5);
    expect(governor.stats()).toMatchObject({
      aggregateBudget: 500_000,
      selectionBudget: 1_000_000,
    });

    motion.release();
    vi.advanceTimersByTime(99);
    expect(setDensity).toHaveBeenLastCalledWith(0.5);
    vi.advanceTimersByTime(1);
    expect(setBudget).not.toHaveBeenCalled();
    expect(setDensity).toHaveBeenLastCalledWith(1);
  });

  it("reduces on the second consecutive severely missed frame", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const setBudget = vi.fn();
    const setDensity = vi.fn();
    governor.register(memberOptions(setBudget, setDensity));
    governor.beginMotion("explicit");
    // Displayed cadence quantizes to whole vsyncs, so one missed 60 Hz frame
    // reads as ~33 ms — past 2× the moving target. An isolated miss is a
    // hitch (a tile upload, a GC pause), not a sustained cadence: it must be
    // measured, never answered with a halving.
    governor.recordHostFrame({ hostFrameMs: 80, now: tick() });
    expect(setDensity).not.toHaveBeenCalledWith(0.5);
    expect(governor.stats().samples).toBe(1);
    governor.recordHostFrame({ hostFrameMs: 80, now: tick() });
    expect(setBudget).toHaveBeenLastCalledWith(1_000_000);
    expect(setDensity).toHaveBeenLastCalledWith(0.5);
  });

  it("does not cut when a healthy frame interrupts the missed ones", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const setBudget = vi.fn();
    const setDensity = vi.fn();
    governor.register(memberOptions(setBudget, setDensity));
    governor.beginMotion("explicit");
    governor.recordHostFrame({ hostFrameMs: 80, now: tick() });
    governor.recordHostFrame({ hostFrameMs: 10, now: tick() });
    governor.recordHostFrame({ hostFrameMs: 80, now: tick() });
    expect(setDensity).not.toHaveBeenCalledWith(0.5);
    // The misses were measured, not discarded: the damped controller answers
    // a scene that hitches every other frame.
    expect(governor.stats().samples).toBe(3);
  });

  it("does not carry a missed frame across gesture bursts", () => {
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      interactionSettleMs: 100,
    });
    const setBudget = vi.fn();
    const setDensity = vi.fn();
    governor.register(memberOptions(setBudget, setDensity));

    const first = governor.beginMotion("explicit");
    governor.recordCameraChange();
    governor.recordHostFrame({ hostFrameMs: 80, now: tick() });
    first.release();
    vi.advanceTimersByTime(100);

    // A fresh burst from rest starts with a clean streak: its first missed
    // frame is this gesture's isolated hitch, not the second half of a pair
    // split across two gestures.
    governor.beginMotion("explicit");
    governor.recordHostFrame({ hostFrameMs: 80, now: tick() });
    expect(setDensity).not.toHaveBeenCalledWith(0.5);
  });

  it("halves the budget on sustained severe input delay while moving", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const setBudget = vi.fn();
    const setDensity = vi.fn();
    governor.register(memberOptions(setBudget, setDensity));
    governor.beginMotion("explicit");
    // 5ms host frame is well under target*2, so only the input-delay signal
    // can trigger the cut — this isolates the severe-input branch.
    governor.recordHostFrame({ hostFrameMs: 5, inputDelayMs: 80, now: tick() });
    governor.recordHostFrame({ hostFrameMs: 5, inputDelayMs: 80, now: tick() });
    expect(setBudget).toHaveBeenLastCalledWith(1_000_000);
    expect(setDensity).toHaveBeenLastCalledWith(0.5);
  });

  it("halves the budget on a sustained severe long task while moving", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const setBudget = vi.fn();
    const setDensity = vi.fn();
    governor.register(memberOptions(setBudget, setDensity));
    governor.beginMotion("inferred");
    governor.recordHostFrame({ hostFrameMs: 5, longTaskMs: 120, now: tick() });
    governor.recordHostFrame({ hostFrameMs: 5, longTaskMs: 120, now: tick() });
    expect(setBudget).toHaveBeenLastCalledWith(1_000_000);
    expect(setDensity).toHaveBeenLastCalledWith(0.5);
  });

  it("feeds a severely slow stationary frame through the damped path", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const setBudget = vi.fn();
    governor.register(memberOptions(setBudget));

    governor.recordHostFrame({ hostFrameMs: 90, longTaskMs: 120, now: tick() });
    expect(governor.stats().aggregateBudget).toBe(1_000_000);

    frames(governor, 90, 7);
    expect(governor.stats().aggregateBudget).toBe(500_000);
  });

  it("holds an emergency cut through its recovery cooldown", () => {
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      cooldownMs: 400,
      minSamples: 1,
    });
    const setBudget = vi.fn();
    governor.register(memberOptions(setBudget));
    governor.beginMotion("explicit");

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
      cooldownMs: 0,
    });
    governor.register(memberOptions(vi.fn()));

    for (let index = 0; index < 80; index += 1) {
      governor.recordHostFrame({
        hostFrameMs: 5,
        longTaskMs: index % 10 === 0 ? 120 : undefined,
        now: tick(),
      });
      expect(governor.stats().aggregateBudget).toBeGreaterThanOrEqual(
        1_000_000,
      );
    }
  });

  it("does not cut the budget on a fast frame with mild input delay", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const setBudget = vi.fn();
    const setDensity = vi.fn();
    governor.register(memberOptions(setBudget, setDensity));
    // The emergency cut only runs in the moving regime, so hold a motion
    // reference: without one the mild-delay assertion passes whatever the
    // threshold says, because the branch under test is never reached.
    const motion = governor.beginMotion("explicit");
    governor.recordHostFrame({ hostFrameMs: 5, inputDelayMs: 20, now: tick() });
    expect(setDensity).not.toHaveBeenCalledWith(0.5);
    // The same frame with a sustained severe delay must cut — this is what
    // proves the mild case exercised a live branch rather than a skipped one.
    governor.recordHostFrame({ hostFrameMs: 5, inputDelayMs: 80, now: tick() });
    governor.recordHostFrame({ hostFrameMs: 5, inputDelayMs: 80, now: tick() });
    expect(setDensity).toHaveBeenCalledWith(0.5);
    motion.release();
  });

  it("keeps the moving window across bursts inside the settle window", () => {
    const governor = createViewGovernor(FAST_ADAPT);
    const setBudget = vi.fn();
    governor.register(memberOptions(setBudget));
    // Eight begin/release pairs with one fast frame each — the shape wheel
    // ticks and scrub steps arrive in. Restarting the moving track on every
    // burst would clear its window before `minSamples` frames could ever
    // accumulate, so the moving regime would never adapt at all.
    for (let burst = 0; burst < 8; burst += 1) {
      const motion = governor.beginMotion("explicit");
      governor.recordCameraChange();
      governor.recordHostFrame({ hostFrameMs: 5, now: tick() });
      motion.release();
      vi.advanceTimersByTime(50);
    }
    // The eighth burst's frame completes the window: 5 ms against the 16 ms
    // moving target grows by the +25% step cap.
    expect(setBudget).toHaveBeenLastCalledWith(1_250_000);
  });

  it("does not reselect for ceiling jitter smaller than the dead-band", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const setBudget = vi.fn();
    const member = governor.register(memberOptions(setBudget));
    member.update({ memoryCeilingPoints: 900_000 });
    expect(setBudget).toHaveBeenLastCalledWith(900_000);
    const applied = setBudget.mock.calls.length;
    // The reported ceiling is derived from measured bytes-per-point, which
    // drifts a fraction of a percent with every tile that lands or leaves.
    // Applying each drift would run a synchronous reselection per frame.
    for (let index = 0; index < 20; index += 1) {
      member.update({
        memoryCeilingPoints: 900_000 + (index % 2 === 0 ? 800 : -800),
      });
    }
    expect(setBudget.mock.calls.length).toBe(applied);
    // A move past the band still applies immediately.
    member.update({ memoryCeilingPoints: 700_000 });
    expect(setBudget).toHaveBeenLastCalledWith(700_000);
  });

  it("resumes stationary refinement when a ceiling rise returns headroom", () => {
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      minSamples: 1,
      cooldownMs: 0,
    });
    const setBudget = vi.fn();
    const member = governor.register(memberOptions(setBudget));
    member.update({ memoryCeilingPoints: 500_000 });
    // Fast frames walk the budget onto the ceiling; once pinned, the track
    // reports "clamped", which is convergence for as long as the bound stands.
    frames(governor, 5, 3);
    expect(governor.stats().needsFrame).toBe(false);
    // The bound moves — another cloud left the view and its share came back.
    // A governor that still reads "clamped" as converged would never ask for
    // the frame that lets refinement use the returned headroom.
    member.update({ memoryCeilingPoints: 2_000_000 });
    expect(governor.stats().needsFrame).toBe(true);
    frames(governor, 5, 2);
    expect(setBudget.mock.calls.at(-1)![0]).toBeGreaterThan(500_000);
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
    governor.register(memberOptions(setBudget));
    // 24ms of VTK paint is 48ms of frame at half the frame — past the 39.6ms
    // stationary dead-band even though the host frame itself was 10ms.
    governor.recordHostFrame({ hostFrameMs: 10, vtkFrameMs: 24, now: tick() });
    expect(setBudget).toHaveBeenLastCalledWith(750_000);
  });
});

describe("createViewGovernor capacity eligibility", () => {
  it("keeps a contaminated 500 ms frame out of lasting capacity", () => {
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      minSamples: 1,
      cooldownMs: 0,
    });
    governor.register(memberOptions(vi.fn()));

    governor.recordCapacitySample({
      frameMs: 500,
      regime: "stationary",
      eligible: false,
      now: tick(),
    });

    expect(governor.stats()).toMatchObject({
      trackBudget: 1_000_000,
      samples: 0,
      capacitySamples: { eligible: 0, rejected: 1, lastEligible: false },
    });
  });

  it("derives fallback rejection from required core work", () => {
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      minSamples: 1,
      cooldownMs: 0,
    });
    const member = governor.register(memberOptions(vi.fn()));
    member.update({ workPending: true });

    governor.recordHostFrame({ hostFrameMs: 500, now: tick() });

    expect(governor.stats()).toMatchObject({
      trackBudget: 1_000_000,
      samples: 0,
      capacitySamples: { eligible: 0, rejected: 1, lastEligible: false },
    });
  });

  it("still reduces capacity on a sustained clean GPU cost", () => {
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      minSamples: 2,
      cooldownMs: 0,
    });
    governor.register(memberOptions(vi.fn()));

    for (let sample = 0; sample < 2; sample += 1) {
      governor.recordCapacitySample({
        frameMs: 80,
        regime: "stationary",
        eligible: true,
        now: tick(),
      });
    }

    expect(governor.stats()).toMatchObject({
      trackBudget: 500_000,
      capacitySamples: { eligible: 2, rejected: 0, lastEligible: true },
    });
  });

  it("reports input, camera stability, and required work independently", () => {
    const governor = createViewGovernor({ interactionSettleMs: 100 });
    const member = governor.register(memberOptions(vi.fn()));
    const motion = governor.beginMotion("explicit");
    governor.recordCameraChange();
    member.update({ workPending: true });

    expect(governor.stats().activity).toEqual({
      inputActive: true,
      cameraStable: false,
      workPending: true,
      measurementEligible: false,
    });

    vi.advanceTimersByTime(100);
    member.update({ workPending: false });
    expect(governor.stats().activity).toEqual({
      inputActive: true,
      cameraStable: true,
      workPending: false,
      measurementEligible: true,
    });

    motion.release();
    expect(governor.stats().regime).toBe("stationary");
  });
});

describe("createViewGovernor ceilings", () => {
  it("caps the budget at a configured maximum below the memory ceiling", () => {
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      maxBudget: 800_000,
    });
    const setBudget = vi.fn();
    const member = governor.register(memberOptions(setBudget));
    member.update({ memoryCeilingPoints: 5_000_000 });

    const stats = governor.stats();
    expect(stats.trackBudget).toBe(800_000);
    expect(stats.aggregateBudget).toBe(800_000);
    expect(stats.activeConstraint).toBe("configured-maximum");
    expect(setBudget).toHaveBeenLastCalledWith(800_000);
  });

  it("keeps the memory ceiling authoritative under a higher maximum", () => {
    const governor = createViewGovernor({
      initialBudget: 2_000_000,
      maxBudget: 5_000_000,
    });
    const setBudget = vi.fn();
    const member = governor.register(memberOptions(setBudget));
    member.update({ memoryCeilingPoints: 900_000 });

    const stats = governor.stats();
    expect(stats.aggregateBudget).toBe(900_000);
    expect(stats.activeConstraint).toBe("memory");
    expect(setBudget).toHaveBeenLastCalledWith(900_000);
  });

  it("leaves memory as the only ceiling when no maximum is configured", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const setBudget = vi.fn();
    const member = governor.register(memberOptions(setBudget));
    expect(governor.stats().configuredMaxPoints).toBeNull();
    expect(governor.stats().memoryCeilingPoints).toBeNull();
    expect(governor.stats().aggregateBudget).toBe(1_000_000);

    member.update({ memoryCeilingPoints: 400_000 });
    expect(governor.stats().aggregateBudget).toBe(400_000);
    expect(governor.stats().activeConstraint).toBe("memory");
    expect(setBudget).toHaveBeenLastCalledWith(400_000);
  });

  it("stops growing the track once the memory ceiling binds", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const setBudget = vi.fn();
    const member = governor.register(memberOptions(setBudget));
    member.update({ memoryCeilingPoints: 1_200_000 });

    // Frames far under target: on a machine with headroom the loop would grow
    // by maxIncreaseStep for ever, because only the aggregate was clamped.
    for (let round = 0; round < 30; round += 1) {
      frames(governor, 1, 10);
      vi.advanceTimersByTime(500);
    }

    const stats = governor.stats();
    expect(stats.trackBudget).toBe(1_200_000);
    expect(stats.aggregateBudget).toBe(1_200_000);
    expect(stats.activeConstraint).toBe("memory");
    expect(setBudget).toHaveBeenLastCalledWith(1_200_000);
  });

  it("stops asking for frames when only memory holds the budget down", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const member = governor.register(memberOptions(vi.fn()));
    member.update({
      memoryCeilingPoints: 1_200_000,
      physicalTileOperations: 0,
      physicalHierarchyOperations: 0,
    });

    for (let round = 0; round < 30; round += 1) {
      frames(governor, 1, 10);
      vi.advanceTimersByTime(500);
    }

    // Pinned against memory is converged: the effective budget cannot move,
    // so a host that repaints while needsFrame() is true would never idle.
    expect(governor.stats().trackBudget).toBe(1_200_000);
    expect(governor.needsFrame()).toBe(false);
  });

  it("walks a budget grown past a newly reported memory ceiling back down", () => {
    const governor = createViewGovernor({ initialBudget: 4_000_000 });
    const setBudget = vi.fn();
    const member = governor.register(memberOptions(setBudget));
    member.update({ memoryCeilingPoints: 500_000 });

    // The member never sees more than memory allows, even before the loop
    // has adjusted anything.
    expect(setBudget).toHaveBeenLastCalledWith(500_000);

    for (let round = 0; round < 5; round += 1) {
      frames(governor, 1, 10);
      vi.advanceTimersByTime(500);
    }
    expect(governor.stats().trackBudget).toBe(500_000);
  });

  it("lets memory hold the budget under the adaptive floor", () => {
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      minBudget: 200_000,
    });
    const setBudget = vi.fn();
    const member = governor.register(memberOptions(setBudget));
    member.update({ memoryCeilingPoints: 50_000 });

    // The floor bounds what the loop may choose, not what memory permits:
    // drawing 200,000 points out of a 50,000-point budget is the failure the
    // memory ceiling exists to prevent.
    expect(governor.stats().aggregateBudget).toBe(50_000);
    expect(setBudget).toHaveBeenLastCalledWith(50_000);
  });

  it("applies the ceiling to the aggregate before splitting it", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const near = vi.fn();
    const far = vi.fn();
    // Each controller owns a byte share of one pool, so their point ceilings
    // add up to the view's.
    governor
      .register({ ...memberOptions(far), id: "far" })
      .update({ projectedImportance: 1, memoryCeilingPoints: 300_000 });
    governor
      .register({ ...memberOptions(near), id: "near" })
      .update({ projectedImportance: 3, memoryCeilingPoints: 300_000 });

    const stats = governor.stats();
    expect(stats.memoryCeilingPoints).toBe(600_000);
    expect(stats.aggregateBudget).toBe(600_000);
    expect(far).toHaveBeenLastCalledWith(150_000);
    expect(near).toHaveBeenLastCalledWith(450_000);
    // Importance can hand a member more than its own share of memory; its
    // controller clamps locally, and the diagnostics say so.
    const nearStats = stats.members.find((member) => member.id === "near")!;
    expect(nearStats.allocatedShare).toBe(450_000);
    expect(nearStats.effectiveBudget).toBe(300_000);
    expect(nearStats.activeConstraint).toBe("memory");
    // The far cloud is inside its own ceiling, so what bounds it is the view's
    // memory ceiling, not its own.
    const farStats = stats.members.find((member) => member.id === "far")!;
    expect(farStats.effectiveBudget).toBe(150_000);
    expect(farStats.activeConstraint).toBe("memory");
  });

  it("stops asking for frames when nothing reports a ceiling at all", () => {
    // Every member report is optional: a member may only ever report its
    // importance. With no configured maximum either, a scene where more
    // points cost no frame time — one already fully resident, or smaller than
    // its budget — must still converge instead of integrating for ever and
    // repainting at full rate against a budget past exact integers.
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const setPointBudget = vi.fn();
    governor
      .register(memberOptions(setPointBudget))
      .update({ projectedImportance: 1 });
    moveAndSettle(governor, 750);

    let rounds = 0;
    while (governor.needsFrame() && rounds < 500) {
      rounds += 1;
      frames(governor, 2, 10);
      vi.advanceTimersByTime(450);
    }

    expect(governor.needsFrame()).toBe(false);
    const stats = governor.stats();
    expect(stats.lastAdjustment?.reason).toBe("clamped");
    expect(Number.isSafeInteger(stats.trackBudget)).toBe(true);
    expect(Number.isSafeInteger(setPointBudget.mock.calls.at(-1)![0])).toBe(
      true,
    );
    governor.dispose();
  });
});

describe("createViewGovernor diagnostics", () => {
  it("reports the constraint the arithmetic actually produced", () => {
    const cases = [
      { maxBudget: undefined, memory: 5_000_000, expect: "adaptive" },
      { maxBudget: 800_000, memory: 5_000_000, expect: "configured-maximum" },
      { maxBudget: 5_000_000, memory: 400_000, expect: "memory" },
    ] as const;
    for (const scenario of cases) {
      const governor = createViewGovernor({
        initialBudget: 1_000_000,
        maxBudget: scenario.maxBudget,
      });
      const member = governor.register(memberOptions(vi.fn()));
      member.update({ memoryCeilingPoints: scenario.memory });
      const stats = governor.stats();
      expect(stats.aggregateBudget).toBe(
        Math.min(
          stats.trackBudget,
          stats.configuredMaxPoints ?? Number.POSITIVE_INFINITY,
          stats.memoryCeilingPoints ?? Number.POSITIVE_INFINITY,
        ),
      );
      expect(stats.activeConstraint).toBe(scenario.expect);
    }

    const idle = createViewGovernor({ initialBudget: 1_000_000 });
    const setBudget = vi.fn();
    const member = idle.register(memberOptions(setBudget));
    member.update({ active: false, memoryCeilingPoints: 400_000 });
    const stats = idle.stats();
    expect(stats.activeConstraint).toBe("inactive");
    expect(stats.memoryCeilingPoints).toBeNull();
    expect(stats.members[0]!.activeConstraint).toBe("inactive");
    expect(setBudget).toHaveBeenLastCalledWith(0);
  });

  it("explains a drawn point count while rejecting active-work samples", () => {
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      maxBudget: 4_000_000,
      minSamples: 8,
      cooldownMs: 0,
    });
    const member = governor.register({
      setPointBudget: vi.fn(),
      setDensityFraction: vi.fn(),
      id: "cloud-1",
    });
    member.update({
      projectedImportance: 2.5,
      memoryCeilingPoints: 9_000_000,
      physicalTileOperations: 3,
      physicalHierarchyOperations: 1,
    });
    frames(governor, 1, 8);

    const stats = governor.stats();
    expect(stats).toMatchObject({
      regime: "stationary",
      targetFrameTimeMs: 33,
      estimateMs: null,
      samples: 0,
      trackBudget: 1_000_000,
      configuredMaxPoints: 4_000_000,
      memoryCeilingPoints: 9_000_000,
      aggregateBudget: 1_000_000,
      selectionBudget: 1_000_000,
      activeConstraint: "adaptive",
      activeMembers: 1,
      physicalTileOperations: 3,
      physicalHierarchyOperations: 1,
      needsFrame: false,
      activity: { workPending: true, measurementEligible: false },
      capacitySamples: { eligible: 0, rejected: 8, lastEligible: false },
    });
    expect(stats.motion.source).toBeNull();
    expect(stats.lastAdjustment).toBeNull();
    expect(stats.members).toEqual([
      {
        id: "cloud-1",
        active: true,
        projectedImportance: 2.5,
        allocatedShare: 1_000_000,
        memoryCeilingPoints: 9_000_000,
        effectiveBudget: 1_000_000,
        selectionShare: 1_000_000,
        effectiveSelectionBudget: 1_000_000,
        densityFraction: 1,
        activeConstraint: "adaptive",
        physicalTileOperations: 3,
        physicalHierarchyOperations: 1,
      },
    ]);
  });
});

describe("createViewGovernor numeric configuration", () => {
  const NON_FINITE = [
    Number.NaN,
    Number.POSITIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
  ];

  for (const value of [...NON_FINITE, 0, 0.04, 1.01]) {
    it(`rejects vtkFrameFraction = ${value}`, () => {
      expect(() => createViewGovernor({ vtkFrameFraction: value })).toThrow(
        /^vtkFrameFraction must be/,
      );
    });
  }

  for (const value of [...NON_FINITE, -1]) {
    it(`rejects interactionSettleMs = ${value}`, () => {
      expect(() => createViewGovernor({ interactionSettleMs: value })).toThrow(
        /^interactionSettleMs must be/,
      );
    });
  }

  it("fails construction on any invalid budget option it forwards", () => {
    expect(() => createViewGovernor({ minBudget: 0 })).toThrow(/^minBudget/);
    expect(() =>
      createViewGovernor({ interactionTargetMs: Number.NaN }),
    ).toThrow(/^interactionTargetMs/);
    expect(() => createViewGovernor({ maxBudget: 100 })).toThrow(/^maxBudget/);
  });

  it("ignores unusable member reports instead of poisoning the split", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const setBudget = vi.fn();
    const member = governor.register(memberOptions(setBudget));
    member.update({ projectedImportance: 4, memoryCeilingPoints: 800_000 });
    expect(governor.stats().aggregateBudget).toBe(800_000);

    for (const bad of [...NON_FINITE, -1]) {
      member.update({
        projectedImportance: bad,
        memoryCeilingPoints: bad,
        physicalTileOperations: bad,
        physicalHierarchyOperations: bad,
      });
    }
    const stats = governor.stats();
    expect(stats.members[0]).toMatchObject({
      projectedImportance: 4,
      memoryCeilingPoints: 800_000,
      physicalTileOperations: 0,
      physicalHierarchyOperations: 0,
    });
    expect(stats.aggregateBudget).toBe(800_000);
    expect(setBudget).toHaveBeenLastCalledWith(800_000);
  });

  it("ignores unusable frame metrics", () => {
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      minSamples: 1,
      cooldownMs: 0,
    });
    governor.register(memberOptions(vi.fn()));
    for (const bad of [...NON_FINITE, -1]) {
      governor.recordHostFrame({ hostFrameMs: bad, now: tick() });
    }
    expect(governor.stats().samples).toBe(0);
    expect(governor.stats().aggregateBudget).toBe(1_000_000);
  });

  it("stops distributing and requesting frames once disposed", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const setBudget = vi.fn();
    governor.register(memberOptions(setBudget));
    setBudget.mockClear();
    governor.dispose();
    governor.beginMotion("explicit").release();
    governor.recordHostFrame({ hostFrameMs: 1, now: tick() });
    expect(setBudget).not.toHaveBeenCalled();
    expect(governor.needsFrame()).toBe(false);
  });
});
