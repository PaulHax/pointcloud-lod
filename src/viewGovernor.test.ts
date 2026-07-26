import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createViewGovernor, type ViewGovernor } from "./viewGovernor";

/**
 * One monotonic clock for both the governor's own `Date.now()` (settle timers)
 * and the timestamps frames are reported with: a frame stamped before the last
 * adjustment would read as a cooldown that never elapses.
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
  governor.beginMotion("explicit").release();
  vi.advanceTimersByTime(settleMs);
};

describe("createViewGovernor", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    clock = 0;
  });
  afterEach(() => vi.useRealTimers());

  it("distributes one aggregate budget by projected importance", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const a = vi.fn();
    const b = vi.fn();
    governor.register({ setPointBudget: a }).update({ projectedImportance: 1 });
    governor.register({ setPointBudget: b }).update({ projectedImportance: 3 });
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
      .register({ setPointBudget: showing })
      .update({ projectedImportance: 0.04 });
    governor
      .register({ setPointBudget: culled })
      .update({ projectedImportance: 0 });

    const showingBudget: number = showing.mock.calls.at(-1)![0];
    const culledBudget: number = culled.mock.calls.at(-1)![0];
    expect(showingBudget).toBeGreaterThan(culledBudget);
    expect(showingBudget + culledBudget).toBeLessThanOrEqual(1_000_000);
  });

  it("splits evenly while no active member has reported anything", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const a = vi.fn();
    const b = vi.fn();
    governor.register({ setPointBudget: a });
    governor.register({ setPointBudget: b });
    expect(a).toHaveBeenLastCalledWith(500_000);
    expect(b).toHaveBeenLastCalledWith(500_000);
  });

  it("never starves an active view when another's importance dominates", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const quiet = vi.fn();
    const dominant = vi.fn();
    governor
      .register({ setPointBudget: quiet })
      .update({ projectedImportance: 1 });
    governor
      .register({ setPointBudget: dominant })
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
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    clock = 0;
  });
  afterEach(() => vi.useRealTimers());

  it("holds the moving regime until the last source releases", () => {
    const governor = createViewGovernor({ interactionSettleMs: 100 });
    governor.register({ setPointBudget: vi.fn() });

    const gesture = governor.beginMotion("explicit");
    const playback = governor.beginMotion("inferred");
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
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    clock = 0;
  });
  afterEach(() => vi.useRealTimers());

  it("starts stationary at the last moving budget without a jump", () => {
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      interactionSettleMs: 100,
      minSamples: 1,
      cooldownMs: 0,
      maxIncreaseStep: 1,
    });
    const setBudget = vi.fn();
    governor.register({ setPointBudget: setBudget });

    const gesture = governor.beginMotion("explicit");
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

  it("raises quality gradually while stationary frames are fast", () => {
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      interactionSettleMs: 100,
      minSamples: 8,
      cooldownMs: 0,
    });
    const setBudget = vi.fn();
    governor.register({ setPointBudget: setBudget });
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
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      interactionSettleMs: 100,
      minSamples: 8,
      cooldownMs: 0,
    });
    governor.register({ setPointBudget: vi.fn() });
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
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      interactionSettleMs: 100,
      minSamples: 8,
      cooldownMs: 0,
    });
    governor.register({ setPointBudget: vi.fn() });
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

  it("keeps asking for frames while tiles and pages are still landing", () => {
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      interactionSettleMs: 100,
      minSamples: 8,
      cooldownMs: 0,
    });
    const member = governor.register({ setPointBudget: vi.fn() });
    moveAndSettle(governor, 100);
    frames(governor, 33, 8); // straight into the dead-band
    expect(governor.needsFrame()).toBe(false);

    member.update({ physicalTileOperations: 2 });
    expect(governor.needsFrame()).toBe(true);
    member.update({ physicalTileOperations: 0, physicalHierarchyOperations: 1 });
    expect(governor.needsFrame()).toBe(true);
    member.update({ physicalHierarchyOperations: 0 });
    expect(governor.needsFrame()).toBe(false);
  });

  it("returns to the moving track the moment motion resumes", () => {
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      interactionSettleMs: 100,
      minSamples: 8,
      cooldownMs: 0,
    });
    const setBudget = vi.fn();
    governor.register({ setPointBudget: setBudget });
    moveAndSettle(governor, 100);
    frames(governor, 1, 16); // stationary refines up to 1.5625M
    expect(governor.stats().aggregateBudget).toBe(1_562_500);

    governor.beginMotion("explicit");
    const stats = governor.stats();
    expect(stats.regime).toBe("interaction");
    expect(stats.targetFrameTimeMs).toBe(16);
    // The moving track kept its own learned budget; the refined stationary one
    // does not follow the camera into motion.
    expect(stats.aggregateBudget).toBe(1_000_000);
    expect(setBudget).toHaveBeenLastCalledWith(1_000_000);
    expect(governor.needsFrame()).toBe(true);
  });

  it("decides each regime only from frames measured in that regime", () => {
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      interactionSettleMs: 100,
      minSamples: 8,
      cooldownMs: 0,
    });
    governor.register({ setPointBudget: vi.fn() });

    const gesture = governor.beginMotion("explicit");
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

describe("createViewGovernor emergency response", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    clock = 0;
  });
  afterEach(() => vi.useRealTimers());

  it("reduces immediately on a severely missed frame while moving", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const setBudget = vi.fn();
    governor.register({ setPointBudget: setBudget });
    governor.beginMotion("explicit");
    governor.recordHostFrame({ hostFrameMs: 80, now: tick() });
    expect(setBudget).toHaveBeenLastCalledWith(500_000);
  });

  it("halves the budget on severe input delay while moving", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const setBudget = vi.fn();
    governor.register({ setPointBudget: setBudget });
    governor.beginMotion("explicit");
    // 5ms host frame is well under target*2, so only the input-delay signal
    // can trigger the cut — this isolates the severe-input branch.
    governor.recordHostFrame({ hostFrameMs: 5, inputDelayMs: 80, now: tick() });
    expect(setBudget).toHaveBeenLastCalledWith(500_000);
  });

  it("halves the budget on a severe long task while moving", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const setBudget = vi.fn();
    governor.register({ setPointBudget: setBudget });
    governor.beginMotion("inferred");
    governor.recordHostFrame({ hostFrameMs: 5, longTaskMs: 120, now: tick() });
    expect(setBudget).toHaveBeenLastCalledWith(500_000);
  });

  it("feeds a severely slow stationary frame through the damped path", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const setBudget = vi.fn();
    governor.register({ setPointBudget: setBudget });

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
    governor.register({ setPointBudget: setBudget });
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
    governor.register({ setPointBudget: vi.fn() });

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
    governor.register({ setPointBudget: setBudget });
    governor.recordHostFrame({ hostFrameMs: 5, inputDelayMs: 20, now: tick() });
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
    // 24ms of VTK paint is 48ms of frame at half the frame — past the 39.6ms
    // stationary dead-band even though the host frame itself was 10ms.
    governor.recordHostFrame({ hostFrameMs: 10, vtkFrameMs: 24, now: tick() });
    expect(setBudget).toHaveBeenLastCalledWith(750_000);
  });
});

describe("createViewGovernor ceilings", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    clock = 0;
  });
  afterEach(() => vi.useRealTimers());

  it("caps the budget at a configured maximum below the memory ceiling", () => {
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      maxBudget: 800_000,
    });
    const setBudget = vi.fn();
    const member = governor.register({ setPointBudget: setBudget });
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
    const member = governor.register({ setPointBudget: setBudget });
    member.update({ memoryCeilingPoints: 900_000 });

    const stats = governor.stats();
    expect(stats.aggregateBudget).toBe(900_000);
    expect(stats.activeConstraint).toBe("memory");
    expect(setBudget).toHaveBeenLastCalledWith(900_000);
  });

  it("leaves memory as the only ceiling when no maximum is configured", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const setBudget = vi.fn();
    const member = governor.register({ setPointBudget: setBudget });
    expect(governor.stats().configuredMaxPoints).toBeNull();
    expect(governor.stats().memoryCeilingPoints).toBeNull();
    expect(governor.stats().aggregateBudget).toBe(1_000_000);

    member.update({ memoryCeilingPoints: 400_000 });
    expect(governor.stats().aggregateBudget).toBe(400_000);
    expect(governor.stats().activeConstraint).toBe("memory");
    expect(setBudget).toHaveBeenLastCalledWith(400_000);
  });

  it("applies the ceiling to the aggregate before splitting it", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const near = vi.fn();
    const far = vi.fn();
    // Each controller owns a byte share of one pool, so their point ceilings
    // add up to the view's.
    governor
      .register({ setPointBudget: far, id: "far" })
      .update({ projectedImportance: 1, memoryCeilingPoints: 300_000 });
    governor
      .register({ setPointBudget: near, id: "near" })
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
});

describe("createViewGovernor diagnostics", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    clock = 0;
  });
  afterEach(() => vi.useRealTimers());

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
      const member = governor.register({ setPointBudget: vi.fn() });
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
    const member = idle.register({ setPointBudget: setBudget });
    member.update({ active: false, memoryCeilingPoints: 400_000 });
    const stats = idle.stats();
    expect(stats.activeConstraint).toBe("inactive");
    expect(stats.memoryCeilingPoints).toBeNull();
    expect(stats.members[0]!.activeConstraint).toBe("inactive");
    expect(setBudget).toHaveBeenLastCalledWith(0);
  });

  it("explains a drawn point count without exposing internal state", () => {
    const governor = createViewGovernor({
      initialBudget: 1_000_000,
      maxBudget: 4_000_000,
      minSamples: 8,
      cooldownMs: 0,
    });
    const member = governor.register({ setPointBudget: vi.fn(), id: "cloud-1" });
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
      trackBudget: 1_250_000,
      configuredMaxPoints: 4_000_000,
      memoryCeilingPoints: 9_000_000,
      aggregateBudget: 1_250_000,
      activeConstraint: "adaptive",
      activeMembers: 1,
      physicalTileOperations: 3,
      physicalHierarchyOperations: 1,
      needsFrame: true,
    });
    expect(stats.motion.source).toBeNull();
    expect(stats.lastAdjustment).toMatchObject({
      direction: "increase",
      reason: "below-target",
      fromBudget: 1_000_000,
      toBudget: 1_250_000,
      estimateMs: 1,
    });
    expect(stats.lastAdjustment!.atMs).toBe(clock);
    expect(stats.members).toEqual([
      {
        id: "cloud-1",
        active: true,
        projectedImportance: 2.5,
        allocatedShare: 1_250_000,
        memoryCeilingPoints: 9_000_000,
        effectiveBudget: 1_250_000,
        activeConstraint: "adaptive",
        physicalTileOperations: 3,
        physicalHierarchyOperations: 1,
      },
    ]);
  });
});

describe("createViewGovernor numeric configuration", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    clock = 0;
  });
  afterEach(() => vi.useRealTimers());

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
    const member = governor.register({ setPointBudget: setBudget });
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
    governor.register({ setPointBudget: vi.fn() });
    for (const bad of [...NON_FINITE, -1]) {
      governor.recordHostFrame({ hostFrameMs: bad, now: tick() });
    }
    expect(governor.stats().samples).toBe(0);
    expect(governor.stats().aggregateBudget).toBe(1_000_000);
  });

  it("stops distributing and requesting frames once disposed", () => {
    const governor = createViewGovernor({ initialBudget: 1_000_000 });
    const setBudget = vi.fn();
    governor.register({ setPointBudget: setBudget });
    setBudget.mockClear();
    governor.dispose();
    governor.beginMotion("explicit").release();
    governor.recordHostFrame({ hostFrameMs: 1, now: tick() });
    expect(setBudget).not.toHaveBeenCalled();
    expect(governor.needsFrame()).toBe(false);
  });
});
