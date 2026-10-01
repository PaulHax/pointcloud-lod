import { afterEach, describe, expect, it, vi } from "vitest";

import { perspectiveView } from "../test/helpers";
import { createViewGovernor } from "./viewGovernor";

const VIEW = perspectiveView({ fovY: 1 });

afterEach(() => vi.useRealTimers());

describe("createViewGovernor", () => {
  it("reports a direct normalized fraction with no point-budget surface", () => {
    const governor = createViewGovernor();
    expect(governor.qualityFraction()).toBe(1);
    expect(governor.stats()).toMatchObject({
      regime: "stationary",
      viewQualityFraction: 1,
      targetFrameTimeMs: 33,
    });
    expect(Object.keys(governor.stats()).join(" ")).not.toMatch(
      /point|budget|member/i,
    );
  });

  it("learns stationary fraction directly from eligible host frames", () => {
    const governor = createViewGovernor({
      initialFraction: 0.8,
      minSamples: 2,
      cooldownMs: 0,
    });
    governor.recordHostFrame({ hostFrameMs: 66, now: 0 });
    governor.recordHostFrame({ hostFrameMs: 66, now: 1 });
    expect(governor.qualityFraction()).toBeCloseTo(0.4);
    expect(governor.stats().lastAdjustment).toMatchObject({
      reason: "above-target",
      fromFraction: 0.8,
      toFraction: 0.4,
    });
  });

  it("normalizes VTK and GPU spans against their host-frame share", () => {
    const governor = createViewGovernor({
      initialFraction: 0.8,
      minSamples: 1,
      cooldownMs: 0,
      vtkFrameFraction: 0.5,
    });
    governor.recordHostFrame({ hostFrameMs: 5, vtkFrameMs: 33, now: 0 });
    expect(governor.stats().lastAdjustment?.estimateMs).toBe(66);
    expect(governor.qualityFraction()).toBeCloseTo(0.4);
  });

  it("keeps full quality when presentation and rendering meet the target", () => {
    const governor = createViewGovernor();
    for (let index = 0; index < 12; index += 1) {
      governor.recordHostFrame({
        hostFrameMs: 33.4,
        vtkFrameMs: 30,
        gpuMs: 20,
        now: index * 33.4,
      });
    }
    expect(governor.qualityFraction()).toBe(1);
    expect(governor.stats().estimateMs).toBe(33.4);
  });

  it.each([
    { hostFrameMs: 50, vtkFrameMs: 5, gpuMs: 1 },
    { hostFrameMs: 16.7, vtkFrameMs: 50, gpuMs: 1 },
    { hostFrameMs: 16.7, vtkFrameMs: 5, gpuMs: 50 },
  ])("reduces quality for an actual slow measured span: %o", (metrics) => {
    const governor = createViewGovernor();
    for (let index = 0; index < 10; index += 1) {
      governor.recordHostFrame({ ...metrics, now: index * 50 });
    }
    // Two refreshes are on time for the stationary target; 50 ms is three.
    expect(governor.qualityFraction()).toBeCloseTo(2 / Math.sqrt(6), 1);
    expect(governor.stats().lastAdjustment).toMatchObject({
      reason: "above-target",
      estimateMs: 50,
    });
  });

  it("remeasures a completed workload without reusing the old capacity window", () => {
    const governor = createViewGovernor({ minSamples: 2, cooldownMs: 0 });
    governor.recordHostFrame({ hostFrameMs: 16.7, now: 0 });
    governor.recordHostFrame({ hostFrameMs: 16.7, now: 1 });
    expect(governor.needsFrame()).toBe(false);
    governor.setWorkState({
      workPending: true,
      physicalTileOperations: 0,
      physicalHierarchyOperations: 0,
    });
    governor.setWorkState({
      workPending: false,
      physicalTileOperations: 0,
      physicalHierarchyOperations: 0,
    });
    expect(governor.stats()).toMatchObject({
      viewQualityFraction: 1,
      samples: 0,
      needsFrame: true,
    });
    governor.recordHostFrame({ hostFrameMs: 66, now: 2 });
    governor.recordHostFrame({ hostFrameMs: 66, now: 3 });
    expect(governor.qualityFraction()).toBeCloseTo(2 / Math.sqrt(12), 1);
    governor.dispose();
  });

  it("rejects capacity samples while any work remains", () => {
    const governor = createViewGovernor({ minSamples: 1, cooldownMs: 0 });
    governor.setWorkState({
      workPending: true,
      physicalTileOperations: 0,
      physicalHierarchyOperations: 0,
    });
    governor.recordHostFrame({ hostFrameMs: 100, now: 0 });
    expect(governor.qualityFraction()).toBe(1);
    expect(governor.stats()).toMatchObject({
      activity: { workPending: true, measurementEligible: false },
      capacitySamples: { eligible: 0, rejected: 1, lastEligible: false },
      needsFrame: false,
    });
  });

  it("treats physical tile and hierarchy operations as pending capacity work", () => {
    const governor = createViewGovernor();
    governor.setWorkState({
      workPending: false,
      physicalTileOperations: 2.9,
      physicalHierarchyOperations: 3.2,
    });
    expect(governor.stats()).toMatchObject({
      activity: { workPending: true, measurementEligible: false },
      physicalTileOperations: 2,
      physicalHierarchyOperations: 3,
    });
  });

  it("counts nested and overlapping motion references", () => {
    const governor = createViewGovernor();
    const a = governor.beginMotion("explicit");
    const b = governor.beginMotion("explicit");
    const inferred = governor.beginMotion("inferred");
    expect(governor.stats()).toMatchObject({
      regime: "interaction",
      motion: {
        explicitReferences: 2,
        inferredReferences: 1,
        source: "both",
      },
    });
    b.release();
    b.release();
    inferred.release();
    expect(governor.stats().regime).toBe("interaction");
    a.release();
    expect(governor.stats().regime).toBe("stationary");
  });

  /** Frames one interval apart, all of that interval, from `start`. */
  const present = (
    governor: ReturnType<typeof createViewGovernor>,
    hostFrameMs: number,
    count: number,
    start: number,
  ): number => {
    let now = start;
    for (let index = 0; index < count; index += 1) {
      now += hostFrameMs;
      governor.recordHostFrame({ hostFrameMs, now });
    }
    return now;
  };

  it("cuts only when most of a short window has collapsed", () => {
    const governor = createViewGovernor({ minSamples: 30, cooldownMs: 0 });
    const motion = governor.beginMotion("explicit");
    // Two refreshes per frame is slow, but it is not a collapse.
    let now = present(governor, 33.3, 20, 0);
    expect(governor.qualityFraction()).toBe(1);
    // Two collapsed frames of the last five are still a minority.
    now = present(governor, 100, 2, now);
    expect(governor.qualityFraction()).toBe(1);
    present(governor, 100, 1, now);
    expect(governor.qualityFraction()).toBe(0.5);
    expect(governor.stats().lastAdjustment?.reason).toBe("emergency-cut");
    motion.release();
  });

  it("ignores frames drawn while a GPU wakes from idle", () => {
    const governor = createViewGovernor({ minSamples: 30, cooldownMs: 0 });
    let now = present(governor, 16.7, 10, 0);
    const motion = governor.beginMotion("explicit");
    now = present(governor, 100, 4, now + 2000);
    expect(governor.stats().activity.warmingUp).toBe(true);
    now = present(governor, 16.7, 30, now);
    expect(governor.stats().activity.warmingUp).toBe(false);
    expect(governor.qualityFraction()).toBe(1);
    expect(governor.stats().capacitySamples.rejected).toBeGreaterThan(0);
    motion.release();
  });

  it("takes back a cut that did not make frames faster", () => {
    const governor = createViewGovernor({ minSamples: 30, cooldownMs: 0 });
    governor.setWorkState({
      workPending: true,
      physicalTileOperations: 4,
      physicalHierarchyOperations: 0,
    });
    const motion = governor.beginMotion("explicit");
    let now = present(governor, 100, 5, 0);
    expect(governor.qualityFraction()).toBe(0.5);
    // The time goes somewhere quality does not reach.
    now = present(governor, 100, 7, now);
    expect(governor.qualityFraction()).toBe(1);
    expect(governor.stats().lastAdjustment?.reason).toBe("emergency-restore");
    expect(governor.stats().emergency.suspended).toBe(true);
    present(governor, 100, 20, now);
    expect(governor.qualityFraction()).toBe(1);
    motion.release();
  });

  it("gives a cut back after sustained calm while tiles stream", () => {
    const governor = createViewGovernor({ minSamples: 30, cooldownMs: 0 });
    // Tiles are streaming, which is what withholds eligibility: the sampling
    // loop that would otherwise raise quality back is shut off for as long as
    // this holds, while the emergency path is not.
    governor.setWorkState({
      workPending: true,
      physicalTileOperations: 4,
      physicalHierarchyOperations: 0,
    });
    const motion = governor.beginMotion("explicit");
    let now = present(governor, 100, 5, 0);
    expect(governor.qualityFraction()).toBe(0.5);
    // The cut worked; a calm second later it is handed back.
    now = present(governor, 16.7, 30, now);
    expect(governor.qualityFraction()).toBe(0.5);
    present(governor, 16.7, 60, now);
    expect(governor.qualityFraction()).toBe(1);
    expect(governor.stats().lastAdjustment).toMatchObject({
      reason: "emergency-restore",
      fromFraction: 0.5,
      toFraction: 1,
    });
    // Still no eligible sample anywhere: the recovery came from the frames
    // themselves, not from the loop that was shut off.
    expect(governor.stats().capacitySamples).toMatchObject({ eligible: 0 });
    motion.release();
  });

  it("preserves emergency relief across capacity invalidation", () => {
    const governor = createViewGovernor({ minSamples: 30, cooldownMs: 0 });
    governor.setWorkState({
      workPending: true,
      physicalTileOperations: 1,
      physicalHierarchyOperations: 0,
    });
    const motion = governor.beginMotion("explicit");
    const now = present(governor, 100, 5, 0);
    expect(governor.qualityFraction()).toBe(0.5);
    governor.invalidateCapacity();
    expect(governor.qualityFraction()).toBe(0.5);
    expect(governor.stats().samples).toBe(0);
    present(governor, 16.7, 90, now);
    expect(governor.qualityFraction()).toBe(1);
    motion.release();
    governor.dispose();
  });

  it("restores no more quality than the emergency cuts took away", () => {
    const governor = createViewGovernor({
      minSamples: 30,
      cooldownMs: 0,
      initialFraction: 0.4,
    });
    governor.setWorkState({
      workPending: true,
      physicalTileOperations: 1,
      physicalHierarchyOperations: 0,
    });
    const motion = governor.beginMotion("explicit");
    let now = present(governor, 100, 5, 0);
    expect(governor.qualityFraction()).toBeCloseTo(0.2);
    for (let frame = 0; frame < 300; frame += 1) {
      now = present(governor, 10, 1, now);
      // Back to where the cut started and never past it: quality above that
      // has to be earned by an eligible capacity sample.
      expect(governor.qualityFraction()).toBeLessThanOrEqual(0.4);
    }
    expect(governor.qualityFraction()).toBeCloseTo(0.4);
    motion.release();
  });

  it("continues a gesture begun inside the previous settle window", () => {
    vi.useFakeTimers();
    const governor = createViewGovernor({
      minSamples: 30,
      cooldownMs: 0,
      interactionSettleMs: 1000,
    });
    const first = governor.beginMotion("explicit");
    governor.recordCameraChange();
    present(governor, 200, 5, 0);
    expect(governor.qualityFraction()).toBe(0.5);

    first.release();
    // The settle timer is still pending, so the governor never left the
    // interaction regime, and the next drag carries on at the cut level
    // rather than reseeding into the collapse that caused it.
    expect(governor.stats().regime).toBe("interaction");
    const second = governor.beginMotion("explicit");
    expect(governor.qualityFraction()).toBe(0.5);
    expect(governor.stats().lastAdjustment?.reason).toBe("emergency-cut");
    second.release();
  });

  it("leaves an uncut interaction track alone when a gesture repeats", () => {
    vi.useFakeTimers();
    const governor = createViewGovernor({
      minSamples: 2,
      cooldownMs: 0,
      interactionSettleMs: 1000,
    });
    const sample = (now: number) =>
      governor.recordCapacitySample({
        frameMs: 66,
        regime: "interaction",
        eligible: true,
        now,
      });

    const first = governor.beginMotion("explicit");
    governor.recordCameraChange();
    sample(10);
    first.release();

    // Nothing cut the track, so a repeated nudge must not clear the window it
    // is still filling. Otherwise a user making many gestures each shorter than
    // `minSamples` could never accumulate the samples that argue for less
    // quality.
    const second = governor.beginMotion("explicit");
    sample(11);
    expect(governor.stats().lastAdjustment?.reason).toBe("above-target");
    second.release();
  });

  it("infers rendered-camera motion and schedules one stationary frame", () => {
    vi.useFakeTimers();
    const scheduleRender = vi.fn();
    const governor = createViewGovernor({
      motionDebounceMs: 10,
      interactionSettleMs: 10,
    });
    governor.noteRenderedCameras(new Map([["view", VIEW]]), scheduleRender);
    governor.noteRenderedCameras(
      new Map([["view", { ...VIEW, position: [1, 0, 0] }]]),
      scheduleRender,
    );
    expect(governor.stats()).toMatchObject({
      regime: "interaction",
      motion: { inferredReferences: 1 },
    });
    vi.advanceTimersByTime(10);
    expect(scheduleRender).toHaveBeenCalledOnce();
    expect(governor.stats().regime).toBe("stationary");
  });

  it.each(["position", "matrix"])(
    "infers motion when the host mutates its reused %s",
    (field) => {
      vi.useFakeTimers();
      const governor = createViewGovernor({ motionDebounceMs: 10 });
      const scheduleRender = vi.fn();
      const matrix = new Float64Array(VIEW.viewProj);
      const position: [number, number, number] = [0, 0, 0];
      const camera = { ...VIEW, viewProj: matrix, position };
      const cameras = new Map([["view", camera]]);
      governor.noteRenderedCameras(cameras, scheduleRender);
      if (field === "position") position[0] = 1;
      else matrix[12] = 1;
      governor.noteRenderedCameras(cameras, scheduleRender);
      expect(governor.stats()).toMatchObject({
        regime: "interaction",
        motion: { inferredReferences: 1 },
      });
      vi.advanceTimersByTime(10);
      expect(scheduleRender).toHaveBeenCalledOnce();
      governor.dispose();
    },
  );

  it("retargets in place while preserving held motion", () => {
    const governor = createViewGovernor();
    const held = governor.beginMotion("explicit");
    governor.setOptions({ interactionTargetMs: 20, stationaryTargetMs: 40 });
    expect(governor.stats()).toMatchObject({
      regime: "interaction",
      targetFrameTimeMs: 20,
      motion: { explicitReferences: 1 },
    });
    held.release();
    expect(governor.stats().targetFrameTimeMs).toBe(40);
  });

  it("validates a replacement before disturbing the running governor", () => {
    const governor = createViewGovernor({ stationaryTargetMs: 40 });
    expect(() => governor.setOptions({ initialFraction: 2 })).toThrow(
      "initialFraction",
    );
    expect(governor.stats().targetFrameTimeMs).toBe(40);
    expect(governor.qualityFraction()).toBe(1);
  });

  it("converges at the upper fraction and stops requesting capacity frames", () => {
    const governor = createViewGovernor({ minSamples: 1, cooldownMs: 0 });
    expect(governor.needsFrame()).toBe(true);
    governor.recordHostFrame({ hostFrameMs: 1, now: 0 });
    expect(governor.stats().lastAdjustment?.reason).toBe("clamped");
    expect(governor.needsFrame()).toBe(false);
  });

  it("answers regime, convergence and frame count as its snapshot does", () => {
    const governor = createViewGovernor({ minSamples: 1, cooldownMs: 0 });
    const read = () => ({
      regime: governor.regime(),
      converged: governor.converged(),
      frames: governor.frameCount(),
    });
    const snapshot = () => {
      const stats = governor.stats();
      const reason = stats.lastAdjustment?.reason;
      return {
        regime: stats.regime,
        converged: reason === "within-hysteresis" || reason === "clamped",
        frames: stats.frameMetrics.frames,
      };
    };
    expect(read()).toEqual({
      regime: "stationary",
      converged: false,
      frames: 0,
    });
    expect(read()).toEqual(snapshot());
    governor.recordHostFrame({ hostFrameMs: 1, now: 0 });
    expect(read()).toEqual({
      regime: "stationary",
      converged: true,
      frames: 1,
    });
    expect(read()).toEqual(snapshot());
    const motion = governor.beginMotion("explicit");
    expect(read()).toEqual({
      regime: "interaction",
      converged: false,
      frames: 1,
    });
    expect(read()).toEqual(snapshot());
    motion.release();
    expect(read()).toEqual(snapshot());
    governor.dispose();
  });

  it("learns the display quantum and grants the target whole refreshes", () => {
    const governor = createViewGovernor({
      initialFraction: 0.5,
      interactionTargetMs: 16,
      minSamples: 2,
      cooldownMs: 0,
    });
    const motion = governor.beginMotion("explicit");
    for (let frame = 0; frame < 2; frame += 1) {
      governor.recordHostFrame({ hostFrameMs: 16.7, now: frame });
    }
    expect(governor.stats().displayQuantumMs).toBeNull();

    // The third short interval settles what the display is. A 16 ms target
    // then means one refresh, and a frame that made its refresh is on time
    // rather than a miss the track would cut for.
    governor.recordHostFrame({ hostFrameMs: 16.7, now: 2 });
    expect(governor.stats().displayQuantumMs).toBeCloseTo(16.7, 5);
    expect(governor.stats().configuredFrameTimeMs).toBe(16);
    expect(governor.stats().targetFrameTimeMs).toBeCloseTo(16.7, 5);
    expect(governor.stats().lastAdjustment).toMatchObject({
      direction: "none",
      lateFraction: 0,
    });
    expect(governor.qualityFraction()).toBe(0.5);
    motion.release();
    governor.dispose();
  });

  it("does not mistake a slow page's best frame for the display's floor", () => {
    const governor = createViewGovernor({
      initialFraction: 0.5,
      interactionTargetMs: 16,
      stationaryTargetMs: 33,
      minSamples: 2,
      cooldownMs: 0,
    });
    // A software rasteriser never reaches a refresh boundary, so its shortest
    // interval says what it managed, not what the display can show. Believing
    // it would grant the settled target a 100 ms refresh and read these frames
    // as on time.
    for (let frame = 0; frame < 4; frame += 1) {
      governor.recordHostFrame({ hostFrameMs: 100, now: frame });
    }
    expect(governor.stats().displayQuantumMs).toBeLessThanOrEqual(17);
    expect(governor.stats().targetFrameTimeMs).toBeLessThanOrEqual(34);
    expect(governor.stats().lastAdjustment?.reason).toBe("above-target");
    governor.dispose();
  });

  it("keeps what it learned about the display across a settings change", () => {
    const governor = createViewGovernor({
      initialFraction: 0.5,
      interactionTargetMs: 16,
      minSamples: 2,
      cooldownMs: 0,
    });
    const motion = governor.beginMotion("explicit");
    for (let frame = 0; frame < 3; frame += 1) {
      governor.recordHostFrame({ hostFrameMs: 16.7, now: frame });
    }
    expect(governor.stats().targetFrameTimeMs).toBeCloseTo(16.7, 1);

    // New targets build new tracks. The display did not change with them.
    governor.setOptions({
      initialFraction: 0.5,
      interactionTargetMs: 16,
      minSamples: 2,
      cooldownMs: 0,
      stationaryTargetMs: 40,
    });
    expect(governor.stats().displayQuantumMs).toBeCloseTo(16.7, 5);
    expect(governor.stats().targetFrameTimeMs).toBeCloseTo(16.7, 1);
    motion.release();
    governor.dispose();
  });

  it("does not let stray short intervals pin the quantum for the session", () => {
    const governor = createViewGovernor({
      initialFraction: 0.5,
      interactionTargetMs: 16,
      minSamples: 2,
      cooldownMs: 0,
    });
    // A 60 Hz session with three compositor hiccups spread across it. A
    // session-lifetime minimum would take those three as the display and
    // settle at a 4 ms quantum and grant the 16 ms target four refreshes the
    // display never shows.
    const motion = governor.beginMotion("explicit");
    const hiccups = new Set([5, 900, 2500]);
    for (let frame = 0; frame < 3000; frame += 1) {
      governor.recordHostFrame({
        hostFrameMs: hiccups.has(frame) ? 4 : 16.7,
        now: frame * 16.7,
      });
    }

    expect(governor.stats().displayQuantumMs).toBeCloseTo(16.7, 5);
    expect(governor.stats().targetFrameTimeMs).toBeCloseTo(16.7, 1);
    motion.release();
    governor.dispose();
  });

  it("relearns the quantum when the display it is presenting on changes", () => {
    const governor = createViewGovernor({
      initialFraction: 0.5,
      interactionTargetMs: 16,
      minSamples: 2,
      cooldownMs: 0,
    });
    // A window dragged from a 120 Hz panel to a 60 Hz one. The 8.3 ms
    // intervals were true when measured and are simply no longer reachable.
    for (let frame = 0; frame < 50; frame += 1) {
      governor.recordHostFrame({ hostFrameMs: 8.3, now: frame * 8.3 });
    }
    expect(governor.stats().displayQuantumMs).toBeCloseTo(8.3, 5);

    for (let frame = 0; frame < 5000; frame += 1) {
      governor.recordHostFrame({ hostFrameMs: 16.7, now: 415 + frame * 16.7 });
    }
    expect(governor.stats().displayQuantumMs).toBeCloseTo(16.7, 5);
    governor.dispose();
  });

  it("will not believe an interval no display could have produced", () => {
    const governor = createViewGovernor({ minSamples: 2, cooldownMs: 0 });
    // Two callbacks inside one refresh say nothing about the refresh period.
    for (let frame = 0; frame < 4; frame += 1) {
      governor.recordHostFrame({ hostFrameMs: 0.5, now: frame });
    }
    expect(governor.stats().displayQuantumMs).toBeNull();
    governor.dispose();
  });
});
