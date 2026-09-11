import { afterEach, describe, expect, it, vi } from "vitest";

import type { CameraView } from "./camera";
import { createViewGovernor } from "./viewGovernor";

const VIEW: CameraView = {
  projection: "perspective",
  viewProj: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
  position: [0, 0, 0],
  fovY: 1,
  viewportWidthCssPx: 100,
  viewportHeightCssPx: 100,
};

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
      hysteresis: 0,
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
      hysteresis: 0,
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
    expect(governor.qualityFraction()).toBeCloseTo(0.66);
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
    expect(governor.qualityFraction()).toBeCloseTo(0.5);
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

  it("requires two consecutive severe interaction frames for an emergency cut", () => {
    const governor = createViewGovernor({ minSamples: 30, cooldownMs: 0 });
    const motion = governor.beginMotion("explicit");
    governor.recordTransientFrame({ hostFrameMs: 40, now: 0 });
    expect(governor.qualityFraction()).toBe(1);
    governor.recordTransientFrame({ hostFrameMs: 40, now: 1 });
    expect(governor.qualityFraction()).toBe(0.5);
    expect(governor.stats().lastAdjustment?.reason).toBe("emergency-cut");
    motion.release();
  });

  it("gives an emergency cut back inside the gesture that caused it", () => {
    const governor = createViewGovernor({ minSamples: 30, cooldownMs: 0 });
    // Tiles are streaming, which is what withholds eligibility: the sampling
    // loop that would otherwise raise quality back is shut off for as long as
    // this holds, while the emergency cut below is not.
    governor.setWorkState({
      workPending: true,
      physicalTileOperations: 4,
      physicalHierarchyOperations: 0,
    });
    const motion = governor.beginMotion("explicit");
    expect(governor.stats().activity).toMatchObject({
      workPending: true,
      measurementEligible: false,
    });

    governor.recordHostFrame({ hostFrameMs: 40, now: 0 });
    governor.recordHostFrame({ hostFrameMs: 40, now: 1 });
    expect(governor.qualityFraction()).toBe(0.5);
    expect(governor.stats().lastAdjustment?.reason).toBe("emergency-cut");
    expect(governor.stats().capacitySamples).toMatchObject({ eligible: 0 });

    // The same frames the cut reads, now comfortably inside the target.
    governor.recordHostFrame({ hostFrameMs: 10, now: 2 });
    expect(governor.qualityFraction()).toBe(0.5);
    governor.recordHostFrame({ hostFrameMs: 10, now: 3 });
    expect(governor.qualityFraction()).toBe(1);
    expect(governor.stats().lastAdjustment).toMatchObject({
      reason: "emergency-restore",
      direction: "increase",
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
    governor.recordHostFrame({ hostFrameMs: 40, now: 0 });
    governor.recordHostFrame({ hostFrameMs: 40, now: 1 });
    expect(governor.qualityFraction()).toBe(0.5);
    governor.invalidateCapacity();
    expect(governor.qualityFraction()).toBe(0.5);
    expect(governor.stats().samples).toBe(0);
    governor.recordHostFrame({ hostFrameMs: 10, now: 2 });
    governor.recordHostFrame({ hostFrameMs: 10, now: 3 });
    expect(governor.qualityFraction()).toBe(1);
    expect(governor.stats().lastAdjustment?.reason).toBe("emergency-restore");
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
    governor.recordHostFrame({ hostFrameMs: 40, now: 0 });
    governor.recordHostFrame({ hostFrameMs: 40, now: 1 });
    expect(governor.qualityFraction()).toBeCloseTo(0.2);

    for (let at = 2; at < 40; at += 1) {
      governor.recordHostFrame({ hostFrameMs: 10, now: at });
      // Back to where the cut started and never past it: quality above that
      // has to be earned by an eligible capacity sample.
      expect(governor.qualityFraction()).toBeLessThanOrEqual(0.4);
    }
    expect(governor.qualityFraction()).toBeCloseTo(0.4);
    motion.release();
  });

  it("reseeds a gesture that begins inside the previous settle window", () => {
    vi.useFakeTimers();
    const governor = createViewGovernor({
      minSamples: 30,
      cooldownMs: 0,
      interactionSettleMs: 1000,
    });
    const first = governor.beginMotion("explicit");
    governor.recordCameraChange();
    // Two consecutive severe frames per cut, spaced past the emergency
    // cooldown so all three land.
    for (const at of [0, 1, 500, 501, 1000, 1001]) {
      governor.recordTransientFrame({ hostFrameMs: 200, now: at });
    }
    expect(governor.qualityFraction()).toBeCloseTo(0.125);

    first.release();
    // The settle timer is still pending, so the governor never left the
    // interaction regime and the cut floor is still in force.
    expect(governor.stats().regime).toBe("interaction");

    const second = governor.beginMotion("explicit");
    expect(governor.qualityFraction()).toBeCloseTo(0.25);
    expect(governor.stats().lastAdjustment?.reason).toBe("seeded");
    second.release();
  });

  it("leaves an uncut interaction track alone when a gesture repeats", () => {
    vi.useFakeTimers();
    const governor = createViewGovernor({
      minSamples: 2,
      cooldownMs: 0,
      hysteresis: 0,
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
    // quality, and the emergency streak could never reach two frames either.
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

  it("learns the display quantum and stops asking for what it cannot show", () => {
    const governor = createViewGovernor({
      initialFraction: 0.5,
      interactionTargetMs: 16,
      minSamples: 2,
      cooldownMs: 0,
      hysteresis: 0.2,
    });
    const motion = governor.beginMotion("explicit");
    // A 60 Hz display presents no faster than this however cheap the frame,
    // so at a 16 ms target every one of these lands inside the hysteresis
    // band and the interaction track can only ever be cut.
    for (let frame = 0; frame < 2; frame += 1) {
      governor.recordHostFrame({ hostFrameMs: 16.7, now: frame });
    }
    expect(governor.stats().displayQuantumMs).toBeNull();
    expect(governor.stats().lastAdjustment?.direction).toBe("none");

    // The third short interval settles what the display is: the target moves
    // to one the same measurement can fall below, and the very frame that
    // taught it that is evidence of headroom.
    governor.recordHostFrame({ hostFrameMs: 16.7, now: 2 });
    expect(governor.stats().displayQuantumMs).toBeCloseTo(16.7, 5);
    expect(governor.stats().configuredFrameTimeMs).toBe(16);
    expect(governor.stats().targetFrameTimeMs).toBeCloseTo(24, 1);
    expect(governor.stats().lastAdjustment).toMatchObject({
      direction: "increase",
      reason: "below-target",
    });
    expect(governor.qualityFraction()).toBeGreaterThan(0.5);
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
      hysteresis: 0.2,
    });
    // A software rasteriser never reaches a refresh boundary, so its shortest
    // interval says what it managed, not what the display can show. Believing
    // it would raise the settled target past 100 ms and read these frames as
    // headroom.
    for (let frame = 0; frame < 4; frame += 1) {
      governor.recordHostFrame({ hostFrameMs: 100, now: frame });
    }
    expect(governor.stats().displayQuantumMs).toBeLessThanOrEqual(17);
    expect(governor.stats().targetFrameTimeMs).toBe(33);
    expect(governor.stats().lastAdjustment?.reason).toBe("above-target");
    governor.dispose();
  });

  it("keeps what it learned about the display across a settings change", () => {
    const governor = createViewGovernor({
      initialFraction: 0.5,
      interactionTargetMs: 16,
      minSamples: 2,
      cooldownMs: 0,
      hysteresis: 0.2,
    });
    const motion = governor.beginMotion("explicit");
    for (let frame = 0; frame < 3; frame += 1) {
      governor.recordHostFrame({ hostFrameMs: 16.7, now: frame });
    }
    expect(governor.stats().targetFrameTimeMs).toBeCloseTo(24, 1);

    // New targets build new tracks. The display did not change with them.
    governor.setOptions({
      initialFraction: 0.5,
      interactionTargetMs: 16,
      minSamples: 2,
      cooldownMs: 0,
      hysteresis: 0.2,
      stationaryTargetMs: 40,
    });
    expect(governor.stats().displayQuantumMs).toBeCloseTo(16.7, 5);
    expect(governor.stats().targetFrameTimeMs).toBeCloseTo(24, 1);
    motion.release();
    governor.dispose();
  });

  it("does not let stray short intervals pin the quantum for the session", () => {
    const governor = createViewGovernor({
      initialFraction: 0.5,
      interactionTargetMs: 16,
      minSamples: 2,
      cooldownMs: 0,
      hysteresis: 0.2,
    });
    // A 60 Hz session with three compositor hiccups spread across it. A
    // session-lifetime minimum would take those three as the display and
    // settle at a 4 ms quantum, whose reachable target is 5.75 ms — under the
    // 16 ms interaction target, so the correction silently disappears.
    const motion = governor.beginMotion("explicit");
    const hiccups = new Set([5, 900, 2500]);
    for (let frame = 0; frame < 3000; frame += 1) {
      governor.recordHostFrame({
        hostFrameMs: hiccups.has(frame) ? 4 : 16.7,
        now: frame * 16.7,
      });
    }

    expect(governor.stats().displayQuantumMs).toBeCloseTo(16.7, 5);
    expect(governor.stats().targetFrameTimeMs).toBeCloseTo(24, 1);
    motion.release();
    governor.dispose();
  });

  it("relearns the quantum when the display it is presenting on changes", () => {
    const governor = createViewGovernor({
      initialFraction: 0.5,
      interactionTargetMs: 16,
      minSamples: 2,
      cooldownMs: 0,
      hysteresis: 0.2,
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
