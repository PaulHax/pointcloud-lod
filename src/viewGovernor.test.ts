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
