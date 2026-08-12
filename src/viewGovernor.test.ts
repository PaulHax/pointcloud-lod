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
});
