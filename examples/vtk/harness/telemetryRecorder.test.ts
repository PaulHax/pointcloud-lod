import { describe, expect, it } from "vitest";

import {
  createTelemetryRecorder,
  isSoftwareRenderer,
  type TelemetryEnvironment,
  type TelemetryFrameEvent,
} from "./telemetryRecorder";

const ENVIRONMENT: TelemetryEnvironment = {
  capturedAt: "2026-08-02T00:00:00.000Z",
  userAgent: "test browser",
  platform: "test platform",
  logicalProcessors: 8,
  deviceMemoryGiB: 16,
  viewport: {
    widthCssPx: 1280,
    heightCssPx: 720,
    devicePixelRatio: 1,
  },
  webgl: {
    version: "WebGL 2",
    vendor: "vendor",
    renderer: "renderer",
    unmaskedVendor: "GPU vendor",
    unmaskedRenderer: "GPU renderer",
    softwareRenderer: false,
    timerQuerySupported: true,
  },
};

describe("example telemetry", () => {
  it("recognizes common software WebGL renderers without flagging hardware", () => {
    expect(isSoftwareRenderer("ANGLE (Google, Vulkan SwiftShader)")).toBe(true);
    expect(isSoftwareRenderer("Mesa/X.org", "llvmpipe (LLVM 19.1.1)")).toBe(
      true,
    );
    expect(isSoftwareRenderer("NVIDIA Corporation", "NVIDIA RTX 4090")).toBe(
      false,
    );
  });

  it("classifies the complete interval between presentations by work revision", () => {
    let clock = 0;
    const recorder = createTelemetryRecorder({
      environment: () => ENVIRONMENT,
      now: () => clock,
      wallNow: () => new Date(clock),
    });
    const frame = (contamination: string[] = []): TelemetryFrameEvent => {
      const event = recorder.recordFrame({
        presentedAtMs: clock,
        rafIntervalMs: 16,
        vtkCpuMs: 4,
        governorFrameMs: 16,
        reportedToGovernor: true,
        contamination,
        state: { budget: 1_000_000 },
      });
      expect(event).not.toBeNull();
      return event!;
    };

    recorder.start();
    clock = 10;
    expect(frame().contamination).toEqual(["first-frame"]);

    clock = 20;
    const finish = recorder.beginWork("tile-load", { key: "1-0-0-0" });
    clock = 25;
    expect(frame(["selected-tiles-undecoded"]).contamination).toEqual([
      "selected-tiles-undecoded",
      "streaming-work-overlap",
      "streaming-work-pending",
    ]);

    clock = 30;
    finish("ok", { points: 100 });
    clock = 35;
    expect(frame().contamination).toEqual(["streaming-work-overlap"]);

    clock = 50;
    expect(frame().clean).toBe(true);
    recorder.stop();

    const trace = recorder.trace();
    expect(trace.schemaVersion).toBe(1);
    expect(trace.summary).toMatchObject({
      active: false,
      frames: 4,
      cleanFrames: 1,
      contaminatedFrames: 3,
      workEvents: 2,
      pendingWork: 0,
      workRevision: 2,
    });
    expect(trace.environment.webgl.softwareRenderer).toBe(false);
  });

  it("bounds retained events and reports what was dropped", () => {
    let clock = 0;
    const recorder = createTelemetryRecorder({
      environment: () => ENVIRONMENT,
      maxEvents: 3,
      now: () => clock,
      wallNow: () => new Date(clock),
    });
    recorder.start();
    for (let index = 0; index < 5; index += 1) {
      clock += 1;
      recorder.recordState(`state-${index}`, { index });
    }
    expect(recorder.summary()).toMatchObject({
      events: 3,
      droppedEvents: 3,
    });
    expect(recorder.trace().events.map((event) => event.sequence)).toEqual([
      4, 5, 6,
    ]);
  });

  it("attaches asynchronous GPU timing to the original frame", () => {
    let clock = 0;
    const recorder = createTelemetryRecorder({
      environment: () => ENVIRONMENT,
      now: () => clock,
      wallNow: () => new Date(clock),
    });
    recorder.start();
    const frame = recorder.recordFrame({
      presentedAtMs: 0,
      rafIntervalMs: 16,
      vtkCpuMs: 3,
      governorFrameMs: 16,
      gpuPending: true,
      reportedToGovernor: true,
      capacitySampleEligible: true,
      capacitySamplePending: true,
      state: null,
    })!;
    expect(frame.gpuStatus).toBe("pending");

    recorder.resolveGpuFrame(frame, { status: "valid", gpuMs: 7.5 });
    expect(recorder.trace().events).toContainEqual(
      expect.objectContaining({
        type: "frame",
        sequence: frame.sequence,
        gpuStatus: "valid",
        gpuMs: 7.5,
        capacitySampleStatus: "accepted",
      }),
    );
  });

  it("does not let work from a cleared recording finish into the new one", () => {
    let clock = 0;
    const recorder = createTelemetryRecorder({
      environment: () => ENVIRONMENT,
      now: () => clock,
      wallNow: () => new Date(clock),
    });
    recorder.start();
    const finishOldWork = recorder.beginWork("tile-load");
    clock = 5;
    recorder.clear();
    clock = 10;
    finishOldWork();
    expect(recorder.summary()).toMatchObject({
      active: true,
      events: 1,
      pendingWork: 0,
      workRevision: 0,
    });
  });
});
