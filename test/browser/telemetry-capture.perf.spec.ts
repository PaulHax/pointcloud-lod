import { mkdir, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, resolve } from "node:path";

import { afterAll, describe, expect, it } from "vitest";

import type { TelemetryTrace } from "../../examples/vtk/telemetry";
import { closeBrowser, openExample, usingRealGpu } from "./harness";

const captureUrl = (process.env.POINTCLOUD_LOD_TELEMETRY_URL ?? "").trim();
const configuredOutput = (
  process.env.POINTCLOUD_LOD_TELEMETRY_OUT ?? ""
).trim();
const defaultName = `pointcloud-trace-${new Date()
  .toISOString()
  .replaceAll(":", "-")
  .replaceAll(".", "-")}.json`;
const outputPath = configuredOutput
  ? isAbsolute(configuredOutput)
    ? configuredOutput
    : resolve(process.cwd(), configuredOutput)
  : resolve(process.cwd(), "artifacts/telemetry", defaultName);

const writeTrace = async (trace: TelemetryTrace): Promise<void> => {
  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(trace, null, 2)}\n`, "utf8");
};

const cameraDistance = (camera: {
  readonly position: readonly number[];
  readonly focalPoint: readonly number[];
}): number =>
  Math.hypot(
    camera.position[0]! - camera.focalPoint[0]!,
    camera.position[1]! - camera.focalPoint[1]!,
    camera.position[2]! - camera.focalPoint[2]!,
  );

const cameraPitchDegrees = (camera: {
  readonly position: readonly number[];
  readonly focalPoint: readonly number[];
}): number => {
  const distance = cameraDistance(camera);
  return (
    (Math.asin((camera.position[2]! - camera.focalPoint[2]!) / distance) *
      180) /
    Math.PI
  );
};

describe("hardware telemetry capture", { tags: ["perf"] }, () => {
  afterAll(async () => {
    await closeBrowser();
  });

  it.skipIf(captureUrl.length === 0)(
    "captures overview, ROI zoom, close orbit, horizontal pan, and return",
    async () => {
      if (!usingRealGpu()) {
        throw new Error(
          "telemetry capture requires POINTCLOUD_LOD_BROWSER_GPU=1",
        );
      }

      const session = await openExample({
        cloud: captureUrl,
        telemetry: true,
      });
      try {
        const environment = await session.telemetryEnvironment();
        if (environment.webgl.softwareRenderer) {
          throw new Error(
            `telemetry capture requires a hardware WebGL renderer; got ${
              environment.webgl.unmaskedRenderer ??
              environment.webgl.renderer ??
              "unknown"
            }`,
          );
        }
        expect((await session.telemetrySummary()).active).toBe(true);

        await session.markTelemetry("source-opened");
        await session.settle(300_000);
        // Prove the converged view can produce a clean presentation, then
        // converge again in case that sample caused a governor adjustment.
        // Marking before a train of forced renders would let the measurement
        // driver change the state after calling it settled.
        await session.render();
        await session.frame();
        await session.settle(300_000);
        await session.markTelemetry("initial-settled");

        await session.markTelemetry("gesture-start");
        await session.drag(
          [
            { dx: 18, dy: -6 },
            { dx: 18, dy: -5 },
            { dx: 18, dy: -4 },
            { dx: 18, dy: -2 },
            { dx: 18, dy: 2 },
            { dx: 18, dy: 4 },
            { dx: 18, dy: 5 },
            { dx: 18, dy: 6 },
          ],
          20,
        );
        await session.markTelemetry("drag-ended");

        const box = await session.page.locator("#viewer").boundingBox();
        if (box === null) throw new Error("the viewer has no box to zoom in");
        await session.page.mouse.move(
          box.x + box.width / 2,
          box.y + box.height / 2,
        );
        for (let index = 0; index < 4; index += 1) {
          await session.page.mouse.wheel(0, -120);
          await session.frame();
        }
        await session.markTelemetry("gesture-ended");

        await session.settle(300_000);
        await session.render();
        await session.frame();
        await session.settle(300_000);
        await session.markTelemetry("settled-after-gesture");

        const beforePreZoomPan = await session.readCamera();
        await session.markTelemetry("pre-zoom-pan-start");
        await session.drag(
          Array.from({ length: 8 }, () => ({ dx: 0, dy: -8 })),
          20,
        );
        await session.markTelemetry("pre-zoom-pan-ended");
        const beforeTightZoom = await session.readCamera();
        await session.page.mouse.move(
          box.x + box.width / 2,
          box.y + box.height / 2,
        );
        await session.markTelemetry("tight-zoom-start");
        for (let index = 0; index < 40; index += 1) {
          await session.page.mouse.wheel(0, -120);
          await session.frame();
        }
        await session.markTelemetry("tight-zoom-ended");
        await session.settle(300_000);
        await session.render();
        await session.frame();
        await session.settle(300_000);
        await session.markTelemetry("tight-zoom-settled");
        const afterTightZoom = await session.readCamera();

        await session.markTelemetry("close-orbit-start");
        let gestureX = box.x + box.width / 2;
        let gestureY = box.y + box.height / 2;
        await session.page.mouse.move(gestureX, gestureY);
        await session.page.mouse.down({ button: "right" });
        for (let index = 0; index < 8; index += 1) {
          gestureX += 14;
          if (index < 2) gestureY -= 14;
          await session.page.mouse.move(gestureX, gestureY);
          await session.frame();
        }
        await session.page.mouse.up({ button: "right" });
        await session.markTelemetry("close-orbit-ended");
        await session.settle(300_000);
        await session.render();
        await session.frame();
        await session.settle(300_000);
        await session.markTelemetry("close-orbit-settled");
        const afterCloseOrbit = await session.readCamera();

        await session.markTelemetry("horizontal-rotation-start");
        gestureX = box.x + box.width / 2;
        gestureY = box.y + box.height / 2;
        await session.page.mouse.move(gestureX, gestureY);
        await session.page.mouse.down({ button: "right" });
        for (let index = 0; index < 6; index += 1) {
          gestureY -= 14;
          await session.page.mouse.move(gestureX, gestureY);
          await session.frame();
        }
        await session.page.mouse.up({ button: "right" });
        await session.markTelemetry("horizontal-rotation-ended");
        const beforeHorizontalPan = await session.readCamera();

        await session.markTelemetry("horizontal-pan-start");
        gestureX = box.x + box.width / 2;
        gestureY = box.y + box.height * 0.9;
        await session.page.mouse.move(gestureX, gestureY);
        await session.page.mouse.down();
        for (let index = 0; index < 40; index += 1) {
          gestureY -= 14;
          await session.page.mouse.move(gestureX, gestureY);
          await session.page.waitForTimeout(20);
          await session.frame();
        }
        await session.page.mouse.up();
        await session.markTelemetry("horizontal-pan-ended");
        await session.settle(300_000);
        await session.render();
        await session.frame();
        const horizontalStats = await session.settle(300_000);
        await session.markTelemetry("horizontal-roi-settled");
        const horizontalRoiCamera = await session.readCamera();

        // The return to overview is deliberately a separate phase after the
        // close horizontal view has settled and been measured.
        await session.markTelemetry("return-overview-start");
        gestureX = box.x + box.width / 2;
        gestureY = box.y + box.height / 2;
        await session.page.mouse.move(gestureX, gestureY);
        for (let index = 0; index < 44; index += 1) {
          await session.page.mouse.wheel(0, 120);
          await session.frame();
        }
        await session.markTelemetry("return-overview-gesture-ended");
        await session.settle(300_000);
        await session.render();
        await session.frame();
        await session.settle(300_000);
        await session.markTelemetry("return-overview-settled");
        const overviewCamera = await session.readCamera();

        await session.stopTelemetry();
        const trace = await session.telemetryTrace();
        await writeTrace(trace);

        const markerReasons = trace.events
          .filter((event) => event.type === "state")
          .map((event) => event.reason)
          .filter((reason) => reason.startsWith("marker:"));
        expect(trace.schemaVersion).toBe(1);
        expect(trace.summary).toMatchObject({
          active: false,
          pendingWork: 0,
        });
        expect(trace.environment.webgl.softwareRenderer).toBe(false);
        expect(
          trace.environment.webgl.unmaskedRenderer ??
            trace.environment.webgl.renderer,
        ).not.toBeNull();
        expect(trace.summary.frames).toBeGreaterThan(0);
        expect(trace.summary.cleanFrames).toBeGreaterThan(0);
        expect(trace.summary.workEvents).toBeGreaterThan(0);
        const validGpuFrames = trace.events.filter(
          (event) => event.type === "frame" && event.gpuStatus === "valid",
        );
        expect(validGpuFrames.length).toBeGreaterThan(0);
        expect(markerReasons).toEqual(
          expect.arrayContaining([
            "marker:source-opened",
            "marker:initial-settled",
            "marker:gesture-start",
            "marker:drag-ended",
            "marker:gesture-ended",
            "marker:settled-after-gesture",
            "marker:pre-zoom-pan-start",
            "marker:pre-zoom-pan-ended",
            "marker:tight-zoom-start",
            "marker:tight-zoom-ended",
            "marker:tight-zoom-settled",
            "marker:close-orbit-start",
            "marker:close-orbit-ended",
            "marker:close-orbit-settled",
            "marker:horizontal-rotation-start",
            "marker:horizontal-rotation-ended",
            "marker:horizontal-pan-start",
            "marker:horizontal-pan-ended",
            "marker:horizontal-roi-settled",
            "marker:return-overview-start",
            "marker:return-overview-gesture-ended",
            "marker:return-overview-settled",
          ]),
        );
        expect(cameraDistance(afterTightZoom)).toBeLessThan(
          cameraDistance(beforeTightZoom),
        );
        expect(beforeTightZoom.focalPoint[2]).toBeLessThan(
          beforePreZoomPan.focalPoint[2]!,
        );
        expect(cameraDistance(afterCloseOrbit)).toBeCloseTo(
          cameraDistance(afterTightZoom),
          5,
        );
        expect(Math.abs(cameraPitchDegrees(afterCloseOrbit))).toBeGreaterThan(
          35,
        );
        expect(Math.abs(cameraPitchDegrees(afterCloseOrbit))).toBeLessThan(55);
        expect(Math.abs(cameraPitchDegrees(horizontalRoiCamera))).toBeLessThan(
          15,
        );
        expect(horizontalRoiCamera.focalPoint[2]).toBeLessThan(
          beforeHorizontalPan.focalPoint[2]!,
        );
        expect(cameraDistance(horizontalRoiCamera)).toBeCloseTo(
          cameraDistance(afterTightZoom),
          5,
        );
        expect(
          horizontalStats.controller?.selection.targetTiles,
        ).toBeGreaterThan(0);
        expect(horizontalStats.adapter?.drawnPoints).toBeGreaterThan(0);
        expect(cameraDistance(overviewCamera)).toBeGreaterThan(
          cameraDistance(beforeTightZoom) * 0.5,
        );
        expect(session.failures).toEqual([]);

        process.stdout.write(
          `PERF_METRICS ${JSON.stringify({
            frames: trace.summary.frames,
            cleanFrames: trace.summary.cleanFrames,
            contaminatedFrames: trace.summary.contaminatedFrames,
            validGpuFrames: validGpuFrames.length,
            acceptedCapacitySamples: trace.summary.acceptedCapacitySamples,
            rejectedCapacitySamples: trace.summary.rejectedCapacitySamples,
            workEvents: trace.summary.workEvents,
            longTasks: trace.summary.longTasks,
            durationMs: trace.summary.durationMs,
          })}\nTELEMETRY_ARTIFACT ${outputPath}\n`,
        );
      } finally {
        await session.close();
      }
    },
  );
});
