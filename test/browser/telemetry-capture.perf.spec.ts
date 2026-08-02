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

describe("hardware telemetry capture", { tags: ["perf"] }, () => {
  afterAll(async () => {
    await closeBrowser();
  });

  it.skipIf(captureUrl.length === 0)(
    "captures an initial load, user gesture, and settled view",
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

        const beforeTightZoom = await session.readCamera();
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
        expect(markerReasons).toEqual(
          expect.arrayContaining([
            "marker:source-opened",
            "marker:initial-settled",
            "marker:gesture-start",
            "marker:drag-ended",
            "marker:gesture-ended",
            "marker:settled-after-gesture",
            "marker:tight-zoom-start",
            "marker:tight-zoom-ended",
            "marker:tight-zoom-settled",
          ]),
        );
        expect(cameraDistance(afterTightZoom)).toBeLessThan(
          cameraDistance(beforeTightZoom),
        );
        expect(session.failures).toEqual([]);

        process.stdout.write(
          `PERF_METRICS ${JSON.stringify({
            frames: trace.summary.frames,
            cleanFrames: trace.summary.cleanFrames,
            contaminatedFrames: trace.summary.contaminatedFrames,
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
