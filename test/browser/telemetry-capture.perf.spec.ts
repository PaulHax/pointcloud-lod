import { mkdir, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, resolve } from "node:path";

import { afterAll, describe, expect, it } from "vitest";

import type { TelemetryTrace } from "../../src/telemetry";
import {
  cameraDistance,
  cameraPitchDegrees,
  closeBrowser,
  cross,
  dot,
  openExample,
  subtract,
  usingRealGpu,
  type CameraReading,
  type ExampleSession,
} from "./harness";
import { ensureReplayCloud, REPLAY_URL_PATH } from "./networkReplay";

const captureUrl = (process.env.POINTCLOUD_LOD_TELEMETRY_URL ?? "").trim();
const INPUT_INTERVAL_MS = 12;
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

const networkMode = (
  process.env.POINTCLOUD_LOD_TELEMETRY_NETWORK ?? "replay"
).trim();
if (networkMode !== "replay" && networkMode !== "live") {
  throw new Error(
    "POINTCLOUD_LOD_TELEMETRY_NETWORK must be either replay or live",
  );
}

const finiteEnvironmentNumber = (
  name: string,
  fallback: number,
  minimum: number,
): number => {
  const raw = process.env[name]?.trim();
  const value = raw === undefined || raw === "" ? fallback : Number(raw);
  if (!Number.isFinite(value) || value < minimum) {
    throw new Error(`${name} must be a finite number >= ${minimum}`);
  }
  return value;
};

const replayLatencyMs = finiteEnvironmentNumber(
  "POINTCLOUD_LOD_TELEMETRY_LATENCY_MS",
  40,
  0,
);
const replayMbps = finiteEnvironmentNumber(
  "POINTCLOUD_LOD_TELEMETRY_MBPS",
  80,
  Number.MIN_VALUE,
);
const replayCacheDirectory = resolve(
  process.env.POINTCLOUD_LOD_TELEMETRY_CACHE_DIR?.trim() ||
    "artifacts/telemetry/cache",
);
const budgetMode = (
  process.env.POINTCLOUD_LOD_TELEMETRY_BUDGET_MODE ?? "adaptive"
).trim();
if (budgetMode !== "adaptive" && budgetMode !== "fixed") {
  throw new Error(
    "POINTCLOUD_LOD_TELEMETRY_BUDGET_MODE must be either adaptive or fixed",
  );
}
const fixedPointBudget = finiteEnvironmentNumber(
  "POINTCLOUD_LOD_TELEMETRY_POINT_BUDGET",
  2_000_000,
  1,
);
const fixedInteractionDensity = finiteEnvironmentNumber(
  "POINTCLOUD_LOD_TELEMETRY_INTERACTION_DENSITY",
  1,
  0,
);
if (fixedInteractionDensity > 1) {
  throw new Error(
    "POINTCLOUD_LOD_TELEMETRY_INTERACTION_DENSITY must be at most 1",
  );
}

const captureSource = async (): Promise<Parameters<typeof openExample>[0]> => {
  if (networkMode === "live") {
    return { cloud: captureUrl, telemetry: true };
  }
  const file = await ensureReplayCloud(
    captureUrl,
    replayCacheDirectory,
    (process.env.POINTCLOUD_LOD_TELEMETRY_REFRESH ?? "") !== "",
  );
  return {
    cloud: REPLAY_URL_PATH,
    telemetry: true,
    files: { [REPLAY_URL_PATH]: file },
    network: {
      paths: [REPLAY_URL_PATH],
      latencyMs: replayLatencyMs,
      bytesPerSecond: (replayMbps * 1_000_000) / 8,
    },
  };
};

const writeTrace = async (trace: TelemetryTrace): Promise<void> => {
  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(trace, null, 2)}\n`, "utf8");
};

const distanceBetween = (
  left: readonly number[],
  right: readonly number[],
): number => Math.hypot(...subtract(left, right));

const normalised = (value: readonly number[]): [number, number, number] => {
  const length = Math.hypot(...value);
  if (!Number.isFinite(length) || length === 0) {
    throw new Error("cannot normalise a zero-length camera vector");
  }
  return [value[0]! / length, value[1]! / length, value[2]! / length];
};

type Projection = {
  readonly clientX: number;
  readonly clientY: number;
  readonly normalisedX: number;
  readonly normalisedY: number;
  readonly depth: number;
};

type ViewerBox = {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
};

/** Project a world point through the current perspective camera into CSS. */
const projectPoint = (
  camera: CameraReading,
  point: readonly number[],
  box: ViewerBox,
): Projection => {
  if (camera.parallelProjection) {
    throw new Error(
      "the telemetry camera path requires perspective projection",
    );
  }
  const forward = normalised(subtract(camera.focalPoint, camera.position));
  const right = normalised(cross(forward, camera.viewUp));
  const up = normalised(cross(right, forward));
  const eyeToPoint = subtract(point, camera.position);
  const depth = dot(eyeToPoint, forward);
  const halfHeight = depth * Math.tan((camera.viewAngle * Math.PI) / 360);
  const normalisedX =
    dot(eyeToPoint, right) / (halfHeight * (box.width / box.height));
  const normalisedY = dot(eyeToPoint, up) / halfHeight;
  return {
    clientX: box.x + ((normalisedX + 1) * box.width) / 2,
    clientY: box.y + ((1 - normalisedY) * box.height) / 2,
    normalisedX,
    normalisedY,
    depth,
  };
};

/**
 * A straight gesture as the ~16 px steps `session.drag` takes, so a capture
 * gesture is paced to painted frames like every other one in the suite.
 */
const lineSteps = (
  start: { readonly x: number; readonly y: number },
  end: { readonly x: number; readonly y: number },
): { dx: number; dy: number }[] => {
  const length = Math.hypot(end.x - start.x, end.y - start.y);
  const steps = Math.max(1, Math.ceil(length / 16));
  return Array.from({ length: steps }, () => ({
    dx: (end.x - start.x) / steps,
    dy: (end.y - start.y) / steps,
  }));
};

const centerPointWithPan = async (
  session: ExampleSession,
  box: ViewerBox,
  target: readonly [number, number, number],
): Promise<void> => {
  const beforeCamera = await session.readCamera();
  const before = projectPoint(beforeCamera, target, box);
  if (
    before.depth <= 0 ||
    Math.abs(before.normalisedX) >= 0.9 ||
    Math.abs(before.normalisedY) >= 0.9
  ) {
    throw new Error(
      `camera target is outside the usable view: ${JSON.stringify(before)}`,
    );
  }
  const depthRatio = before.depth / cameraDistance(beforeCamera);
  const center = { x: box.x + box.width / 2, y: box.y + box.height / 2 };
  const start = { x: before.clientX, y: before.clientY };
  await session.drag(
    lineSteps(start, {
      // vtk.js converts pointer travel into a translation on the focal plane.
      // A sampled target in front of or behind that plane crosses the screen
      // faster or slower by this exact perspective depth ratio.
      x: before.clientX + (center.x - before.clientX) * depthRatio,
      y: before.clientY + (center.y - before.clientY) * depthRatio,
    }),
    { start, button: "left", pauseMs: INPUT_INTERVAL_MS },
  );
  const panned = await session.readCamera();
  const afterProjection = projectPoint(panned, target, box);
  if (
    Math.abs(afterProjection.normalisedX) > 0.12 ||
    Math.abs(afterProjection.normalisedY) > 0.12
  ) {
    throw new Error(
      `pan did not center its world target: ${JSON.stringify(afterProjection)}`,
    );
  }

  // A screen pan establishes direction but cannot infer surface depth. The
  // coarse sample supplies that missing datum. Translate eye and focus by the
  // same residual so orientation and eye-to-focus distance remain unchanged,
  // while subsequent orbit is exactly about an actual point.
  const residual = subtract(target, panned.focalPoint);
  await session.place({
    position: panned.position.map((value, axis) => value + residual[axis]!),
    focalPoint: target,
  });
  await session.frame();
};

const pickBelowFocus = async (
  session: ExampleSession,
  box: ViewerBox,
  preferredYRatio: number,
): Promise<readonly [number, number, number]> => {
  const camera = await session.readCamera();
  const focalDepth = cameraDistance(camera);
  for (const yRatio of [preferredYRatio, 0.6, 0.7, 0.55, 0.75]) {
    for (const xRatio of [0.5, 0.45, 0.55]) {
      const result = await session.pickPoint(
        box.width * xRatio,
        box.height * yRatio,
      );
      if (result?.status !== "hit") continue;
      const depthRatio =
        distanceBetween(result.pointOnRay, camera.position) / focalDepth;
      if (depthRatio >= 0.4 && depthRatio <= 2.5) {
        return result.pointOnRay;
      }
    }
  }
  throw new Error(
    "no nearby drawn point supports a target below the current focus",
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

      const session = await openExample(await captureSource());
      try {
        if (budgetMode === "fixed") {
          await session.setBudgetMode("fixed");
          await session.page
            .locator("#point-budget")
            .fill(String(Math.floor(fixedPointBudget)));
          await session.page.locator("#point-budget").dispatchEvent("change");
        }
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

        await session.markTelemetry(
          networkMode === "live"
            ? "network-live"
            : `network-replay-${replayLatencyMs}ms-${replayMbps}mbps`,
        );

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

        const box = await session.page.locator("#viewer").boundingBox();
        if (box === null)
          throw new Error("the viewer has no box to interact in");
        const viewerCenter = {
          x: box.x + box.width / 2,
          y: box.y + box.height / 2,
        };
        const markMotionBoundary = async (
          label: string,
          moving: boolean,
        ): Promise<void> => {
          if (moving && budgetMode === "fixed") {
            await session.setDensityFraction(fixedInteractionDensity);
          }
          await session.markTelemetry(label);
          if (!moving && budgetMode === "fixed") {
            await session.setDensityFraction(1);
          }
        };

        // Pick a support depth from the point prefixes actually being drawn.
        // Pan its projection to the view centre, then use that sampled depth
        // as the orbit focus. No endpoint comes from a fixed pixel drag.
        const roiTarget = await pickBelowFocus(session, box, 0.62);
        await markMotionBoundary("gesture-start", true);
        await session.markTelemetry("roi-pan-start");
        await centerPointWithPan(session, box, roiTarget);
        await session.markTelemetry("roi-pan-ended");
        await session.markTelemetry("roi-centered");

        await session.page.mouse.move(viewerCenter.x, viewerCenter.y);
        for (let index = 0; index < 4; index += 1) {
          await session.page.mouse.wheel(0, -120);
          await session.page.waitForTimeout(INPUT_INTERVAL_MS);
        }
        await session.frame();
        await markMotionBoundary("gesture-ended", false);

        await session.settle(300_000);
        await session.render();
        await session.frame();
        await session.settle(300_000);
        await session.markTelemetry("settled-after-gesture");

        await session.page.mouse.move(viewerCenter.x, viewerCenter.y);
        await markMotionBoundary("tight-zoom-start", true);
        for (let index = 0; index < 40; index += 1) {
          await session.page.mouse.wheel(0, -120);
          await session.page.waitForTimeout(INPUT_INTERVAL_MS);
        }
        await session.frame();
        await markMotionBoundary("tight-zoom-ended", false);
        await session.settle(300_000);
        await session.render();
        await session.frame();
        await session.settle(300_000);
        await session.markTelemetry("tight-zoom-settled");
        const afterTightZoom = await session.readCamera();

        // Establish an oblique close view, then sweep broadly around its ROI.
        // The half turn is a repeatable workload length, not a camera-control
        // requirement: it exchanges tiles across the frustum and changes
        // which parts of the cloud are near and far from the camera.
        await markMotionBoundary("close-orbit-start", true);
        const obliquePitch = 45;
        const orbitSweepDegrees = -180;
        await session.drag(
          lineSteps(viewerCenter, {
            x: viewerCenter.x,
            y:
              viewerCenter.y +
              ((obliquePitch - cameraPitchDegrees(afterTightZoom)) / 360) *
                box.height,
          }),
          { button: "right", pauseMs: INPUT_INTERVAL_MS },
        );
        await session.markTelemetry("close-orbit-oblique");
        await session.drag(
          lineSteps(viewerCenter, {
            x: viewerCenter.x - (orbitSweepDegrees / 360) * box.width,
            y: viewerCenter.y,
          }),
          { button: "right", pauseMs: INPUT_INTERVAL_MS },
        );
        await markMotionBoundary("close-orbit-ended", false);
        await session.settle(300_000);
        await session.render();
        await session.frame();
        await session.settle(300_000);
        await session.markTelemetry("close-orbit-settled");
        const afterCloseOrbit = await session.readCamera();

        await markMotionBoundary("horizontal-rotation-start", true);
        const horizontalPitch = 4;
        await session.drag(
          lineSteps(viewerCenter, {
            x: viewerCenter.x,
            y:
              viewerCenter.y +
              ((horizontalPitch - cameraPitchDegrees(afterCloseOrbit)) / 360) *
                box.height,
          }),
          { button: "right", pauseMs: INPUT_INTERVAL_MS },
        );
        await session.markTelemetry("horizontal-rotation-ended");

        // Pick another support depth below the horizontal focal point and drag
        // that projected point to centre. This is the close-view pan the
        // previous fixed 560 px gesture failed to express.
        await session.frame();
        const horizontalTarget = await pickBelowFocus(session, box, 0.6);
        await session.markTelemetry("horizontal-pan-start");
        await centerPointWithPan(session, box, horizontalTarget);
        await markMotionBoundary("horizontal-pan-ended", false);
        await session.settle(300_000);
        await session.render();
        await session.frame();
        const horizontalStats = await session.settle(300_000);
        await session.markTelemetry("horizontal-roi-settled");

        // The return to overview is deliberately a separate phase after the
        // close horizontal view has settled and been measured.
        await markMotionBoundary("return-overview-start", true);
        await session.page.mouse.move(viewerCenter.x, viewerCenter.y);
        for (let index = 0; index < 44; index += 1) {
          await session.page.mouse.wheel(0, 120);
          await session.page.waitForTimeout(INPUT_INTERVAL_MS);
        }
        await session.frame();
        await markMotionBoundary("return-overview-gesture-ended", false);
        await session.settle(300_000);
        await session.render();
        await session.frame();
        await session.settle(300_000);
        await session.markTelemetry("return-overview-settled");

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
            "marker:roi-pan-start",
            "marker:roi-pan-ended",
            "marker:roi-centered",
            "marker:gesture-ended",
            "marker:settled-after-gesture",
            "marker:tight-zoom-start",
            "marker:tight-zoom-ended",
            "marker:tight-zoom-settled",
            "marker:close-orbit-start",
            "marker:close-orbit-oblique",
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
        expect(
          horizontalStats.controller?.selection.targetTiles,
        ).toBeGreaterThan(0);
        expect(horizontalStats.adapter?.drawnPoints).toBeGreaterThan(0);
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
            network:
              networkMode === "live"
                ? { mode: "live" }
                : {
                    mode: "replay",
                    latencyMs: replayLatencyMs,
                    mbpsPerRequest: replayMbps,
                  },
            budget:
              budgetMode === "fixed"
                ? {
                    mode: "fixed",
                    points: Math.floor(fixedPointBudget),
                    interactionDensity: fixedInteractionDensity,
                  }
                : { mode: "adaptive" },
          })}\nTELEMETRY_ARTIFACT ${outputPath}\n`,
        );
      } finally {
        await session.close();
      }
    },
  );
});
