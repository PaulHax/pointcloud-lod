/**
 * The replay machinery itself, checked against the committed fixture.
 *
 * The benchmark it serves runs headed on real hardware over remote datasets
 * and asserts no speed, so nothing in that path would notice the harness
 * quietly breaking — a replay that dispatched into empty space, a recorder
 * that stopped seeing wheel events, a drift comparison that always returned
 * zero all produce a clean-looking run. These checks are about mechanism, on
 * the small fixture, under the same software rasteriser as the rest of the
 * suite: a gesture is captured, replayed into a second page, and the camera it
 * produced there is compared with the camera it produced here.
 */

import { afterAll, describe, expect, it } from "vitest";

import type { InputRecording } from "../../examples/vtk/scene/inputRecorder";
import { compareCameraTracks, replayInput } from "./inputReplay";
import {
  closeBenchmarkBrowser,
  openScene,
  replayPath,
  type SceneSession,
} from "./sceneHarness";

const FIXTURE_SCENE = "/index.html?url=/fixtures/fixture.copc.laz";
const VIEWER = { width: 800, height: 600 };

const openFixtureScene = async (): Promise<
  Awaited<ReturnType<typeof openScene>>
> => {
  const session = await openScene({ path: FIXTURE_SCENE, headless: true });
  await session.resizeViewer(VIEWER);
  await session.settle(120_000);
  return session;
};

/**
 * A gesture with all three of the things a replay has to carry: a held-button
 * drag, a wheel turn with nothing held, and a pause between them.
 */
const captureGesture = async (
  session: SceneSession,
): Promise<InputRecording> => {
  const box = await session.viewerBox();
  const centre = { x: box.x + box.width / 2, y: box.y + box.height / 2 };
  await session.startInputRecorder();

  await session.page.mouse.move(centre.x, centre.y);
  await session.page.mouse.down({ button: "right" });
  for (let step = 1; step <= 12; step += 1) {
    await session.page.mouse.move(centre.x + step * 9, centre.y + step * 3);
    await session.page.waitForTimeout(16);
  }
  await session.page.mouse.up({ button: "right" });
  await session.page.waitForTimeout(80);
  for (let turn = 0; turn < 3; turn += 1) {
    await session.page.mouse.wheel(0, -120);
    await session.page.waitForTimeout(32);
  }
  await session.page.waitForTimeout(120);

  await session.stopInputRecorder();
  return (await session.inputRecording()) as InputRecording;
};

describe("recorded-gesture replay harness", () => {
  afterAll(async () => {
    await closeBenchmarkBrowser();
  });

  it("captures a gesture and replays it into an equivalent camera path", async () => {
    const capture = await openFixtureScene();
    let recording: InputRecording;
    let capturedPoses: InputRecording["poses"];
    let capturedFinalPose: InputRecording["startPose"];
    try {
      recording = await captureGesture(capture);
      capturedPoses = recording.poses;
      capturedFinalPose = await capture.readCamera();

      expect(recording.schemaVersion).toBe(1);
      expect(recording.viewer).toMatchObject(
        // The recorder reports the viewer it watched, which is what a replay
        // has to reproduce; a recording of a differently sized view aims its
        // coordinates somewhere else.
        { widthCssPx: VIEWER.width, heightCssPx: VIEWER.height },
      );
      expect(
        recording.events.filter((event) => event.type === "pointermove").length,
      ).toBeGreaterThanOrEqual(12);
      expect(
        recording.events.filter((event) => event.type === "wheel").length,
      ).toBe(3);
      expect(recording.truncated).toBe(false);
      // A resize mid-capture would make every coordinate after it aim at a
      // differently sized view, and the recorder reports one as an event.
      expect(
        recording.events.filter((event) => event.type === "viewport"),
      ).toEqual([]);
      // The gesture has to have moved the camera, or the comparison below is
      // between two cameras that both did nothing.
      expect(capturedPoses.length).toBeGreaterThan(1);
      expect(capturedFinalPose.position).not.toEqual(
        recording.startPose.position,
      );
    } finally {
      await capture.close();
    }

    const replay = await openScene({
      path: replayPath(`http://recorded.invalid${FIXTURE_SCENE}`),
      headless: true,
    });
    try {
      await replay.resizeViewer(VIEWER);
      await replay.settle(120_000);
      await replay.placeCamera({
        position: recording.startPose.position,
        focalPoint: recording.startPose.focalPoint,
        viewUp: recording.startPose.viewUp,
      });
      await replay.settle(120_000);

      const marks: string[] = [];
      await replay.startTelemetry();
      await replay.startInputRecorder();
      const result = await replayInput({
        page: replay.page,
        recording,
        viewer: await replay.viewerBox(),
        onMarker: async (label) => {
          marks.push(label);
        },
      });
      await replay.stopInputRecorder();
      await replay.stopTelemetry();

      expect(result.dispatch.events).toBe(
        recording.events.filter((event) => event.type !== "viewport").length,
      );
      expect(marks).toEqual(recording.markers.map((marker) => marker.label));

      const replayedPose = await replay.readCamera();
      // The same input from the same camera has to reach the same place. It is
      // not bit-identical: the interactor applies moves on its own animation
      // frame, so the two runs coalesce a differently sized tail of the
      // gesture. A whole-radius error would mean the gesture missed entirely.
      const radius = Math.hypot(
        ...capturedFinalPose.position.map(
          (value, axis) => value - capturedFinalPose.focalPoint[axis]!,
        ),
      );
      const separation = Math.hypot(
        ...replayedPose.position.map(
          (value, axis) => value - capturedFinalPose.position[axis]!,
        ),
      );
      expect(separation / radius).toBeLessThan(0.25);

      const replayedRecording =
        (await replay.inputRecording()) as InputRecording;
      const drift = compareCameraTracks(capturedPoses, replayedRecording.poses);
      expect(drift).not.toBeNull();
      expect(drift!.samples).toBeGreaterThan(0);
      expect(Number.isFinite(drift!.maxRelativeError)).toBe(true);
      expect(drift!.meanRelativeError).toBeLessThan(
        drift!.maxRelativeError + 1,
      );

      const trace = await replay.telemetryTrace();
      process.stdout.write(
        `replay harness: ${recording.events.length} events, ` +
          `${capturedPoses.length} captured poses, ` +
          `${replayedRecording.poses.length} replayed poses, ` +
          `separation ${(separation / radius).toFixed(3)} radii, ` +
          `timed drift ${(drift!.meanRelativeError * 100).toFixed(2)}%/` +
          `${(drift!.maxRelativeError * 100).toFixed(2)}%, ` +
          `path drift ${(drift!.meanRelativePathError * 100).toFixed(2)}%/` +
          `${(drift!.maxRelativePathError * 100).toFixed(2)}%, ` +
          `${trace.summary.frames} frames\n`,
      );
      expect(trace.summary.frames).toBeGreaterThan(0);
      expect(
        trace.events.some(
          (event) =>
            event.type === "state" && event.reason === "recording-started",
        ),
      ).toBe(true);
      expect(replay.failures).toEqual([]);
    } finally {
      await replay.close();
    }
  });

  it("reports drift between two genuinely different camera tracks", () => {
    const at = (atMs: number, x: number): InputRecording["poses"][number] => ({
      atMs,
      position: [x, 0, 10],
      focalPoint: [0, 0, 0],
      viewUp: [0, 0, 1],
      viewAngle: 30,
      parallelScale: 1,
      parallelProjection: false,
    });
    const identical = compareCameraTracks(
      [at(0, 0), at(10, 1)],
      [at(0, 0), at(10, 1)],
    );
    expect(identical!.maxPositionError).toBe(0);
    expect(identical!.maxViewAngleError).toBe(0);
    expect(identical!.maxRelativePathError).toBe(0);

    // The same route walked late: every sample lands somewhere the replay did
    // reach, so path error is zero while time-aligned error is not.
    const delayed = compareCameraTracks(
      [at(0, 0), at(10, 1), at(20, 2)],
      [at(0, 0), at(15, 1), at(30, 2)],
    );
    expect(delayed!.maxRelativePathError).toBeCloseTo(0);
    expect(delayed!.maxPositionError).toBeGreaterThan(0);

    const apart = compareCameraTracks(
      [at(0, 0), at(10, 1)],
      [at(0, 0), at(10, 4)],
    );
    // Sampled on the recorded track's clock: the first pair agree, the second
    // are three units apart, so the mean is half of the maximum.
    expect(apart!.maxPositionError).toBeCloseTo(3);
    expect(apart!.meanPositionError).toBeCloseTo(1.5);
    expect(apart!.maxViewAngleError).toBeGreaterThan(0);
  });
});
