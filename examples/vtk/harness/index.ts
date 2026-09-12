/**
 * Everything that watches a scene, and nothing that draws one.
 *
 * This module is the only thing under `harness/` a page names, and it is
 * reached through a dynamic import guarded by `harnessRequested`, so a session
 * nobody is measuring never downloads, parses or runs any of it. That is the
 * point of keeping it here rather than beside the explorer: instrumentation
 * that ships with the thing it instruments eventually gets called from it, and
 * then the measured code and the measuring code are the same code.
 *
 * Attaching is one call. What it installs:
 *
 * - a telemetry recorder, and the frame probe that feeds it — the host issues
 *   GPU timer queries only while a probe is attached, so the frame loop an
 *   unmeasured session runs is the one it always ran;
 * - the capture overlay, when the URL asked to record a gesture;
 * - `window.pointCloudScene` and `window.pointCloudRecorder`, the surface the
 *   replay benchmark drives the page through.
 */

import { createControlSurface } from "./controlSurface";
import { installRecorderOverlay, recordingRequested } from "./captureOverlay";
import { createInputRecorder, type InputRecorder } from "./inputRecorder";
import type { SceneHandle } from "./sceneHandle";
import { createSceneTelemetry } from "./telemetry";

export type { SceneHandle } from "./sceneHandle";
export type {
  HarnessActivity,
  HarnessDataset,
  HarnessDatasetControls,
} from "./sceneHandle";

const telemetryAutostart = (): boolean =>
  new URLSearchParams(window.location.search).get("telemetry") === "1";

export const attachHarness = (handle: SceneHandle): void => {
  const recorder: InputRecorder = createInputRecorder({
    viewer: handle.viewer,
    pose: () => ({
      position: [...handle.host.camera.getPosition()] as [
        number,
        number,
        number,
      ],
      focalPoint: [...handle.host.camera.getFocalPoint()] as [
        number,
        number,
        number,
      ],
      viewUp: [...handle.host.camera.getViewUp()] as [number, number, number],
      viewAngle: handle.host.camera.getViewAngle(),
      parallelScale: handle.host.camera.getParallelScale(),
      parallelProjection: !!handle.host.camera.getParallelProjection(),
    }),
    environment: () => telemetry.environment(),
  });

  const telemetry = createSceneTelemetry({
    host: handle.host,
    members: () =>
      handle.datasets().map((dataset) => ({
        id: dataset.id,
        label: dataset.label,
        stats: dataset.stats(),
      })),
    settings: () => ({ preset: handle.preset, ...handle.qualityTargets() }),
  });

  if (recordingRequested()) {
    installRecorderOverlay(recorder, { telemetry });
  }
  if (telemetryAutostart()) telemetry.start();

  Object.assign(window, {
    pointCloudRecorder: recorder,
    pointCloudScene: createControlSurface(handle, telemetry),
  });
};
