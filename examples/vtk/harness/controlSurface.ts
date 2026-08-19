/**
 * The automation surface a benchmark drives the page through.
 *
 * Every setting is applied through the control a hand would use, so a swept
 * configuration and a clicked one take the same path through the page. A
 * benchmark that wrote to a member's config directly would measure a code path
 * no user reaches, and would keep passing after the control that reaches it
 * broke.
 */

import type { SceneTelemetry } from "./telemetry";
import type { HarnessDataset, SceneHandle } from "./sceneHandle";

const required = <T>(
  dataset: HarnessDataset,
  control: T | undefined,
  kind: string,
): T => {
  if (control === undefined) {
    throw new Error(`${dataset.id} is not a ${kind} dataset`);
  }
  return control;
};

export const createControlSurface = (
  handle: SceneHandle,
  telemetry: SceneTelemetry,
) => {
  const { host } = handle;

  const datasetById = (id: string): HarnessDataset => {
    const found = handle.datasets().find((candidate) => candidate.id === id);
    if (!found) throw new Error(`no dataset ${id} in this scene`);
    return found;
  };

  return {
    preset: handle.preset,
    datasets: () =>
      handle.datasets().map(({ id, kind, label }) => ({ id, kind, label })),
    /** Coordinator and per-member numbers, as the diagnostics panel reads them. */
    stats: () => ({
      coordinator: host.coordinator.stats(),
      members: handle.datasets().map((dataset) => ({
        id: dataset.id,
        kind: dataset.kind,
        label: dataset.label,
        stats: dataset.stats(),
      })),
      lastFrameMs: host.lastFrameMs(),
      paints: host.paintCount(),
      loading: handle.loading(),
      error: handle.error(),
    }),
    /**
     * What each dataset is busy with. Convergence is every dataset settled
     * with nothing loading — the same condition the page's own activity light
     * shows, rather than a second definition beside it.
     */
    activity: () => ({
      loading: handle.loading(),
      datasets: handle.datasets().map((dataset) => ({
        id: dataset.id,
        ...dataset.activity(),
      })),
    }),
    viewport: () => ({
      width: handle.viewer.clientWidth,
      height: handle.viewer.clientHeight,
    }),
    camera: {
      read: () => ({
        position: [...host.camera.getPosition()],
        focalPoint: [...host.camera.getFocalPoint()],
        viewUp: [...host.camera.getViewUp()],
        parallelScale: host.camera.getParallelScale(),
        viewAngle: host.camera.getViewAngle(),
        parallelProjection: !!host.camera.getParallelProjection(),
      }),
      place: (next: {
        position?: readonly number[];
        focalPoint?: readonly number[];
        viewUp?: readonly number[];
      }) => {
        if (next.position) host.camera.setPosition(...next.position);
        if (next.focalPoint) host.camera.setFocalPoint(...next.focalPoint);
        if (next.viewUp) host.camera.setViewUp(...next.viewUp);
        host.scheduleRender();
      },
      reset: () => handle.resetCamera(),
    },
    settings: {
      read: () => handle.qualityTargets(),
      setQualityTargets: (targets: {
        interactionTargetMs?: number;
        stationaryTargetMs?: number;
      }) => handle.setQualityTargets(targets),
      setVisible: (id: string, visible: boolean) =>
        datasetById(id).controls.setVisible(visible),
      setScreenSpaceErrorPx: (id: string, px: number) => {
        const dataset = datasetById(id);
        required(
          dataset,
          dataset.controls.setScreenSpaceErrorPx,
          "3D Tiles",
        )(px);
      },
      setBudgetMode: (id: string, mode: "adaptive" | "fixed") => {
        const dataset = datasetById(id);
        required(dataset, dataset.controls.setBudgetMode, "point-cloud")(mode);
      },
      setFixedPointBudget: (id: string, points: number) => {
        const dataset = datasetById(id);
        required(
          dataset,
          dataset.controls.setFixedPointBudget,
          "point-cloud",
        )(points);
      },
      setMaximumPoints: (id: string, points: number | null) => {
        const dataset = datasetById(id);
        required(
          dataset,
          dataset.controls.setMaximumPoints,
          "point-cloud",
        )(points);
      },
      setPointSize: (id: string, mode: "auto" | "fixed", value: number) => {
        const dataset = datasetById(id);
        required(
          dataset,
          dataset.controls.setPointSize,
          "point-cloud",
        )(mode, value);
      },
    },
    telemetry: {
      start: () => telemetry.start(),
      stop: () => telemetry.stop(),
      clear: () => telemetry.clear(),
      isActive: () => telemetry.isActive(),
      mark: (label: string) => telemetry.mark(label),
      environment: () => telemetry.environment(),
      summary: () => telemetry.summary(),
      trace: () => telemetry.trace(),
      download: () => telemetry.download(),
      gpuTimingSupported: () => host.gpuTimingSupported(),
    },
    render: () => host.scheduleRender(),
    needsFrame: () => host.coordinator.needsFrame(),
  };
};
