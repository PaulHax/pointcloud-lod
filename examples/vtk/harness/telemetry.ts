/**
 * Telemetry for the streamed-scene examples.
 *
 * The point-cloud example records its own trace because it owns its governor
 * and drives its own frame loop. The scene examples do not: the coordinator
 * owns quality, and `SceneHost` owns the loop. So this attaches to the host's
 * frame probe instead of rebuilding a loop beside it, and produces the same
 * `TelemetryTrace` the point example writes — one schema, one analyzer.
 *
 * Nothing here is fed back into the coordinator. GPU durations are recorded as
 * observations only, while the governor keeps learning from exactly the host
 * frame intervals it always did, so a trace describes a session that behaved
 * as it would have with no recorder attached.
 */

import {
  captureTelemetryEnvironment,
  createTelemetryRecorder,
  type TelemetryEnvironment,
  type TelemetryFrameEvent,
  type TelemetryRecorder,
  type TelemetrySummary,
  type TelemetryTrace,
} from "./telemetryRecorder";
import type { PointCloudMemberStats } from "../../../src/pointCloudMember";
import type { Tiles3dMemberStats } from "../../../src/tiles3d/memberTypes";
import type {
  FrameProbe,
  FrameReport,
  GpuFrameResolution,
  SceneHost,
} from "../scene/host";

export type SceneMemberSnapshot = {
  readonly id: string;
  readonly label: string;
  readonly stats: PointCloudMemberStats | Tiles3dMemberStats;
};

export type SceneTelemetry = {
  start(): void;
  stop(): void;
  clear(): void;
  isActive(): boolean;
  mark(label: string): void;
  environment(): TelemetryEnvironment;
  summary(): TelemetrySummary;
  trace(): TelemetryTrace;
  download(filename?: string): void;
  /** Record a labelled state snapshot, for a load or a setting change. */
  note(reason: string): void;
};

/**
 * Why this presentation cannot describe steady-state cost.
 *
 * A frame drawn while tiles were decoding, uploading, or being swapped into
 * the renderer measures that work, not the cost of the view. The coordinator's
 * own activity gate covers the governor's needs; this is the finer-grained
 * account an analysis needs to say *what* a slow frame was busy with.
 */
const contaminationOf = (members: readonly SceneMemberSnapshot[]): string[] => {
  if (members.length === 0) return ["no-members"];
  const reasons: string[] = [];
  for (const { id, stats } of members) {
    if (stats.kind === "pointCloud") {
      const controller = stats.controller;
      if (controller.physicalTileOperations > 0) reasons.push(`${id}:tiles`);
      if (controller.physicalHierarchyOperations > 0)
        reasons.push(`${id}:hierarchy`);
      if (controller.queuedTiles > 0 || controller.queuedPages > 0)
        reasons.push(`${id}:queued`);
      if (controller.selectionPending) reasons.push(`${id}:selecting`);
      continue;
    }
    if (stats.sourceState === "loading") reasons.push(`${id}:source`);
    if ((stats.queue?.active ?? 0) > 0) reasons.push(`${id}:fetching`);
    if ((stats.queue?.queued ?? 0) + (stats.queue?.retrying ?? 0) > 0)
      reasons.push(`${id}:queued`);
    if ((stats.decode?.activeJobs ?? 0) + (stats.decode?.queuedJobs ?? 0) > 0)
      reasons.push(`${id}:decoding`);
    if (stats.submissions.queuedJobs > 0 || stats.renderer.pendingJobs > 0)
      reasons.push(`${id}:submitting`);
  }
  return reasons;
};

/**
 * The per-member numbers a frame is worth storing with.
 *
 * The full member stats carry per-tile entry lists, which at one snapshot per
 * frame would dominate a trace and describe queue membership rather than what
 * was drawn. What survives is what an analysis compares across runs: the
 * detail delivered, the memory it took, and the work still owed.
 */
const memberSnapshot = (
  member: SceneMemberSnapshot,
): Record<string, unknown> => {
  const { id, label, stats } = member;
  if (stats.kind === "pointCloud") {
    return {
      id,
      label,
      kind: "pointCloud",
      adaptive: stats.adaptive,
      allocation: stats.allocation,
      drawnTiles: stats.renderer.drawnTiles,
      drawnPoints: stats.renderer.drawnPoints,
      submittedPoints: stats.renderer.submittedPoints,
      gpuResidentBytes: stats.renderer.gpuResidentBytes,
      rendererWorkRevision: stats.renderer.workRevision,
      densityFraction: stats.controller.densityFraction,
      pointBudget: stats.controller.pointBudget,
      selection: stats.controller.selection,
      queuedTiles: stats.controller.queuedTiles,
      queuedPages: stats.controller.queuedPages,
      physicalTileOperations: stats.controller.physicalTileOperations,
      physicalHierarchyOperations: stats.controller.physicalHierarchyOperations,
      selectionPending: stats.controller.selectionPending,
      workRevision: stats.controller.workRevision,
    };
  }
  return {
    id,
    label,
    kind: "tiles3d",
    active: stats.active,
    allocation: stats.allocation,
    sourceState: stats.sourceState,
    irreducibleBudget: stats.irreducibleBudget,
    maximumScreenSpaceErrorPx: stats.maximumScreenSpaceErrorPx,
    effectiveScreenSpaceErrorPx: stats.effectiveScreenSpaceErrorPx,
    sseMultiplier: stats.sseMultiplier,
    memoryConstrained: stats.memoryConstrained,
    selectedTiles: stats.selectedTiles,
    requestedTiles: stats.requestedTiles,
    drawnTiles: stats.renderer.drawnTiles,
    drawnTriangles: stats.renderer.drawnTriangles,
    residentGeometryBytes: stats.renderer.residentGeometryBytes,
    residentTextureBytes: stats.renderer.residentTextureBytes,
    pendingJobs: stats.renderer.pendingJobs,
    queuedSubmissionBytes: stats.submissions.queuedBytes,
    queue:
      stats.queue === null || stats.queue === undefined
        ? null
        : {
            selected: stats.queue.selected,
            active: stats.queue.active,
            queued: stats.queue.queued,
            retrying: stats.queue.retrying,
            ready: stats.queue.ready,
            failed: stats.queue.failed,
            decodedBytes: stats.queue.decodedBytes,
            cacheHits: stats.queue.cacheHits,
            cacheMisses: stats.queue.cacheMisses,
            cacheEvictions: stats.queue.cacheEvictions,
            cacheRevisits: stats.queue.cacheRevisits,
            workPending: stats.queue.workPending,
          },
    decode: stats.decode ?? null,
    errorCount: stats.errorCount,
  };
};

export const createSceneTelemetry = (options: {
  readonly host: SceneHost;
  /** Every member currently in the scene, in panel order. */
  readonly members: () => readonly SceneMemberSnapshot[];
  /** Page-level settings worth carrying into the trace. */
  readonly settings?: () => Record<string, unknown>;
}): SceneTelemetry => {
  const environment = (): TelemetryEnvironment =>
    captureTelemetryEnvironment(options.host.glContext());

  const recorder: TelemetryRecorder = createTelemetryRecorder({ environment });
  const pendingGpuFrames = new Map<number, TelemetryFrameEvent>();

  const state = (): unknown => {
    const members = options.members();
    return {
      coordinator: options.host.coordinator.stats(),
      camera: options.host.cameraView(),
      devicePixelRatio: options.host.devicePixelRatio(),
      rendererName: options.host.rendererName,
      settings: options.settings?.() ?? {},
      members: members.map(memberSnapshot),
    };
  };

  const probe: FrameProbe = {
    onFrame: (report: FrameReport): void => {
      const members = options.members();
      const frame = recorder.recordFrame({
        presentedAtMs: report.presentedAtMs,
        // A rejected interval is reported as the frame's own cost, which is
        // the only duration that frame is known to have taken.
        rafIntervalMs: report.hostFrameMs ?? report.vtkFrameMs,
        vtkCpuMs: report.vtkFrameMs,
        governorFrameMs: report.hostFrameMs ?? report.vtkFrameMs,
        gpuPending: report.gpuPending,
        reportedToGovernor: report.reportedToGovernor,
        capacitySampleEligible: report.capacitySampleEligible,
        // The coordinator, not this recorder, decides what the governor keeps,
        // so a sample is never left pending on a GPU result nothing consumes.
        capacitySamplePending: false,
        contamination: [
          ...contaminationOf(members),
          ...(report.reportedToGovernor ? [] : ["interval-rejected"]),
        ],
        state: state(),
      });
      if (frame !== null && report.gpuPending) {
        pendingGpuFrames.set(report.id, frame);
      }
    },
    onGpuResolved: (resolution: GpuFrameResolution): void => {
      const frame = pendingGpuFrames.get(resolution.id);
      if (frame === undefined) return;
      pendingGpuFrames.delete(resolution.id);
      recorder.resolveGpuFrame(frame, {
        status: resolution.status,
        gpuMs: resolution.gpuMs,
      });
    },
  };

  return {
    start() {
      if (recorder.isActive()) return;
      recorder.start();
      pendingGpuFrames.clear();
      options.host.setFrameProbe(probe);
      recorder.recordState("recording-started", state());
      options.host.scheduleRender();
    },
    stop() {
      if (!recorder.isActive()) return;
      recorder.recordState("recording-stopped", state());
      options.host.setFrameProbe(null);
      pendingGpuFrames.clear();
      recorder.stop();
    },
    clear() {
      recorder.clear();
      pendingGpuFrames.clear();
      if (recorder.isActive())
        recorder.recordState("recording-cleared", state());
    },
    isActive: () => recorder.isActive(),
    mark(label) {
      const normalized = label.trim().slice(0, 128);
      if (normalized.length === 0 || !recorder.isActive()) return;
      recorder.recordState(`marker:${normalized}`, state());
    },
    note(reason) {
      if (!recorder.isActive()) return;
      recorder.recordState(reason, state());
    },
    environment: () => recorder.summary().environment,
    summary: () => recorder.summary(),
    trace: () => recorder.trace(),
    download: (filename) => recorder.download(filename),
  };
};
