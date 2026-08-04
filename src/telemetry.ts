/**
 * A record of what a session actually did: work spans, per-frame timings with
 * their asynchronous GPU resolution, and the machine that produced them.
 *
 * Host-independent — nothing here knows about vtk.js, the controller, or the
 * governor. A host reports events and the recorder assembles a downloadable
 * trace. The environment capture reads a WebGL context only to name the
 * renderer and flag the software rasterizers whose timings are not evidence
 * of anything.
 */

export type TelemetryEnvironment = {
  readonly capturedAt: string;
  readonly userAgent: string;
  readonly platform: string;
  readonly logicalProcessors: number | null;
  readonly deviceMemoryGiB: number | null;
  readonly viewport: {
    readonly widthCssPx: number;
    readonly heightCssPx: number;
    readonly devicePixelRatio: number;
  };
  readonly webgl: {
    readonly version: string | null;
    readonly vendor: string | null;
    readonly renderer: string | null;
    readonly unmaskedVendor: string | null;
    readonly unmaskedRenderer: string | null;
    readonly softwareRenderer: boolean;
    readonly timerQuerySupported: boolean;
  };
};

type TelemetryBaseEvent = {
  readonly sequence: number;
  readonly atMs: number;
};

export type TelemetrySessionEvent = TelemetryBaseEvent & {
  readonly type: "session";
  readonly phase: "start" | "stop";
};

export type TelemetryWorkKind =
  | "source-open"
  | "hierarchy"
  | "tile-load"
  | "renderer-batch";

export type TelemetryDetail = Readonly<
  Record<string, string | number | boolean | null>
>;

export type TelemetryWorkEvent = TelemetryBaseEvent & {
  readonly type: "work";
  readonly workId: number;
  readonly kind: TelemetryWorkKind;
  readonly phase: "start" | "finish";
  readonly revision: number;
  readonly pendingWork: number;
  readonly durationMs: number | null;
  readonly status: "running" | "ok" | "error" | "cancelled";
  readonly detail: TelemetryDetail;
};

export type TelemetryFrameEvent = TelemetryBaseEvent & {
  readonly type: "frame";
  readonly presentedAtMs: number;
  readonly rafIntervalMs: number;
  readonly vtkCpuMs: number;
  readonly governorFrameMs: number;
  readonly gpuMs: number | null;
  readonly gpuStatus:
    | "unsupported"
    | "pending"
    | "valid"
    | "disjoint"
    | "error";
  /** True when this measurement was handed to the governor. */
  readonly reportedToGovernor: boolean;
  readonly capacitySampleEligible: boolean;
  readonly capacitySampleStatus: "pending" | "accepted" | "rejected";
  readonly workRevision: number;
  readonly pendingWork: number;
  readonly clean: boolean;
  readonly contamination: readonly string[];
  readonly state: unknown;
};

export type TelemetryStateEvent = TelemetryBaseEvent & {
  readonly type: "state";
  readonly reason: string;
  readonly state: unknown;
};

export type TelemetryLongTaskEvent = TelemetryBaseEvent & {
  readonly type: "long-task";
  readonly startedAtMs: number;
  readonly durationMs: number;
};

export type TelemetryEvent =
  | TelemetrySessionEvent
  | TelemetryWorkEvent
  | TelemetryFrameEvent
  | TelemetryStateEvent
  | TelemetryLongTaskEvent;

export type TelemetrySummary = {
  readonly active: boolean;
  readonly events: number;
  readonly droppedEvents: number;
  readonly frames: number;
  readonly cleanFrames: number;
  readonly contaminatedFrames: number;
  readonly acceptedCapacitySamples: number;
  readonly rejectedCapacitySamples: number;
  readonly pendingCapacitySamples: number;
  readonly workEvents: number;
  readonly longTasks: number;
  readonly durationMs: number;
  readonly pendingWork: number;
  readonly workRevision: number;
  readonly environment: TelemetryEnvironment;
};

export type TelemetryTrace = {
  readonly schemaVersion: 1;
  readonly startedAt: string;
  readonly stoppedAt: string | null;
  readonly environment: TelemetryEnvironment;
  readonly summary: TelemetrySummary;
  readonly events: readonly TelemetryEvent[];
};

export type TelemetryWorkFinish = (
  status?: "ok" | "error" | "cancelled",
  detail?: TelemetryDetail,
) => void;

export type TelemetryRecorder = {
  start(): void;
  stop(): void;
  clear(): void;
  dispose(): void;
  isActive(): boolean;
  beginWork(
    kind: TelemetryWorkKind,
    detail?: TelemetryDetail,
  ): TelemetryWorkFinish;
  recordFrame(input: {
    readonly presentedAtMs: number;
    readonly rafIntervalMs: number;
    readonly vtkCpuMs: number;
    readonly governorFrameMs: number;
    readonly gpuMs?: number | null;
    readonly gpuPending?: boolean;
    readonly reportedToGovernor: boolean;
    readonly capacitySampleEligible?: boolean;
    readonly capacitySamplePending?: boolean;
    readonly contamination?: readonly string[];
    readonly state: unknown;
  }): TelemetryFrameEvent | null;
  resolveGpuFrame(
    frame: TelemetryFrameEvent,
    result: {
      readonly status: "valid" | "disjoint" | "error";
      readonly gpuMs: number | null;
    },
  ): void;
  recordState(reason: string, state: unknown): void;
  summary(): TelemetrySummary;
  trace(): TelemetryTrace;
  download(filename?: string): void;
};

type WebGlDebugRendererInfo = {
  readonly UNMASKED_VENDOR_WEBGL: number;
  readonly UNMASKED_RENDERER_WEBGL: number;
};

const textParameter = (
  gl: WebGLRenderingContext | WebGL2RenderingContext,
  parameter: number,
): string | null => {
  try {
    const value: unknown = gl.getParameter(parameter);
    return typeof value === "string" && value.length > 0 ? value : null;
  } catch {
    return null;
  }
};

export const isSoftwareRenderer = (
  ...descriptions: readonly (string | null)[]
): boolean =>
  descriptions.some((description) =>
    /swiftshader|llvmpipe|softpipe|lavapipe|software rasterizer|software renderer/i.test(
      description ?? "",
    ),
  );

export const captureTelemetryEnvironment = (
  gl: WebGLRenderingContext | WebGL2RenderingContext | null,
): TelemetryEnvironment => {
  const nav = navigator as Navigator & { readonly deviceMemory?: number };
  const debug = gl?.getExtension(
    "WEBGL_debug_renderer_info",
  ) as WebGlDebugRendererInfo | null;
  const vendor = gl === null ? null : textParameter(gl, gl.VENDOR);
  const renderer = gl === null ? null : textParameter(gl, gl.RENDERER);
  const version = gl === null ? null : textParameter(gl, gl.VERSION);
  const unmaskedVendor =
    gl === null || debug === null
      ? null
      : textParameter(gl, debug.UNMASKED_VENDOR_WEBGL);
  const unmaskedRenderer =
    gl === null || debug === null
      ? null
      : textParameter(gl, debug.UNMASKED_RENDERER_WEBGL);

  return {
    capturedAt: new Date().toISOString(),
    userAgent: nav.userAgent,
    platform: nav.platform,
    logicalProcessors:
      Number.isFinite(nav.hardwareConcurrency) && nav.hardwareConcurrency > 0
        ? nav.hardwareConcurrency
        : null,
    deviceMemoryGiB:
      Number.isFinite(nav.deviceMemory) && (nav.deviceMemory ?? 0) > 0
        ? nav.deviceMemory!
        : null,
    viewport: {
      widthCssPx: window.innerWidth,
      heightCssPx: window.innerHeight,
      devicePixelRatio: window.devicePixelRatio,
    },
    webgl: {
      version,
      vendor,
      renderer,
      unmaskedVendor,
      unmaskedRenderer,
      softwareRenderer: isSoftwareRenderer(
        vendor,
        renderer,
        unmaskedVendor,
        unmaskedRenderer,
      ),
      timerQuerySupported:
        gl !== null &&
        gl.getExtension("EXT_disjoint_timer_query_webgl2") !== null,
    },
  };
};

const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T;

const mergedDetail = (
  initial: TelemetryDetail,
  final: TelemetryDetail,
): TelemetryDetail => ({ ...initial, ...final });

const filenameTimestamp = (date: Date): string =>
  date.toISOString().replaceAll(":", "-").replaceAll(".", "-");

export const createTelemetryRecorder = (options: {
  readonly environment: () => TelemetryEnvironment;
  readonly maxEvents?: number;
  readonly now?: () => number;
  readonly wallNow?: () => Date;
}): TelemetryRecorder => {
  const maxEvents = Math.max(1, Math.floor(options.maxEvents ?? 20_000));
  const now = options.now ?? (() => performance.now());
  const wallNow = options.wallNow ?? (() => new Date());
  let environment = options.environment();
  let active = false;
  let events: TelemetryEvent[] = [];
  let droppedEvents = 0;
  let sequence = 0;
  let workId = 0;
  let workRevision = 0;
  let pendingWork = 0;
  let lastFrameRevision: number | null = null;
  let startedAt = wallNow();
  let startedAtMs = now();
  let stoppedAt: Date | null = null;
  let longTaskObserver: PerformanceObserver | null = null;
  let recordingGeneration = 0;

  const append = <T extends TelemetryEvent>(event: T): T => {
    if (events.length === maxEvents) {
      events.shift();
      droppedEvents += 1;
    }
    events.push(event);
    return event;
  };

  const baseEvent = (): TelemetryBaseEvent => ({
    sequence: ++sequence,
    atMs: Math.max(0, now() - startedAtMs),
  });

  const observeLongTasks = (): void => {
    if (
      typeof PerformanceObserver === "undefined" ||
      !PerformanceObserver.supportedEntryTypes?.includes("longtask")
    ) {
      return;
    }
    longTaskObserver = new PerformanceObserver((list) => {
      if (!active) return;
      for (const entry of list.getEntries()) {
        if (entry.startTime < startedAtMs) continue;
        append({
          ...baseEvent(),
          type: "long-task",
          startedAtMs: entry.startTime - startedAtMs,
          durationMs: entry.duration,
        });
      }
    });
    longTaskObserver.observe({ type: "longtask", buffered: true });
  };

  const reset = (): void => {
    recordingGeneration += 1;
    longTaskObserver?.disconnect();
    longTaskObserver = null;
    events = [];
    droppedEvents = 0;
    sequence = 0;
    workId = 0;
    workRevision = 0;
    pendingWork = 0;
    lastFrameRevision = null;
    startedAt = wallNow();
    startedAtMs = now();
    stoppedAt = null;
    environment = options.environment();
  };

  const summary = (): TelemetrySummary => {
    let frames = 0;
    let cleanFrames = 0;
    let workEvents = 0;
    let longTasks = 0;
    let acceptedCapacitySamples = 0;
    let rejectedCapacitySamples = 0;
    let pendingCapacitySamples = 0;
    for (const event of events) {
      if (event.type === "frame") {
        frames += 1;
        if (event.clean) cleanFrames += 1;
        if (event.capacitySampleStatus === "accepted") {
          acceptedCapacitySamples += 1;
        } else if (event.capacitySampleStatus === "rejected") {
          rejectedCapacitySamples += 1;
        } else {
          pendingCapacitySamples += 1;
        }
      } else if (event.type === "work") {
        workEvents += 1;
      } else if (event.type === "long-task") {
        longTasks += 1;
      }
    }
    return {
      active,
      events: events.length,
      droppedEvents,
      frames,
      cleanFrames,
      contaminatedFrames: frames - cleanFrames,
      acceptedCapacitySamples,
      rejectedCapacitySamples,
      pendingCapacitySamples,
      workEvents,
      longTasks,
      durationMs: Math.max(
        0,
        (stoppedAt?.getTime() ?? wallNow().getTime()) - startedAt.getTime(),
      ),
      pendingWork,
      workRevision,
      environment: clone(environment),
    };
  };

  const trace = (): TelemetryTrace =>
    clone({
      schemaVersion: 1 as const,
      startedAt: startedAt.toISOString(),
      stoppedAt: stoppedAt?.toISOString() ?? null,
      environment,
      summary: summary(),
      events,
    });

  const recorder: TelemetryRecorder = {
    start() {
      if (active) return;
      reset();
      active = true;
      append({ ...baseEvent(), type: "session", phase: "start" });
      observeLongTasks();
    },

    stop() {
      if (!active) return;
      append({ ...baseEvent(), type: "session", phase: "stop" });
      active = false;
      stoppedAt = wallNow();
      longTaskObserver?.disconnect();
      longTaskObserver = null;
    },

    clear() {
      const wasActive = active;
      reset();
      active = wasActive;
      if (active) {
        append({ ...baseEvent(), type: "session", phase: "start" });
        observeLongTasks();
      }
    },

    dispose() {
      recorder.stop();
      longTaskObserver?.disconnect();
      longTaskObserver = null;
    },

    isActive: () => active,

    beginWork(kind, detail = {}) {
      if (!active) return () => {};
      const id = ++workId;
      const generation = recordingGeneration;
      const beganAt = now();
      workRevision += 1;
      pendingWork += 1;
      append({
        ...baseEvent(),
        type: "work",
        workId: id,
        kind,
        phase: "start",
        revision: workRevision,
        pendingWork,
        durationMs: null,
        status: "running",
        detail,
      });
      let finished = false;
      return (status = "ok", finalDetail = {}) => {
        if (finished) return;
        finished = true;
        if (!active || generation !== recordingGeneration) return;
        pendingWork = Math.max(0, pendingWork - 1);
        workRevision += 1;
        append({
          ...baseEvent(),
          type: "work",
          workId: id,
          kind,
          phase: "finish",
          revision: workRevision,
          pendingWork,
          durationMs: Math.max(0, now() - beganAt),
          status,
          detail: mergedDetail(detail, finalDetail),
        });
      };
    },

    recordFrame(input) {
      if (!active) return null;
      const contamination = new Set(input.contamination ?? []);
      if (lastFrameRevision === null) contamination.add("first-frame");
      else if (lastFrameRevision !== workRevision)
        contamination.add("streaming-work-overlap");
      if (pendingWork > 0) contamination.add("streaming-work-pending");
      const event = append({
        ...baseEvent(),
        type: "frame",
        presentedAtMs: input.presentedAtMs - startedAtMs,
        rafIntervalMs: input.rafIntervalMs,
        vtkCpuMs: input.vtkCpuMs,
        governorFrameMs: input.governorFrameMs,
        gpuMs: input.gpuMs ?? null,
        gpuStatus:
          input.gpuMs !== undefined && input.gpuMs !== null
            ? "valid"
            : input.gpuPending
              ? "pending"
              : "unsupported",
        reportedToGovernor: input.reportedToGovernor,
        capacitySampleEligible: input.capacitySampleEligible ?? false,
        capacitySampleStatus: input.capacitySamplePending
          ? "pending"
          : input.capacitySampleEligible
            ? "accepted"
            : "rejected",
        workRevision,
        pendingWork,
        clean: contamination.size === 0,
        contamination: [...contamination].sort(),
        state: input.state,
      });
      lastFrameRevision = workRevision;
      return event;
    },

    resolveGpuFrame(frame, result) {
      const index = events.indexOf(frame);
      if (index < 0) return;
      const event = events[index];
      if (event?.type !== "frame") return;
      events[index] = {
        ...event,
        gpuMs: result.status === "valid" ? result.gpuMs : null,
        gpuStatus: result.status,
        capacitySampleStatus:
          event.capacitySampleStatus !== "pending"
            ? event.capacitySampleStatus
            : event.capacitySampleEligible && result.status === "valid"
              ? "accepted"
              : "rejected",
      };
    },

    recordState(reason, state) {
      if (!active) return;
      append({ ...baseEvent(), type: "state", reason, state });
    },

    summary,
    trace,

    download(filename) {
      const documentRef = document;
      const urlRef = URL;
      const blob = new Blob([JSON.stringify(trace(), null, 2)], {
        type: "application/json",
      });
      const url = urlRef.createObjectURL(blob);
      const anchor = documentRef.createElement("a");
      anchor.href = url;
      anchor.download =
        filename ?? `pointcloud-telemetry-${filenameTimestamp(wallNow())}.json`;
      anchor.hidden = true;
      documentRef.body.append(anchor);
      anchor.click();
      anchor.remove();
      setTimeout(() => urlRef.revokeObjectURL(url), 0);
    },
  };

  return recorder;
};
