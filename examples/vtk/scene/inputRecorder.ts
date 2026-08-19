/**
 * Capture of a real pointer gesture, so a benchmark can replay one instead of
 * synthesising a camera path.
 *
 * A synthesised path is a straight drag at a constant rate over a target the
 * driver picked. A hand does none of those things: it accelerates, overshoots,
 * pauses mid-orbit, and aims at whatever looked interesting. Those are the
 * inputs that make streaming decisions hard, and none of them survive being
 * described in code.
 *
 * What is recorded is the *input*, not the camera. The camera a gesture
 * produces depends on how many animation frames the interactor got through it,
 * which is exactly the thing under measurement — so replaying poses would
 * erase the effect being measured. The poses are recorded alongside as a
 * reference track: the replay compares against them and reports the drift, so
 * a run whose camera wandered somewhere else is visible as such rather than
 * being quietly averaged into the result.
 *
 * Host-independent: it needs a viewer element to watch and a way to read the
 * camera. It knows nothing about vtk.js, the coordinator, or the controller.
 */

import type { TelemetryEnvironment } from "../../../src/telemetry";

export const INPUT_RECORDING_SCHEMA_VERSION = 1;

/** The camera state the replay compares its own camera against. */
export type RecordedPose = {
  readonly position: readonly [number, number, number];
  readonly focalPoint: readonly [number, number, number];
  readonly viewUp: readonly [number, number, number];
  readonly viewAngle: number;
  readonly parallelScale: number;
  readonly parallelProjection: boolean;
};

export type RecordedPoseSample = RecordedPose & {
  /** Milliseconds since the recording started. */
  readonly atMs: number;
};

type RecordedEventBase = {
  readonly atMs: number;
  /**
   * CSS pixels from the viewer's top-left, not the page's. The replay window
   * has a differently sized panel beside it, and the recording is meaningless
   * if a gesture aimed at a building lands on the control column instead.
   */
  readonly x: number;
  readonly y: number;
  /** The DOM `buttons` bitmask, so a replay knows what was held mid-drag. */
  readonly buttons: number;
  readonly shiftKey: boolean;
  readonly ctrlKey: boolean;
  readonly altKey: boolean;
  readonly metaKey: boolean;
};

export type RecordedPointerEvent = RecordedEventBase & {
  readonly type: "pointerdown" | "pointermove" | "pointerup" | "pointercancel";
  /** DOM `button`: 0 left, 1 middle, 2 right. -1 when no button changed. */
  readonly button: number;
};

export type RecordedWheelEvent = RecordedEventBase & {
  readonly type: "wheel";
  readonly deltaX: number;
  readonly deltaY: number;
  /** 0 pixels, 1 lines, 2 pages — a trackpad and a wheel report differently. */
  readonly deltaMode: number;
};

/**
 * A viewer resize mid-recording. Every coordinate before it was aimed at a
 * differently sized view, so a replay cannot honour both; it is recorded so
 * the mismatch is reported rather than silently replayed at the wrong scale.
 */
export type RecordedViewportEvent = {
  readonly type: "viewport";
  readonly atMs: number;
  readonly widthCssPx: number;
  readonly heightCssPx: number;
};

export type RecordedInputEvent =
  | RecordedPointerEvent
  | RecordedWheelEvent
  | RecordedViewportEvent;

export type RecordedMarker = {
  readonly atMs: number;
  readonly label: string;
};

export type InputRecording = {
  readonly schemaVersion: typeof INPUT_RECORDING_SCHEMA_VERSION;
  readonly recordedAt: string;
  readonly label: string;
  /**
   * The page URL at the moment recording started, query string included. Both
   * example shells keep their dataset selection in the address bar, so this is
   * what lets a replay reopen the same scene without a second description of
   * it that could disagree.
   */
  readonly href: string;
  readonly viewer: {
    readonly widthCssPx: number;
    readonly heightCssPx: number;
    readonly devicePixelRatio: number;
  };
  readonly environment: TelemetryEnvironment;
  /** Where the camera stood when the first event was recorded. */
  readonly startPose: RecordedPose;
  readonly events: readonly RecordedInputEvent[];
  readonly poses: readonly RecordedPoseSample[];
  readonly markers: readonly RecordedMarker[];
  readonly durationMs: number;
  /** True when the caps below discarded anything; the tail is not the whole gesture. */
  readonly truncated: boolean;
};

export type InputRecorderStatus = {
  readonly active: boolean;
  readonly events: number;
  readonly poses: number;
  readonly markers: number;
  readonly durationMs: number;
  readonly truncated: boolean;
  readonly viewerResized: boolean;
};

export type InputRecorder = {
  start(): void;
  stop(): void;
  clear(): void;
  isActive(): boolean;
  mark(label: string): void;
  status(): InputRecorderStatus;
  /** The recording so far. Safe to read while still recording. */
  recording(): InputRecording;
  download(filename?: string): void;
};

const DEFAULT_MAX_EVENTS = 200_000;
const DEFAULT_MAX_POSES = 60_000;

const filenameTimestamp = (date: Date): string =>
  date.toISOString().replaceAll(":", "-").replaceAll(".", "-");

const samePose = (left: RecordedPose, right: RecordedPose): boolean =>
  left.viewAngle === right.viewAngle &&
  left.parallelScale === right.parallelScale &&
  left.parallelProjection === right.parallelProjection &&
  ([0, 1, 2] as const).every(
    (axis) =>
      left.position[axis] === right.position[axis] &&
      left.focalPoint[axis] === right.focalPoint[axis] &&
      left.viewUp[axis] === right.viewUp[axis],
  );

export const createInputRecorder = (options: {
  readonly viewer: HTMLElement;
  readonly pose: () => RecordedPose;
  readonly environment: () => TelemetryEnvironment;
  /** Called whenever the counts a UI would show have changed. */
  readonly onChange?: () => void;
  readonly maxEvents?: number;
  readonly maxPoses?: number;
  readonly now?: () => number;
  readonly wallNow?: () => Date;
}): InputRecorder => {
  const maxEvents = Math.max(
    1,
    Math.floor(options.maxEvents ?? DEFAULT_MAX_EVENTS),
  );
  const maxPoses = Math.max(
    1,
    Math.floor(options.maxPoses ?? DEFAULT_MAX_POSES),
  );
  const now = options.now ?? ((): number => performance.now());
  const wallNow = options.wallNow ?? ((): Date => new Date());

  let active = false;
  let startedAtMs = 0;
  let startedAtWall: Date | null = null;
  let href = "";
  let label = "";
  let startPose: RecordedPose | null = null;
  let startViewer = { widthCssPx: 0, heightCssPx: 0, devicePixelRatio: 1 };
  let events: RecordedInputEvent[] = [];
  let poses: RecordedPoseSample[] = [];
  let markers: RecordedMarker[] = [];
  let lastPose: RecordedPose | null = null;
  let lastViewerSize = { width: 0, height: 0 };
  let viewerResized = false;
  let truncated = false;
  let poseFrame: number | null = null;

  const changed = (): void => options.onChange?.();

  const elapsed = (): number => (active ? now() - startedAtMs : lastEndMs);
  let lastEndMs = 0;

  const push = (event: RecordedInputEvent): void => {
    if (events.length >= maxEvents) {
      truncated = true;
      return;
    }
    events.push(event);
  };

  const viewerBox = (): DOMRect => options.viewer.getBoundingClientRect();

  const base = (
    event: MouseEvent,
    box: DOMRect,
  ): RecordedEventBase & { readonly atMs: number } => ({
    atMs: now() - startedAtMs,
    x: event.clientX - box.left,
    y: event.clientY - box.top,
    buttons: event.buttons,
    shiftKey: event.shiftKey,
    ctrlKey: event.ctrlKey,
    altKey: event.altKey,
    metaKey: event.metaKey,
  });

  const recordPointer = (event: PointerEvent): void => {
    if (!active) return;
    push({
      ...base(event, viewerBox()),
      type: event.type as RecordedPointerEvent["type"],
      button: event.button,
    });
    // A press or release is a gesture boundary, and the pose either side of it
    // is what a replay has to line up with. The pose loop samples on its own
    // frame, which can fall on the wrong side of one.
    samplePose();
  };

  const recordWheel = (event: WheelEvent): void => {
    if (!active) return;
    push({
      ...base(event, viewerBox()),
      type: "wheel",
      deltaX: event.deltaX,
      deltaY: event.deltaY,
      deltaMode: event.deltaMode,
    });
  };

  const samplePose = (): void => {
    if (!active) return;
    const pose = options.pose();
    if (lastPose !== null && samePose(pose, lastPose)) return;
    if (poses.length >= maxPoses) {
      truncated = true;
      return;
    }
    lastPose = pose;
    poses.push({ ...pose, atMs: now() - startedAtMs });
  };

  const checkViewerSize = (): void => {
    const box = viewerBox();
    const width = Math.round(box.width);
    const height = Math.round(box.height);
    if (width === lastViewerSize.width && height === lastViewerSize.height) {
      return;
    }
    lastViewerSize = { width, height };
    viewerResized = true;
    push({
      type: "viewport",
      atMs: now() - startedAtMs,
      widthCssPx: width,
      heightCssPx: height,
    });
  };

  /**
   * The pose track is sampled on the browser's own frame clock rather than on
   * the render loop's, so a frame the page skipped still leaves a gap the
   * replay comparison can see. Poses that did not change are not stored.
   */
  const poseLoop = (): void => {
    poseFrame = requestAnimationFrame(() => {
      if (!active) return;
      checkViewerSize();
      samplePose();
      poseLoop();
    });
  };

  // Capture phase, so nothing the page or vtk.js does with the event can keep
  // it from being recorded; passive, so recording can never change how a
  // gesture behaves — a recording that alters the thing it records is not one.
  const listen = { capture: true, passive: true } as const;
  for (const type of [
    "pointerdown",
    "pointermove",
    "pointerup",
    "pointercancel",
  ] as const) {
    options.viewer.addEventListener(
      type,
      recordPointer as EventListener,
      listen,
    );
  }
  options.viewer.addEventListener(
    "wheel",
    recordWheel as EventListener,
    listen,
  );

  const recorder: InputRecorder = {
    start() {
      if (active) return;
      active = true;
      startedAtMs = now();
      startedAtWall = wallNow();
      href = window.location.href;
      label = document.title;
      const box = viewerBox();
      startViewer = {
        widthCssPx: Math.round(box.width),
        heightCssPx: Math.round(box.height),
        devicePixelRatio: window.devicePixelRatio,
      };
      lastViewerSize = {
        width: startViewer.widthCssPx,
        height: startViewer.heightCssPx,
      };
      startPose = options.pose();
      lastPose = null;
      samplePose();
      poseLoop();
      changed();
    },
    stop() {
      if (!active) return;
      samplePose();
      lastEndMs = now() - startedAtMs;
      active = false;
      if (poseFrame !== null) cancelAnimationFrame(poseFrame);
      poseFrame = null;
      changed();
    },
    clear() {
      events = [];
      poses = [];
      markers = [];
      lastPose = null;
      truncated = false;
      viewerResized = false;
      lastEndMs = 0;
      if (active) {
        startedAtMs = now();
        startedAtWall = wallNow();
        startPose = options.pose();
        samplePose();
      }
      changed();
    },
    isActive: () => active,
    mark(text) {
      const normalized = text.trim().slice(0, 128);
      if (normalized.length === 0 || !active) return;
      markers.push({ atMs: now() - startedAtMs, label: normalized });
      changed();
    },
    status: () => ({
      active,
      events: events.length,
      poses: poses.length,
      markers: markers.length,
      durationMs: elapsed(),
      truncated,
      viewerResized,
    }),
    recording: () => ({
      schemaVersion: INPUT_RECORDING_SCHEMA_VERSION,
      recordedAt: (startedAtWall ?? wallNow()).toISOString(),
      label,
      href,
      viewer: startViewer,
      environment: options.environment(),
      startPose: startPose ?? options.pose(),
      events: [...events],
      poses: [...poses],
      markers: [...markers],
      durationMs: elapsed(),
      truncated,
    }),
    download(filename) {
      const blob = new Blob([JSON.stringify(recorder.recording(), null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download =
        filename ??
        `pointcloud-input-${filenameTimestamp(startedAtWall ?? wallNow())}.json`;
      anchor.hidden = true;
      document.body.append(anchor);
      anchor.click();
      anchor.remove();
      setTimeout(() => URL.revokeObjectURL(url), 0);
    },
  };

  return recorder;
};

/** True when the page was opened with `?record` or `?record=1`. */
export const recordingRequested = (): boolean => {
  const value = new URLSearchParams(window.location.search).get("record");
  return value !== null && value !== "0" && value !== "false";
};

const OVERLAY_STYLE = `
.input-recorder {
  position: fixed;
  top: 12px;
  right: 12px;
  z-index: 9999;
  display: grid;
  gap: 6px;
  padding: 10px 12px;
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.18);
  background: rgba(12, 16, 22, 0.88);
  color: #e8edf3;
  font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace;
  box-shadow: 0 6px 24px rgba(0, 0, 0, 0.45);
  min-width: 210px;
}
.input-recorder-title {
  display: flex;
  align-items: center;
  gap: 6px;
  font-weight: 600;
  letter-spacing: 0.02em;
}
.input-recorder-dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: #64748b;
}
.input-recorder[data-active="true"] .input-recorder-dot {
  background: #ef4444;
  box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.25);
}
.input-recorder-actions {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}
.input-recorder button {
  flex: 1 1 auto;
  padding: 4px 8px;
  border-radius: 5px;
  border: 1px solid rgba(255, 255, 255, 0.22);
  background: rgba(255, 255, 255, 0.08);
  color: inherit;
  font: inherit;
  cursor: pointer;
}
.input-recorder button:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.16);
}
.input-recorder button:disabled {
  opacity: 0.45;
  cursor: default;
}
.input-recorder-status {
  white-space: pre-line;
  color: #a8b6c6;
}
.input-recorder-warning {
  color: #fbbf24;
}
.input-recorder-hint {
  color: #7c8b9c;
}
`;

/**
 * The trace side of a capture, when the page has one.
 *
 * Structural, not nominal, so the overlay can drive either example's recorder
 * without either of them having to agree on a shared type.
 */
export type OverlayTelemetry = {
  start(): void;
  stop(): void;
  isActive(): boolean;
  download(filename?: string): void;
  summary(): { readonly events: number; readonly frames: number };
};

/**
 * A floating panel, deliberately not part of either example's own layout.
 *
 * The three pages it has to appear on have three different shells, and a
 * capture tool that had to be laid out into each of them would be three
 * chances to record from a viewer of a different size than the one being
 * measured. Fixed positioning also keeps it out of the viewer element, so its
 * own clicks are never part of the gesture it is recording.
 */
export const installRecorderOverlay = (
  recorder: InputRecorder,
  options: {
    readonly markLabel?: () => string | null;
    /**
     * Recorded alongside the input when the page can. One button starts both:
     * a gesture and the frames it produced are one measurement, and a capture
     * that started them separately would describe two overlapping sessions.
     */
    readonly telemetry?: OverlayTelemetry;
  } = {},
): HTMLElement => {
  const style = document.createElement("style");
  style.textContent = OVERLAY_STYLE;
  document.head.append(style);

  const panel = document.createElement("aside");
  panel.className = "input-recorder";
  panel.dataset.active = "false";
  panel.innerHTML = `
    <div class="input-recorder-title">
      <span class="input-recorder-dot"></span><span>Input capture</span>
    </div>
    <div class="input-recorder-actions">
      <button type="button" data-role="toggle">Record</button>
      <button type="button" data-role="mark" disabled>Mark</button>
    </div>
    <div class="input-recorder-actions">
      <button type="button" data-role="download" disabled>Download</button>
      <button type="button" data-role="clear" disabled>Clear</button>
    </div>
    <div class="input-recorder-status" data-role="status">idle</div>
    <div class="input-recorder-hint">F9 record · F10 mark</div>`;
  if (options.telemetry) {
    const traceRow = document.createElement("div");
    traceRow.className = "input-recorder-actions";
    traceRow.innerHTML =
      '<button type="button" data-role="trace" disabled>Download trace</button>';
    panel.querySelector('[data-role="status"]')!.before(traceRow);
  }
  document.body.append(panel);

  const control = <T extends HTMLElement>(role: string): T => {
    const found = panel.querySelector<T>(`[data-role="${role}"]`);
    if (!found) throw new Error(`the recorder overlay is missing ${role}`);
    return found;
  };
  const toggle = control<HTMLButtonElement>("toggle");
  const mark = control<HTMLButtonElement>("mark");
  const download = control<HTMLButtonElement>("download");
  const clear = control<HTMLButtonElement>("clear");
  const status = control<HTMLElement>("status");
  const trace = options.telemetry ? control<HTMLButtonElement>("trace") : null;

  const refresh = (): void => {
    const state = recorder.status();
    panel.dataset.active = String(state.active);
    toggle.textContent = state.active ? "Stop" : "Record";
    mark.disabled = !state.active;
    download.disabled = state.events === 0;
    clear.disabled = state.events === 0 && state.poses === 0;
    const warnings = [
      state.truncated ? "truncated — capture limit reached" : null,
      state.viewerResized ? "viewer resized mid-capture" : null,
    ].filter((entry): entry is string => entry !== null);
    const traceSummary = options.telemetry?.summary();
    if (trace !== null) trace.disabled = (traceSummary?.frames ?? 0) === 0;
    status.textContent =
      `${(state.durationMs / 1000).toFixed(1)} s · ${state.events} events\n` +
      `${state.poses} poses · ${state.markers} marks` +
      (traceSummary ? `\n${traceSummary.frames} frames traced` : "");
    status.classList.toggle("input-recorder-warning", warnings.length > 0);
    if (warnings.length > 0) status.textContent += `\n${warnings.join("\n")}`;
  };

  toggle.addEventListener("click", () => {
    if (recorder.isActive()) {
      recorder.stop();
      options.telemetry?.stop();
    } else {
      options.telemetry?.start();
      recorder.start();
    }
    refresh();
  });
  trace?.addEventListener("click", () => options.telemetry?.download());
  mark.addEventListener("click", () => {
    recorder.mark(
      options.markLabel?.() ?? `mark-${recorder.status().markers + 1}`,
    );
    refresh();
  });
  download.addEventListener("click", () => recorder.download());
  clear.addEventListener("click", () => {
    recorder.clear();
    refresh();
  });

  window.addEventListener("keydown", (event) => {
    if (event.repeat || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === "F9") {
      event.preventDefault();
      toggle.click();
    } else if (event.key === "F10") {
      event.preventDefault();
      mark.click();
    }
  });

  // The counters move while a gesture is in flight, and refreshing them from
  // the event handlers would put this panel's own layout work inside the frame
  // being measured.
  setInterval(refresh, 200);
  refresh();
  return panel;
};
