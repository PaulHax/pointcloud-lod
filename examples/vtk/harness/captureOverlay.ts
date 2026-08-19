/**
 * The capture panel a hand drives, and the URL flag that asks for it.
 *
 * Separate from the recorder because they answer to different things: the
 * recorder is a passive listener whose correctness is about not perturbing
 * what it watches, and this is a piece of user interface whose correctness is
 * about a person being able to start a capture, mark it, and end up with both
 * halves of it on disk.
 */

import { filenameTimestamp, type InputRecorder } from "./inputRecorder";

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
 * A capture is two files and they are only useful together: the input is what
 * a benchmark replays, the trace is what this machine did with it. Naming both
 * from one stamp is what lets a pair be recognised later as one session.
 */
const captureBasename = (recorder: InputRecorder): string =>
  `pointcloud-capture-${filenameTimestamp(
    new Date(recorder.recording().recordedAt),
  )}`;

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
      <button type="button" data-role="download" disabled>Save capture</button>
      <button type="button" data-role="clear" disabled>Clear</button>
    </div>
    <div class="input-recorder-status" data-role="status">idle</div>
    <div class="input-recorder-hint">F9 record · F10 mark</div>`;
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
  mark.addEventListener("click", () => {
    recorder.mark(
      options.markLabel?.() ?? `mark-${recorder.status().markers + 1}`,
    );
    refresh();
  });
  /**
   * Both halves, under one name. Stopping first because a trace downloaded
   * mid-recording has no end, and the pair would describe a session that was
   * still running when it was written out.
   */
  download.addEventListener("click", () => {
    if (recorder.isActive()) {
      recorder.stop();
      options.telemetry?.stop();
    }
    const base = captureBasename(recorder);
    recorder.download(`${base}-input.json`);
    options.telemetry?.download(`${base}-trace.json`);
    refresh();
  });
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
