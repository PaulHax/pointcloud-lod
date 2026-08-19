/**
 * Replays a recorded gesture into a live page, in real time.
 *
 * Real time is the point. The interactor applies pointer moves on its own
 * animation frame, so a configuration that paints slowly gets fewer of them
 * through the same gesture — which is exactly the cost being measured. Pacing
 * the input to painted frames instead, as the deterministic checks in this
 * suite do, would hand every configuration the same number of camera updates
 * and hide the thing the benchmark exists to see.
 *
 * The price is that the camera no longer lands in exactly the same place every
 * run, so this reports two fidelity numbers alongside the measurement rather
 * than assuming them:
 *
 * - dispatch lateness, how far behind the recorded schedule the driver fell,
 *   which bounds how faithfully the gesture was reproduced at all;
 * - camera drift, how far the replayed pose track wandered from the recorded
 *   one, which bounds how comparable two runs' workloads are.
 *
 * A run whose drift is large is not wrong, but it did not traverse the same
 * scene, and the analysis says so instead of averaging it in.
 *
 * Events reach the page through CDP, so they arrive trusted and carry a real
 * pointer id. Synthetic DOM events would be cheaper to schedule and would fail
 * the moment the interactor asked to capture a pointer that never existed.
 */

import type { CDPSession, Page } from "playwright";

import type {
  InputRecording,
  RecordedInputEvent,
  RecordedPointerEvent,
  RecordedPoseSample,
  RecordedWheelEvent,
} from "../../examples/vtk/scene/inputRecorder";

/** Where the viewer sits on the page, so recorded viewer coordinates land. */
export type ViewerBox = {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
};

export type DispatchFidelity = {
  readonly events: number;
  /** Milliseconds behind schedule, over every dispatched event. */
  readonly meanLatenessMs: number;
  readonly maxLatenessMs: number;
  /** Events that went out more than 16 ms late — a missed display frame. */
  readonly lateEvents: number;
  readonly recordedDurationMs: number;
  readonly actualDurationMs: number;
};

export type CameraDrift = {
  readonly samples: number;
  /** Eye distance from the recorded eye at the same instant, in scene units. */
  readonly meanPositionError: number;
  readonly maxPositionError: number;
  /**
   * The same, as a fraction of the recorded eye-to-focus distance — the only
   * form comparable between a city overview and a street-level orbit.
   */
  readonly meanRelativeError: number;
  readonly maxRelativeError: number;
  /** Angle between the replayed and recorded view directions, degrees. */
  readonly meanViewAngleError: number;
  readonly maxViewAngleError: number;
  /**
   * Distance from each recorded eye to the nearest point on the replayed eye
   * path, ignoring when it got there — relative to the recorded orbit radius.
   *
   * Time-aligned error and path error answer different questions, and a replay
   * routinely scores badly on the first while scoring perfectly on the second:
   * the interactor coalesces a gesture into whatever frames it got, so a run
   * that painted fewer of them arrives at each point slightly later while
   * travelling exactly the same route. Only path error says the two runs
   * traversed different scenery, which is what makes their workloads
   * incomparable.
   */
  readonly meanRelativePathError: number;
  readonly maxRelativePathError: number;
};

export type ReplayResult = {
  readonly dispatch: DispatchFidelity;
  readonly drift: CameraDrift | null;
  /** Markers emitted, in the order the replay reached them. */
  readonly markers: readonly string[];
};

const sleep = (ms: number): Promise<void> =>
  ms <= 0 ? Promise.resolve() : new Promise((done) => setTimeout(done, ms));

/** The CDP button name for a DOM `button` index. */
const buttonName = (
  button: number,
): "left" | "middle" | "right" | "back" | "forward" | "none" => {
  switch (button) {
    case 0:
      return "left";
    case 1:
      return "middle";
    case 2:
      return "right";
    case 3:
      return "back";
    case 4:
      return "forward";
    default:
      return "none";
  }
};

/** CDP wants the held buttons as its own bitmask, which matches the DOM's. */
const heldButtons = (buttons: number): number => buttons;

const modifiers = (event: {
  readonly altKey: boolean;
  readonly ctrlKey: boolean;
  readonly metaKey: boolean;
  readonly shiftKey: boolean;
}): number =>
  (event.altKey ? 1 : 0) |
  (event.ctrlKey ? 2 : 0) |
  (event.metaKey ? 4 : 0) |
  (event.shiftKey ? 8 : 0);

const isPointer = (event: RecordedInputEvent): event is RecordedPointerEvent =>
  event.type === "pointerdown" ||
  event.type === "pointermove" ||
  event.type === "pointerup" ||
  event.type === "pointercancel";

const isWheel = (event: RecordedInputEvent): event is RecordedWheelEvent =>
  event.type === "wheel";

/**
 * A wheel turn's strength, in notches, as the interactor will count it.
 *
 * CDP can say that a wheel turned and where, but not how hard: Chrome pins
 * `wheelDelta` to a single notch on every injected wheel event regardless of
 * the `deltaY` it carries, and `normalizeWheel` — which is what vtk.js reads —
 * prefers `wheelDelta`. A hand's flick of two notches therefore replays as
 * one, and the dolly it drives compounds the shortfall over the rest of the
 * path.
 *
 * What CDP can express is count, so magnitude is converted into it. The
 * interactor normalises a burst of wheel events by the first one's strength
 * and treats a 200 ms gap as the end of a burst, so the same rule is applied
 * here: each event is worth its delta as a multiple of the delta that opened
 * its burst. Fractional turns — a trackpad's, rather than a wheel's — cannot
 * be dispatched at all, so the remainder is carried into the next event of the
 * burst rather than rounded away one event at a time.
 */
const WHEEL_BURST_GAP_MS = 200;

export const wheelNotches = (
  events: readonly RecordedInputEvent[],
): ReadonlyMap<RecordedWheelEvent, number> => {
  const notches = new Map<RecordedWheelEvent, number>();
  let base = 0;
  let previousAtMs = Number.NEGATIVE_INFINITY;
  let carry = 0;
  for (const event of events) {
    if (!isWheel(event)) continue;
    const magnitude = Math.hypot(event.deltaX, event.deltaY);
    if (event.atMs - previousAtMs > WHEEL_BURST_GAP_MS || base === 0) {
      base = magnitude;
      carry = 0;
    }
    previousAtMs = event.atMs;
    if (magnitude === 0) {
      notches.set(event, 0);
      continue;
    }
    const wanted = carry + magnitude / base;
    const whole = Math.max(0, Math.round(wanted));
    carry = wanted - whole;
    notches.set(event, whole);
  }
  return notches;
};

const CDP_TYPE = {
  pointerdown: "mousePressed",
  pointerup: "mouseReleased",
  pointermove: "mouseMoved",
  pointercancel: "mouseReleased",
} as const;

const subtract = (
  left: readonly number[],
  right: readonly number[],
): [number, number, number] => [
  left[0]! - right[0]!,
  left[1]! - right[1]!,
  left[2]! - right[2]!,
];

const length = (value: readonly number[]): number => Math.hypot(...value);

const unit = (value: readonly number[]): [number, number, number] => {
  const size = length(value);
  return size === 0
    ? [0, 0, 0]
    : [value[0]! / size, value[1]! / size, value[2]! / size];
};

const angleBetweenDegrees = (
  left: readonly number[],
  right: readonly number[],
): number => {
  const a = unit(left);
  const b = unit(right);
  const dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  return (Math.acos(Math.min(1, Math.max(-1, dot))) * 180) / Math.PI;
};

/** Distance from `point` to the segment `start`-`end`. */
const distanceToSegment = (
  point: readonly number[],
  start: readonly number[],
  end: readonly number[],
): number => {
  const segment = subtract(end, start);
  const toPoint = subtract(point, start);
  const segmentLengthSquared =
    segment[0] * segment[0] + segment[1] * segment[1] + segment[2] * segment[2];
  if (segmentLengthSquared === 0) return length(toPoint);
  const projection =
    (toPoint[0] * segment[0] +
      toPoint[1] * segment[1] +
      toPoint[2] * segment[2]) /
    segmentLengthSquared;
  const clamped = Math.min(1, Math.max(0, projection));
  return length([
    toPoint[0] - segment[0] * clamped,
    toPoint[1] - segment[1] * clamped,
    toPoint[2] - segment[2] * clamped,
  ]);
};

/** Closest approach of `point` to the polyline through `track`. */
const distanceToTrack = (
  point: readonly number[],
  track: readonly RecordedPoseSample[],
): number => {
  if (track.length === 1) return length(subtract(point, track[0]!.position));
  let closest = Number.POSITIVE_INFINITY;
  for (let index = 1; index < track.length; index += 1) {
    closest = Math.min(
      closest,
      distanceToSegment(
        point,
        track[index - 1]!.position,
        track[index]!.position,
      ),
    );
  }
  return closest;
};

/** The pose the recording held at `atMs`, held constant between samples. */
const poseAt = (
  poses: readonly RecordedPoseSample[],
  atMs: number,
): RecordedPoseSample | null => {
  if (poses.length === 0) return null;
  let found: RecordedPoseSample | null = null;
  for (const pose of poses) {
    if (pose.atMs > atMs) break;
    found = pose;
  }
  return found ?? poses[0]!;
};

/**
 * Compare two pose tracks on the recorded track's own sample times.
 *
 * Sampling on the recording's clock rather than the replay's keeps the
 * comparison from being dominated by whichever run happened to paint more
 * frames — the question is where the camera was at a given instant, not how
 * often each run wrote that down.
 */
export const compareCameraTracks = (
  recorded: readonly RecordedPoseSample[],
  replayed: readonly RecordedPoseSample[],
): CameraDrift | null => {
  if (recorded.length === 0 || replayed.length === 0) return null;
  let positionSum = 0;
  let positionMax = 0;
  let relativeSum = 0;
  let relativeMax = 0;
  let angleSum = 0;
  let angleMax = 0;
  let pathSum = 0;
  let pathMax = 0;
  let samples = 0;
  for (const pose of recorded) {
    const other = poseAt(replayed, pose.atMs);
    if (other === null) continue;
    const positionError = length(subtract(pose.position, other.position));
    const radius = Math.max(
      1e-6,
      length(subtract(pose.position, pose.focalPoint)),
    );
    const relative = positionError / radius;
    const angle = angleBetweenDegrees(
      subtract(pose.focalPoint, pose.position),
      subtract(other.focalPoint, other.position),
    );
    positionSum += positionError;
    positionMax = Math.max(positionMax, positionError);
    relativeSum += relative;
    relativeMax = Math.max(relativeMax, relative);
    angleSum += angle;
    angleMax = Math.max(angleMax, angle);
    const pathError = distanceToTrack(pose.position, replayed) / radius;
    pathSum += pathError;
    pathMax = Math.max(pathMax, pathError);
    samples += 1;
  }
  if (samples === 0) return null;
  return {
    samples,
    meanPositionError: positionSum / samples,
    maxPositionError: positionMax,
    meanRelativeError: relativeSum / samples,
    maxRelativeError: relativeMax,
    meanViewAngleError: angleSum / samples,
    maxViewAngleError: angleMax,
    meanRelativePathError: pathSum / samples,
    maxRelativePathError: pathMax,
  };
};

export type ReplayOptions = {
  readonly page: Page;
  readonly recording: InputRecording;
  /** Where the viewer element is now, in page pixels. */
  readonly viewer: ViewerBox;
  /** Called for each recorded marker, at the instant it was recorded. */
  readonly onMarker?: (label: string) => void | Promise<void>;
};

/**
 * Dispatch the recording's events against the page, on the recorded schedule.
 *
 * Sends are issued on time and not awaited. Awaiting them would tie the hand
 * to the page: a CDP acknowledgement comes back through the renderer's main
 * thread, so a configuration that blocks that thread for 300 ms would receive
 * a gesture performed 300 ms more slowly — and would be rewarded for stalling
 * with extra time to catch up, which is the opposite of what the benchmark is
 * asking. A real mouse keeps moving while a page is busy, and the browser
 * queues and coalesces what it could not deliver. Issuing on schedule
 * reproduces that; the events still arrive in order, because CDP messages are
 * processed in the order they are written.
 *
 * The schedule is never rewound: each event aims at its own recorded instant,
 * so lateness measures this driver's own jitter rather than accumulating into
 * a slow-motion replay.
 */
export const replayInput = async (
  options: ReplayOptions,
): Promise<ReplayResult> => {
  const { page, recording, viewer } = options;
  const session: CDPSession = await page.context().newCDPSession(page);
  const markers = [...recording.markers].sort((a, b) => a.atMs - b.atMs);
  const emitted: string[] = [];
  let nextMarker = 0;

  const pointX = (x: number): number => viewer.x + x;
  const pointY = (y: number): number => viewer.y + y;

  /**
   * Pointer coordinates cross CDP in CSS pixels, but wheel deltas cross it in
   * device pixels and reach the page divided by the device pixel ratio. A
   * recording carries the CSS-pixel deltas the page saw, so they are scaled
   * back up here; without it, every notch of a gesture captured on a
   * fractionally scaled display replays at 1/dpr of its recorded strength and
   * the dolly it drives compounds that shortfall over the whole path.
   */
  const wheelScale = await page.evaluate(() => window.devicePixelRatio);
  const notches = wheelNotches(recording.events);

  let latenessSum = 0;
  let latenessMax = 0;
  let lateEvents = 0;
  let dispatched = 0;
  const sends: Promise<unknown>[] = [];
  const sendFailures: string[] = [];
  const issue = (
    method: "Input.dispatchMouseEvent",
    parameters: Record<string, unknown>,
  ): void => {
    sends.push(
      session
        .send(method, parameters as never)
        .catch((error: unknown) =>
          sendFailures.push(
            error instanceof Error ? error.message : String(error),
          ),
        ),
    );
  };

  const startedAt = performance.now();
  for (const event of recording.events) {
    if (event.type === "viewport") continue;
    const due = startedAt + event.atMs;
    await sleep(due - performance.now());

    while (
      nextMarker < markers.length &&
      markers[nextMarker]!.atMs <= event.atMs
    ) {
      const label = markers[nextMarker]!.label;
      emitted.push(label);
      await options.onMarker?.(label);
      nextMarker += 1;
    }

    const lateness = Math.max(0, performance.now() - due);
    latenessSum += lateness;
    latenessMax = Math.max(latenessMax, lateness);
    if (lateness > 16) lateEvents += 1;

    if (isPointer(event)) {
      issue("Input.dispatchMouseEvent", {
        type: CDP_TYPE[event.type],
        x: pointX(event.x),
        y: pointY(event.y),
        button: buttonName(event.button),
        buttons: heldButtons(event.buttons),
        modifiers: modifiers(event),
        clickCount: event.type === "pointerdown" ? 1 : 0,
        pointerType: "mouse",
      });
    } else if (isWheel(event)) {
      // One message per notch, each carrying its share of the recorded delta,
      // so both the count the interactor reads and the distance anything else
      // reads come out at what was recorded.
      const turns = notches.get(event) ?? 1;
      for (let turn = 0; turn < turns; turn += 1) {
        issue("Input.dispatchMouseEvent", {
          type: "mouseWheel",
          x: pointX(event.x),
          y: pointY(event.y),
          button: "none",
          buttons: heldButtons(event.buttons),
          modifiers: modifiers(event),
          deltaX: (event.deltaX / turns) * wheelScale,
          deltaY: (event.deltaY / turns) * wheelScale,
          pointerType: "mouse",
        });
      }
    }
    dispatched += 1;
  }

  for (; nextMarker < markers.length; nextMarker += 1) {
    const label = markers[nextMarker]!.label;
    emitted.push(label);
    await options.onMarker?.(label);
  }

  // Every event has been issued; the page may still be working through the
  // tail of them, and detaching the session first would drop those.
  await Promise.all(sends);
  await session.detach().catch(() => undefined);
  if (sendFailures.length > 0) {
    throw new Error(
      `the replay could not deliver ${sendFailures.length} event(s): ${sendFailures[0]}`,
    );
  }
  return {
    dispatch: {
      events: dispatched,
      meanLatenessMs: dispatched === 0 ? 0 : latenessSum / dispatched,
      maxLatenessMs: latenessMax,
      lateEvents,
      recordedDurationMs: recording.durationMs,
      actualDurationMs: performance.now() - startedAt,
    },
    drift: null,
    markers: emitted,
  };
};
