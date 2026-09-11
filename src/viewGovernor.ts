/**
 * One normalized adaptive-quality governor per rendered view.
 *
 * The governor owns frame-time learning, motion classification, and capacity
 * sample eligibility. It deliberately knows nothing about points, members, or
 * memory; the streamed-scene coordinator supplies aggregate work state and
 * distributes the resulting fraction to format-specific members.
 */

import {
  ADAPTIVE_QUALITY_DEFAULTS,
  createAdaptiveQuality,
  type AdaptiveQuality,
  type AdaptiveQualityOptions,
  type AdaptiveQualityTrackStats,
  type DisplayQuantumSupplier,
  type QualityAdjustment,
  type QualityRegime,
} from "./adaptiveBudget";
import { cameraMoved, type CameraView } from "./camera";
import { finiteAtLeast, finiteNonNegative, finiteWithin } from "./numeric";

export type TransientFrameMetrics = {
  /**
   * Interval between this presentation and the previous one, in milliseconds.
   *
   * A presentation interval, not a render duration — unlike its
   * `vtkFrameMs`/`gpuMs` siblings, which are durations. The governor reads the
   * shortest intervals it sees as the display's refresh period, so a host that
   * reports how long its frame took to build instead reports sub-refresh
   * values on every cheap frame and teaches the governor a display quantum
   * far below the real one. Measure it between successive
   * `requestAnimationFrame` callbacks, or from a presentation timestamp.
   */
  readonly hostFrameMs: number;
  readonly vtkFrameMs?: number;
  readonly gpuMs?: number;
  readonly inputDelayMs?: number;
  readonly longTaskMs?: number;
  readonly now?: number;
};

export type HostFrameMetrics = TransientFrameMetrics & {
  readonly capacitySampleEligible?: boolean;
};

export type CapacitySampleMetrics = {
  readonly frameMs: number;
  readonly regime: QualityRegime;
  readonly eligible: boolean;
  readonly now?: number;
};

export type GovernorWorkState = {
  readonly workPending: boolean;
  readonly physicalTileOperations: number;
  readonly physicalHierarchyOperations: number;
};

export type MotionSourceKind = "explicit" | "inferred";

export type MotionReference = {
  /** Idempotent: releasing twice does not double-decrement. */
  release(): void;
};

export type ViewGovernorOptions = AdaptiveQualityOptions & {
  /** Maximum fraction of a host frame assigned to streamed VTK work. Default 1. */
  readonly vtkFrameFraction?: number;
  /** Delay before returning a changed camera to stationary quality. */
  readonly interactionSettleMs?: number;
  /** Stillness required to release inferred rendered-camera motion. */
  readonly motionDebounceMs?: number;
};

export type ViewGovernorStats = {
  readonly regime: QualityRegime;
  readonly viewQualityFraction: number;
  readonly motion: {
    readonly explicitReferences: number;
    readonly inferredReferences: number;
    readonly source: "explicit" | "inferred" | "both" | null;
    readonly settling: boolean;
  };
  readonly activity: {
    readonly inputActive: boolean;
    readonly cameraStable: boolean;
    readonly workPending: boolean;
    readonly measurementEligible: boolean;
  };
  readonly capacitySamples: {
    readonly eligible: number;
    readonly rejected: number;
    readonly lastEligible: boolean | null;
  };
  /** The target being steered to, raised if the display cannot beat it. */
  readonly targetFrameTimeMs: number;
  /** The configured target, before the display quantum is taken into account. */
  readonly configuredFrameTimeMs: number;
  /**
   * The refresh period the targets are held above, once it is known: the
   * shortest interval presented, and never above the slowest refresh worth
   * assuming.
   */
  readonly displayQuantumMs: number | null;
  readonly estimateMs: number | null;
  readonly samples: number;
  readonly lastAdjustment: QualityAdjustment | null;
  readonly trial: AdaptiveQualityTrackStats["trial"];
  readonly increaseCeiling: number;
  readonly trialAttempts: number;
  readonly physicalTileOperations: number;
  readonly physicalHierarchyOperations: number;
  readonly needsFrame: boolean;
  readonly frameMetrics: {
    readonly frames: number;
    readonly peakHostFrameMs: number;
    readonly peakObservedFrameMs: number;
    readonly lastHostFrameMs: number | null;
    readonly lastObservedFrameMs: number | null;
  };
};

export type ViewGovernor = {
  /** Current fraction for the active motion regime. */
  qualityFraction(): number;
  /** Aggregate work from every active member, including fixed-quality members. */
  setWorkState(state: GovernorWorkState): void;
  setOptions(options?: ViewGovernorOptions): void;
  beginMotion(kind: MotionSourceKind): MotionReference;
  recordCameraChange(): void;
  noteRenderedCameras(
    views: ReadonlyMap<unknown, CameraView | null | undefined>,
    scheduleRender?: () => void,
  ): void;
  resetMotionBaselines(): void;
  recordTransientFrame(metrics: TransientFrameMetrics): void;
  recordCapacitySample(metrics: CapacitySampleMetrics): void;
  recordHostFrame(metrics: HostFrameMetrics): void;
  invalidateCapacity(): void;
  needsFrame(): boolean;
  stats(): ViewGovernorStats;
  dispose(): void;
};

const EMERGENCY_CONSECUTIVE_FRAMES = 2;
/** Faster than any display refreshes: a shorter interval is a doubled tick. */
const MIN_DISPLAY_QUANTUM_MS = 3;
/**
 * The slowest refresh worth assuming, a little above 60 Hz's 16.67 ms.
 *
 * The shortest interval a page has presented is only the refresh period if the
 * page was ever quick enough to hit it. One that never is — a software
 * rasteriser, a throttled tab — would otherwise report its own best frame as
 * the display's floor and have every target raised to match, which is the
 * opposite of what the floor is for. Capping it lets the estimate be too low,
 * never too high: a genuinely slower display just gets less of the correction
 * than it could have had.
 */
const MAX_DISPLAY_QUANTUM_MS = 17;
/** How many short intervals must agree before the quantum is believed. */
const DISPLAY_QUANTUM_SAMPLES = 3;
/**
 * Frames the display-quantum estimate looks back over: about four seconds at
 * 60 Hz.
 *
 * A session-lifetime minimum can only ever ratchet down, and the ratchet has
 * no way back up. Three stray short intervals anywhere in a long session — a
 * compositor hiccup at frame 5, another at frame 900 — would pin the quantum
 * at 4 ms for good, and a window dragged from a 120 Hz panel to a 60 Hz one
 * would keep answering 8.3 ms forever. Either puts the reachable target below
 * the interaction target, which withdraws the correction entirely. A window
 * forgets both, while the k-th smallest within it keeps a single hiccup from
 * claiming a faster display than there is.
 */
const DISPLAY_QUANTUM_WINDOW = 240;
const INTERACTION_SEED_OF_STATIONARY = 0.25;

const GOVERNOR_DEFAULTS = {
  // The measured presentation already includes the host's other work.
  // Callers can still request a smaller share as an explicit sub-budget.
  vtkFrameFraction: 1,
  interactionSettleMs: 250,
  motionDebounceMs: 250,
} as const;

const RESOLVED_OPTION_DEFAULTS = {
  ...ADAPTIVE_QUALITY_DEFAULTS,
  ...GOVERNOR_DEFAULTS,
} as const;

const sameOptions = (a: ViewGovernorOptions, b: ViewGovernorOptions): boolean =>
  (
    Object.keys(RESOLVED_OPTION_DEFAULTS) as (keyof ViewGovernorOptions)[]
  ).every(
    (key) =>
      (a[key] ?? RESOLVED_OPTION_DEFAULTS[key]) ===
      (b[key] ?? RESOLVED_OPTION_DEFAULTS[key]),
  );

type GovernorConfiguration = {
  readonly vtkFrameFraction: number;
  readonly interactionSettleMs: number;
  readonly motionDebounceMs: number;
  readonly quality: AdaptiveQuality;
  readonly emergencyCooldownMs: number;
};

const resolveConfiguration = (
  options: ViewGovernorOptions,
  displayQuantum: DisplayQuantumSupplier,
): GovernorConfiguration => {
  const {
    vtkFrameFraction: rawVtkFraction = GOVERNOR_DEFAULTS.vtkFrameFraction,
    interactionSettleMs: rawSettleMs = GOVERNOR_DEFAULTS.interactionSettleMs,
    motionDebounceMs: rawDebounceMs = GOVERNOR_DEFAULTS.motionDebounceMs,
    ...qualityOptions
  } = options;
  const vtkFrameFraction = finiteWithin(
    "vtkFrameFraction",
    rawVtkFraction,
    0.05,
    1,
  );
  const interactionSettleMs = finiteAtLeast(
    "interactionSettleMs",
    rawSettleMs,
    0,
  );
  const motionDebounceMs = finiteAtLeast("motionDebounceMs", rawDebounceMs, 0);
  const quality = createAdaptiveQuality(qualityOptions, displayQuantum);
  return {
    vtkFrameFraction,
    interactionSettleMs,
    motionDebounceMs,
    quality,
    emergencyCooldownMs: quality.stats().cooldownMs,
  };
};

export const createViewGovernor = (
  options: ViewGovernorOptions = {},
): ViewGovernor => {
  let appliedOptions = { ...options };
  // What the display has been measured to refresh at, kept here rather than
  // in the quality tracks: new tracks built by `setOptions` read the same
  // measurement, and the tracks read it on every target, so the window scan
  // behind it runs once a frame instead of once a read.
  let currentQuantum: number | null = null;
  let {
    vtkFrameFraction,
    interactionSettleMs,
    motionDebounceMs,
    quality,
    emergencyCooldownMs,
  } = resolveConfiguration(options, () => currentQuantum);
  let work: GovernorWorkState = {
    workPending: false,
    physicalTileOperations: 0,
    physicalHierarchyOperations: 0,
  };
  let explicitMotion = 0;
  let inferredMotion = 0;
  let disposed = false;
  let cameraStable = true;
  let cameraStabilityTimer: ReturnType<typeof setTimeout> | null = null;
  let emergencyCooldownUntil = Number.NEGATIVE_INFINITY;
  let lastEmergencyCutAt = Number.NEGATIVE_INFINITY;
  let emergencyStreak = 0;
  let reliefStreak = 0;
  let eligibleCapacitySamples = 0;
  let rejectedCapacitySamples = 0;
  let lastCapacitySampleEligible: boolean | null = null;
  let hostEpochOffsetMs: number | null = null;
  let frameCount = 0;
  let peakHostFrameMs = 0;
  let peakObservedFrameMs = 0;
  let lastHostFrameMs: number | null = null;
  let lastObservedFrameMs: number | null = null;
  // The most recent presentation intervals, oldest overwritten. Frames are
  // reported between presentations, so the smallest interval a display can
  // produce is its refresh period however cheap the frame was — which is what
  // the quality tracks need in order to tell headroom from a vsync floor.
  const recentFrameMs = new Float64Array(DISPLAY_QUANTUM_WINDOW);
  let recentFrames = 0;
  let recentFrameAt = 0;
  const smallestFrameMs = new Float64Array(DISPLAY_QUANTUM_SAMPLES);

  const stampNow = (): number => Date.now() + (hostEpochOffsetMs ?? 0);
  const moving = (): boolean => explicitMotion + inferredMotion > 0;
  const interacting = (): boolean => moving() || !cameraStable;
  const regime = (): QualityRegime =>
    interacting() ? "interaction" : "stationary";
  const pendingWork = (): boolean =>
    work.workPending ||
    work.physicalTileOperations > 0 ||
    work.physicalHierarchyOperations > 0;
  const coreEligible = (): boolean =>
    !pendingWork() && (moving() || cameraStable);

  const clearCameraStabilityTimer = (): void => {
    if (cameraStabilityTimer !== null) clearTimeout(cameraStabilityTimer);
    cameraStabilityTimer = null;
  };

  const enterInteraction = (): void => {
    emergencyStreak = 0;
    reliefStreak = 0;
    quality.invalidateCapacity(false, stampNow());
    quality.restartAt(
      true,
      Math.max(
        quality.fraction(true),
        INTERACTION_SEED_OF_STATIONARY * quality.fraction(false),
      ),
      stampNow(),
    );
  };

  const enterStationary = (): void => {
    quality.restartAt(
      false,
      Math.max(quality.fraction(false), quality.fraction(true)),
      stampNow(),
    );
  };

  const markCameraChanged = (): void => {
    if (disposed) return;
    const wasInteracting = interacting();
    cameraStable = false;
    clearCameraStabilityTimer();
    cameraStabilityTimer = setTimeout(() => {
      cameraStabilityTimer = null;
      cameraStable = true;
      if (!moving()) enterStationary();
    }, interactionSettleMs);
    if (!wasInteracting) enterInteraction();
  };

  const takeMotionReference = (kind: MotionSourceKind): MotionReference => {
    let held = !disposed;
    if (held) {
      const wasInteracting = interacting();
      // A fresh gesture reseeds even while the previous one's settle window is
      // still open, but only while an emergency cut is actually holding the
      // track under its seed floor. Otherwise a gesture begun inside
      // `interactionSettleMs` inherits that floor and only rest can lift it: no
      // capacity sample is eligible while tiles are streaming, so the very
      // gestures that trigger cuts are the ones that cannot undo them.
      //
      // Reseeding unconditionally would instead clear the sample window and the
      // emergency streak on every nudge, and a user making many gestures each
      // shorter than the window could never adapt downward at all.
      const newGesture =
        kind === "explicit" &&
        explicitMotion === 0 &&
        quality.fraction(true) <
          INTERACTION_SEED_OF_STATIONARY * quality.fraction(false);
      if (kind === "explicit") explicitMotion += 1;
      else inferredMotion += 1;
      if (!wasInteracting || newGesture) enterInteraction();
    }
    return {
      release() {
        if (!held) return;
        held = false;
        if (kind === "explicit") explicitMotion -= 1;
        else inferredMotion -= 1;
        if (!disposed && !moving() && cameraStable) enterStationary();
      },
    };
  };

  let renderedCameras: ReadonlyMap<unknown, CameraView | null | undefined> =
    new Map();
  let inferredBurst: MotionReference | null = null;
  let inferredBurstTimer: ReturnType<typeof setTimeout> | null = null;

  const endInferredBurst = (): void => {
    if (inferredBurstTimer !== null) clearTimeout(inferredBurstTimer);
    inferredBurstTimer = null;
    inferredBurst?.release();
    inferredBurst = null;
  };

  const converged = (): boolean => {
    const current =
      regime() === "interaction"
        ? quality.stats().interaction
        : quality.stats().stationary;
    const reason = current.lastAdjustment?.reason;
    return (
      reason === "within-hysteresis" ||
      reason === "clamped" ||
      (reason === "trial-accepted" && current.fraction === 1)
    );
  };

  const shouldRender = (): boolean =>
    !disposed && (interacting() || (!pendingWork() && !converged()));

  const stampFrom = (metrics: { readonly now?: number }): number => {
    const localNow = Date.now();
    if (finiteNonNegative(metrics.now)) {
      hostEpochOffsetMs = metrics.now - localNow;
    }
    return localNow + (hostEpochOffsetMs ?? 0);
  };

  const observedFrameMs = (metrics: TransientFrameMetrics): number => {
    const candidates = [metrics.hostFrameMs];
    if (finiteNonNegative(metrics.vtkFrameMs)) {
      candidates.push(metrics.vtkFrameMs / vtkFrameFraction);
    }
    if (finiteNonNegative(metrics.gpuMs)) {
      candidates.push(metrics.gpuMs / vtkFrameFraction);
    }
    return Math.max(...candidates);
  };

  /** The k-th smallest interval still inside the window, capped. */
  const believedQuantumMs = (): number | null => {
    const held = Math.min(recentFrames, DISPLAY_QUANTUM_WINDOW);
    if (held < DISPLAY_QUANTUM_SAMPLES) return null;
    smallestFrameMs.fill(Number.POSITIVE_INFINITY);
    for (let index = 0; index < held; index += 1) {
      const seen = recentFrameMs[index]!;
      for (let rank = 0; rank < DISPLAY_QUANTUM_SAMPLES; rank += 1) {
        if (seen >= smallestFrameMs[rank]!) continue;
        smallestFrameMs.copyWithin(rank + 1, rank);
        smallestFrameMs[rank] = seen;
        break;
      }
    }
    return Math.min(
      MAX_DISPLAY_QUANTUM_MS,
      smallestFrameMs[DISPLAY_QUANTUM_SAMPLES - 1]!,
    );
  };

  const noteDisplayQuantum = (hostFrameMs: number): void => {
    // Below this is faster than any display presents, so it is evidence of a
    // doubled callback rather than of a refresh period.
    if (hostFrameMs < MIN_DISPLAY_QUANTUM_MS) return;
    recentFrameMs[recentFrameAt] = hostFrameMs;
    recentFrameAt = (recentFrameAt + 1) % DISPLAY_QUANTUM_WINDOW;
    recentFrames += 1;
    const quantum = believedQuantumMs();
    if (quantum !== null) currentQuantum = quantum;
  };

  const recordTransientFrame = (metrics: TransientFrameMetrics): number => {
    if (disposed || !finiteNonNegative(metrics?.hostFrameMs)) return 0;
    noteDisplayQuantum(metrics.hostFrameMs);
    const now = stampFrom(metrics);
    const severeInput =
      (finiteNonNegative(metrics.inputDelayMs) && metrics.inputDelayMs > 50) ||
      (finiteNonNegative(metrics.longTaskMs) && metrics.longTaskMs > 50);
    const target = quality.target(interacting());
    const observedMs = observedFrameMs(metrics);
    frameCount += 1;
    lastHostFrameMs = metrics.hostFrameMs;
    lastObservedFrameMs = observedMs;
    peakHostFrameMs = Math.max(peakHostFrameMs, metrics.hostFrameMs);
    peakObservedFrameMs = Math.max(peakObservedFrameMs, observedMs);
    const emergency = moving() && (severeInput || observedMs > target * 2);
    if (!emergency) emergencyStreak = 0;
    if (
      emergency &&
      now >= emergencyCooldownUntil &&
      ++emergencyStreak >= EMERGENCY_CONSECUTIVE_FRAMES
    ) {
      emergencyStreak = 0;
      reliefStreak = 0;
      quality.reduceNow(true, now);
      lastEmergencyCutAt = now;
      emergencyCooldownUntil = now + emergencyCooldownMs;
    }
    // The cut above answers a frame no capacity sample is allowed to answer:
    // it fires while work is pending, which is exactly when eligibility is
    // withheld. Without a relief that reads the same frames under the same
    // conditions, a gesture long enough to keep tiles streaming can only
    // ratchet quality down, and recovers no earlier than the next gesture or
    // rest. Relief hands back one cut and stops at what the cuts took, so the
    // eligible loop still owns every fraction above that.
    const relief =
      moving() &&
      !severeInput &&
      observedMs <= quality.increaseThresholdMs(interacting());
    if (!relief) reliefStreak = 0;
    if (
      relief &&
      now >= emergencyCooldownUntil &&
      ++reliefStreak >= EMERGENCY_CONSECUTIVE_FRAMES
    ) {
      reliefStreak = 0;
      const before = quality.fraction(true);
      if (quality.restoreNow(true, now) !== before) {
        emergencyCooldownUntil = now + emergencyCooldownMs;
      }
    }
    return observedMs;
  };

  const recordCapacitySample = (metrics: CapacitySampleMetrics): void => {
    if (disposed || !finiteNonNegative(metrics?.frameMs)) return;
    const now = stampFrom(metrics);
    lastCapacitySampleEligible = metrics.eligible;
    if (!metrics.eligible) {
      rejectedCapacitySamples += 1;
      return;
    }
    eligibleCapacitySamples += 1;
    if (now <= lastEmergencyCutAt || now < emergencyCooldownUntil) return;
    quality.recordFrame(metrics.frameMs, {
      interacting: metrics.regime === "interaction",
      now,
    });
  };

  return {
    qualityFraction: () => quality.fraction(interacting()),

    invalidateCapacity() {
      if (disposed) return;
      const now = stampNow();
      // A different workload invalidates both regimes' measurements, but
      // preserves their quality, cooldowns, and outstanding emergency relief.
      quality.invalidateCapacity(false, now);
      quality.invalidateCapacity(true, now);
    },

    setWorkState(next) {
      if (disposed) return;
      const wasPending = pendingWork();
      work = {
        workPending: !!next.workPending,
        physicalTileOperations: finiteNonNegative(next.physicalTileOperations)
          ? Math.floor(next.physicalTileOperations)
          : 0,
        physicalHierarchyOperations: finiteNonNegative(
          next.physicalHierarchyOperations,
        )
          ? Math.floor(next.physicalHierarchyOperations)
          : 0,
      };
      if (wasPending && !pendingWork()) {
        quality.clearSamples(interacting(), stampNow());
      }
    },

    setOptions(next = {}) {
      if (disposed || sameOptions(appliedOptions, next)) return;
      const configuration = resolveConfiguration(next, () => currentQuantum);
      appliedOptions = { ...next };
      ({
        vtkFrameFraction,
        interactionSettleMs,
        motionDebounceMs,
        quality,
        emergencyCooldownMs,
      } = configuration);
      emergencyStreak = 0;
      reliefStreak = 0;
      emergencyCooldownUntil = Number.NEGATIVE_INFINITY;
      lastEmergencyCutAt = Number.NEGATIVE_INFINITY;
      eligibleCapacitySamples = 0;
      rejectedCapacitySamples = 0;
      lastCapacitySampleEligible = null;
      if (interacting()) enterInteraction();
    },

    beginMotion: takeMotionReference,
    recordCameraChange: markCameraChanged,

    noteRenderedCameras(views, scheduleRender) {
      if (disposed) return;
      let moved = false;
      for (const [key, view] of views) {
        if (view && cameraMoved(renderedCameras.get(key), view)) moved = true;
      }
      renderedCameras = new Map(views);
      if (!moved) return;
      markCameraChanged();
      if (!inferredBurst) inferredBurst = takeMotionReference("inferred");
      if (inferredBurstTimer !== null) clearTimeout(inferredBurstTimer);
      inferredBurstTimer = setTimeout(() => {
        endInferredBurst();
        scheduleRender?.();
      }, motionDebounceMs);
    },

    resetMotionBaselines() {
      renderedCameras = new Map();
    },

    recordTransientFrame(metrics) {
      recordTransientFrame(metrics);
    },

    recordCapacitySample,

    recordHostFrame(metrics) {
      if (disposed || !finiteNonNegative(metrics?.hostFrameMs)) return;
      const observedMs = recordTransientFrame(metrics);
      recordCapacitySample({
        frameMs: observedMs,
        regime: regime(),
        eligible: (metrics.capacitySampleEligible ?? true) && coreEligible(),
        now: metrics.now,
      });
    },

    needsFrame: shouldRender,

    stats() {
      const currentRegime = regime();
      const adaptive = quality.stats();
      const track =
        currentRegime === "interaction"
          ? adaptive.interaction
          : adaptive.stationary;
      return {
        regime: currentRegime,
        viewQualityFraction: track.fraction,
        motion: {
          explicitReferences: explicitMotion,
          inferredReferences: inferredMotion,
          source:
            explicitMotion > 0 && inferredMotion > 0
              ? "both"
              : explicitMotion > 0
                ? "explicit"
                : inferredMotion > 0
                  ? "inferred"
                  : null,
          settling: !cameraStable && !moving(),
        },
        activity: {
          inputActive: moving(),
          cameraStable,
          workPending: pendingWork(),
          measurementEligible: coreEligible(),
        },
        capacitySamples: {
          eligible: eligibleCapacitySamples,
          rejected: rejectedCapacitySamples,
          lastEligible: lastCapacitySampleEligible,
        },
        targetFrameTimeMs: track.effectiveTargetMs,
        configuredFrameTimeMs: track.targetMs,
        displayQuantumMs: adaptive.displayQuantumMs,
        estimateMs: track.estimateMs,
        samples: track.samples,
        lastAdjustment: track.lastAdjustment,
        trial: track.trial,
        increaseCeiling: track.increaseCeiling,
        trialAttempts: track.trialAttempts,
        physicalTileOperations: work.physicalTileOperations,
        physicalHierarchyOperations: work.physicalHierarchyOperations,
        needsFrame: shouldRender(),
        frameMetrics: {
          frames: frameCount,
          peakHostFrameMs,
          peakObservedFrameMs,
          lastHostFrameMs,
          lastObservedFrameMs,
        },
      };
    },

    dispose() {
      if (disposed) return;
      disposed = true;
      endInferredBurst();
      clearCameraStabilityTimer();
    },
  };
};
