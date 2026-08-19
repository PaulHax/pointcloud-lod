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
  type QualityAdjustment,
  type QualityRegime,
} from "./adaptiveBudget";
import { cameraMoved, type CameraView } from "./camera";
import { finiteAtLeast, finiteNonNegative, finiteWithin } from "./numeric";

export type TransientFrameMetrics = {
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
  /** Maximum fraction of a host frame assigned to streamed VTK work. */
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
  /** Shortest interval this display has presented, once it is known. */
  readonly displayQuantumMs: number | null;
  readonly estimateMs: number | null;
  readonly samples: number;
  readonly lastAdjustment: QualityAdjustment | null;
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
  needsFrame(): boolean;
  stats(): ViewGovernorStats;
  dispose(): void;
};

const EMERGENCY_CONSECUTIVE_FRAMES = 2;
/** Faster than any display refreshes: a shorter interval is a doubled tick. */
const MIN_DISPLAY_QUANTUM_MS = 3;
/** How many short intervals must agree before the quantum is believed. */
const DISPLAY_QUANTUM_SAMPLES = 3;
const INTERACTION_SEED_OF_STATIONARY = 0.25;

const GOVERNOR_DEFAULTS = {
  vtkFrameFraction: 0.7,
  interactionSettleMs: 750,
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
  const quality = createAdaptiveQuality(qualityOptions);
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
  let {
    vtkFrameFraction,
    interactionSettleMs,
    motionDebounceMs,
    quality,
    emergencyCooldownMs,
  } = resolveConfiguration(options);
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
  let eligibleCapacitySamples = 0;
  let rejectedCapacitySamples = 0;
  let lastCapacitySampleEligible: boolean | null = null;
  let hostEpochOffsetMs: number | null = null;
  let frameCount = 0;
  let peakHostFrameMs = 0;
  let peakObservedFrameMs = 0;
  let lastHostFrameMs: number | null = null;
  let lastObservedFrameMs: number | null = null;
  // The shortest presentation intervals seen this session, ascending. Frames
  // are reported between presentations, so the smallest interval a display can
  // produce is its refresh period however cheap the frame was — which is what
  // the quality tracks need in order to tell headroom from a vsync floor. The
  // k-th smallest rather than the smallest, so one stray short interval from a
  // compositor hiccup cannot claim a faster display than there is.
  const shortestFrameMs: number[] = [];

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
      if (kind === "explicit") explicitMotion += 1;
      else inferredMotion += 1;
      if (!wasInteracting) enterInteraction();
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
    return reason === "within-hysteresis" || reason === "clamped";
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

  const noteDisplayQuantum = (hostFrameMs: number): void => {
    // Below this is faster than any display presents, so it is evidence of a
    // doubled callback rather than of a refresh period.
    if (hostFrameMs < MIN_DISPLAY_QUANTUM_MS) return;
    const at = shortestFrameMs.findIndex((seen) => hostFrameMs < seen);
    if (at < 0) {
      if (shortestFrameMs.length >= DISPLAY_QUANTUM_SAMPLES) return;
      shortestFrameMs.push(hostFrameMs);
    } else {
      shortestFrameMs.splice(at, 0, hostFrameMs);
      shortestFrameMs.length = Math.min(
        shortestFrameMs.length,
        DISPLAY_QUANTUM_SAMPLES,
      );
    }
    if (shortestFrameMs.length === DISPLAY_QUANTUM_SAMPLES) {
      quality.setDisplayQuantumMs(
        shortestFrameMs[DISPLAY_QUANTUM_SAMPLES - 1]!,
      );
    }
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
      quality.reduceNow(true, now);
      lastEmergencyCutAt = now;
      emergencyCooldownUntil = now + emergencyCooldownMs;
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

    setWorkState(next) {
      if (disposed) return;
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
    },

    setOptions(next = {}) {
      if (disposed || sameOptions(appliedOptions, next)) return;
      const configuration = resolveConfiguration(next);
      appliedOptions = { ...next };
      ({
        vtkFrameFraction,
        interactionSettleMs,
        motionDebounceMs,
        quality,
        emergencyCooldownMs,
      } = configuration);
      emergencyStreak = 0;
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
