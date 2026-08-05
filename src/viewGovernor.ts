/**
 * One adaptive point budget per view.
 *
 * The governor owns the regime (camera moving vs settled) and the adaptive
 * loop that sizes the aggregate view budget. A shared view-budget coordinator
 * applies memory bounds and distributes that target across clouds. The
 * governor does not own the render loop: the host reports every completed
 * frame through frame pressure and clean capacity samples, then asks
 * `needsFrame()` whether another one is required.
 */

import {
  createAdaptiveBudget,
  DEFAULTS,
  type AdaptiveBudget,
  type AdaptiveBudgetOptions,
  type BudgetAdjustment,
  type BudgetRegime,
} from "./adaptiveBudget";
import { cameraMoved, type CameraView } from "./camera";
import { finiteAtLeast, finiteNonNegative, finiteWithin } from "./numeric";
import {
  createViewBudgetCoordinator,
  memberConstraint,
  type ViewBudgetMember,
  type ViewBudgetMemberOptions,
  type ViewBudgetMemberStats,
  type ViewBudgetMemberUpdate,
} from "./viewBudget";

export type TransientFrameMetrics = {
  /** Complete host frame time, including non-VTK work. */
  readonly hostFrameMs: number;
  /** Time spent painting VTK content within the host frame. */
  readonly vtkFrameMs?: number;
  /** Asynchronously collected GPU time for the VTK paint. */
  readonly gpuMs?: number;
  readonly inputDelayMs?: number;
  readonly longTaskMs?: number;
  /**
   * When this frame was measured. Any epoch will do — `performance.now()`,
   * `Date.now()`, a test clock — as long as the host stays on one of them:
   * only differences matter, and the governor adopts whatever the host
   * supplies as the timeline it stamps its own decisions with. It need not be
   * on every frame: the epoch is held as an offset from the local clock, so a
   * frame that omits it is stamped by that clock in the host's own epoch.
   */
  readonly now?: number;
};

export type HostFrameMetrics = TransientFrameMetrics & {
  /**
   * Whether this frame is allowed to change lasting capacity. Omit only when
   * the host has no asynchronous work or upload tracking and the frame is its
   * clean fallback sample.
   */
  readonly capacitySampleEligible?: boolean;
};

export type CapacitySampleMetrics = {
  /** Frame cost normalized to the complete host-frame allowance. */
  readonly frameMs: number;
  /** Regime that drew the measured frame, captured before async GPU readback. */
  readonly regime: BudgetRegime;
  /** False keeps the measurement diagnostic without changing either track. */
  readonly eligible: boolean;
  readonly now?: number;
};

/**
 * What is holding the moving regime. Explicit sources are user gestures the
 * host reports directly; inferred sources are rendered-camera motion a
 * classifier detected (playback, scrubbing, programmatic animation). Both
 * count the same toward the regime — the distinction exists so diagnostics can
 * say which one is keeping quality relaxed.
 */
export type MotionSourceKind = "explicit" | "inferred";

export type MotionReference = {
  /** Idempotent: releasing twice does not double-decrement. */
  release(): void;
};

export type ViewGovernorMemberOptions = ViewBudgetMemberOptions;

/** Everything a member reports; each field is ignored when not usable. */
export type ViewGovernorMemberUpdate = ViewBudgetMemberUpdate & {
  /** Tile fetch/decode operations physically running (`fetchConcurrency`). */
  physicalTileOperations?: number;
  /** Hierarchy page operations physically running. */
  physicalHierarchyOperations?: number;
  /** Required selection, request, decode, or submission work remains. */
  workPending?: boolean;
};

export type ViewGovernorMember = {
  update(update: ViewGovernorMemberUpdate): void;
  release(): void;
};

export type ViewGovernorOptions = AdaptiveBudgetOptions & {
  /** Maximum fraction of a host frame assigned to VTK. Default 0.7. */
  vtkFrameFraction?: number;
  /** Delay before switching back to stationary allocation. Default 750 ms. */
  interactionSettleMs?: number;
  /**
   * How long rendered cameras must hold still before the inferred motion
   * reference is released (`noteRenderedCameras`). Long enough to bridge a
   * dropped playback frame or a slow scrub step, short enough that a single
   * camera jump refines almost immediately. Default 250 ms. This is regime
   * policy, not deployment configuration: it exists as an option so a test
   * can shorten it.
   */
  motionDebounceMs?: number;
};

/**
 * Which bound explains the budget a cloud is currently drawing to. `demand` is
 * the answer no budget explains: the cloud is drawing everything the camera
 * asks of it, so raising anything would change nothing.
 */
export type BudgetConstraint =
  | "adaptive"
  | "configured-maximum"
  | "memory"
  | "demand"
  | "inactive";

/**
 * The coordinator's allocation for one cloud, re-classified in the governor's
 * constraint vocabulary and joined to the physical work that cloud reports.
 */
export type ViewGovernorMemberStats = Omit<
  ViewBudgetMemberStats,
  "activeConstraint"
> & {
  readonly activeConstraint: BudgetConstraint;
  readonly physicalTileOperations: number;
  readonly physicalHierarchyOperations: number;
};

export type ViewGovernorStats = {
  readonly regime: BudgetRegime;
  readonly motion: {
    readonly explicitReferences: number;
    readonly inferredReferences: number;
    /** What is holding the moving regime, null when nothing is. */
    readonly source: "explicit" | "inferred" | "both" | null;
    /** True while the settle debounce alone is holding the moving regime. */
    readonly settling: boolean;
  };
  readonly activity: {
    readonly inputActive: boolean;
    readonly cameraStable: boolean;
    readonly workPending: boolean;
    /** Core activity permits a capacity sample; timer validity is host-owned. */
    readonly measurementEligible: boolean;
  };
  readonly capacitySamples: {
    readonly eligible: number;
    readonly rejected: number;
    readonly lastEligible: boolean | null;
  };
  /** Frame-time target of the current regime, ms. */
  readonly targetFrameTimeMs: number;
  /** Percentile estimate of the current regime's window, null when empty. */
  readonly estimateMs: number | null;
  /** Frames the current regime has measured under its current budget. */
  readonly samples: number;
  /** What the adaptive loop asks for, before ceilings. */
  readonly trackBudget: number;
  /** Configured maximum, null when none was configured. */
  readonly configuredMaxPoints: number | null;
  /** Memory-derived ceiling summed over reporting active members. */
  readonly memoryCeilingPoints: number | null;
  /** min(track budget, configured maximum, memory ceiling). */
  readonly aggregateBudget: number;
  /** Aggregate point budget kept selected for progressive drawing. */
  readonly selectionBudget: number;
  readonly activeConstraint: BudgetConstraint;
  /** The current regime's most recent decision, including no-change ones. */
  readonly lastAdjustment: BudgetAdjustment | null;
  readonly activeMembers: number;
  readonly members: readonly ViewGovernorMemberStats[];
  /** Physical work outstanding across active members. */
  readonly physicalTileOperations: number;
  readonly physicalHierarchyOperations: number;
  readonly needsFrame: boolean;
};

export type ViewGovernor = {
  register(options: ViewGovernorMemberOptions): ViewGovernorMember;
  /**
   * Replace the governor's options wholesale — an absent field means the same
   * default it means at construction, and identical options are a no-op.
   * A change restarts both adaptive tracks fresh, because their learned
   * budgets measured the old configuration; memberships, motion references,
   * camera-stability state and the host clock offset all survive, so a
   * re-configured view never reads as a torn-down one. An unusable value
   * throws and leaves the governor exactly as it was.
   */
  setOptions(options?: ViewGovernorOptions): void;
  /**
   * Hold the moving regime. References are counted across kinds, so
   * overlapping pointer, wheel, playback, and programmatic motion compose and
   * the regime ends only when the last of them is released.
   */
  beginMotion(kind: MotionSourceKind): MotionReference;
  /** Report a rendered-camera change; one trailing timer owns camera stability. */
  recordCameraChange(): void;
  /**
   * Feed the cameras a pass actually rendered, keyed by host view (any stable
   * key — a renderer instance, an id). Motion beyond recomputation jitter in
   * any of them holds one inferred motion reference for the burst, released
   * after `motionDebounceMs` of stillness; every camera that reaches LOD
   * passing through here is what covers playback, scrubbing and programmatic
   * animation without any of them having to announce itself. A key's first
   * camera is its baseline, not a movement. `scheduleRender` is invoked once
   * when the burst settles, because the settled regime cannot refine quality
   * it never measures — the governor still never schedules frames on its own.
   */
  noteRenderedCameras(
    views: ReadonlyMap<unknown, CameraView | null | undefined>,
    scheduleRender?: () => void,
  ): void;
  /**
   * Forget the rendered-camera baselines. For a host whose view stops
   * feeding cameras (its last cloud left): without this, the first camera
   * after they return is compared against one from before they left, and all
   * the travel between reads as a gesture nobody made.
   */
  resetMotionBaselines(): void;
  /** Immediate responsiveness signal. Never feeds the damped capacity window. */
  recordTransientFrame(metrics: TransientFrameMetrics): void;
  /** Feed one clean or rejected stable-capacity measurement. */
  recordCapacitySample(metrics: CapacitySampleMetrics): void;
  /** Combined fallback for hosts without asynchronous GPU timing. */
  recordHostFrame(metrics: HostFrameMetrics): void;
  /**
   * Whether the host must schedule another frame. Poll it after reporting a
   * frame: the governor never schedules anything itself.
   *
   * True while the camera moves, or while an uncontaminated adaptive track
   * still needs samples to settle its budget. Required tile or hierarchy work
   * makes samples ineligible but does not by itself make the current pixels
   * dirty; the controller or renderer must request a frame when completion
   * changes presentation. False while a stable view is merely waiting.
   */
  needsFrame(): boolean;
  stats(): ViewGovernorStats;
  dispose(): void;
};

type MemberState = {
  readonly budgetMember: ViewBudgetMember;
  readonly id: string | null;
  active: boolean;
  physicalTileOperations: number;
  physicalHierarchyOperations: number;
  workPending: boolean;
};

/**
 * How many qualifying frames in a row an emergency cut requires. With frame
 * time measured as displayed cadence, intervals quantize to whole vsyncs: on a
 * 60 Hz display one missed vsync reads 33.3 ms, just past the 2× threshold of
 * the 16 ms moving target, so a single-frame trigger halves the budget on any
 * isolated hitch — a tile upload landing, a GC pause — that says nothing about
 * the sustainable point count. Two consecutive misses is a sustained cadence,
 * not a hitch. The isolated frame still enters the sampled window, so a scene
 * that hitches every other frame is answered by the damped controller.
 */
const EMERGENCY_CONSECUTIVE_FRAMES = 2;

/**
 * Seed for the moving track when a gesture begins from stationary rest, as a
 * fraction of the settled budget: the machine just sustained the stationary
 * budget at roughly twice the moving frame target, so a quarter of it is a
 * conservative moving-density floor. The seed only ever raises the track —
 * a moving budget the loop learned to hold higher keeps its value.
 *
 * This is also the moving track's recovery path. A vsync-floored display
 * reports a healthy gesture at exactly the display cadence, which sits inside
 * the moving target's dead-band — the track's own frames can therefore never
 * vote to grow it, and without this seed one spurious cut would decimate every
 * later gesture for the rest of the session. The stationary track has no such
 * blind spot (its target is a multiple of the cadence floor), so it recovers,
 * and each fresh gesture inherits that recovery here.
 */
const INTERACTION_SEED_OF_STATIONARY = 0.25;

/** Governor-level defaults, beside the adaptive loop's own `DEFAULTS`. */
const GOVERNOR_DEFAULTS = {
  vtkFrameFraction: 0.7,
  interactionSettleMs: 750,
  motionDebounceMs: 250,
} as const;

/**
 * Every option resolved to the value it configures, `maxBudget`'s absence
 * resolved to null. Two option bags that resolve equal configure the same
 * governor, whether a field was stated or defaulted — which is what makes
 * `setOptions` with an equivalent bag a no-op instead of a track reset.
 */
const RESOLVED_OPTION_DEFAULTS = {
  ...DEFAULTS,
  maxBudget: null,
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
  vtkFrameFraction: number;
  interactionSettleMs: number;
  motionDebounceMs: number;
  budget: AdaptiveBudget;
  /** Read back from the loop rather than re-defaulted, so the two cannot drift. */
  configuredMaxPoints: number | null;
  emergencyCooldownMs: number;
};

/**
 * Validate one options bag and build the adaptive loop it asks for. Everything
 * that can throw happens in here, before any governor state is touched — a
 * bad `setOptions` bag must leave the running configuration intact.
 */
const resolveConfiguration = (
  options: ViewGovernorOptions,
): GovernorConfiguration => {
  const {
    vtkFrameFraction: rawVtkFraction = GOVERNOR_DEFAULTS.vtkFrameFraction,
    interactionSettleMs: rawSettleMs = GOVERNOR_DEFAULTS.interactionSettleMs,
    motionDebounceMs: rawDebounceMs = GOVERNOR_DEFAULTS.motionDebounceMs,
    ...budgetOptions
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
  const budget = createAdaptiveBudget(budgetOptions);
  const { maxBudget: configuredMaxPoints, cooldownMs: emergencyCooldownMs } =
    budget.stats();
  return {
    vtkFrameFraction,
    interactionSettleMs,
    motionDebounceMs,
    budget,
    configuredMaxPoints,
    emergencyCooldownMs,
  };
};

export const createViewGovernor = (
  options: ViewGovernorOptions = {},
): ViewGovernor => {
  let appliedOptions: ViewGovernorOptions = { ...options };
  let {
    vtkFrameFraction,
    interactionSettleMs,
    motionDebounceMs,
    budget,
    configuredMaxPoints,
    emergencyCooldownMs,
  } = resolveConfiguration(options);
  const viewBudget = createViewBudgetCoordinator({
    pointBudget: budget.budget(false),
  });
  const members = new Set<MemberState>();
  let explicitMotion = 0;
  let inferredMotion = 0;
  let disposed = false;
  let cameraStable = true;
  let cameraStabilityTimer: ReturnType<typeof setTimeout> | null = null;
  const settling = (): boolean => !cameraStable && !moving();
  let emergencyCooldownUntil = Number.NEGATIVE_INFINITY;
  let lastEmergencyCutAt = Number.NEGATIVE_INFINITY;
  /** Qualifying moving frames seen in a row; see EMERGENCY_CONSECUTIVE_FRAMES. */
  let emergencyStreak = 0;
  let eligibleCapacitySamples = 0;
  let rejectedCapacitySamples = 0;
  let lastCapacitySampleEligible: boolean | null = null;

  /**
   * The governor stamps cooldowns and restarts on the same timeline the host
   * stamps its frames with. Mixing epochs — host frames on `performance.now()`
   * against `Date.now()` stamps — makes every cooldown comparison meaningless:
   * the tracks either freeze in permanent cooldown or never wait at all.
   * The host's epoch is therefore adopted as an OFFSET from the local clock,
   * not as a stored instant. Holding the instant would freeze time for a host
   * that supplies `metrics.now` only sometimes: every later stamp would repeat
   * the last one it sent, so `now >= emergencyCooldownUntil` could never come
   * true again and the loop would stop adapting for the rest of the session.
   * With an offset a frame that omits the stamp still advances, in the host's
   * own epoch. `Date.now()` is not monotonic, which is the same hazard on a
   * clock step, and is why the offset is re-derived from every stamped frame.
   */
  let hostEpochOffsetMs: number | null = null;
  const stampNow = (): number => Date.now() + (hostEpochOffsetMs ?? 0);

  const moving = (): boolean => explicitMotion + inferredMotion > 0;
  const interacting = (): boolean => moving() || !cameraStable;
  const regime = (): BudgetRegime =>
    interacting() ? "interaction" : "stationary";

  /** Fold the active members without materializing them. */
  const reduceActive = <T>(
    seed: T,
    step: (accumulated: T, member: MemberState) => T,
  ): T => {
    let accumulated = seed;
    for (const member of members) {
      if (member.active) accumulated = step(accumulated, member);
    }
    return accumulated;
  };

  const someActive = (test: (member: MemberState) => boolean): boolean => {
    for (const member of members) {
      if (member.active && test(member)) return true;
    }
    return false;
  };

  const activeCount = (): number => reduceActive(0, (count) => count + 1);

  /**
   * The effective aggregate:
   * `min(track budget, configured maximum, memory-derived ceiling)`.
   * The configured maximum is the adaptive loop's own upper range bound, so
   * the track budget already carries it; only the memory ceiling is left to
   * apply here — and it applies to the aggregate, before the split, so no
   * member's share can be sized against memory another member owns.
   */
  const cappedTrackBudget = (
    interactionTrack: boolean,
    ceiling: number | null,
  ): number => {
    const track = budget.budget(interactionTrack);
    return ceiling === null ? track : Math.min(track, ceiling);
  };

  const aggregateBudget = (ceiling: number | null): number =>
    cappedTrackBudget(interacting(), ceiling);

  /**
   * Motion thins the selected set instead of replacing it. Usually the
   * stationary track is denser; if the interaction track has proved it can
   * draw more, retain enough points to honor that budget too.
   */
  const selectionBudget = (ceiling: number | null): number =>
    interacting()
      ? Math.max(
          cappedTrackBudget(false, ceiling),
          cappedTrackBudget(true, ceiling),
        )
      : cappedTrackBudget(false, ceiling);

  const constraintOf = (
    active: number,
    memory: number | null,
  ): BudgetConstraint => {
    if (active === 0) return "inactive";
    const track = budget.budget(interacting());
    if (memory !== null && memory <= track) {
      // Ties go to the harder constraint: when memory and the configured
      // maximum both sit exactly at the loop's budget, memory is what raising
      // the configured maximum would fail to lift.
      if (configuredMaxPoints === null || memory <= configuredMaxPoints) {
        return "memory";
      }
    }
    if (configuredMaxPoints !== null && configuredMaxPoints <= track) {
      return "configured-maximum";
    }
    return "adaptive";
  };

  const distribute = (): void => {
    if (disposed) return;
    const ceiling = viewBudget.memoryCeiling();
    // Hand the loop the memory ceiling before reading it. Clamping only the
    // aggregate would let the track integrate toward frame-time headroom the
    // memory ceiling never allows it to spend: the effective budget would sit
    // still while the track climbed for ever, so it would never report itself
    // pinned and a host watching `needsFrame()` would repaint for ever.
    budget.setCeiling(ceiling);
    const drawTotal = aggregateBudget(ceiling);
    const selectionTotal = selectionBudget(ceiling);
    viewBudget.setTargets(selectionTotal, drawTotal);
  };

  const clearCameraStabilityTimer = (): void => {
    if (cameraStabilityTimer !== null) clearTimeout(cameraStabilityTimer);
    cameraStabilityTimer = null;
  };

  const enterInteraction = (): void => {
    emergencyStreak = 0;
    budget.restartAt(
      true,
      Math.max(
        budget.budget(true),
        INTERACTION_SEED_OF_STATIONARY * budget.budget(false),
      ),
      stampNow(),
    );
    distribute();
  };

  const enterStationary = (): void => {
    budget.restartAt(
      false,
      Math.max(budget.budget(false), budget.budget(true)),
      stampNow(),
    );
    distribute();
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
        if (disposed || moving() || !cameraStable) return;
        enterStationary();
      },
    };
  };

  /**
   * The inferred-motion classifier: baselines of the cameras each host view
   * last rendered, and the single burst reference their motion holds. One
   * reference per burst of motion, never one per frame — the governor
   * restarts its moving track whenever the first reference is taken, so a
   * per-frame reference would keep resetting the window it needs to learn
   * from.
   */
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

  const pendingWork = (): boolean =>
    someActive(
      (member) =>
        member.workPending ||
        member.physicalTileOperations > 0 ||
        member.physicalHierarchyOperations > 0,
    );

  /**
   * A track has converged once a full window measured under the current budget
   * lands inside the dead-band, or the budget is pinned at a clamp bound it
   * cannot move off. Every other outcome — a change, a cooldown, a half-filled
   * window, a fresh seed — means the loop is still working.
   */
  const converged = (): boolean => {
    const last = budget.stats().stationary.lastAdjustment;
    return (
      last !== null &&
      (last.reason === "within-hysteresis" || last.reason === "clamped")
    );
  };

  const needsFrame = (pending: boolean): boolean =>
    !disposed && (interacting() || (!pending && !converged()));

  /**
   * Adopt the host's clock offset from this metrics packet and return "now" on
   * the host's timeline. Every timestamp the governor compares — cooldowns,
   * emergency cuts, budget samples — has to come from one epoch.
   */
  const stampFrom = (metrics: { readonly now?: number }): number => {
    const localNow = Date.now();
    if (finiteNonNegative(metrics.now)) {
      hostEpochOffsetMs = metrics.now - localNow;
    }
    return localNow + (hostEpochOffsetMs ?? 0);
  };

  /**
   * What the frame really cost: the host's own frame time, and the vtk and GPU
   * spans grossed up by the fraction of the frame vtk is expected to own. The
   * worst of the three is the one the budget must answer to.
   */
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

  /**
   * A frame may train the budget only when nothing is still streaming into it
   * and the camera is in a settled state — either genuinely moving, or stable.
   */
  const coreEligible = (pending: boolean): boolean =>
    !pending && (moving() || cameraStable);

  const recordTransientFrame = (metrics: TransientFrameMetrics): number => {
    if (disposed || !finiteNonNegative(metrics?.hostFrameMs)) return 0;
    const now = stampFrom(metrics);
    const severeInput =
      (finiteNonNegative(metrics.inputDelayMs) && metrics.inputDelayMs > 50) ||
      (finiteNonNegative(metrics.longTaskMs) && metrics.longTaskMs > 50);
    const inMotion = moving();
    const target = budget.target(interacting());
    const observedMs = observedFrameMs(metrics);
    const emergency = inMotion && (severeInput || observedMs > target * 2);
    if (!emergency) emergencyStreak = 0;
    if (
      emergency &&
      now >= emergencyCooldownUntil &&
      ++emergencyStreak >= EMERGENCY_CONSECUTIVE_FRAMES
    ) {
      emergencyStreak = 0;
      budget.reduceNow(true, now);
      lastEmergencyCutAt = now;
      emergencyCooldownUntil = now + emergencyCooldownMs;
      distribute();
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
    budget.recordFrame(metrics.frameMs, {
      interacting: metrics.regime === "interaction",
      now,
    });
    distribute();
  };

  return {
    register(memberOptions) {
      const budgetMember = viewBudget.register(memberOptions);
      const state: MemberState = {
        budgetMember,
        id: memberOptions.id ?? null,
        active: memberOptions.active ?? true,
        physicalTileOperations: 0,
        physicalHierarchyOperations: 0,
        workPending: false,
      };
      members.add(state);
      distribute();
      return {
        update(next) {
          if (disposed || !members.has(state)) return;
          if (next.active !== undefined) state.active = next.active;
          budgetMember.update(next);
          if (finiteNonNegative(next.physicalTileOperations)) {
            state.physicalTileOperations = Math.floor(
              next.physicalTileOperations,
            );
          }
          if (finiteNonNegative(next.physicalHierarchyOperations)) {
            state.physicalHierarchyOperations = Math.floor(
              next.physicalHierarchyOperations,
            );
          }
          if (next.workPending !== undefined) {
            state.workPending = next.workPending;
          }
          distribute();
        },
        release() {
          if (!members.delete(state)) return;
          budgetMember.release();
          distribute();
        },
      };
    },

    setOptions(next = {}) {
      if (disposed || sameOptions(appliedOptions, next)) return;
      // Resolve first: this is where unusable values throw, and they must do
      // so before the running configuration has been touched.
      const configuration = resolveConfiguration(next);
      appliedOptions = { ...next };
      ({
        vtkFrameFraction,
        interactionSettleMs,
        motionDebounceMs,
        budget,
        configuredMaxPoints,
        emergencyCooldownMs,
      } = configuration);
      // The replaced loop's history measured the old configuration: emergency
      // bookkeeping and the capacity-sample tallies restart with it. The host
      // clock offset survives — the host's timeline did not change.
      emergencyStreak = 0;
      emergencyCooldownUntil = Number.NEGATIVE_INFINITY;
      lastEmergencyCutAt = Number.NEGATIVE_INFINITY;
      eligibleCapacitySamples = 0;
      rejectedCapacitySamples = 0;
      lastCapacitySampleEligible = null;
      // Memberships and motion references were not disturbed, so the fresh
      // tracks re-enter the regime the view is actually in.
      if (interacting()) enterInteraction();
      else distribute();
    },

    beginMotion: takeMotionReference,

    recordCameraChange: markCameraChanged,

    noteRenderedCameras(views, scheduleRender) {
      if (disposed) return;
      let moved = false;
      for (const [key, view] of views) {
        if (view && cameraMoved(renderedCameras.get(key), view)) moved = true;
      }
      // Replace rather than merge: a view that stops reporting must not keep
      // a stale baseline that reads as motion when it comes back.
      renderedCameras = new Map(views);
      if (!moved) return;
      markCameraChanged();
      if (!inferredBurst) inferredBurst = takeMotionReference("inferred");
      if (inferredBurstTimer !== null) clearTimeout(inferredBurstTimer);
      inferredBurstTimer = setTimeout(() => {
        endInferredBurst();
        // The settled regime cannot refine quality it never measures, so
        // hand the host one frame to start from.
        scheduleRender?.();
      }, motionDebounceMs);
    },

    resetMotionBaselines() {
      renderedCameras = new Map();
    },

    recordTransientFrame,

    recordCapacitySample,

    recordHostFrame(metrics) {
      if (disposed || !finiteNonNegative(metrics?.hostFrameMs)) return;
      const observedMs = recordTransientFrame(metrics);
      recordCapacitySample({
        frameMs: observedMs,
        regime: regime(),
        eligible:
          (metrics.capacitySampleEligible ?? true) &&
          coreEligible(pendingWork()),
        now: metrics.now,
      });
    },

    needsFrame: () => needsFrame(pendingWork()),

    stats() {
      const adaptive = budget.stats();
      const track = interacting() ? adaptive.interaction : adaptive.stationary;
      const active = activeCount();
      const pending = pendingWork();
      const allocation = viewBudget.stats();
      const viewCeiling = allocation.memoryCeilingPoints;
      const viewConstraint = constraintOf(active, viewCeiling);
      const memberStats = [...members].map(
        (member): ViewGovernorMemberStats => {
          const allocated = member.budgetMember.stats();
          return {
            ...allocated,
            activeConstraint: memberConstraint(allocated, viewConstraint),
            physicalTileOperations: member.physicalTileOperations,
            physicalHierarchyOperations: member.physicalHierarchyOperations,
          };
        },
      );
      const sum = (pick: (member: MemberState) => number): number =>
        reduceActive(0, (total, member) => total + pick(member));
      return {
        regime: regime(),
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
          settling: settling(),
        },
        activity: {
          inputActive: moving(),
          cameraStable,
          workPending: pending,
          measurementEligible: coreEligible(pending),
        },
        capacitySamples: {
          eligible: eligibleCapacitySamples,
          rejected: rejectedCapacitySamples,
          lastEligible: lastCapacitySampleEligible,
        },
        targetFrameTimeMs: track.targetMs,
        estimateMs: track.estimateMs,
        samples: track.samples,
        trackBudget: track.budget,
        configuredMaxPoints,
        memoryCeilingPoints: viewCeiling,
        aggregateBudget: allocation.aggregateBudget,
        selectionBudget: allocation.selectionBudget,
        activeConstraint: viewConstraint,
        lastAdjustment: track.lastAdjustment,
        activeMembers: active,
        members: memberStats,
        physicalTileOperations: sum((m) => m.physicalTileOperations),
        physicalHierarchyOperations: sum((m) => m.physicalHierarchyOperations),
        needsFrame: needsFrame(pending),
      };
    },

    dispose() {
      if (disposed) return;
      disposed = true;
      endInferredBurst();
      clearCameraStabilityTimer();
      members.clear();
      viewBudget.dispose();
    },
  };
};
