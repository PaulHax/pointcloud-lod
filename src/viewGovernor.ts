/**
 * One adaptive point budget per view.
 *
 * The governor owns the regime (camera moving vs settled), the adaptive loop
 * that sizes the aggregate view budget, the ceilings that bound it, and the
 * split of that aggregate across the clouds drawing into the view. It does not
 * own the render loop: the host reports every completed frame through
 * frame pressure and clean capacity samples, then asks `needsFrame()` whether
 * another one is required.
 */

import {
  createAdaptiveBudget,
  type AdaptiveBudget,
  type AdaptiveBudgetOptions,
  type BudgetAdjustment,
  type BudgetRegime,
} from "./adaptiveBudget";
import { finiteAtLeast, finiteNonNegative, finiteWithin } from "./numeric";

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

export type ViewGovernorMemberOptions = {
  setPointBudget(points: number): void;
  setDensityFraction(densityFraction: number): void;
  /** Names this cloud in the diagnostics. */
  id?: string;
};

/** Everything a member reports; each field is ignored when not usable. */
export type ViewGovernorMemberUpdate = {
  active?: boolean;
  /** Root projected screen-space error from the controller's selection stats. */
  projectedImportance?: number;
  /** The controller's memory-derived point ceiling. */
  memoryCeilingPoints?: number;
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
};

/** Which bound explains the budget a cloud is currently drawing to. */
export type BudgetConstraint =
  | "adaptive"
  | "configured-maximum"
  | "memory"
  | "inactive";

export type ViewGovernorMemberStats = {
  readonly id: string | null;
  readonly active: boolean;
  /** Null until the member reports; 0 is a real measurement, not "unknown". */
  readonly projectedImportance: number | null;
  /** Points this member was allocated from the aggregate view budget. */
  readonly allocatedShare: number;
  /** Last memory-derived ceiling the member reported, null if it never has. */
  readonly memoryCeilingPoints: number | null;
  /**
   * What the cloud can actually draw: its share capped by its own memory
   * ceiling. The controller re-applies its live ceiling to whatever it is
   * given, so this is the governor's view of the same arithmetic.
   */
  readonly effectiveBudget: number;
  /** Points kept selected and resident so density changes need no tile churn. */
  readonly selectionShare: number;
  /** Selection share capped by this member's memory ceiling. */
  readonly effectiveSelectionBudget: number;
  /** Effective draw budget divided by the effective selection budget. */
  readonly densityFraction: number;
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
   * Hold the moving regime. References are counted across kinds, so
   * overlapping pointer, wheel, playback, and programmatic motion compose and
   * the regime ends only when the last of them is released.
   */
  beginMotion(kind: MotionSourceKind): MotionReference;
  /** Report a rendered-camera change; one trailing timer owns camera stability. */
  recordCameraChange(): void;
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
  setPointBudget(points: number): void;
  setDensityFraction(densityFraction: number): void;
  readonly id: string | null;
  active: boolean;
  /** Null until the member reports; 0 is a real measurement, not "unknown". */
  importance: number | null;
  memoryCeilingPoints: number | null;
  physicalTileOperations: number;
  physicalHierarchyOperations: number;
  workPending: boolean;
  drawBudget: number;
  selectionBudget: number;
  densityFraction: number;
};

/**
 * Projected importance is unbounded, so a strictly proportional split lets one
 * dominant view drive another to a handful of points. Clamping the weight
 * spread guarantees every active view at least this fraction of an even split.
 */
const MIN_SHARE_OF_EVEN_SPLIT = 0.25;

/**
 * A member's share moves with its memory ceiling, and the ceiling is derived
 * from measured bytes-per-point, which shifts a little with every tile that
 * lands or leaves. Applying a share is a synchronous, undebounced reselection
 * in the controller, so a change has to be worth one before it is applied.
 * Shares are recomputed from the fresh aggregate every pass, so skipped drift
 * accumulates and is applied once it crosses the band rather than being lost.
 */
const DISTRIBUTE_DEADBAND = 0.01;

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

export const createViewGovernor = (
  options: ViewGovernorOptions = {},
): ViewGovernor => {
  const {
    vtkFrameFraction: rawVtkFraction = 0.7,
    interactionSettleMs: rawSettleMs = 750,
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
  const budget: AdaptiveBudget = createAdaptiveBudget(budgetOptions);
  // Configuration, so these are fixed for the governor's life. The cooldown is
  // read back from the loop instead of re-defaulted here, so the two cannot
  // drift apart.
  const { maxBudget: configuredMaxPoints, cooldownMs: emergencyCooldownMs } =
    budget.stats();
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

  const activeMembers = (): MemberState[] =>
    [...members].filter((member) => member.active);

  /**
   * The view's memory ceiling is the sum of the ceilings its active members
   * report: each controller owns a byte share of one pool, so their point
   * ceilings add. Members that have never reported are left out — an unknown
   * ceiling contributes no known headroom — and if nobody reports, the memory
   * bound lives entirely in the controllers.
   */
  const memoryCeilingPoints = (
    active: readonly MemberState[],
  ): number | null => {
    let total = 0;
    let reported = false;
    for (const member of active) {
      if (member.memoryCeilingPoints === null) continue;
      total += member.memoryCeilingPoints;
      reported = true;
    }
    return reported ? total : null;
  };

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
    active: readonly MemberState[],
    memory: number | null,
  ): BudgetConstraint => {
    if (active.length === 0) return "inactive";
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
    const active = activeMembers();
    const ceiling = memoryCeilingPoints(active);
    // Hand the loop the memory ceiling before reading it. Clamping only the
    // aggregate would let the track integrate toward frame-time headroom the
    // memory ceiling never allows it to spend: the effective budget would sit
    // still while the track climbed for ever, so it would never report itself
    // pinned and a host watching `needsFrame()` would repaint for ever.
    budget.setCeiling(ceiling);
    const drawTotal = aggregateBudget(ceiling);
    const selectionTotal = selectionBudget(ceiling);
    // Importance is root screen-space error in CSS px, so it is legitimately
    // below 1 for a distant cloud and exactly 0 for one that is fully culled
    // or still loading. Neither may be treated as "unknown": a member that
    // has never reported is the only one that needs a stand-in, and it gets
    // the largest reported weight so registering does not starve it.
    const maxReported = active.reduce(
      (max, member) => Math.max(max, member.importance ?? 0),
      0,
    );
    // Every active member reporting zero means nothing is on screen, so there
    // is no basis to prefer one over another: equal weights split evenly.
    const floorWeight =
      maxReported > 0 ? maxReported * MIN_SHARE_OF_EVEN_SPLIT : 1;
    const weightOf = (member: MemberState): number =>
      Math.max(member.importance ?? maxReported, floorWeight);
    const weightTotal = active.reduce((sum, m) => sum + weightOf(m), 0);
    const shareOf = (member: MemberState, total: number): number =>
      member.active
        ? Math.max(1, Math.floor((total * weightOf(member)) / weightTotal))
        : 0;
    for (const member of members) {
      const nextSelection = shareOf(member, selectionTotal);
      if (nextSelection !== member.selectionBudget) {
        // Deactivation (0) and first assignment always apply; between live
        // budgets, jitter smaller than the dead-band is not worth the
        // synchronous reselection it would trigger. The stored value is the
        // applied one, so skipped drift accumulates for the next comparison.
        const jitter =
          member.selectionBudget > 0 &&
          nextSelection > 0 &&
          Math.abs(nextSelection - member.selectionBudget) <
            member.selectionBudget * DISTRIBUTE_DEADBAND;
        if (!jitter) {
          member.selectionBudget = nextSelection;
          member.setPointBudget(nextSelection);
        }
      }

      const nextDraw = shareOf(member, drawTotal);
      const memberCeiling = member.memoryCeilingPoints;
      const effectiveSelection =
        memberCeiling === null
          ? Math.max(member.selectionBudget, 0)
          : Math.min(Math.max(member.selectionBudget, 0), memberCeiling);
      const effectiveDraw =
        memberCeiling === null ? nextDraw : Math.min(nextDraw, memberCeiling);
      const nextDensity =
        effectiveSelection > 0
          ? Math.min(1, effectiveDraw / effectiveSelection)
          : 0;
      const densityJitter =
        member.densityFraction >= 0 &&
        Math.abs(nextDensity - member.densityFraction) < DISTRIBUTE_DEADBAND;
      member.drawBudget = nextDraw;
      if (!densityJitter) {
        member.densityFraction = nextDensity;
        member.setDensityFraction(nextDensity);
      }
    }
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

  const pendingWork = (): boolean =>
    activeMembers().some(
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

  const needsFrame = (): boolean =>
    !disposed && (interacting() || (!pendingWork() && !converged()));

  const recordTransientFrame = (metrics: TransientFrameMetrics): void => {
    if (disposed || !finiteNonNegative(metrics?.hostFrameMs)) return;
    const localNow = Date.now();
    if (finiteNonNegative(metrics.now)) {
      hostEpochOffsetMs = metrics.now - localNow;
    }
    const now = localNow + (hostEpochOffsetMs ?? 0);
    const candidates = [metrics.hostFrameMs];
    if (finiteNonNegative(metrics.vtkFrameMs)) {
      candidates.push(metrics.vtkFrameMs / vtkFrameFraction);
    }
    if (finiteNonNegative(metrics.gpuMs)) {
      candidates.push(metrics.gpuMs / vtkFrameFraction);
    }
    const severeInput =
      (finiteNonNegative(metrics.inputDelayMs) && metrics.inputDelayMs > 50) ||
      (finiteNonNegative(metrics.longTaskMs) && metrics.longTaskMs > 50);
    const inMotion = moving();
    const target = budget.target(interacting());
    const observedMs = Math.max(...candidates);
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
  };

  const recordCapacitySample = (metrics: CapacitySampleMetrics): void => {
    if (disposed || !finiteNonNegative(metrics?.frameMs)) return;
    const localNow = Date.now();
    if (finiteNonNegative(metrics.now)) {
      hostEpochOffsetMs = metrics.now - localNow;
    }
    const now = localNow + (hostEpochOffsetMs ?? 0);
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
      const state: MemberState = {
        setPointBudget: memberOptions.setPointBudget,
        setDensityFraction: memberOptions.setDensityFraction,
        id: memberOptions.id ?? null,
        active: true,
        importance: null,
        memoryCeilingPoints: null,
        physicalTileOperations: 0,
        physicalHierarchyOperations: 0,
        workPending: false,
        drawBudget: -1,
        selectionBudget: -1,
        densityFraction: -1,
      };
      members.add(state);
      distribute();
      return {
        update(next) {
          if (disposed || !members.has(state)) return;
          if (next.active !== undefined) state.active = next.active;
          if (finiteNonNegative(next.projectedImportance)) {
            state.importance = next.projectedImportance;
          }
          if (finiteNonNegative(next.memoryCeilingPoints)) {
            state.memoryCeilingPoints = Math.floor(next.memoryCeilingPoints);
          }
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
          distribute();
        },
      };
    },

    beginMotion(kind) {
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
    },

    recordCameraChange: markCameraChanged,

    recordTransientFrame,

    recordCapacitySample,

    recordHostFrame(metrics) {
      if (disposed || !finiteNonNegative(metrics?.hostFrameMs)) return;
      recordTransientFrame(metrics);
      const candidates = [metrics.hostFrameMs];
      if (finiteNonNegative(metrics.vtkFrameMs)) {
        candidates.push(metrics.vtkFrameMs / vtkFrameFraction);
      }
      if (finiteNonNegative(metrics.gpuMs)) {
        candidates.push(metrics.gpuMs / vtkFrameFraction);
      }
      const observedMs = Math.max(...candidates);
      const coreEligible = !pendingWork() && (moving() || cameraStable);
      recordCapacitySample({
        frameMs: observedMs,
        regime: regime(),
        eligible: (metrics.capacitySampleEligible ?? true) && coreEligible,
        now: metrics.now,
      });
    },

    needsFrame,

    stats() {
      const adaptive = budget.stats();
      const track = interacting() ? adaptive.interaction : adaptive.stationary;
      const active = activeMembers();
      const viewCeiling = memoryCeilingPoints(active);
      const viewConstraint = constraintOf(active, viewCeiling);
      const aggregate = aggregateBudget(viewCeiling);
      const selected = selectionBudget(viewCeiling);
      const memberStats = [...members].map(
        (member): ViewGovernorMemberStats => {
          const ceiling = member.memoryCeilingPoints;
          const share = member.active ? Math.max(member.drawBudget, 0) : 0;
          const selectionShare = member.active
            ? Math.max(member.selectionBudget, 0)
            : 0;
          return {
            id: member.id,
            active: member.active,
            projectedImportance: member.importance,
            allocatedShare: share,
            memoryCeilingPoints: ceiling,
            effectiveBudget:
              ceiling === null ? share : Math.min(share, ceiling),
            selectionShare,
            effectiveSelectionBudget:
              ceiling === null
                ? selectionShare
                : Math.min(selectionShare, ceiling),
            densityFraction: Math.max(member.densityFraction, 0),
            activeConstraint: !member.active
              ? "inactive"
              : ceiling !== null && ceiling < share
                ? "memory"
                : viewConstraint,
            physicalTileOperations: member.physicalTileOperations,
            physicalHierarchyOperations: member.physicalHierarchyOperations,
          };
        },
      );
      const sum = (pick: (member: MemberState) => number): number =>
        active.reduce((total, member) => total + pick(member), 0);
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
          workPending: pendingWork(),
          measurementEligible: !pendingWork() && (moving() || cameraStable),
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
        aggregateBudget: aggregate,
        selectionBudget: selected,
        activeConstraint: viewConstraint,
        lastAdjustment: track.lastAdjustment,
        activeMembers: active.length,
        members: memberStats,
        physicalTileOperations: sum((m) => m.physicalTileOperations),
        physicalHierarchyOperations: sum((m) => m.physicalHierarchyOperations),
        needsFrame: needsFrame(),
      };
    },

    dispose() {
      if (disposed) return;
      disposed = true;
      clearCameraStabilityTimer();
      members.clear();
    },
  };
};
