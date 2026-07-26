/**
 * One adaptive point budget per view.
 *
 * The governor owns the regime (camera moving vs settled), the adaptive loop
 * that sizes the aggregate view budget, the ceilings that bound it, and the
 * split of that aggregate across the clouds drawing into the view. It does not
 * own the render loop: the host reports every completed frame through
 * `recordHostFrame` and asks `needsFrame()` whether another one is required.
 */

import {
  createAdaptiveBudget,
  DEFAULTS,
  type AdaptiveBudget,
  type AdaptiveBudgetOptions,
  type AdaptiveBudgetStats,
  type BudgetAdjustment,
  type BudgetRegime,
} from "./adaptiveBudget";
import { finiteAtLeast, finiteNonNegative, finiteWithin } from "./numeric";

export interface HostFrameMetrics {
  /** Complete host frame time, including non-VTK work. */
  readonly hostFrameMs: number;
  /** Time spent painting VTK content within the host frame. */
  readonly vtkFrameMs?: number;
  /** Asynchronously collected GPU time for the VTK paint. */
  readonly gpuMs?: number;
  readonly inputDelayMs?: number;
  readonly longTaskMs?: number;
  readonly now?: number;
}

/**
 * What is holding the moving regime. Explicit sources are user gestures the
 * host reports directly; inferred sources are rendered-camera motion a
 * classifier detected (playback, scrubbing, programmatic animation). Both
 * count the same toward the regime — the distinction exists so diagnostics can
 * say which one is keeping quality relaxed.
 */
export type MotionSourceKind = "explicit" | "inferred";

export interface MotionReference {
  /** Idempotent: releasing twice does not double-decrement. */
  release(): void;
}

export interface ViewGovernorMemberOptions {
  setPointBudget(points: number): void;
  active?: boolean;
  /** Names this cloud in the diagnostics. */
  id?: string;
}

/** Everything a member reports; each field is ignored when not usable. */
export interface ViewGovernorMemberUpdate {
  active?: boolean;
  /** Root projected screen-space error from the controller's selection stats. */
  projectedImportance?: number;
  /** The controller's memory-derived point ceiling. */
  memoryCeilingPoints?: number;
  /** Tile fetch/decode operations physically running (`fetchConcurrency`). */
  physicalTileOperations?: number;
  /** Hierarchy page operations physically running. */
  physicalHierarchyOperations?: number;
}

export interface ViewGovernorMember {
  update(update: ViewGovernorMemberUpdate): void;
  release(): void;
}

export interface ViewGovernorOptions extends AdaptiveBudgetOptions {
  /** Maximum fraction of a host frame assigned to VTK. Default 0.7. */
  vtkFrameFraction?: number;
  /** Delay before switching back to stationary allocation. Default 750 ms. */
  interactionSettleMs?: number;
}

/** Which bound explains the budget a cloud is currently drawing to. */
export type BudgetConstraint =
  | "adaptive"
  | "configured-maximum"
  | "memory"
  | "inactive";

export interface ViewGovernorMemberStats {
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
  readonly activeConstraint: BudgetConstraint;
  readonly physicalTileOperations: number;
  readonly physicalHierarchyOperations: number;
}

export interface ViewGovernorStats {
  readonly regime: BudgetRegime;
  readonly motion: {
    readonly explicitReferences: number;
    readonly inferredReferences: number;
    /** What is holding the moving regime, null when nothing is. */
    readonly source: "explicit" | "inferred" | "both" | null;
    /** True while the settle debounce alone is holding the moving regime. */
    readonly settling: boolean;
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
  readonly activeConstraint: BudgetConstraint;
  /** The current regime's most recent decision, including no-change ones. */
  readonly lastAdjustment: BudgetAdjustment | null;
  readonly activeMembers: number;
  readonly members: readonly ViewGovernorMemberStats[];
  /** Physical work outstanding across active members. */
  readonly physicalTileOperations: number;
  readonly physicalHierarchyOperations: number;
  readonly needsFrame: boolean;
  readonly adaptive: AdaptiveBudgetStats;
}

export interface ViewGovernor {
  register(options: ViewGovernorMemberOptions): ViewGovernorMember;
  /**
   * Hold the moving regime. References are counted across kinds, so
   * overlapping pointer, wheel, playback, and programmatic motion compose and
   * the regime ends only when the last of them is released.
   */
  beginMotion(kind: MotionSourceKind): MotionReference;
  recordHostFrame(metrics: HostFrameMetrics): void;
  /**
   * Whether the host must schedule another frame. Poll it after reporting a
   * frame: the governor never schedules anything itself.
   *
   * True while the camera moves, while the stationary track is still moving
   * its budget (or has not measured enough to decide), and while any member
   * still has physical tile or hierarchy work running. False once selection
   * and loading converge and the budget lands inside the dead-band — at which
   * point repainting would show exactly the same pixels.
   */
  needsFrame(): boolean;
  stats(): ViewGovernorStats;
  dispose(): void;
}

interface MemberState {
  setPointBudget(points: number): void;
  readonly id: string | null;
  active: boolean;
  /** Null until the member reports; 0 is a real measurement, not "unknown". */
  importance: number | null;
  memoryCeilingPoints: number | null;
  physicalTileOperations: number;
  physicalHierarchyOperations: number;
  budget: number;
}

/**
 * Projected importance is unbounded, so a strictly proportional split lets one
 * dominant view drive another to a handful of points. Clamping the weight
 * spread guarantees every active view at least this fraction of an even split.
 */
const MIN_SHARE_OF_EVEN_SPLIT = 0.25;

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
  // Configuration, so it is fixed for the governor's life.
  const configuredMaxPoints = budget.stats().maxBudget;
  // Shared with the loop's own cooldown so the two cannot drift apart.
  const emergencyCooldownMs = budgetOptions.cooldownMs ?? DEFAULTS.cooldownMs;
  const members = new Set<MemberState>();
  let explicitMotion = 0;
  let inferredMotion = 0;
  let settling = false;
  let disposed = false;
  let settleTimer: ReturnType<typeof setTimeout> | null = null;
  let emergencyCooldownUntil = Number.NEGATIVE_INFINITY;

  const moving = (): boolean => explicitMotion + inferredMotion > 0;
  const interacting = (): boolean => moving() || settling;
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
  const memoryCeilingPoints = (): number | null => {
    let total = 0;
    let reported = false;
    for (const member of activeMembers()) {
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
  const aggregateBudget = (): number => {
    const ceiling = memoryCeilingPoints();
    const track = budget.budget(interacting());
    return ceiling === null ? track : Math.min(track, ceiling);
  };

  const constraintOf = (): BudgetConstraint => {
    const active = activeMembers();
    if (active.length === 0) return "inactive";
    const track = budget.budget(interacting());
    const memory = memoryCeilingPoints();
    // Ties go to the harder constraint: when memory and the configured maximum
    // both sit exactly at the loop's budget, memory is what raising the
    // configured maximum would fail to lift.
    const underConfigured =
      configuredMaxPoints === null || memory === null
        ? true
        : memory <= configuredMaxPoints;
    if (memory !== null && memory <= track && underConfigured) return "memory";
    if (configuredMaxPoints !== null && configuredMaxPoints <= track) {
      return "configured-maximum";
    }
    return "adaptive";
  };

  const distribute = (): void => {
    if (disposed) return;
    const active = activeMembers();
    // Hand the loop the memory ceiling before reading it. Clamping only the
    // aggregate would let the track integrate toward frame-time headroom the
    // memory ceiling never allows it to spend: the effective budget would sit
    // still while the track climbed for ever, so it would never report itself
    // pinned and a host watching `needsFrame()` would repaint for ever.
    budget.setCeiling(memoryCeilingPoints());
    const total = aggregateBudget();
    // Importance is root screen-space error in CSS px, so it is legitimately
    // below 1 for a distant cloud and exactly 0 for one that is fully culled
    // or still loading. Neither may be treated as "unknown": a member that
    // has never reported is the only one that needs a stand-in, and it gets
    // the largest reported weight so registering does not starve it.
    const reported = active.filter((member) => member.importance !== null);
    const maxReported = Math.max(
      0,
      ...reported.map((member) => member.importance ?? 0),
    );
    const rawWeight = (member: MemberState): number =>
      member.importance ?? maxReported;
    const floorWeight = maxReported * MIN_SHARE_OF_EVEN_SPLIT;
    const weightOf = (member: MemberState): number =>
      Math.max(rawWeight(member), floorWeight);
    const weightTotal = active.reduce((sum, m) => sum + weightOf(m), 0);
    for (const member of members) {
      const next = member.active
        ? weightTotal > 0
          ? Math.max(1, Math.floor((total * weightOf(member)) / weightTotal))
          : // Every active member reports zero: nothing is on screen, so
            // there is no basis to prefer one over another.
            Math.max(1, Math.floor(total / active.length))
        : 0;
      if (next === member.budget) continue;
      member.budget = next;
      member.setPointBudget(next);
    }
  };

  const clearSettle = (): void => {
    if (settleTimer !== null) clearTimeout(settleTimer);
    settleTimer = null;
  };

  const startSettle = (): void => {
    settling = true;
    clearSettle();
    settleTimer = setTimeout(() => {
      settleTimer = null;
      settling = false;
      // Stationary refinement starts from exactly the density the moving
      // regime just sustained, so releasing the camera changes nothing on
      // screen, and measures it fresh: samples taken while moving describe a
      // different regime and must not decide the next stationary step.
      budget.restartAt(false, budget.budget(true), Date.now());
      distribute();
    }, interactionSettleMs);
  };

  const pendingWork = (): boolean =>
    activeMembers().some(
      (member) =>
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
    !disposed && (interacting() || !converged() || pendingWork());

  return {
    register(memberOptions) {
      const state: MemberState = {
        setPointBudget: memberOptions.setPointBudget,
        id: memberOptions.id ?? null,
        active: memberOptions.active ?? true,
        importance: null,
        memoryCeilingPoints: null,
        physicalTileOperations: 0,
        physicalHierarchyOperations: 0,
        budget: -1,
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
        if (kind === "explicit") explicitMotion += 1;
        else inferredMotion += 1;
        if (explicitMotion + inferredMotion === 1) {
          clearSettle();
          settling = false;
          // The moving track's samples are from the previous burst of motion;
          // the scene and the load have moved on. Restart it at the density it
          // learned so this burst decides on its own frames. A motion source
          // holds its reference for the whole burst, so this runs once per
          // burst rather than once per frame.
          budget.restartAt(true, budget.budget(true), Date.now());
          distribute();
        }
      }
      return {
        release() {
          if (!held) return;
          held = false;
          if (kind === "explicit") explicitMotion -= 1;
          else inferredMotion -= 1;
          if (disposed || moving()) return;
          startSettle();
        },
      };
    },

    recordHostFrame(metrics) {
      if (disposed || !finiteNonNegative(metrics?.hostFrameMs)) return;
      const now = finiteNonNegative(metrics.now) ? metrics.now : Date.now();
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
      const inInteractionRegime = interacting();
      const target = budget.target(inInteractionRegime);
      const observedMs = Math.max(...candidates);
      // Emergency cuts protect live motion. Once motion has stopped, isolated
      // long tasks and missed frames use the sampled stationary controller so
      // they cannot drive a fast grow/halve density sawtooth.
      const emergency = moving() && (severeInput || observedMs > target * 2);
      // After an emergency, hold the cut long enough to measure the cheaper
      // rendering regime before allowing the normal controller to grow again.
      if (emergency) {
        if (now >= emergencyCooldownUntil) {
          budget.reduceNow(inInteractionRegime, now, 0.5);
          emergencyCooldownUntil = now + emergencyCooldownMs;
        }
      } else if (now >= emergencyCooldownUntil) {
        budget.recordFrame(observedMs, {
          interacting: inInteractionRegime,
          now,
        });
      }
      distribute();
    },

    needsFrame,

    stats() {
      const adaptive = budget.stats();
      const track = interacting() ? adaptive.interaction : adaptive.stationary;
      const viewConstraint = constraintOf();
      const aggregate = aggregateBudget();
      const memberStats = [...members].map(
        (member): ViewGovernorMemberStats => {
          const ceiling = member.memoryCeilingPoints;
          const share = member.active ? Math.max(member.budget, 0) : 0;
          return {
            id: member.id,
            active: member.active,
            projectedImportance: member.importance,
            allocatedShare: share,
            memoryCeilingPoints: ceiling,
            effectiveBudget:
              ceiling === null ? share : Math.min(share, ceiling),
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
      const active = activeMembers();
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
          settling,
        },
        targetFrameTimeMs: track.targetMs,
        estimateMs: track.estimateMs,
        samples: track.samples,
        trackBudget: track.budget,
        configuredMaxPoints,
        memoryCeilingPoints: memoryCeilingPoints(),
        aggregateBudget: aggregate,
        activeConstraint: viewConstraint,
        lastAdjustment: track.lastAdjustment,
        activeMembers: active.length,
        members: memberStats,
        physicalTileOperations: sum((m) => m.physicalTileOperations),
        physicalHierarchyOperations: sum((m) => m.physicalHierarchyOperations),
        needsFrame: needsFrame(),
        adaptive,
      };
    },

    dispose() {
      if (disposed) return;
      disposed = true;
      clearSettle();
      members.clear();
    },
  };
};
