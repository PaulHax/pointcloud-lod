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
  type AdaptiveBudget,
  type AdaptiveBudgetOptions,
  type BudgetAdjustment,
  type BudgetRegime,
} from "./adaptiveBudget";
import { finiteAtLeast, finiteNonNegative, finiteWithin } from "./numeric";

export type HostFrameMetrics = {
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
};

export type ViewGovernor = {
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
};

type MemberState = {
  setPointBudget(points: number): void;
  readonly id: string | null;
  active: boolean;
  /** Null until the member reports; 0 is a real measurement, not "unknown". */
  importance: number | null;
  memoryCeilingPoints: number | null;
  physicalTileOperations: number;
  physicalHierarchyOperations: number;
  budget: number;
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
  let settleTimer: ReturnType<typeof setTimeout> | null = null;
  const settling = (): boolean => settleTimer !== null;
  let emergencyCooldownUntil = Number.NEGATIVE_INFINITY;
  /** Qualifying moving frames seen in a row; see EMERGENCY_CONSECUTIVE_FRAMES. */
  let emergencyStreak = 0;

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
  const interacting = (): boolean => moving() || settling();
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
  const aggregateBudget = (ceiling: number | null): number => {
    const track = budget.budget(interacting());
    return ceiling === null ? track : Math.min(track, ceiling);
  };

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
    const total = aggregateBudget(ceiling);
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
    for (const member of members) {
      const next = member.active
        ? Math.max(1, Math.floor((total * weightOf(member)) / weightTotal))
        : 0;
      if (next === member.budget) continue;
      // Deactivation (0) and first assignment always apply; between live
      // budgets, jitter smaller than the dead-band is not worth the
      // reselection it would trigger. `member.budget` keeps the applied
      // value, so the skipped drift stays visible to the next comparison.
      if (
        member.budget > 0 &&
        next > 0 &&
        Math.abs(next - member.budget) < member.budget * DISTRIBUTE_DEADBAND
      ) {
        continue;
      }
      member.budget = next;
      member.setPointBudget(next);
    }
  };

  const clearSettle = (): void => {
    if (settleTimer !== null) clearTimeout(settleTimer);
    settleTimer = null;
  };

  const startSettle = (): void => {
    clearSettle();
    settleTimer = setTimeout(() => {
      settleTimer = null;
      // Stationary refinement starts from the denser of the two tracks. Its
      // own budget is capacity this machine already proved at the stationary
      // target, so a settle jumps straight back to that density in one
      // reselection — the deselected tiles are usually still in the CPU cache
      // — instead of climbing out of the moving budget one bounded increase
      // at a time, several hundred milliseconds each. When the moving regime
      // sustained more (it grew), that is the proven number instead. Either
      // way the seed is measured fresh: samples taken while moving describe a
      // different regime and must not decide the next stationary step. The
      // converse holds as well, and `recordHostFrame` enforces it: the frames
      // painted inside this window are not moving frames either, so they must
      // not decide the next interaction step.
      budget.restartAt(
        false,
        Math.max(budget.budget(false), budget.budget(true)),
        stampNow(),
      );
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
        active: true,
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
          // Motion that resumes inside the settle window is the same burst
          // for sampling purposes: wheel ticks and scrub steps arrive as many
          // short begin/release pairs, and restarting the track on every one
          // would clear its window faster than `minSamples` frames can fill
          // it — the moving regime would never adapt at all. Only motion that
          // begins from genuine stationary rest restarts the track: those
          // samples measured a scene and a load that have since moved on.
          // The restart seeds from the settled track when that raises the
          // moving budget — see INTERACTION_SEED_OF_STATIONARY — and a fresh
          // burst starts with a clean emergency streak: a hitch in the last
          // gesture must not pre-arm a cut in this one.
          const resuming = settling();
          clearSettle();
          if (!resuming) {
            emergencyStreak = 0;
            budget.restartAt(
              true,
              Math.max(
                budget.budget(true),
                INTERACTION_SEED_OF_STATIONARY * budget.budget(false),
              ),
              stampNow(),
            );
          }
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
      // One local reading for both the offset and this frame's stamp, so a
      // stamped frame is stamped with exactly the instant the host reported.
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
        (finiteNonNegative(metrics.inputDelayMs) &&
          metrics.inputDelayMs > 50) ||
        (finiteNonNegative(metrics.longTaskMs) && metrics.longTaskMs > 50);
      const inMotion = moving();
      const inInteractionRegime = interacting();
      const target = budget.target(inInteractionRegime);
      const observedMs = Math.max(...candidates);
      // Emergency cuts protect live motion. Once motion has stopped, isolated
      // long tasks and missed frames use the sampled stationary controller so
      // they cannot drive a fast grow/halve density sawtooth.
      const emergency = inMotion && (severeInput || observedMs > target * 2);
      // The streak resets on any frame that is not a qualifying moving frame,
      // wherever it lands: only literally consecutive misses reach the cut.
      if (!emergency) emergencyStreak = 0;
      // A frame painted inside the settle window measures neither regime, so
      // it decides neither track. It was drawn at the interaction budget, so
      // it cannot size the stationary one; and it carries none of the cost of
      // motion — no gesture handling, no camera-driven reselection — so
      // folding it into the interaction track can only ever find headroom.
      // With emergency cuts reserved for live motion, that leaves the moving
      // budget with a growth path and no matching shrink path: every hover
      // after a gesture ratchets it above the density real motion sustains,
      // and the next gesture opens over target and has to halve back down.
      // A settle window is under a second; discarding it costs one adjustment
      // step, and the stationary track is seeded from the same budget the
      // moment the window closes.
      const settleWindowOnly = inInteractionRegime && !inMotion;
      // After an emergency, hold the cut long enough to measure the cheaper
      // rendering regime before allowing the normal controller to grow again:
      // inside that window a frame decides nothing, emergency or not.
      if (!settleWindowOnly && now >= emergencyCooldownUntil) {
        if (emergency && ++emergencyStreak >= EMERGENCY_CONSECUTIVE_FRAMES) {
          emergencyStreak = 0;
          budget.reduceNow(inInteractionRegime, now);
          emergencyCooldownUntil = now + emergencyCooldownMs;
        } else {
          // A first qualifying frame is measured, not acted on: it still
          // belongs in the window so a scene that hitches every other frame
          // is answered by the damped controller rather than never at all.
          budget.recordFrame(observedMs, {
            interacting: inInteractionRegime,
            now,
          });
        }
      }
      distribute();
    },

    needsFrame,

    stats() {
      const adaptive = budget.stats();
      const track = interacting() ? adaptive.interaction : adaptive.stationary;
      const active = activeMembers();
      const viewCeiling = memoryCeilingPoints(active);
      const viewConstraint = constraintOf(active, viewCeiling);
      const aggregate = aggregateBudget(viewCeiling);
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
        targetFrameTimeMs: track.targetMs,
        estimateMs: track.estimateMs,
        samples: track.samples,
        trackBudget: track.budget,
        configuredMaxPoints,
        memoryCeilingPoints: viewCeiling,
        aggregateBudget: aggregate,
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
      clearSettle();
      members.clear();
    },
  };
};
