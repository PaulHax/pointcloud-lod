/**
 * View-wide point-budget allocation.
 *
 * A fixed target and an adaptive frame-time policy both end here. The
 * coordinator caps one aggregate selection budget by known memory, splits it
 * among active point clouds by projected importance, and derives each cloud's
 * draw fraction. It deliberately knows nothing about frames or camera motion.
 */

import { finiteNonNegative, finiteWithin, wholeAtLeast } from "./numeric";

export type ViewBudgetMemberOptions = {
  setPointBudget(points: number): void;
  setDensityFraction(densityFraction: number): void;
  /** Names this cloud in diagnostics. */
  id?: string;
};

export type ViewBudgetMemberUpdate = {
  active?: boolean;
  /** Root projected screen-space error from the controller. */
  projectedImportance?: number;
  /** The controller's memory-derived point ceiling. */
  memoryCeilingPoints?: number;
};

export type ViewBudgetMemberStats = {
  readonly id: string | null;
  readonly active: boolean;
  /** Null until the member reports; 0 is a real measurement, not "unknown". */
  readonly projectedImportance: number | null;
  /** This member's share of the aggregate draw budget. */
  readonly allocatedShare: number;
  /** Last memory-derived ceiling the member reported, null if it never has. */
  readonly memoryCeilingPoints: number | null;
  /** What the member can actually draw: its share capped by its own ceiling. */
  readonly effectiveBudget: number;
  /** Points kept selected and resident so density changes need no tile churn. */
  readonly selectionShare: number;
  /** Selection share capped by this member's memory ceiling. */
  readonly effectiveSelectionBudget: number;
  /** Effective draw budget divided by the effective selection budget. */
  readonly densityFraction: number;
  readonly activeConstraint: ViewBudgetConstraint;
};

export type ViewBudgetMember = {
  update(update: ViewBudgetMemberUpdate): void;
  stats(): ViewBudgetMemberStats;
  release(): void;
};

export type ViewBudgetConstraint = "target" | "memory" | "inactive";

/**
 * Which bound explains one member's allocation, given whichever bound explains
 * the view's. A member's own memory ceiling outranks the view-wide answer; an
 * inactive member is bound by nothing else. Hosts that wrap this coordinator
 * classify their view with a vocabulary of their own, so the view's constraint
 * passes through unnarrowed.
 */
export const memberConstraint = <Constraint extends string>(
  member: {
    readonly active: boolean;
    readonly memoryCeilingPoints: number | null;
    readonly allocatedShare: number;
  },
  viewConstraint: Constraint,
): Constraint | "memory" | "inactive" =>
  !member.active
    ? "inactive"
    : member.memoryCeilingPoints !== null &&
        member.memoryCeilingPoints < member.allocatedShare
      ? "memory"
      : viewConstraint;

export type ViewBudgetCoordinatorOptions = {
  /** Aggregate selected-point target. Default 2,000,000. */
  pointBudget?: number;
  /** Fraction of the selected target to draw. Default 1. */
  densityFraction?: number;
};

export type ViewBudgetStats = {
  /** Requested aggregate selection target before memory limits. */
  readonly pointBudget: number;
  readonly densityFraction: number;
  /** Requested aggregate draw target before memory limits. */
  readonly drawBudget: number;
  readonly memoryCeilingPoints: number | null;
  /** Effective aggregate draw budget after memory limits. */
  readonly aggregateBudget: number;
  /** Effective aggregate selection budget after memory limits. */
  readonly selectionBudget: number;
  readonly activeConstraint: ViewBudgetConstraint;
  readonly activeMembers: number;
  readonly members: readonly ViewBudgetMemberStats[];
};

export type ViewBudgetCoordinator = {
  register(options: ViewBudgetMemberOptions): ViewBudgetMember;
  /** Set the aggregate selected-point target. */
  setPointBudget(points: number): void;
  /** Set the fraction of selected points to draw without changing residency. */
  setDensityFraction(densityFraction: number): void;
  /** Update selection and draw targets atomically. Used by adaptive policies. */
  setTargets(pointBudget: number, drawBudget: number): void;
  /**
   * The aggregate residency ceiling, or null when no member declares one.
   * The one field of {@link stats} a policy reads per pass, without building
   * the rest of the diagnostic snapshot.
   */
  memoryCeiling(): number | null;
  stats(): ViewBudgetStats;
  dispose(): void;
};

type MemberState = {
  setPointBudget(points: number): void;
  setDensityFraction(densityFraction: number): void;
  readonly id: string | null;
  active: boolean;
  importance: number | null;
  memoryCeilingPoints: number | null;
  drawBudget: number;
  selectionBudget: number;
  densityFraction: number;
};

/** A dominant cloud cannot reduce another below this fraction of an even split. */
const MIN_SHARE_OF_EVEN_SPLIT = 0.25;

/**
 * Ignore sub-percent share drift that would synchronously reselect a cloud.
 * Applied *relatively*, against the member's own selection budget.
 */
const DISTRIBUTE_DEADBAND = 0.01;

/**
 * The same idea for the draw density, which is already a 0..1 fraction and so
 * is compared *absolutely*. It needs its own, wider number: re-applying a
 * density re-plans the draw prefixes, rebuilds the auto diameter and rewalks
 * the ready frontier, and the fraction is recomputed on every distribute() —
 * at one hundredth those three tree walks run for changes no one can see.
 */
const DENSITY_DEADBAND = 0.05;

export const createViewBudgetCoordinator = (
  options: ViewBudgetCoordinatorOptions = {},
): ViewBudgetCoordinator => {
  let pointBudget = wholeAtLeast(
    "pointBudget",
    options.pointBudget ?? 2_000_000,
    1,
  );
  let densityFraction = finiteWithin(
    "densityFraction",
    options.densityFraction ?? 1,
    0,
    1,
  );
  let drawBudget = Math.floor(pointBudget * densityFraction);
  const members = new Set<MemberState>();
  let disposed = false;

  const activeMembers = (): MemberState[] =>
    [...members].filter((member) => member.active);

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

  const effectiveSelectionBudget = (active: readonly MemberState[]): number => {
    const ceiling = memoryCeilingPoints(active);
    return ceiling === null ? pointBudget : Math.min(pointBudget, ceiling);
  };

  const effectiveDrawBudget = (active: readonly MemberState[]): number =>
    Math.min(drawBudget, effectiveSelectionBudget(active));

  const constraint = (active: readonly MemberState[]): ViewBudgetConstraint => {
    if (active.length === 0) return "inactive";
    const ceiling = memoryCeilingPoints(active);
    return ceiling !== null && ceiling <= pointBudget ? "memory" : "target";
  };

  const memberStats = (
    member: MemberState,
    viewConstraint: ViewBudgetConstraint,
  ): ViewBudgetMemberStats => {
    const share = member.active ? Math.max(member.drawBudget, 0) : 0;
    const selectionShare = member.active
      ? Math.max(member.selectionBudget, 0)
      : 0;
    const ceiling = member.memoryCeilingPoints;
    const allocated = {
      id: member.id,
      active: member.active,
      projectedImportance: member.importance,
      allocatedShare: share,
      memoryCeilingPoints: ceiling,
      effectiveBudget: ceiling === null ? share : Math.min(share, ceiling),
      selectionShare,
      effectiveSelectionBudget:
        ceiling === null ? selectionShare : Math.min(selectionShare, ceiling),
      densityFraction: Math.max(member.densityFraction, 0),
    };
    return {
      ...allocated,
      activeConstraint: memberConstraint(allocated, viewConstraint),
    };
  };

  const distribute = (): void => {
    if (disposed) return;
    const active = activeMembers();
    // One ceiling walk for both totals: effectiveDrawBudget would otherwise
    // recompute the selection budget, and every member below reads the
    // ceiling again.
    const selectionTotal = effectiveSelectionBudget(active);
    const drawTotal = Math.min(drawBudget, selectionTotal);
    const maxReported = active.reduce(
      (max, member) => Math.max(max, member.importance ?? 0),
      0,
    );
    const floorWeight =
      maxReported > 0 ? maxReported * MIN_SHARE_OF_EVEN_SPLIT : 1;
    const weightOf = (member: MemberState): number =>
      Math.max(member.importance ?? maxReported, floorWeight);
    const weightTotal = active.reduce(
      (sum, member) => sum + weightOf(member),
      0,
    );
    const shareOf = (member: MemberState, total: number): number =>
      member.active && total > 0
        ? Math.max(1, Math.floor((total * weightOf(member)) / weightTotal))
        : 0;

    for (const member of members) {
      const nextSelection = shareOf(member, selectionTotal);
      if (nextSelection !== member.selectionBudget) {
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
      const ceiling = member.memoryCeilingPoints;
      const effectiveSelection =
        ceiling === null
          ? Math.max(member.selectionBudget, 0)
          : Math.min(Math.max(member.selectionBudget, 0), ceiling);
      const effectiveDraw =
        ceiling === null ? nextDraw : Math.min(nextDraw, ceiling);
      const nextDensity =
        effectiveSelection > 0
          ? Math.min(1, effectiveDraw / effectiveSelection)
          : 0;
      // The endpoints are never jitter: undrawn and fully drawn are visible
      // states the deadband must not strand the cloud just short of, however
      // small the last step is.
      const densityEndpoint = nextDensity >= 1 || nextDensity <= 0;
      const densityJitter =
        member.densityFraction >= 0 &&
        !densityEndpoint &&
        Math.abs(nextDensity - member.densityFraction) < DENSITY_DEADBAND;
      member.drawBudget = nextDraw;
      if (!densityJitter) {
        member.densityFraction = nextDensity;
        member.setDensityFraction(nextDensity);
      }
    }
  };

  const setTargets = (points: number, drawnPoints: number): void => {
    if (
      disposed ||
      !finiteNonNegative(points) ||
      !finiteNonNegative(drawnPoints) ||
      drawnPoints > points
    ) {
      return;
    }
    const nextPoints = Math.floor(points);
    const nextDrawnPoints = Math.floor(drawnPoints);
    if (nextPoints === pointBudget && nextDrawnPoints === drawBudget) return;
    pointBudget = nextPoints;
    drawBudget = nextDrawnPoints;
    if (pointBudget > 0) densityFraction = drawBudget / pointBudget;
    distribute();
  };

  return {
    register(options) {
      const state: MemberState = {
        setPointBudget: options.setPointBudget,
        setDensityFraction: options.setDensityFraction,
        id: options.id ?? null,
        active: true,
        importance: null,
        memoryCeilingPoints: null,
        drawBudget: -1,
        selectionBudget: -1,
        densityFraction: -1,
      };
      if (disposed) {
        return {
          update() {},
          stats: () => memberStats(state, "inactive"),
          release() {},
        };
      }
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
          distribute();
        },
        stats() {
          const active = activeMembers();
          return memberStats(state, constraint(active));
        },
        release() {
          if (!members.delete(state)) return;
          distribute();
        },
      };
    },

    setPointBudget(points) {
      if (disposed || !finiteNonNegative(points)) return;
      const nextPoints = Math.floor(points);
      setTargets(nextPoints, Math.floor(nextPoints * densityFraction));
    },

    setDensityFraction(fraction) {
      if (
        disposed ||
        !finiteNonNegative(fraction) ||
        fraction > 1 ||
        fraction === densityFraction
      ) {
        return;
      }
      densityFraction = fraction;
      drawBudget = Math.floor(pointBudget * densityFraction);
      distribute();
    },

    setTargets,

    memoryCeiling() {
      return memoryCeilingPoints(activeMembers());
    },

    stats() {
      const active = activeMembers();
      const viewConstraint = constraint(active);
      return {
        pointBudget,
        densityFraction,
        drawBudget,
        memoryCeilingPoints: memoryCeilingPoints(active),
        aggregateBudget: effectiveDrawBudget(active),
        selectionBudget: effectiveSelectionBudget(active),
        activeConstraint: viewConstraint,
        activeMembers: active.length,
        members: [...members].map((member) =>
          memberStats(member, viewConstraint),
        ),
      };
    },

    dispose() {
      if (disposed) return;
      disposed = true;
      members.clear();
    },
  };
};
