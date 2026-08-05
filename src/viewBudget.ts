/**
 * View-wide point-budget allocation.
 *
 * A fixed target and an adaptive frame-time policy both end here. The
 * coordinator caps one aggregate selection budget by known memory, splits it
 * among active point clouds by projected importance and what each can actually
 * use, and derives each cloud's draw fraction. It deliberately knows nothing
 * about frames or camera motion.
 */

import { finiteNonNegative, finiteWithin, wholeAtLeast } from "./numeric";

export type ViewBudgetMemberOptions = {
  setPointBudget(points: number): void;
  setDensityFraction(densityFraction: number): void;
  /** Names this cloud in diagnostics. */
  id?: string;
  /** Whether the cloud draws; defaults to true. A hidden cloud takes no share. */
  active?: boolean;
};

export type ViewBudgetMemberUpdate = {
  active?: boolean;
  /** Root projected screen-space error from the controller. */
  projectedImportance?: number;
  /** The controller's memory-derived point ceiling. */
  memoryCeilingPoints?: number;
  /**
   * Points this cloud could use at the current camera. No allocation exceeds
   * it, so a cloud is never handed a share it has nothing to spend on.
   * Zero reads as "not reported yet", not as "wants nothing": a member whose
   * first selection has not run must not be allocated the nothing that would
   * stop it ever running one.
   */
  demandPoints?: number;
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
  /** Last demand the member reported, null if it never has. */
  readonly demandPoints: number | null;
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

export type ViewBudgetConstraint = "target" | "memory" | "demand" | "inactive";

/**
 * Which bound explains one member's allocation, given whichever bound explains
 * the view's. A member's own memory ceiling outranks the view-wide answer, and
 * a member drawing everything it asked for is bound by its own demand rather
 * than by any budget; an inactive member is bound by nothing else. Hosts that
 * wrap this coordinator classify their view with a vocabulary of their own, so
 * the view's constraint passes through unnarrowed.
 */
export const memberConstraint = <Constraint extends string>(
  member: {
    readonly active: boolean;
    readonly memoryCeilingPoints: number | null;
    readonly demandPoints: number | null;
    readonly allocatedShare: number;
  },
  viewConstraint: Constraint,
): Constraint | "memory" | "demand" | "inactive" =>
  !member.active
    ? "inactive"
    : member.memoryCeilingPoints !== null &&
        member.memoryCeilingPoints < member.allocatedShare
      ? "memory"
      : member.demandPoints !== null &&
          member.demandPoints <= member.allocatedShare
        ? "demand"
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
  demand: number | null;
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

/** A null ceiling is no ceiling: the value passes through uncapped. */
const cappedBy = (value: number, ceiling: number | null): number =>
  ceiling === null ? value : Math.min(value, ceiling);

/**
 * Split `total` among `contenders` by weight, giving no member more than the
 * demand it reported. A member whose demand sits under its weighted share is
 * satisfied exactly and its surplus re-divides among the rest, which is then
 * settled the same way — a sparse cloud must not sit on points it can never
 * spend while a dense one is thinned to fit around it.
 *
 * Each pass either settles the whole remainder or removes at least one member
 * from contention, so it runs at most once per member.
 */
const allocateByDemand = (
  contenders: readonly MemberState[],
  total: number,
  weightOf: (member: MemberState) => number,
): Map<MemberState, number> => {
  const shares = new Map<MemberState, number>();
  let contending = contenders;
  let remaining = total;
  while (contending.length > 0) {
    const weightTotal = contending.reduce(
      (sum, member) => sum + weightOf(member),
      0,
    );
    // An active member always gets a point to select, so a rounding remainder
    // never leaves a visible cloud with nothing at all.
    const wants = contending.map((member) =>
      remaining > 0 && weightTotal > 0
        ? Math.max(1, Math.floor((remaining * weightOf(member)) / weightTotal))
        : 0,
    );
    const settled = contending.filter(
      (member, index) =>
        member.demand !== null && member.demand <= wants[index]!,
    );
    if (settled.length === 0) {
      contending.forEach((member, index) => shares.set(member, wants[index]!));
      break;
    }
    for (const member of settled) {
      const demand = member.demand!;
      shares.set(member, demand);
      remaining = Math.max(0, remaining - demand);
    }
    contending = contending.filter((member) => !shares.has(member));
  }
  return shares;
};

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

  const activeCount = (): number => reduceActive(0, (count) => count + 1);

  const memoryCeilingPoints = (): number | null =>
    reduceActive<number | null>(null, (total, member) =>
      member.memoryCeilingPoints === null
        ? total
        : (total ?? 0) + member.memoryCeilingPoints,
    );

  const effectiveSelectionBudget = (): number =>
    cappedBy(pointBudget, memoryCeilingPoints());

  const effectiveDrawBudget = (): number =>
    Math.min(drawBudget, effectiveSelectionBudget());

  const constraint = (active: number): ViewBudgetConstraint => {
    if (active === 0) return "inactive";
    const ceiling = memoryCeilingPoints();
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
      demandPoints: member.demand,
      effectiveBudget: cappedBy(share, ceiling),
      selectionShare,
      effectiveSelectionBudget: cappedBy(selectionShare, ceiling),
      densityFraction: Math.max(member.densityFraction, 0),
    };
    return {
      ...allocated,
      activeConstraint: memberConstraint(allocated, viewConstraint),
    };
  };

  const distribute = (): void => {
    if (disposed) return;
    // One ceiling walk for both totals: effectiveDrawBudget would otherwise
    // recompute the selection budget, and every member below reads the
    // ceiling again.
    const selectionTotal = effectiveSelectionBudget();
    const drawTotal = Math.min(drawBudget, selectionTotal);
    const maxReported = reduceActive(0, (max, member) =>
      Math.max(max, member.importance ?? 0),
    );
    const floorWeight =
      maxReported > 0 ? maxReported * MIN_SHARE_OF_EVEN_SPLIT : 1;
    const weightOf = (member: MemberState): number =>
      Math.max(member.importance ?? maxReported, floorWeight);
    const contenders = [...members].filter((member) => member.active);
    // Water-filling is monotone in the total, and both totals are capped by
    // the same demands, so a member's draw share can never exceed its
    // selection share: the density fraction below stays a real fraction.
    const selectionShares = allocateByDemand(
      contenders,
      selectionTotal,
      weightOf,
    );
    const drawShares = allocateByDemand(contenders, drawTotal, weightOf);
    const shareOf = (
      member: MemberState,
      shares: Map<MemberState, number>,
    ): number => shares.get(member) ?? 0;

    for (const member of members) {
      const nextSelection = shareOf(member, selectionShares);
      if (nextSelection !== member.selectionBudget) {
        // Landing exactly on a member's demand is never jitter. A cloud
        // reaches its last few points one small step at a time, and a relative
        // deadband is widest for the clouds those points matter most to: a
        // sparse one would be stranded a percent short of complete for ever.
        const satisfied = nextSelection === member.demand;
        const jitter =
          !satisfied &&
          member.selectionBudget > 0 &&
          nextSelection > 0 &&
          Math.abs(nextSelection - member.selectionBudget) <
            member.selectionBudget * DISTRIBUTE_DEADBAND;
        if (!jitter) {
          member.selectionBudget = nextSelection;
          member.setPointBudget(nextSelection);
        }
      }

      const nextDraw = shareOf(member, drawShares);
      const ceiling = member.memoryCeilingPoints;
      const effectiveSelection = cappedBy(
        Math.max(member.selectionBudget, 0),
        ceiling,
      );
      const effectiveDraw = cappedBy(nextDraw, ceiling);
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
        active: options.active ?? true,
        importance: null,
        memoryCeilingPoints: null,
        demand: null,
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
          if (finiteNonNegative(next.demandPoints)) {
            const demand = Math.floor(next.demandPoints);
            state.demand = demand > 0 ? demand : null;
          }
          distribute();
        },
        stats() {
          return memberStats(state, constraint(activeCount()));
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
      return memoryCeilingPoints();
    },

    stats() {
      const active = activeCount();
      const viewConstraint = constraint(active);
      return {
        pointBudget,
        densityFraction,
        drawBudget,
        memoryCeilingPoints: memoryCeilingPoints(),
        aggregateBudget: effectiveDrawBudget(),
        selectionBudget: effectiveSelectionBudget(),
        activeConstraint: viewConstraint,
        activeMembers: active,
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
