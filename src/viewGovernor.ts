import {
  createAdaptiveBudget,
  type AdaptiveBudget,
  type AdaptiveBudgetOptions,
  type AdaptiveBudgetStats,
} from "./adaptiveBudget";

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

export interface ViewGovernorMemberOptions {
  setPointBudget(points: number): void;
  active?: boolean;
  projectedImportance?: number;
}

export interface ViewGovernorMember {
  update(options: {
    active?: boolean;
    projectedImportance?: number;
  }): void;
  budget(): number;
  release(): void;
}

export interface ViewGovernorOptions extends AdaptiveBudgetOptions {
  /** Maximum fraction of a host frame assigned to VTK. Default 0.7. */
  vtkFrameFraction?: number;
  /** Delay before switching back to stationary allocation. Default 300 ms. */
  interactionSettleMs?: number;
}

export interface ViewGovernorStats {
  readonly interacting: boolean;
  readonly interactionDepth: number;
  readonly aggregateBudget: number;
  readonly activeMembers: number;
  readonly adaptive: AdaptiveBudgetStats;
}

export interface ViewGovernor {
  register(options: ViewGovernorMemberOptions): ViewGovernorMember;
  beginInteraction(): void;
  endInteraction(): void;
  recordHostFrame(metrics: HostFrameMetrics): void;
  stats(): ViewGovernorStats;
  dispose(): void;
}

interface MemberState {
  setPointBudget(points: number): void;
  active: boolean;
  importance: number;
  budget: number;
}

const finiteNonNegative = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value) && value >= 0;

export const createViewGovernor = (
  options: ViewGovernorOptions = {},
): ViewGovernor => {
  const {
    vtkFrameFraction: rawVtkFraction = 0.7,
    interactionSettleMs = 300,
    ...budgetOptions
  } = options;
  const vtkFrameFraction = Math.min(Math.max(rawVtkFraction, 0.05), 1);
  const emergencyCooldownMs = Math.max(0, budgetOptions.cooldownMs ?? 400);
  const budget: AdaptiveBudget = createAdaptiveBudget(budgetOptions);
  const members = new Set<MemberState>();
  let interactionDepth = 0;
  let settling = false;
  let disposed = false;
  let settleTimer: ReturnType<typeof setTimeout> | null = null;
  let emergencyCooldownUntil = Number.NEGATIVE_INFINITY;

  const interacting = (): boolean => interactionDepth > 0 || settling;

  const distribute = (): void => {
    if (disposed) return;
    const active = [...members].filter((member) => member.active);
    const total = budget.budget(interacting());
    const weightTotal = active.reduce(
      (sum, member) => sum + (member.importance > 0 ? member.importance : 1),
      0,
    );
    for (const member of members) {
      const weight = member.importance > 0 ? member.importance : 1;
      const next = member.active
        ? Math.max(1, Math.floor((total * weight) / weightTotal))
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

  return {
    register(memberOptions) {
      const state: MemberState = {
        setPointBudget: memberOptions.setPointBudget,
        active: memberOptions.active ?? true,
        importance: finiteNonNegative(memberOptions.projectedImportance)
          ? memberOptions.projectedImportance
          : 0,
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
          distribute();
        },
        budget: () => Math.max(0, state.budget),
        release() {
          if (!members.delete(state)) return;
          distribute();
        },
      };
    },

    beginInteraction() {
      if (disposed) return;
      interactionDepth += 1;
      if (interactionDepth !== 1) return;
      clearSettle();
      settling = false;
      distribute();
    },

    endInteraction() {
      if (disposed || interactionDepth === 0) return;
      interactionDepth -= 1;
      if (interactionDepth !== 0) return;
      settling = true;
      clearSettle();
      settleTimer = setTimeout(() => {
        settleTimer = null;
        settling = false;
        distribute();
      }, Math.max(0, interactionSettleMs));
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
      const target = inInteractionRegime
        ? budgetOptions.interactionTargetMs ?? 33
        : budgetOptions.stationaryTargetMs ?? 16;
      const observedMs = Math.max(...candidates);
      // Emergency cuts protect live gestures. Once input has stopped, isolated
      // long tasks and missed frames use the sampled stationary controller so
      // they cannot drive a fast grow/halve density sawtooth.
      const emergency =
        interactionDepth > 0 && (severeInput || observedMs > target * 2);
      // After an emergency, hold the cut long enough to measure the cheaper
      // rendering regime before allowing the normal controller to grow again.
      if (emergency) {
        if (now >= emergencyCooldownUntil) {
          budget.reduceNow(inInteractionRegime, 0.5);
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

    stats() {
      return {
        interacting: interacting(),
        interactionDepth,
        aggregateBudget: budget.budget(interacting()),
        activeMembers: [...members].filter((member) => member.active).length,
        adaptive: budget.stats(),
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
