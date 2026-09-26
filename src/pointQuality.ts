/** Pure mapping from a view allocation to a point cloud's budgets. */

import type { Allocation } from "./streamedMember";

export type PointBudgets = {
  /** Points selection may keep resident. */
  readonly pointBudget: number;
  /** Fraction of the selected points drawn. */
  readonly densityFraction: number;
};

/**
 * Map normalized view quality onto a cloud's useful point ceiling.
 *
 * A stationary allocation changes selection and draws all of it. While
 * moving, the last stationary selection stays resident and only its point
 * prefixes thin, so returning to full quality neither fetches nor rebuilds
 * tiles. Every budget is capped by the camera's demand when it has one,
 * never falls below the minimum, and never exceeds the ceiling.
 */
export const pointBudgets = (input: {
  readonly allocation: Allocation;
  /** Quality fraction of the latest stationary allocation. */
  readonly stationaryFraction: number;
  /** Points the cloud could usefully draw at full quality. */
  readonly fullCeiling: number;
  /** Points the camera asks for; zero before anything was selected. */
  readonly demandPoints: number;
  readonly minimum: number;
}): PointBudgets => {
  const { allocation, fullCeiling, demandPoints, minimum } = input;
  const budgetAt = (qualityFraction: number): number => {
    const requested = Math.floor(fullCeiling * qualityFraction);
    const demandCapped =
      demandPoints > 0 ? Math.min(requested, demandPoints) : requested;
    return Math.min(fullCeiling, Math.max(minimum, demandCapped));
  };
  if (allocation.regime === "stationary") {
    return {
      pointBudget: budgetAt(allocation.qualityFraction),
      densityFraction: 1,
    };
  }
  const drawPoints = budgetAt(allocation.qualityFraction);
  const selectionPoints = budgetAt(
    Math.max(input.stationaryFraction, allocation.qualityFraction),
  );
  return {
    pointBudget: selectionPoints,
    densityFraction: selectionPoints > 0 ? drawPoints / selectionPoints : 0,
  };
};
