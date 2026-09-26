/** Pure mappings from a view allocation and a byte share to point budgets. */

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

/** Assumed until enough points are resident to measure bytes per point. */
const FALLBACK_BYTES_PER_POINT = 16;
const MEASURE_MIN_POINTS = 100_000;
const BYTES_PER_POINT_STEPS = 64;

/**
 * The most points a byte share can hold, at the bytes per point measured over
 * the resident tiles, or an estimate until enough are resident to measure.
 * Never below one point unless the share is zero.
 *
 * The measure is quantized to 1/64 of a byte. It moves in its last digits
 * whenever the resident set changes, and the budget that chose that set is
 * derived from this ceiling: unquantized, a share that exactly covers the
 * view can flip one tile in and out on every allocation, forever.
 */
export const pointCapacity = (
  budgetBytes: number,
  residentPoints: number,
  residentBytes: number,
): number => {
  if (budgetBytes === 0) return 0;
  const bytesPerPoint =
    residentPoints >= MEASURE_MIN_POINTS
      ? Math.ceil((residentBytes / residentPoints) * BYTES_PER_POINT_STEPS) /
        BYTES_PER_POINT_STEPS
      : FALLBACK_BYTES_PER_POINT;
  return Math.max(1, Math.floor(budgetBytes / bytesPerPoint));
};
