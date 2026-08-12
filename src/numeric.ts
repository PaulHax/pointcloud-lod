/**
 * One numeric policy for every public boundary in this package:
 *
 * - construction options are programmer errors — an invalid one throws naming
 *   the option and the offending value;
 * - live setters and reported measurements carry wire input — an invalid value
 *   is ignored and leaves state exactly as it was.
 *
 * Either way nothing non-finite reaches a budget, a `Math.min`, a selection
 * comparison, or a statistic: one NaN silently empties a cloud, and every
 * comparison downstream of it answers false forever.
 */

/** Reported measurements and wire input: true only for a usable value. */
export const finiteNonNegative = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value) && value >= 0;

/** The same, for quantities a zero would make meaningless. */
export const finitePositive = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value) && value > 0;

/** Construction guard for continuous quantities (milliseconds, pixels). */
export const finiteAtLeast = (
  name: string,
  value: number,
  min: number,
): number => {
  if (!Number.isFinite(value) || value < min) {
    throw new Error(`${name} must be a finite number >= ${min}, got ${value}`);
  }
  return value;
};

/**
 * Construction guard for counts that must be stated exactly. Unlike
 * `wholeAtLeast`, a fractional value is rejected rather than truncated.
 */
export const integerAtLeast = (
  name: string,
  value: number,
  minimum: number,
): number => {
  if (!Number.isInteger(value) || value < minimum) {
    throw new RangeError(`${name} must be an integer >= ${minimum}`);
  }
  return value;
};

/** Construction guard for counts and point budgets; fractions truncate. */
export const wholeAtLeast = (
  name: string,
  value: number,
  min: number,
): number => Math.floor(finiteAtLeast(name, value, min));

/** Construction guard for quantities a zero would make meaningless. */
export const finiteAbove = (
  name: string,
  value: number,
  min: number,
): number => {
  if (!Number.isFinite(value) || value <= min) {
    throw new Error(`${name} must be a finite number > ${min}, got ${value}`);
  }
  return value;
};

/**
 * Nearest-rank percentile of `values` (0..1). Does not mutate the input.
 * `percentile(xs, 0.9)` of ten values returns the 9th-smallest.
 */
export const percentile = (values: readonly number[], p: number): number => {
  if (values.length === 0) return Number.NaN;
  const sorted = [...values].sort((a, b) => a - b);
  const clampedP = Math.min(Math.max(p, 0), 1);
  const rank = Math.ceil(clampedP * sorted.length);
  const index = Math.min(Math.max(rank - 1, 0), sorted.length - 1);
  return sorted[index]!;
};

/** The empty-window case reports "nothing measured" as null, never NaN. */
export const percentileOrNull = (
  values: readonly number[],
  p: number,
): number | null => (values.length === 0 ? null : percentile(values, p));

/** Construction guard for bounded fractions (percentile, hysteresis, steps). */
export const finiteWithin = (
  name: string,
  value: number,
  min: number,
  max: number,
): number => {
  if (!Number.isFinite(value) || value < min || value > max) {
    throw new Error(
      `${name} must be a finite number in [${min}, ${max}], got ${value}`,
    );
  }
  return value;
};
