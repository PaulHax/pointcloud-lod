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

/** Construction guard for counts and point budgets; fractions truncate. */
export const wholeAtLeast = (
  name: string,
  value: number,
  min: number,
): number => {
  if (!Number.isFinite(value) || value < min) {
    throw new Error(`${name} must be a finite number >= ${min}, got ${value}`);
  }
  return Math.floor(value);
};

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
