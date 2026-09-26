/**
 * Retry arithmetic for request queues. A key gets a few attempts, then rests
 * before it may try again, and each spent allowance rests longer than the
 * one before. Resting is a backoff, not an eviction: a transient outage must
 * not blank a key for good, so a rested key gets its allowance back.
 *
 * Pure: the clock is an argument, so asking whether a key rests changes
 * nothing about what is fetched next.
 */

export type RetryPolicy = {
  /** Failed attempts a key may make before it rests. */
  readonly attempts: number;
  /** Delay before the next attempt, after `failedAttempts` in this round. */
  readonly retryDelayMs: (failedAttempts: number) => number;
  /** The rest after `round` earlier spent allowances. */
  readonly restMs: (round: number) => number;
};

export type FailureRecord = {
  /** Failures in the current round. */
  readonly count: number;
  /** Allowances already spent; each one lengthens the rest. */
  readonly rounds: number;
  readonly lastMs: number;
};

export const recordFailure = (
  prior: FailureRecord | undefined,
  policy: RetryPolicy,
  nowMs: number,
): FailureRecord => {
  // A failure on a key whose allowance is spent means its rest elapsed and
  // the attempt that followed failed too: a new round, with a longer rest.
  const spent = prior !== undefined && prior.count >= policy.attempts;
  return {
    count: spent ? 1 : (prior?.count ?? 0) + 1,
    rounds: spent ? prior.rounds + 1 : (prior?.rounds ?? 0),
    lastMs: nowMs,
  };
};

/** Whether the key has spent its allowance and its rest is not over. */
export const isResting = (
  record: FailureRecord | undefined,
  policy: RetryPolicy,
  nowMs: number,
): boolean =>
  record !== undefined &&
  record.count >= policy.attempts &&
  nowMs - record.lastMs < policy.restMs(record.rounds);

/** When the key may be asked for again: soon, or when its rest is over. */
export const retryDelayMs = (
  record: FailureRecord | undefined,
  policy: RetryPolicy,
  nowMs: number,
): number =>
  record === undefined || record.count < policy.attempts
    ? policy.retryDelayMs(record?.count ?? 0)
    : Math.max(0, record.lastMs + policy.restMs(record.rounds) - nowMs);
