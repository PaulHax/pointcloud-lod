import { describe, expect, it } from "vitest";

import {
  isResting,
  recordFailure,
  retryDelayMs,
  type FailureRecord,
  type RetryPolicy,
} from "./retryPolicy";

const POLICY: RetryPolicy = {
  attempts: 3,
  retryDelayMs: (failedAttempts) => 100 * failedAttempts,
  restMs: (round) => 1_000 * 2 ** round,
};

const failTimes = (times: readonly number[]): FailureRecord | undefined =>
  times.reduce<FailureRecord | undefined>(
    (record, nowMs) => recordFailure(record, POLICY, nowMs),
    undefined,
  );

describe("recordFailure", () => {
  it("counts failures within a round, then opens a new one", () => {
    expect(failTimes([0])).toEqual({ count: 1, rounds: 0, lastMs: 0 });
    expect(failTimes([0, 10, 20])).toEqual({ count: 3, rounds: 0, lastMs: 20 });
    expect(failTimes([0, 10, 20, 2_000])).toEqual({
      count: 1,
      rounds: 1,
      lastMs: 2_000,
    });
  });
});

describe("isResting", () => {
  it("rests only a key that spent its allowance, until its rest is over", () => {
    expect(isResting(undefined, POLICY, 0)).toBe(false);
    expect(isResting(failTimes([0, 10]), POLICY, 20)).toBe(false);
    const spent = failTimes([0, 10, 20]);
    expect(isResting(spent, POLICY, 1_019)).toBe(true);
    expect(isResting(spent, POLICY, 1_020)).toBe(false);
  });

  it("rests longer after every spent allowance", () => {
    const twice = failTimes([0, 10, 20, 2_000, 2_010, 2_020]);
    expect(isResting(twice, POLICY, 2_020 + 1_999)).toBe(true);
    expect(isResting(twice, POLICY, 2_020 + 2_000)).toBe(false);
  });
});

describe("retryDelayMs", () => {
  it("retries soon while attempts remain, then waits out the rest", () => {
    expect(retryDelayMs(failTimes([0]), POLICY, 0)).toBe(100);
    expect(retryDelayMs(failTimes([0, 10]), POLICY, 10)).toBe(200);
    expect(retryDelayMs(failTimes([0, 10, 20]), POLICY, 520)).toBe(500);
    expect(retryDelayMs(failTimes([0, 10, 20]), POLICY, 5_000)).toBe(0);
  });
});
