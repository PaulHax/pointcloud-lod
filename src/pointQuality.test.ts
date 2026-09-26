import { describe, expect, it } from "vitest";

import { pointBudgets, pointCapacity } from "./pointQuality";
import type { Allocation } from "./streamedMember";

const allocation = (
  qualityFraction: number,
  regime: Allocation["regime"],
): Allocation => ({ qualityFraction, memoryBudgetBytes: 0, regime });

const budgets = (
  input: Partial<Parameters<typeof pointBudgets>[0]> & {
    readonly allocation: Allocation;
  },
) =>
  pointBudgets({
    stationaryFraction: 1,
    fullCeiling: 1_000_000,
    demandPoints: 0,
    minimum: 1,
    ...input,
  });

describe("pointBudgets", () => {
  it("selects the stationary share of the ceiling and draws all of it", () => {
    expect(budgets({ allocation: allocation(0.5, "stationary") })).toEqual({
      pointBudget: 500_000,
      densityFraction: 1,
    });
  });

  it("caps a share at the camera's demand, but never below the minimum", () => {
    expect(
      budgets({
        allocation: allocation(0.5, "stationary"),
        demandPoints: 100_000,
      }).pointBudget,
    ).toBe(100_000);
    expect(
      budgets({
        allocation: allocation(0.5, "stationary"),
        demandPoints: 100_000,
        minimum: 200_000,
      }).pointBudget,
    ).toBe(200_000);
  });

  it("never exceeds the ceiling, even to reach the minimum", () => {
    expect(
      budgets({
        allocation: allocation(1, "stationary"),
        fullCeiling: 50,
        minimum: 100,
      }).pointBudget,
    ).toBe(50);
  });

  it("keeps the stationary selection while moving and thins what it draws", () => {
    expect(
      budgets({
        allocation: allocation(0.25, "moving"),
        stationaryFraction: 1,
      }),
    ).toEqual({ pointBudget: 1_000_000, densityFraction: 0.25 });
    // A moving share above the last stationary one selects at its own level.
    expect(
      budgets({
        allocation: allocation(0.8, "moving"),
        stationaryFraction: 0.5,
      }),
    ).toEqual({ pointBudget: 800_000, densityFraction: 1 });
  });

  it("draws nothing when it may select nothing", () => {
    expect(
      budgets({ allocation: allocation(0.5, "moving"), fullCeiling: 0 }),
    ).toEqual({ pointBudget: 0, densityFraction: 0 });
  });
});

describe("pointCapacity", () => {
  it("holds nothing on a zero share, and never less than a point otherwise", () => {
    expect(pointCapacity(0, 1_000_000, 16_000_000)).toBe(0);
    expect(pointCapacity(1, 0, 0)).toBe(1);
  });

  it("assumes 16 bytes a point until enough points are resident to measure", () => {
    expect(pointCapacity(1_600, 99_999, 99_999)).toBe(100);
  });

  it("divides the share by the measured bytes per point", () => {
    expect(pointCapacity(1_500_000, 100_000, 1_500_000)).toBe(100_000);
  });

  it("holds its value while bytes per point move in their last digits", () => {
    // 14.999 and 14.99895 bytes a point would give 100,006 and 100,007.
    expect(pointCapacity(1_500_000, 100_000, 1_499_900)).toBe(100_000);
    expect(pointCapacity(1_500_000, 100_007, 1_500_000)).toBe(100_000);
  });
});
