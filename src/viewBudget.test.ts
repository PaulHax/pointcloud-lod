import { describe, expect, it, vi } from "vitest";

import { createViewBudgetCoordinator } from "./viewBudget";

const memberOptions = (
  setPointBudget: (points: number) => void,
  setDensityFraction: (fraction: number) => void = vi.fn(),
) => ({ setPointBudget, setDensityFraction });

describe("createViewBudgetCoordinator", () => {
  it("applies a fixed aggregate target directly", () => {
    const coordinator = createViewBudgetCoordinator({
      pointBudget: 2_000_000,
    });
    const setPointBudget = vi.fn();
    const setDensityFraction = vi.fn();
    coordinator.register(memberOptions(setPointBudget, setDensityFraction));

    expect(setPointBudget).toHaveBeenLastCalledWith(2_000_000);
    expect(setDensityFraction).toHaveBeenLastCalledWith(1);

    coordinator.setPointBudget(5_000_000);
    expect(setPointBudget).toHaveBeenLastCalledWith(5_000_000);
    expect(coordinator.stats()).toMatchObject({
      pointBudget: 5_000_000,
      drawBudget: 5_000_000,
      aggregateBudget: 5_000_000,
      selectionBudget: 5_000_000,
      activeConstraint: "target",
    });
  });

  it("splits one target by projected importance", () => {
    const coordinator = createViewBudgetCoordinator({
      pointBudget: 1_000_000,
    });
    const near = vi.fn();
    const far = vi.fn();
    coordinator
      .register({ ...memberOptions(far), id: "far" })
      .update({ projectedImportance: 1 });
    coordinator
      .register({ ...memberOptions(near), id: "near" })
      .update({ projectedImportance: 3 });

    expect(far).toHaveBeenLastCalledWith(250_000);
    expect(near).toHaveBeenLastCalledWith(750_000);
  });

  it("keeps the fixed selection while thinning the draw target", () => {
    const coordinator = createViewBudgetCoordinator({
      pointBudget: 1_000_000,
    });
    const setPointBudget = vi.fn();
    const setDensityFraction = vi.fn();
    coordinator.register(memberOptions(setPointBudget, setDensityFraction));
    setPointBudget.mockClear();

    coordinator.setDensityFraction(0.6);

    expect(setPointBudget).not.toHaveBeenCalled();
    expect(setDensityFraction).toHaveBeenLastCalledWith(0.6);
    expect(coordinator.stats()).toMatchObject({
      pointBudget: 1_000_000,
      drawBudget: 600_000,
      aggregateBudget: 600_000,
      selectionBudget: 1_000_000,
    });

    coordinator.setPointBudget(2_000_000);
    expect(coordinator.stats()).toMatchObject({
      pointBudget: 2_000_000,
      densityFraction: 0.6,
      drawBudget: 1_200_000,
      aggregateBudget: 1_200_000,
      selectionBudget: 2_000_000,
    });
  });

  it("uses zero as an explicit draw-nothing target", () => {
    const coordinator = createViewBudgetCoordinator({ pointBudget: 1_000_000 });
    const setPointBudget = vi.fn();
    const setDensityFraction = vi.fn();
    coordinator.register(memberOptions(setPointBudget, setDensityFraction));

    coordinator.setPointBudget(0);

    expect(setPointBudget).toHaveBeenLastCalledWith(0);
    expect(setDensityFraction).toHaveBeenLastCalledWith(0);
    expect(coordinator.stats()).toMatchObject({
      pointBudget: 0,
      drawBudget: 0,
      aggregateBudget: 0,
      selectionBudget: 0,
    });
  });

  it("caps the aggregate before splitting and reports the local result", () => {
    const coordinator = createViewBudgetCoordinator({
      pointBudget: 1_000_000,
    });
    const near = vi.fn();
    const far = vi.fn();
    coordinator
      .register({ ...memberOptions(far), id: "far" })
      .update({ projectedImportance: 1, memoryCeilingPoints: 300_000 });
    const nearMember = coordinator.register({
      ...memberOptions(near),
      id: "near",
    });
    nearMember.update({
      projectedImportance: 3,
      memoryCeilingPoints: 300_000,
    });

    expect(far).toHaveBeenLastCalledWith(150_000);
    expect(near).toHaveBeenLastCalledWith(450_000);
    expect(coordinator.stats()).toMatchObject({
      memoryCeilingPoints: 600_000,
      aggregateBudget: 600_000,
      selectionBudget: 600_000,
      activeConstraint: "memory",
    });
    expect(nearMember.stats()).toMatchObject({
      allocatedShare: 450_000,
      effectiveBudget: 300_000,
      activeConstraint: "memory",
    });
  });

  it("ignores invalid live targets without changing its state", () => {
    const coordinator = createViewBudgetCoordinator({ pointBudget: 1_000_000 });
    const before = coordinator.stats();
    coordinator.setTargets(Number.NaN, 1);
    coordinator.setTargets(2_000_000, Number.POSITIVE_INFINITY);
    coordinator.setTargets(1_000_000, 1_000_001);
    coordinator.setDensityFraction(-1);
    expect(coordinator.stats()).toEqual(before);
  });

  it("rejects invalid construction options", () => {
    expect(() => createViewBudgetCoordinator({ pointBudget: 0 })).toThrow(
      /^pointBudget/,
    );
    expect(() =>
      createViewBudgetCoordinator({ densityFraction: Number.NaN }),
    ).toThrow(/^densityFraction/);
    expect(() => createViewBudgetCoordinator({ densityFraction: 1.1 })).toThrow(
      /^densityFraction/,
    );
  });

  it("keeps exact adaptive draw targets instead of round-tripping a ratio", () => {
    const coordinator = createViewBudgetCoordinator({ pointBudget: 3 });
    coordinator.setTargets(3, 2);
    expect(coordinator.stats()).toMatchObject({
      pointBudget: 3,
      densityFraction: 2 / 3,
      drawBudget: 2,
      aggregateBudget: 2,
      selectionBudget: 3,
    });
  });
});
