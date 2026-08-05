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

  it("registers a member inactive when it says so", () => {
    const coordinator = createViewBudgetCoordinator({
      pointBudget: 1_000_000,
    });
    const hidden = vi.fn();
    const shown = vi.fn();
    coordinator.register({
      ...memberOptions(hidden),
      id: "hidden",
      active: false,
    });
    coordinator.register({ ...memberOptions(shown), id: "shown" });

    // A hidden cloud that joined as active would halve the visible one's share
    // until its first update landed.
    expect(shown).toHaveBeenLastCalledWith(1_000_000);
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

  it("holds density through sub-deadband drift", () => {
    const coordinator = createViewBudgetCoordinator({ pointBudget: 1_000_000 });
    const setDensityFraction = vi.fn();
    coordinator.register(memberOptions(vi.fn(), setDensityFraction));

    coordinator.setTargets(1_000_000, 500_000);
    expect(setDensityFraction).toHaveBeenLastCalledWith(0.5);
    setDensityFraction.mockClear();

    // Re-planning the draw prefixes, the auto diameter and the ready frontier
    // is not worth a two-point move nobody can see.
    coordinator.setTargets(1_000_000, 520_000);
    expect(setDensityFraction).not.toHaveBeenCalled();
  });

  it("always reaches full density, however small the last step", () => {
    const coordinator = createViewBudgetCoordinator({ pointBudget: 1_000_000 });
    const setDensityFraction = vi.fn();
    coordinator.register(memberOptions(vi.fn(), setDensityFraction));

    coordinator.setTargets(1_000_000, 500_000);
    coordinator.setTargets(1_000_000, 970_000);
    expect(setDensityFraction).toHaveBeenLastCalledWith(0.97);
    setDensityFraction.mockClear();

    // 0.97 -> 1 is inside the deadband, but stopping there would leave the
    // cloud permanently short of the full-quality frame it was asked for.
    coordinator.setTargets(1_000_000, 1_000_000);
    expect(setDensityFraction).toHaveBeenLastCalledWith(1);
  });

  it("reports the aggregate memory ceiling without a full snapshot", () => {
    const coordinator = createViewBudgetCoordinator({ pointBudget: 4_000_000 });
    expect(coordinator.memoryCeiling()).toBeNull();

    coordinator
      .register(memberOptions(vi.fn()))
      .update({ memoryCeilingPoints: 750_000 });
    coordinator
      .register(memberOptions(vi.fn()))
      .update({ memoryCeilingPoints: 250_000 });

    expect(coordinator.memoryCeiling()).toBe(1_000_000);
    expect(coordinator.stats().memoryCeilingPoints).toBe(
      coordinator.memoryCeiling(),
    );
  });

  it("gives a cloud no more than it can use and spends the rest elsewhere", () => {
    const coordinator = createViewBudgetCoordinator({ pointBudget: 1_000_000 });
    const sparse = vi.fn();
    const dense = vi.fn();
    // The sparse cloud's coarser spacing projects the *larger* screen-space
    // error, so importance alone hands the smaller cloud the bigger share.
    coordinator
      .register({ ...memberOptions(sparse), id: "sparse" })
      .update({ projectedImportance: 3, demandPoints: 138_000 });
    coordinator
      .register({ ...memberOptions(dense), id: "dense" })
      .update({ projectedImportance: 1, demandPoints: 9_000_000 });

    expect(sparse).toHaveBeenLastCalledWith(138_000);
    expect(dense).toHaveBeenLastCalledWith(862_000);
  });

  it("draws a satisfied cloud whole while the view is thinned", () => {
    const coordinator = createViewBudgetCoordinator({ pointBudget: 1_000_000 });
    const sparseDensity = vi.fn();
    const denseDensity = vi.fn();
    coordinator
      .register({ ...memberOptions(vi.fn(), sparseDensity), id: "sparse" })
      .update({ projectedImportance: 1, demandPoints: 138_000 });
    const denseMember = coordinator.register({
      ...memberOptions(vi.fn(), denseDensity),
      id: "dense",
    });
    denseMember.update({ projectedImportance: 1, demandPoints: 9_000_000 });

    // Frame time gets tight and the view halves what it draws. The sparse
    // cloud still fits inside its share whole, so it loses nothing: its points
    // cost almost nothing and there are few enough that each one matters.
    coordinator.setTargets(1_000_000, 500_000);

    expect(sparseDensity).toHaveBeenLastCalledWith(1);
    expect(denseDensity).toHaveBeenLastCalledWith(362_000 / 862_000);
    expect(denseMember.stats()).toMatchObject({
      activeConstraint: "target",
    });
  });

  it("names demand as the bound when a cloud draws everything it asked for", () => {
    const coordinator = createViewBudgetCoordinator({ pointBudget: 1_000_000 });
    const member = coordinator.register({
      ...memberOptions(vi.fn()),
      id: "sparse",
    });
    member.update({ demandPoints: 138_000 });

    expect(member.stats()).toMatchObject({
      demandPoints: 138_000,
      allocatedShare: 138_000,
      effectiveSelectionBudget: 138_000,
      densityFraction: 1,
      activeConstraint: "demand",
    });
  });

  it("treats a zero demand as unreported rather than as wanting nothing", () => {
    const coordinator = createViewBudgetCoordinator({ pointBudget: 1_000_000 });
    const setPointBudget = vi.fn();
    const member = coordinator.register(memberOptions(setPointBudget));

    // A cloud reports demand from its own selection, so a member allocated the
    // nothing an empty first selection asks for would never select again.
    member.update({ demandPoints: 0 });

    expect(setPointBudget).toHaveBeenLastCalledWith(1_000_000);
    expect(member.stats().demandPoints).toBeNull();
  });

  it("lets a satisfied cloud grow back when its demand rises", () => {
    const coordinator = createViewBudgetCoordinator({ pointBudget: 1_000_000 });
    const near = vi.fn();
    const far = vi.fn();
    const nearMember = coordinator.register({
      ...memberOptions(near),
      id: "near",
    });
    nearMember.update({ projectedImportance: 1, demandPoints: 100_000 });
    coordinator
      .register({ ...memberOptions(far), id: "far" })
      .update({ projectedImportance: 1, demandPoints: 5_000_000 });
    expect(near).toHaveBeenLastCalledWith(100_000);
    expect(far).toHaveBeenLastCalledWith(900_000);

    // The camera moves in and the near cloud has hierarchy to spend on. A
    // cloud reports demand out of a selection its own share bounded, so being
    // satisfied once must never pin it there. It rises to its weighted half,
    // not to the whole 800_000: the far cloud wants more than a half too.
    nearMember.update({ demandPoints: 800_000 });

    expect(near).toHaveBeenLastCalledWith(500_000);
    expect(far).toHaveBeenLastCalledWith(500_000);
  });

  it("takes the last small step onto a demand the deadband would swallow", () => {
    const coordinator = createViewBudgetCoordinator({ pointBudget: 1_000_000 });
    const setPointBudget = vi.fn();
    const member = coordinator.register(memberOptions(setPointBudget));

    // A cloud finds its last points a handful at a time. The deadband is
    // relative, so for this cloud it spans 1_386 points — enough to strand it
    // just short of whole, which is the one place the thinning is visible.
    member.update({ demandPoints: 138_000 });
    expect(setPointBudget).toHaveBeenLastCalledWith(138_000);
    member.update({ demandPoints: 138_633 });

    expect(setPointBudget).toHaveBeenLastCalledWith(138_633);
    expect(member.stats().effectiveSelectionBudget).toBe(138_633);
  });

  it("shares by weight once no demand is small enough to satisfy", () => {
    const coordinator = createViewBudgetCoordinator({ pointBudget: 1_000_000 });
    const near = vi.fn();
    const far = vi.fn();
    coordinator
      .register({ ...memberOptions(far), id: "far" })
      .update({ projectedImportance: 1, demandPoints: 4_000_000 });
    coordinator
      .register({ ...memberOptions(near), id: "near" })
      .update({ projectedImportance: 3, demandPoints: 4_000_000 });

    expect(far).toHaveBeenLastCalledWith(250_000);
    expect(near).toHaveBeenLastCalledWith(750_000);
  });

  it("keeps a member's own memory ceiling ahead of its demand", () => {
    const coordinator = createViewBudgetCoordinator({ pointBudget: 1_000_000 });
    const member = coordinator.register({
      ...memberOptions(vi.fn()),
      id: "capped",
    });
    member.update({ demandPoints: 900_000, memoryCeilingPoints: 300_000 });

    // The cloud wants more than it can hold, so what it draws is explained by
    // the memory it has, not by everything it asked for.
    expect(member.stats()).toMatchObject({
      demandPoints: 900_000,
      allocatedShare: 300_000,
      effectiveBudget: 300_000,
      activeConstraint: "memory",
    });
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
