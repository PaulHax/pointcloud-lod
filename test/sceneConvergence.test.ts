import { describe, expect, it } from "vitest";
import { sceneConverged as converged } from "./sceneConvergence";

const activity = { loading: 0, datasets: [{ state: "settled" }] };
const governor = {
  regime: "stationary" as const,
  needsFrame: false,
  activity: {
    inputActive: false,
    cameraStable: true,
    workPending: false,
    measurementEligible: true,
  },
};

const members = [{ active: true, qualityManaged: true }];
const sceneConverged = (
  activity: Parameters<typeof converged>[0],
  governor: Parameters<typeof converged>[1],
) => converged(activity, governor, members, 0);

describe("scene convergence", () => {
  it("does not wait for an unused governor in a fixed-only scene", () => {
    expect(
      converged(
        activity,
        { ...governor, needsFrame: true },
        [{ active: true, qualityManaged: false }],
        0,
      ),
    ).toBe(true);
    expect(converged(activity, governor, members, 1)).toBe(false);
  });
  it("waits through gaps in dataset work while quality is adapting", () => {
    expect(sceneConverged(activity, { ...governor, needsFrame: true })).toBe(
      false,
    );
    expect(sceneConverged(activity, governor)).toBe(true);
  });
  it("requires stationary quality and completed work", () => {
    expect(
      sceneConverged(activity, { ...governor, regime: "interaction" }),
    ).toBe(false);
    expect(
      sceneConverged(activity, {
        ...governor,
        activity: { ...governor.activity, workPending: true },
      }),
    ).toBe(false);
    expect(
      sceneConverged(
        { ...activity, datasets: [{ state: "loading" }] },
        governor,
      ),
    ).toBe(false);
    expect(sceneConverged({ ...activity, loading: 1 }, governor)).toBe(false);
    expect(sceneConverged({ ...activity, datasets: [] }, governor)).toBe(false);
  });
  it("returns terminal errors to the caller once adaptation finishes", () => {
    expect(
      sceneConverged({ ...activity, datasets: [{ state: "error" }] }, governor),
    ).toBe(true);
  });
});
