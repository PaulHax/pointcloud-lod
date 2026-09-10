import { describe, expect, it } from "vitest";
import { coordinatorConverged, sceneConverged } from "./sceneConvergence.mjs";

const activity = { loading: 0, datasets: [{ state: "settled" }] };
const coordinator = (overrides = {}) => ({
  governor: {
    regime: "stationary",
    needsFrame: false,
    activity: { workPending: false },
  },
  members: [{ active: true, qualityManaged: true }],
  submissions: { queuedJobs: 0 },
  ...overrides,
});

describe("scene convergence", () => {
  it("waits through gaps in dataset work while quality is adapting", () => {
    const scene = coordinator();
    scene.governor.needsFrame = true;
    expect(sceneConverged(activity, scene)).toBe(false);
    scene.governor.needsFrame = false;
    expect(sceneConverged(activity, scene)).toBe(true);
  });

  it("ignores an unused governor in fixed-only scenes, but waits for mixed scenes", () => {
    const scene = coordinator({
      members: [{ active: true, qualityManaged: false }],
    });
    scene.governor.needsFrame = true;
    expect(sceneConverged(activity, scene)).toBe(true);
    scene.members.push({ active: false, qualityManaged: true });
    expect(sceneConverged(activity, scene)).toBe(true);
    scene.members[1].active = true;
    expect(sceneConverged(activity, scene)).toBe(false);
  });

  it("requires stationary quality, completed work and empty submissions", () => {
    for (const mutate of [
      (scene) => {
        scene.governor.regime = "interaction";
      },
      (scene) => {
        scene.governor.activity.workPending = true;
      },
      (scene) => {
        scene.submissions.queuedJobs = 1;
      },
    ]) {
      const scene = coordinator();
      mutate(scene);
      expect(sceneConverged(activity, scene)).toBe(false);
    }
    expect(coordinatorConverged(undefined)).toBe(false);
  });

  it.each([
    { loading: 1, datasets: [{ state: "settled" }] },
    { loading: 0, datasets: [{ state: "loading" }] },
    { loading: 0, datasets: [] },
  ])("waits for dataset activity: %j", (pending) => {
    expect(sceneConverged(pending, coordinator())).toBe(false);
  });

  it("returns terminal errors to the caller once adaptation finishes", () => {
    expect(
      sceneConverged(
        { loading: 0, datasets: [{ state: "error" }] },
        coordinator(),
      ),
    ).toBe(true);
  });
});
