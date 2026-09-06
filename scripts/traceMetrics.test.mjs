import { describe, expect, it } from "vitest";

import {
  BUSY_LOAD_PER_CORE,
  adjustmentsOf,
  churnMetrics,
  detailKnobOf,
  holdReasonsOf,
  machineLoadOf,
  movementOf,
  settleOf,
} from "./traceMetrics.mjs";

/** One member's per-frame numbers, with only the fields the readers look at. */
const pointCloud = (id, { allocated, density, points, tiles }) => ({
  kind: "pointCloud",
  id,
  allocation: { qualityFraction: allocated },
  densityFraction: density,
  drawnPoints: points ?? 0,
  drawnTiles: tiles ?? 0,
});

const mesh = (id, { allocated, sse, tiles }) => ({
  kind: "tiles3d",
  id,
  allocation: { qualityFraction: allocated },
  sseMultiplier: sse,
  drawnTriangles: 0,
  drawnTiles: tiles ?? 0,
});

const frame = (atMs, members, adjustment) => ({
  type: "frame",
  atMs,
  state: {
    coordinator: {
      governor: {
        regime: "interaction",
        viewQualityFraction: members[0]?.allocation?.qualityFraction ?? 1,
        ...(adjustment === undefined ? {} : { lastAdjustment: adjustment }),
      },
    },
    members,
  },
});

const hold = (atMs, reason, fromFraction = 0.5) => ({
  atMs,
  direction: "none",
  reason,
  fromFraction,
  toFraction: fromFraction,
  estimateMs: 20,
});

describe("movementOf", () => {
  it("separates steady refinement from oscillation", () => {
    expect(movementOf([0.2, 0.4, 0.6, 0.8, 1])).toMatchObject({
      changes: 4,
      reversals: 0,
    });
    expect(movementOf([0.5, 1, 0.5, 1, 0.5])).toMatchObject({
      changes: 4,
      reversals: 3,
    });
  });

  it("counts a held value as no movement however long it is held", () => {
    expect(movementOf([1, 1, 1, 1])).toMatchObject({
      changes: 0,
      reversals: 0,
      distinct: 1,
      turnover: 0,
      read: 4,
    });
  });

  it("measures turnover against the larger of the two values", () => {
    // Halving replaces half of what was shown; doubling back replaces half
    // again. A smaller divisor would score the second step as a whole
    // replacement.
    expect(movementOf([1, 0.5, 1]).turnover).toBeCloseTo(1);
    expect(movementOf([1, 0.5]).turnover).toBeCloseTo(0.5);
  });

  it("ignores gaps rather than treating them as a change", () => {
    expect(movementOf([1, null, 1, undefined, Number.NaN, 1])).toMatchObject({
      changes: 0,
      distinct: 1,
      read: 3,
    });
  });

  it("reports how many frames it could read, so no data is not no churn", () => {
    expect(movementOf([null, null]).read).toBe(0);
    expect(movementOf([]).read).toBe(0);
  });

  it("does not count float noise below the quantum as a change", () => {
    // `distinct` has always quantized to eight places; the change count has to
    // agree with it or the two describe different sequences.
    const noisy = movementOf([0.5, 0.5 + 1e-12, 0.5 - 1e-12]);
    expect(noisy.changes).toBe(0);
    expect(noisy.distinct).toBe(1);
  });
});

describe("settleOf", () => {
  const at = (atMs, density) =>
    frame(atMs, [pointCloud("a", { allocated: 1, density })]);
  const density = (f) => f.state.members[0].densityFraction;

  it("reports when the last change happened, not the first", () => {
    const frames = [0, 10, 20, 30, 40].map((ms, index) =>
      at(ms, index < 3 ? 0.5 : 1),
    );
    expect(settleOf(frames, density)).toMatchObject({
      ms: 30,
      frames: 3,
      settled: true,
    });
  });

  it("does not claim a settle when the value moved on the final frame", () => {
    const frames = [0, 10, 20].map((ms, index) =>
      at(ms, index === 2 ? 1 : 0.5),
    );
    expect(settleOf(frames, density).settled).toBe(false);
  });

  it("does not claim a settle for a value that was never there", () => {
    const frames = [0, 10, 20].map((ms) => frame(ms, []));
    expect(
      settleOf(frames, (f) => f.state.members[0]?.densityFraction ?? null),
    ).toEqual({
      ms: null,
      frames: null,
      read: 0,
      settled: false,
    });
  });

  it("handles the empty and single-frame cases", () => {
    expect(settleOf([], density)).toMatchObject({ read: 0, settled: false });
    expect(settleOf([at(0, 1)], density)).toMatchObject({
      ms: 0,
      read: 1,
      settled: true,
    });
  });
});

describe("adjustmentsOf", () => {
  it("de-duplicates the snapshot repeated across frames", () => {
    const cut = {
      atMs: 5,
      direction: "decrease",
      reason: "emergency-cut",
      fromFraction: 1,
      toFraction: 0.5,
      estimateMs: null,
    };
    const restore = {
      atMs: 9,
      direction: "increase",
      reason: "emergency-restore",
      fromFraction: 0.5,
      toFraction: 1,
      estimateMs: null,
    };
    const frames = [
      frame(0, [], cut),
      frame(1, [], cut),
      frame(2, [], cut),
      frame(3, [], restore),
    ];
    expect(adjustmentsOf(frames).map((entry) => entry.reason)).toEqual([
      "emergency-cut",
      "emergency-restore",
    ]);
  });

  it("keeps two evaluations that agree on everything but their timestamp", () => {
    // The loop re-records a no-op outcome every evaluation. Collapsing them by
    // reason would turn a governor tick count into a single event.
    const frames = [
      hold(1, "within-hysteresis"),
      hold(2, "within-hysteresis"),
    ].map((adjustment, index) => frame(index, [], adjustment));
    expect(adjustmentsOf(frames)).toHaveLength(2);
  });

  it("keeps two records that share a timestamp but did different things", () => {
    const frames = [
      frame(0, [], hold(7, "cooldown")),
      frame(1, [], hold(7, "within-hysteresis")),
    ];
    expect(adjustmentsOf(frames)).toHaveLength(2);
  });
});

describe("holdReasonsOf", () => {
  it("tells contentment apart from a view pinned at the floor", () => {
    const held = holdReasonsOf([
      hold(1, "within-hysteresis", 0.5),
      hold(2, "clamped", 0.05),
      hold(3, "clamped", 0.05),
      hold(4, "clamped", 1),
      hold(5, "cooldown", 0.5),
      hold(6, "insufficient-samples", 0.5),
      {
        ...hold(7, "above-target", 0.5),
        direction: "decrease",
        toFraction: 0.25,
      },
    ]);
    expect(held).toEqual({
      withinHysteresis: 1,
      clampedAtFloor: 2,
      clampedAtCeiling: 1,
      clampedInterior: 0,
      cooldown: 1,
      insufficientSamples: 1,
    });
  });
});

describe("churnMetrics", () => {
  it("measures every member, not just the first", () => {
    // The mesh holds perfectly still while the point cloud oscillates. A
    // reader that only looked at members[0] would call this scene calm.
    const frames = [0, 16, 32, 48].map((atMs, index) =>
      frame(atMs, [
        mesh("m", { allocated: 1, sse: 1, tiles: 4 }),
        pointCloud("p", {
          allocated: 1,
          density: index % 2 === 0 ? 0.5 : 0.25,
          tiles: 3,
        }),
      ]),
    );
    const churn = churnMetrics(frames);
    expect(churn.members.map((member) => member.id)).toEqual(["m", "p"]);
    expect(churn.members[0].detail).toMatchObject({ changes: 0, reversals: 0 });
    expect(churn.members[1].detail).toMatchObject({ changes: 3, reversals: 2 });
    expect(churn.detail).toMatchObject({ changes: 3, reversals: 2 });
  });

  it("reads the detail knob, not the fraction the member was allocated", () => {
    // The two move opposite ways, so a reader that confused them would report
    // the wrong direction as well as the wrong level.
    const frames = [0, 16, 32].map((atMs, index) =>
      frame(atMs, [
        pointCloud("p", {
          allocated: 0.2 + index * 0.1,
          density: 0.9 - index * 0.1,
        }),
      ]),
    );
    const churn = churnMetrics(frames);
    expect(churn.members[0].detail).toMatchObject({
      increases: 0,
      decreases: 2,
    });
    expect(churn.members[0].allocated).toMatchObject({
      increases: 2,
      decreases: 0,
    });
  });

  it("follows a member by id when the scene gains one mid-run", () => {
    // A dataset arriving must not make the first member's series jump onto a
    // different member's axis.
    const cloud = pointCloud("p", { allocated: 1, density: 0.5, tiles: 2 });
    const frames = [
      frame(0, [cloud]),
      frame(16, [mesh("m", { allocated: 1, sse: 8, tiles: 5 }), cloud]),
      frame(32, [mesh("m", { allocated: 1, sse: 8, tiles: 5 }), cloud]),
    ];
    const churn = churnMetrics(frames);
    const byId = Object.fromEntries(churn.members.map((m) => [m.id, m]));
    expect(byId.p.detail).toMatchObject({ changes: 0, read: 3 });
    expect(byId.m.detail).toMatchObject({ changes: 0, read: 2 });
  });

  it("counts tile movement per member rather than across the scene", () => {
    // One member gains four tiles while the other loses four. A scene-wide
    // total would net to zero and report no movement at all.
    const frames = [0, 16].map((atMs, index) =>
      frame(atMs, [
        mesh("m", { allocated: 1, sse: 1, tiles: index === 0 ? 4 : 8 }),
        pointCloud("p", {
          allocated: 1,
          density: 0.5,
          tiles: index === 0 ? 8 : 4,
        }),
      ]),
    );
    const churn = churnMetrics(frames);
    expect(churn.tiles).toMatchObject({ adds: 4, removes: 4 });
  });

  it("counts emergency cuts and restores from the adjustment history", () => {
    const churn = churnMetrics([
      frame(0, [], {
        atMs: 0,
        direction: "decrease",
        reason: "emergency-cut",
        fromFraction: 1,
        toFraction: 0.5,
        estimateMs: null,
      }),
      frame(16, [], hold(16, "within-hysteresis")),
    ]);
    expect(churn.governorMoves).toMatchObject({
      count: 1,
      emergencyCuts: 1,
      emergencyRestores: 0,
    });
    expect(churn.held.withinHysteresis).toBe(1);
  });

  it("reports no readable frames rather than no churn for an empty phase", () => {
    expect(churnMetrics([]).detail).toMatchObject({ read: 0, changes: 0 });
  });
});

describe("detailKnobOf", () => {
  it("reads each member kind on its own detail axis", () => {
    expect(detailKnobOf({ kind: "pointCloud", densityFraction: 0.25 })).toBe(
      0.25,
    );
    expect(detailKnobOf({ kind: "tiles3d", sseMultiplier: 2 })).toBe(2);
  });
});

describe("machineLoadOf", () => {
  const sample = (perCore) => ({ loadAvg1: perCore * 8, cores: 8, perCore });

  it("takes the worst of the samples either side of the run", () => {
    // Contention that arrives halfway through spoils the run just as surely as
    // contention that was there at the start.
    expect(
      machineLoadOf({
        machine: { loadBefore: sample(0.1), loadAfter: sample(1.2) },
      }),
    ).toMatchObject({ perCore: 1.2, busy: true, known: true });
  });

  it("calls a machine with room to spare quiet", () => {
    expect(
      machineLoadOf({
        machine: { loadBefore: sample(0.05), loadAfter: sample(0.2) },
      }),
    ).toMatchObject({ busy: false, known: true });
  });

  it("reports an unmeasured run as unknown rather than quiet", () => {
    // Artifacts written before the bench sampled load must not read as if the
    // machine had been idle.
    expect(machineLoadOf({})).toMatchObject({
      perCore: null,
      busy: false,
      known: false,
    });
    expect(
      machineLoadOf({ machine: { loadBefore: { cores: 8 }, loadAfter: {} } }),
    ).toMatchObject({
      busy: false,
      known: false,
    });
  });

  it("puts the busy line where a run is sharing half its cores", () => {
    expect(BUSY_LOAD_PER_CORE).toBe(0.5);
    expect(
      machineLoadOf({
        machine: { loadBefore: sample(0.51), loadAfter: sample(0.1) },
      }).busy,
    ).toBe(true);
    expect(
      machineLoadOf({
        machine: { loadBefore: sample(0.49), loadAfter: sample(0.1) },
      }).busy,
    ).toBe(false);
  });
});
