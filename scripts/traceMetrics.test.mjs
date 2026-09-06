import { describe, expect, it } from "vitest";

import {
  adjustmentsOf,
  churnMetrics,
  detailKnobOf,
  movementOf,
  settleOf,
} from "./traceMetrics.mjs";

/** A frame carrying only the fields the churn readers look at. */
const frame = (
  atMs,
  { quality, allocated, density, points, tiles, adjustment },
) => ({
  type: "frame",
  atMs,
  state: {
    coordinator: {
      governor: {
        regime: "interaction",
        viewQualityFraction: quality,
        ...(adjustment === undefined ? {} : { lastAdjustment: adjustment }),
      },
    },
    members: [
      {
        kind: "pointCloud",
        id: "cloud",
        allocation: { qualityFraction: allocated ?? quality },
        densityFraction: density ?? quality,
        drawnPoints: points ?? 0,
        drawnTiles: tiles ?? 0,
      },
    ],
  },
});

describe("movementOf", () => {
  it("separates steady refinement from oscillation", () => {
    const climbing = movementOf([0.2, 0.4, 0.6, 0.8, 1]);
    const swinging = movementOf([0.5, 1, 0.5, 1, 0.5]);
    expect(climbing.changes).toBe(4);
    expect(climbing.reversals).toBe(0);
    expect(swinging.changes).toBe(4);
    expect(swinging.reversals).toBe(3);
  });

  it("counts a held value as no movement however long it is held", () => {
    expect(movementOf([1, 1, 1, 1])).toMatchObject({
      changes: 0,
      reversals: 0,
      distinct: 1,
      turnover: 0,
    });
  });

  it("measures turnover against the larger of the two values", () => {
    // Halving replaces half of what was shown; doubling back replaces half
    // again. Dividing by the smaller value would score the two differently.
    const { turnover } = movementOf([1, 0.5, 1]);
    expect(turnover).toBeCloseTo(1);
  });

  it("ignores gaps rather than treating them as a change", () => {
    expect(movementOf([1, null, 1, undefined, Number.NaN, 1])).toMatchObject({
      changes: 0,
      distinct: 1,
    });
  });
});

describe("settleOf", () => {
  it("reports when the last change happened, not the first", () => {
    const frames = [0, 10, 20, 30, 40].map((atMs, index) =>
      frame(atMs, { quality: index < 3 ? 0.5 : 1 }),
    );
    const settle = settleOf(frames, (f) => f.state.members[0].densityFraction);
    expect(settle).toMatchObject({ ms: 30, frames: 3, settled: true });
  });

  it("does not claim a settle when the value moved on the final frame", () => {
    const frames = [0, 10, 20].map((atMs, index) =>
      frame(atMs, { quality: index === 2 ? 1 : 0.5 }),
    );
    expect(
      settleOf(frames, (f) => f.state.members[0].densityFraction).settled,
    ).toBe(false);
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
    const next = {
      ...cut,
      atMs: 9,
      reason: "emergency-restore",
      direction: "increase",
      fromFraction: 0.5,
      toFraction: 1,
    };
    const frames = [
      frame(0, { quality: 1, adjustment: cut }),
      frame(1, { quality: 1, adjustment: cut }),
      frame(2, { quality: 1, adjustment: cut }),
      frame(3, { quality: 1, adjustment: next }),
    ];
    expect(adjustmentsOf(frames).map((a) => a.reason)).toEqual([
      "emergency-cut",
      "emergency-restore",
    ]);
  });
});

describe("churnMetrics", () => {
  it("reads the three levels independently", () => {
    // The governor holds still while the allocated fraction and the visible
    // density both move: the case that says churn entered below the governor.
    const frames = [
      frame(0, {
        quality: 1,
        allocated: 0.5,
        density: 0.5,
        points: 100,
        tiles: 4,
      }),
      frame(16, {
        quality: 1,
        allocated: 0.4,
        density: 0.4,
        points: 80,
        tiles: 5,
      }),
      frame(32, {
        quality: 1,
        allocated: 0.5,
        density: 0.5,
        points: 100,
        tiles: 3,
      }),
    ];
    const churn = churnMetrics(frames);
    expect(churn.governor).toMatchObject({ changes: 0, reversals: 0 });
    expect(churn.allocated).toMatchObject({ changes: 2, reversals: 1 });
    expect(churn.detail).toMatchObject({ changes: 2, reversals: 1 });
    expect(churn.tiles).toMatchObject({ adds: 1, removes: 2 });
  });

  it("counts emergency cuts and restores from the adjustment history", () => {
    const churn = churnMetrics([
      frame(0, {
        quality: 1,
        adjustment: {
          atMs: 0,
          direction: "decrease",
          reason: "emergency-cut",
          fromFraction: 1,
          toFraction: 0.5,
          estimateMs: null,
        },
      }),
      frame(16, {
        quality: 0.5,
        adjustment: {
          atMs: 16,
          direction: "none",
          reason: "within-hysteresis",
          fromFraction: 0.5,
          toFraction: 0.5,
          estimateMs: 20,
        },
      }),
    ]);
    expect(churn.governorMoves).toMatchObject({
      count: 1,
      emergencyCuts: 1,
      emergencyRestores: 0,
    });
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
