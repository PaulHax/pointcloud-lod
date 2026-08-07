import { describe, expect, it } from "vitest";

import { allocatePointPrefixes, type DrawCandidate } from "./drawPlan";

const allocation = (
  candidates: readonly DrawCandidate[],
  pointBudget: number,
) => {
  const byKey = new Map(
    candidates.map((candidate) => [candidate.key, candidate]),
  );
  return allocatePointPrefixes({
    root: "root",
    pointBudget,
    getCandidate: (key) => byKey.get(key),
  });
};

const TREE: readonly DrawCandidate[] = [
  {
    key: "root",
    pointCount: 10,
    priority: 1,
    children: ["near", "far"],
  },
  {
    key: "near",
    pointCount: 20,
    priority: 8,
    children: ["detail"],
  },
  { key: "far", pointCount: 20, priority: 2, children: [] },
  { key: "detail", pointCount: 20, priority: 6, children: [] },
];

describe("allocatePointPrefixes", () => {
  it("draws broad coarse coverage before refining the center branch", () => {
    const result = allocation(TREE, 50);

    expect([...result.prefixes]).toEqual([
      ["root", 10],
      ["near", 20],
      ["far", 20],
    ]);
    expect(result).toMatchObject({
      plannedPoints: 50,
      fullTiles: 3,
      partialTiles: 0,
      skippedTiles: 1,
    });
  });

  it("uses a partial progressive prefix at the budget boundary", () => {
    const result = allocation(TREE, 37);

    expect([...result.prefixes]).toEqual([
      ["root", 10],
      ["near", 20],
      ["far", 7],
    ]);
    expect(result).toMatchObject({
      plannedPoints: 37,
      fullTiles: 2,
      partialTiles: 1,
      skippedTiles: 1,
    });
  });

  it("passes through structural nodes without spending points", () => {
    const result = allocation(
      [
        {
          key: "root",
          pointCount: 0,
          priority: 0,
          children: ["leaf"],
        },
        { key: "leaf", pointCount: 9, priority: 1, children: [] },
      ],
      4,
    );

    expect([...result.prefixes]).toEqual([["leaf", 4]]);
    expect(result).toMatchObject({
      plannedPoints: 4,
      fullTiles: 0,
      partialTiles: 1,
      skippedTiles: 0,
    });
  });

  it("draws nothing for an empty budget and caps at available points", () => {
    expect(allocation(TREE, 0).plannedPoints).toBe(0);
    const full = allocation(TREE, 1_000);
    expect(full.plannedPoints).toBe(70);
    expect(full.fullTiles).toBe(4);
    expect(full.partialTiles).toBe(0);
    expect(full.skippedTiles).toBe(0);
  });

  it("breaks equal-priority ties deterministically", () => {
    const result = allocation(
      [
        {
          key: "root",
          pointCount: 1,
          priority: 1,
          children: ["b", "a"],
        },
        { key: "a", pointCount: 2, priority: 1, children: [] },
        { key: "b", pointCount: 2, priority: 1, children: [] },
      ],
      3,
    );

    expect([...result.prefixes]).toEqual([
      ["root", 1],
      ["a", 2],
    ]);
  });

  it("uses secondary priority only for an equal centre-cone priority", () => {
    const result = allocation(
      [
        {
          key: "root",
          pointCount: 1,
          priority: 0,
          children: ["outer", "center-near", "center-far"],
        },
        { key: "outer", pointCount: 2, priority: -1, children: [] },
        {
          key: "center-near",
          pointCount: 2,
          priority: 0,
          secondaryPriority: 3,
          children: [],
        },
        {
          key: "center-far",
          pointCount: 2,
          priority: 0,
          secondaryPriority: 2,
          children: [],
        },
      ],
      5,
    );

    expect([...result.prefixes].map(([key]) => key)).toEqual([
      "root",
      "center-near",
      "center-far",
    ]);
  });

  it("does not drill into the center cone before coarse peripheral siblings", () => {
    const result = allocation(
      [
        {
          key: "root",
          pointCount: 10,
          priority: 0,
          children: ["center", "edge"],
        },
        {
          key: "center",
          pointCount: 10,
          priority: 10,
          children: ["center-detail"],
        },
        {
          key: "edge",
          pointCount: 10,
          priority: 1,
          children: [],
        },
        {
          key: "center-detail",
          pointCount: 10,
          priority: 100,
          children: [],
        },
      ],
      35,
    );

    expect([...result.prefixes]).toEqual([
      ["root", 10],
      ["center", 10],
      ["edge", 10],
      ["center-detail", 5],
    ]);
  });
});
