import { describe, expect, it } from "vitest";

import { allocateViewQuality } from "./viewBudget";
import type { GovernorInputs, Importance } from "./streamedMember";

const inputs = (
  projectedImportance: number,
  qualityDemand = 1,
): GovernorInputs => ({
  projectedImportance: projectedImportance as Importance,
  qualityDemand,
  work: { operations: 0, progressSerial: 0 },
  physicalTileOperations: 0,
  physicalHierarchyOperations: 0,
  residentBytes: 0,
});

describe("allocateViewQuality", () => {
  it("gives equal-importance members the normalized view fraction", () => {
    const allocation = allocateViewQuality(
      [
        { key: "a", inputs: inputs(1) },
        { key: "b", inputs: inputs(1) },
      ],
      0.4,
    );
    expect(allocation.get("a")).toBeCloseTo(0.4);
    expect(allocation.get("b")).toBeCloseTo(0.4);
  });

  it("splits by importance and caps every member at one", () => {
    const allocation = allocateViewQuality(
      [
        { key: "important", inputs: inputs(0.9) },
        { key: "other", inputs: inputs(0.1) },
      ],
      0.75,
    );
    expect(allocation.get("important")).toBe(1);
    expect(allocation.get("other")).toBeCloseTo(0.5);
  });

  it("water-fills share that a demand-capped member cannot spend", () => {
    const allocation = allocateViewQuality(
      [
        { key: "small", inputs: inputs(1, 0.1) },
        { key: "large", inputs: inputs(1) },
      ],
      0.5,
    );
    expect(allocation.get("small")).toBe(0.1);
    expect(allocation.get("large")).toBe(0.9);
  });

  it("allocates zero to culled and demandless contenders", () => {
    const allocation = allocateViewQuality(
      [
        { key: "culled", inputs: inputs(0) },
        { key: "no-demand", inputs: inputs(1, 0) },
        { key: "open", inputs: inputs(1) },
      ],
      0.5,
    );
    expect([...allocation.values()]).toEqual([0, 0, 1]);
  });

  it("preserves contender order while redistributing through successive demand caps", () => {
    const allocation = allocateViewQuality(
      [
        { key: "small", inputs: inputs(1, 0.1) },
        { key: "open", inputs: inputs(1) },
        { key: "medium", inputs: inputs(1, 0.52) },
        { key: "large", inputs: inputs(1, 0.75) },
        { key: "culled", inputs: inputs(0) },
      ],
      0.4,
    );
    expect([...allocation.keys()]).toEqual([
      "small",
      "open",
      "medium",
      "large",
      "culled",
    ]);
    expect(allocation.get("small")).toBe(0.1);
    expect(allocation.get("medium")).toBe(0.52);
    expect(allocation.get("open")).toBeCloseTo(0.69);
    expect(allocation.get("large")).toBeCloseTo(0.69);
    expect(allocation.get("culled")).toBe(0);
  });
});
