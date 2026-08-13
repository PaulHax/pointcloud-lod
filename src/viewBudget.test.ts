import { describe, expect, it } from "vitest";

import { allocateViewQuality } from "./viewBudget";
import type { GovernorInputs, Importance } from "./streamedMember";

const inputs = (
  projectedImportance: number,
  qualityDemand = 1,
): GovernorInputs => ({
  projectedImportance: projectedImportance as Importance,
  qualityDemand,
  workPending: false,
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

  it("cannot let an out-of-contract importance starve the other members", () => {
    // Importance is contractually [0, 1]. A member reporting on some other
    // scale (a raw screen-space error in pixels, say) must at worst take a
    // full share, never push everyone else down to the quality floor.
    const allocation = allocateViewQuality(
      [
        { key: "misreporting", inputs: inputs(800) },
        { key: "correct", inputs: inputs(1) },
      ],
      0.5,
    );
    expect(allocation.get("correct")).toBeCloseTo(0.5);
    expect(allocation.get("misreporting")).toBeCloseTo(0.5);
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

  it("allocates zero to culled, demandless, and invalid contenders", () => {
    const allocation = allocateViewQuality(
      [
        { key: "culled", inputs: inputs(0) },
        { key: "no-demand", inputs: inputs(1, 0) },
        { key: "invalid", inputs: inputs(Number.NaN) },
      ],
      Number.NaN,
    );
    expect([...allocation.values()]).toEqual([0, 0, 0]);
  });
});
