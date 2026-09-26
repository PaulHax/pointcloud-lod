import { describe, expect, it } from "vitest";

import * as entry from "./index";

describe("package entry", () => {
  it("exposes the adaptive budget floor hosts validate against", () => {
    expect(entry.ADAPTIVE_QUALITY_DEFAULTS.minBudget).toBeGreaterThan(0);
  });
});
