import { describe, expect, it } from "vitest";

import { transferDelayMs } from "./server";

describe("deterministic browser network shaping", () => {
  it("converts response size and rate to a stable transfer delay", () => {
    expect(transferDelayMs(1_000_000, 10_000_000)).toBe(100);
    expect(transferDelayMs(1, 3)).toBe(334);
    expect(transferDelayMs(0, 10_000_000)).toBe(0);
  });
});
