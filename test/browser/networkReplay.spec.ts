import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { replayCachePath } from "./networkReplay";

describe("telemetry network replay", () => {
  it("gives each source a stable opaque cache path", () => {
    const directory = resolve("artifacts/telemetry/cache");
    const first = replayCachePath("https://example.test/a.copc.laz", directory);
    expect(first).toBe(
      replayCachePath("https://example.test/a.copc.laz", directory),
    );
    expect(first).not.toBe(
      replayCachePath("https://example.test/b.copc.laz", directory),
    );
    expect(first).not.toContain("example.test");
    expect(first.endsWith(".copc.laz")).toBe(true);
  });
});
