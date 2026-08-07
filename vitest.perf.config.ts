import { defineConfig } from "vitest/config";

/**
 * Opt-in measurement scenarios. These drive a headed hardware browser and
 * write artifacts; the regular unit and browser suites never collect them.
 */
export default defineConfig({
  test: {
    include: ["test/browser/**/*.perf.spec.ts"],
    tags: [
      {
        name: "perf",
        description: "Opt-in hardware measurement that writes an artifact",
      },
    ],
    fileParallelism: false,
    testTimeout: 600_000,
    hookTimeout: 600_000,
  },
});
