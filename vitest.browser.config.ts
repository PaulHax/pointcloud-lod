import { defineConfig } from "vitest/config";

/**
 * The browser checks: real Chromium, real WebGL, the built example, and a
 * range-serving HTTP origin. They are slow and need `example:build` to have
 * run, so they are a separate command rather than part of `npm test`.
 *
 * Serial by construction — every file drives one shared browser and a software
 * rasteriser, and two clouds streaming at once would make the frame-time and
 * concurrency assertions measure each other.
 */
export default defineConfig({
  test: {
    include: ["test/browser/**/*.spec.ts"],
    fileParallelism: false,
    // Streaming a cloud through SwiftShader is slow; the default 5 s expires
    // during an ordinary load.
    testTimeout: 120_000,
    hookTimeout: 120_000,
  },
});
