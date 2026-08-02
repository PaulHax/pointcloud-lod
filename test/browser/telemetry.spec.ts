import { afterAll, describe, expect, it } from "vitest";

import {
  closeBrowser,
  MULTIPAGE_CLOUD,
  openExample,
  usingRealGpu,
} from "./harness";
import { settleAndAssert } from "./invariants";

describe("local performance telemetry", () => {
  afterAll(async () => {
    await closeBrowser();
  });

  it("labels the renderer and exports streaming plus clean frame evidence", async () => {
    const session = await openExample({
      cloud: MULTIPAGE_CLOUD.urlPath,
      telemetry: true,
    });
    try {
      await session.setBudgetMode("fixed");
      await settleAndAssert(session, 120_000);

      const environment = await session.telemetryEnvironment();
      expect(
        environment.webgl.unmaskedRenderer ?? environment.webgl.renderer,
      ).not.toBeNull();
      expect(environment.webgl.softwareRenderer).toBe(!usingRealGpu());

      expect((await session.telemetrySummary()).active).toBe(true);
      await session.markTelemetry("reload-start");
      await session.load(MULTIPAGE_CLOUD.urlPath);
      await settleAndAssert(session, 120_000);

      // Two quiet presentations give the recorder a baseline followed by an
      // interval whose work revision is unchanged.
      await session.render();
      await session.frame();
      await session.render();
      await session.frame();
      await session.stopTelemetry();

      const trace = await session.telemetryTrace();
      expect(trace.schemaVersion).toBe(1);
      expect(trace.summary).toMatchObject({
        active: false,
        pendingWork: 0,
      });
      expect(trace.summary.frames).toBeGreaterThan(1);
      expect(trace.summary.cleanFrames).toBeGreaterThan(0);
      expect(trace.summary.workEvents).toBeGreaterThan(0);
      expect(trace.events).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            type: "work",
            kind: "source-open",
            phase: "start",
          }),
          expect.objectContaining({
            type: "work",
            kind: "tile-load",
            phase: "finish",
            status: "ok",
          }),
          expect.objectContaining({ type: "frame", clean: true }),
          expect.objectContaining({ type: "state", reason: "source-loaded" }),
          expect.objectContaining({
            type: "state",
            reason: "marker:reload-start",
          }),
        ]),
      );
      expect(JSON.stringify(trace)).toContain("workRevision");
      expect(session.failures).toEqual([]);

      const [download] = await Promise.all([
        session.page.waitForEvent("download"),
        session.page.locator("#telemetry-download").click(),
      ]);
      expect(download.suggestedFilename()).toMatch(
        /^pointcloud-telemetry-.*\.json$/,
      );
      expect(
        await session.page.locator("#telemetry-status").textContent(),
      ).toContain(environment.webgl.softwareRenderer ? "software" : "GPU");

      await session.clearTelemetry();
      expect((await session.telemetrySummary()).events).toBe(0);
    } finally {
      await session.close();
    }
  });
});
