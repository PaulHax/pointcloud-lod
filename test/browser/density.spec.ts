import { afterAll, describe, expect, it } from "vitest";

import { closeBrowser, MULTIPAGE_CLOUD, openExample, shown } from "./harness";
import { settleAndAssert } from "./invariants";

describe("progressive draw density", () => {
  afterAll(async () => {
    await closeBrowser();
  });

  it("thins and restores existing GPU tiles without tile or I/O churn", async () => {
    const session = await openExample({ cloud: MULTIPAGE_CLOUD.urlPath });
    try {
      await session.setBudgetMode("fixed");
      // The fixture contains 3,000 points. The example's interactive default
      // is 2,000,000, so the camera's ~3,000-point demand is already "full"
      // long before the configured target. Put the target below real demand so
      // this gate proves a stated full point allocation is selected first and
      // 0.25 then removes prefixes without changing the resident tile set.
      const fullPointTarget = 2_000;
      await session.page.locator("#point-budget").fill(String(fullPointTarget));
      await session.page.locator("#point-budget").dispatchEvent("change");
      const full = await settleAndAssert(session, 120_000);
      const fullKeys = await session.keys();
      const fullScene = await session.scene();
      const served = session.served.length;

      expect(full.sourcePoints).toBeGreaterThan(fullPointTarget);
      expect(full.controller!.selection.targetPoints).toBeLessThanOrEqual(
        fullPointTarget,
      );
      expect(full.controller!.selection.targetPoints).toBeGreaterThan(
        fullPointTarget * 0.9,
      );
      expect(full.adapter!.drawnPoints).toBe(
        full.controller!.selection.targetPoints,
      );
      expect(full.controller!.densityFraction).toBe(1);
      expect(full.adapter!.drawnFraction).toBe(1);

      await session.setDensityFraction(0.25);
      await session.frame();
      const thin = await session.stats();
      expect(thin.controller!.residentPoints).toBe(
        full.controller!.residentPoints,
      );
      expect(thin.adapter!.submittedPoints).toBe(full.adapter!.submittedPoints);
      expect(thin.adapter!.drawnPoints).toBeLessThan(full.adapter!.drawnPoints);
      expect(thin.adapter!.drawnPoints).toBe(thin.controller!.drawnPoints);
      expect(thin.controller!.densityFraction).toBe(0.25);
      // A tile draws whole points, so a quarter of the submitted set is only
      // reachable to within one point per tile — the fraction is the ratio
      // those integer allocations came out at, not the one that was asked for.
      expect(Math.abs(thin.adapter!.drawnFraction - 0.25)).toBeLessThanOrEqual(
        thin.controller!.residentTiles / full.adapter!.submittedPoints,
      );
      expect(await session.keys()).toEqual(fullKeys);
      expect((await session.scene()).actors).toBe(fullScene.actors);
      expect(
        session.served.length,
        `density-only change issued a range request\n${shown(thin)}`,
      ).toBe(served);

      await session.setDensityFraction(1);
      await session.frame();
      const restored = await session.stats();
      expect(restored.adapter!.drawnPoints).toBe(full.adapter!.drawnPoints);
      expect(restored.adapter!.submittedPoints).toBe(
        full.adapter!.submittedPoints,
      );
      expect(await session.keys()).toEqual(fullKeys);
      expect((await session.scene()).actors).toBe(fullScene.actors);
      expect(session.failures).toEqual([]);
    } finally {
      await session.close();
    }
  });
});
