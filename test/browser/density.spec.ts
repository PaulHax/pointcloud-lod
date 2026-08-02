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
      const full = await settleAndAssert(session, 120_000);
      const fullKeys = await session.keys();
      const fullScene = await session.scene();
      const served = session.served.length;

      await session.setDensityFraction(0.25);
      await session.frame();
      const thin = await session.stats();
      expect(thin.controller!.residentPoints).toBe(
        full.controller!.residentPoints,
      );
      expect(thin.adapter!.submittedPoints).toBe(full.adapter!.submittedPoints);
      expect(thin.adapter!.drawnPoints).toBeLessThan(full.adapter!.drawnPoints);
      expect(thin.adapter!.densityFraction).toBe(0.25);
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
