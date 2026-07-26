/**
 * The matrix runs over whatever clouds this machine has: the committed fixture
 * always, plus anything `POINTCLOUD_LOD_BROWSER_CLOUDS` supplies. That makes
 * the harness's own reach a thing worth checking first — a scenario failing
 * because a supplied cloud never opened would otherwise read as a library
 * defect.
 */

import { afterAll, describe, expect, it } from "vitest";

import { closeBrowser, cloudsUnderTest, openExample } from "./harness";
import { settleAndAssert } from "./invariants";

describe("every cloud under test", () => {
  afterAll(async () => {
    await closeBrowser();
  });

  for (const cloud of cloudsUnderTest()) {
    it(`opens and converges: ${cloud.name}`, async () => {
      const session = await openExample({ cloud: cloud.urlPath });
      try {
        await session.setBudgetMode("fixed");
        const stats = await settleAndAssert(session, 120_000);
        expect(stats.sourcePoints).toBeGreaterThan(0);
        expect(stats.controller!.selection.targetTiles).toBeGreaterThan(0);
        // A deep cloud is the only one that can prove refinement has anywhere
        // to go, so record that the run actually got one.
        if (cloud.deep) {
          expect(stats.sourcePoints).toBeGreaterThan(
            stats.controller!.selection.targetPoints,
          );
        }
        expect(session.failures).toEqual([]);
      } finally {
        await session.close();
      }
    });
  }
});
