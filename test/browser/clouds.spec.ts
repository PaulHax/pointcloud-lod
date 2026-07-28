/**
 * The matrix runs over whatever clouds this machine has: the committed fixture
 * always, plus anything `POINTCLOUD_LOD_BROWSER_CLOUDS` supplies. That makes
 * the harness's own reach a thing worth checking first — a scenario failing
 * because a supplied cloud never opened would otherwise read as a library
 * defect.
 */

import { afterAll, describe, expect, it } from "vitest";

import {
  closeBrowser,
  cloudsUnderTest,
  hierarchyPagesServed,
  openExample,
} from "./harness";
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
        // A multipage cloud is the only thing that makes the bounded page
        // scheduler and the page-blocked branch of selection run at all, and
        // nothing else in the matrix can tell it apart from a single-page one:
        // a cloud whose whole hierarchy arrives in the root page opens and
        // converges perfectly well. So state that pages beyond the root were
        // really fetched: opening any cloud reads the root page, and every
        // page past that one is a page selection had to go and get. The count
        // comes from the whole session's traffic rather than from a stats
        // field, because convergence is defined as no page read outstanding —
        // by the time this line runs, every gauge is back at rest.
        if (cloud.multipage) {
          const pages = await hierarchyPagesServed(session, cloud);
          expect(
            pages,
            `${cloud.name} is meant to span several hierarchy pages, but ` +
              `${pages} were read while it loaded`,
          ).toBeGreaterThan(1);
        }
        expect(session.failures).toEqual([]);
      } finally {
        await session.close();
      }
    });
  }
});
