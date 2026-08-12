/**
 * The committed fixtures' hierarchy shape, which is the thing they are for.
 *
 * `multipage.copc.laz` exists for one reason: every real COPC on hand ships a
 * single hierarchy page, so it is the only asset that makes the bounded page
 * scheduler and the page-blocked branch of selection run against real bytes.
 * That property lives in the file, not in the code — regenerate it without the
 * explicit page assignment `generate_multipage_fixture.py` applies, which is
 * copclib's own default, and every node lands on the root page. Nothing else
 * would notice: a single-page cloud opens, converges and renders, so the whole
 * browser matrix stays green while the page-scheduler coverage disappears.
 *
 * The contract asserted here is "more than one page", not the 73 the file
 * happens to hold. The generator's page split is not stable under a rerun, and
 * pinning the count would report a healthy regeneration as a failure.
 *
 * The walk below is deliberately its own dozen lines rather than shared with
 * the browser harness's twin: this suite is the fast, dependency-free run, and
 * an import of the harness would pull playwright and an HTTP server in behind
 * it. What the browser suite adds is the other half — that the pages this file
 * proves exist are actually fetched while the cloud loads.
 */

import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { Copc, Getter, type Hierarchy } from "copc";
import { describe, expect, it } from "vitest";

import { DEFAULT_MIN_POINT_BUDGET } from "../../src/pointCloudMember";

const fixture = (name: string): string =>
  fileURLToPath(new URL(name, import.meta.url));

type HierarchyShape = {
  /** Hierarchy pages, the root page included. */
  readonly pages: number;
  /**
   * Nodes carried by pages other than the root. These are the ones a selection
   * pass can only reach by fetching a page first, so a multipage file with
   * none of them would exercise nothing.
   */
  readonly nodesBeyondRootPage: number;
};

const hierarchyShape = async (file: string): Promise<HierarchyShape> => {
  const getter = Getter.file(file);
  const copc = await Copc.create(getter);
  const pending: Hierarchy.Page[] = [copc.info.rootHierarchyPage];
  let pages = 0;
  let nodesBeyondRootPage = 0;
  while (pending.length > 0) {
    const page = pending.pop()!;
    const isRoot = pages === 0;
    pages += 1;
    const subtree = await Copc.loadHierarchyPage(getter, page);
    for (const node of Object.values(subtree.nodes)) {
      if (node !== undefined && !isRoot) nodesBeyondRootPage += 1;
    }
    for (const sub of Object.values(subtree.pages)) {
      if (sub !== undefined) pending.push(sub);
    }
  }
  return { pages, nodesBeyondRootPage };
};

describe("the committed COPC fixtures", () => {
  const adaptiveFixture = fixture("adaptive.copc.laz");
  it.skipIf(!existsSync(adaptiveFixture))(
    "gives the generated adaptive fixture demand above the policy floor",
    async () => {
      const getter = Getter.file(adaptiveFixture);
      const copc = await Copc.create(getter);
      expect(copc.header.pointCount).toBeGreaterThan(DEFAULT_MIN_POINT_BUDGET);
    },
  );

  it("give multipage.copc.laz a hierarchy spanning several pages", async () => {
    const shape = await hierarchyShape(fixture("multipage.copc.laz"));
    expect(
      shape.pages,
      "the multipage fixture no longer spans more than one hierarchy page, " +
        "so nothing exercises the page scheduler any more",
    ).toBeGreaterThan(1);
    expect(
      shape.nodesBeyondRootPage,
      "every node of the multipage fixture is on the root page, so no " +
        "selection has to fetch a page to reach one",
    ).toBeGreaterThan(0);
  });

  it("gives the fixed-density browser gate demand above its full target", async () => {
    const getter = Getter.file(fixture("multipage.copc.laz"));
    const copc = await Copc.create(getter);
    // density.spec.ts selects at this target before thinning the existing GPU
    // prefixes to 0.25. A fixture at or below it would make "full" merely the
    // source-demand cap instead of the explicit fixed allocation under test.
    expect(copc.header.pointCount).toBeGreaterThan(2_000);
  });

  it("leaves fixture.copc.laz on the single page it is the contrast for", async () => {
    // The pair is what the browser matrix runs over, one cloud flagged
    // multipage and one not. Both being multipage would leave the flag true
    // everywhere and prove nothing about the branch it selects.
    const shape = await hierarchyShape(fixture("fixture.copc.laz"));
    expect(shape.pages).toBe(1);
  });
});
