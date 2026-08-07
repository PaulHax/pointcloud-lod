/**
 * Rendering demand while asynchronous point-cloud work is unresolved.
 *
 * Pending transport and decode make a frame unsuitable for adaptive capacity,
 * but do not change the current image. The page should sleep until a completed
 * operation changes presentation or makes clean budget measurement possible.
 */

import { resolve } from "node:path";

import { Copc, Getter, type Hierarchy } from "copc";
import { afterAll, describe, expect, it } from "vitest";

import {
  closeBrowser,
  FIXTURES,
  MULTIPAGE_URL_PATH,
  openExample,
} from "./harness";

const deferredHierarchyRanges = async (): Promise<ReadonlySet<string>> => {
  const getter = Getter.file(resolve(FIXTURES, "multipage.copc.laz"));
  const copc = await Copc.create(getter);
  const root = copc.info.rootHierarchyPage;
  const pending: Hierarchy.Page[] = [root];
  const ranges = new Set<string>();
  while (pending.length > 0) {
    const page = pending.pop()!;
    if (
      page.pageOffset !== root.pageOffset ||
      page.pageLength !== root.pageLength
    ) {
      ranges.add(`${page.pageOffset}-${page.pageOffset + page.pageLength}`);
    }
    const subtree = await Copc.loadHierarchyPage(getter, page);
    for (const child of Object.values(subtree.pages)) {
      if (child !== undefined) pending.push(child);
    }
  }
  return ranges;
};

describe("render demand", () => {
  it("sleeps while refinement is held, then wakes and converges", async () => {
    const heldRanges = await deferredHierarchyRanges();
    let releaseGate!: () => void;
    let released = false;
    const gate = new Promise<void>((resolveGate) => {
      releaseGate = resolveGate;
    });
    const release = (): void => {
      if (released) return;
      released = true;
      releaseGate();
    };

    const session = await openExample({
      cloud: MULTIPAGE_URL_PATH,
      telemetry: true,
      preparePage: async (page) => {
        await page.route(`**${MULTIPAGE_URL_PATH}`, async (route) => {
          const range = /^bytes=(\d+)-(\d+)$/.exec(
            route.request().headers().range ?? "",
          );
          if (
            range !== null &&
            heldRanges.has(`${range[1]}-${Number(range[2]) + 1}`)
          ) {
            await gate;
          }
          await route.continue();
        });
      },
    });

    try {
      const waiting = await session.until(
        "a stable view to wait on held refinement without requesting frames",
        (stats) =>
          (stats.controller?.physicalHierarchyOperations ?? 0) > 0 &&
          stats.governor?.activity.cameraStable === true &&
          stats.governor.activity.workPending === true &&
          stats.governor.needsFrame === false,
      );
      const beforeRendererRevision = waiting.adapter?.workRevision ?? 0;

      // The state can become idle while the last already-queued callback is
      // still waiting for its animation tick. Drain that callback, then start
      // the interval in which no new render demand may be created.
      await session.frame();
      await session.clearTelemetry();
      await session.page.waitForTimeout(300);
      expect((await session.telemetrySummary()).frames).toBe(0);

      release();
      await session.until(
        "completed refinement to change renderer resources",
        (stats) =>
          (stats.adapter?.workRevision ?? 0) > beforeRendererRevision &&
          (stats.adapter?.drawnPoints ?? 0) > 0,
      );
      const settled = await session.settle();
      expect((await session.telemetrySummary()).frames).toBeGreaterThan(0);
      expect(settled.governor).toMatchObject({
        needsFrame: false,
        activity: { workPending: false },
      });

      await session.clearTelemetry();
      await session.page.waitForTimeout(300);
      expect((await session.telemetrySummary()).frames).toBe(0);
      expect(session.failures).toEqual([]);
    } finally {
      release();
      await session.close();
    }
  });
});

afterAll(closeBrowser);
