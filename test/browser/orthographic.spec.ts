/**
 * Packet 3.2's browser check: an orthographic camera selects detail from how
 * far it is zoomed, never from how close it sits.
 *
 * The pure math is unit-tested. This exists because the original defect was
 * not in the math — it was that a parallel camera reported no field of view
 * and the bridge substituted 30 degrees, so selection tracked eye distance
 * through vtk.js's real composite projection matrix. Only a real camera
 * driving a real projection can show that.
 *
 * Run over every cloud under test, like the rest of the matrix. The committed
 * fixtures are selected whole, so on them the projection arithmetic is all
 * that can be proved; a deep cloud supplied through
 * POINTCLOUD_LOD_BROWSER_CLOUDS is what makes refinement-on-zoom — the branch
 * this file is named for — actually run.
 */

import { afterAll, describe, expect, it } from "vitest";

import {
  closeBrowser,
  cloudsUnderTest,
  openExample,
  type ExampleSession,
} from "./harness";

const projectedSpacing = async (session: ExampleSession): Promise<number> => {
  const stats = await session.settle();
  const value =
    stats.controller?.selection.readyTerminalFrontier.projectedSpacingCssPx
      .p50 ?? null;
  expect(value, "the frontier reported no projected spacing").not.toBeNull();
  return value as number;
};

describe("an orthographic camera in a real projection", () => {
  let session: ExampleSession | null = null;

  afterAll(async () => {
    await session?.close();
    await closeBrowser();
  });

  /**
   * One session per cloud, opened by the first check that needs it. The three
   * checks below run in order against the same page — reopening for each would
   * pay a full cloud load three times over for nothing.
   */
  const openOrthographic = async (cloud: string): Promise<ExampleSession> => {
    await session?.close();
    session = await openExample({ cloud });
    await session.setBudgetMode("fixed");
    await session.setProjection("orthographic");
    await session.settle();
    return session;
  };

  for (const cloud of cloudsUnderTest()) {
    it(`refines when the view zooms in, with the camera left where it is: ${cloud.name}`, async () => {
      const page = await openOrthographic(cloud.urlPath);
      const before = await page.readCamera();
      const coarse = await projectedSpacing(page);
      const coarseStats = await page.stats();

      // Halving the parallel scale halves the world height the viewport spans,
      // so every spacing projects to twice as many pixels.
      await page.place({ parallelScale: before.parallelScale / 2 });
      const fine = await projectedSpacing(page);
      const fineStats = await page.stats();

      const after = await page.readCamera();
      expect(after.position).toEqual(before.position);
      expect(fine / coarse).toBeGreaterThan(1.6);
      expect(fine / coarse).toBeLessThan(2.4);

      // Finer projected spacing is what makes selection descend the octree —
      // but only a cloud holding more detail than the coarse view selected has
      // anywhere to descend to, so the committed fixtures prove the projection
      // arithmetic above and nothing here. A cloud supplied through
      // POINTCLOUD_LOD_BROWSER_CLOUDS proves both.
      //
      // Note the point count is NOT monotonic in zoom, and no check here may
      // assume it is: halving the parallel scale halves the world height the
      // viewport spans, so the same gesture that refines what stays in view
      // culls what leaves it. On the multipage fixture that trade came out
      // 2,973 against 3,000 — a correct refinement reported as a regression.
      // Only a cloud deep enough for refinement to outweigh the culling can
      // say anything about the total at all.
      const coarsePoints = coarseStats.controller!.selection.targetPoints;
      const finePoints = fineStats.controller!.selection.targetPoints;
      if (cloud.deep && coarsePoints < coarseStats.sourcePoints) {
        expect(finePoints).toBeGreaterThan(coarsePoints);
      }
    });

    it(`selects the same detail however far along its own axis it sits: ${cloud.name}`, async () => {
      const page = session!;
      const start = await page.readCamera();
      const near = await projectedSpacing(page);
      const nearStats = await page.stats();

      // Dolly far along the view direction with the parallel scale untouched.
      // Under perspective this would coarsen selection; under parallel
      // projection there is no eye distance in the arithmetic at all.
      const axis = start.position.map(
        (value, index) => value - start.focalPoint[index]!,
      );
      await page.place({
        position: start.focalPoint.map(
          (value, index) => value + axis[index]! * 8,
        ),
      });

      const far = await projectedSpacing(page);
      const farStats = await page.stats();
      const moved = await page.readCamera();
      expect(moved.parallelScale).toBeCloseTo(start.parallelScale, 6);
      expect(far).toBeCloseTo(near, 6);
      expect(farStats.controller!.selection.targetTiles).toBe(
        nearStats.controller!.selection.targetTiles,
      );
      expect(farStats.controller!.selection.targetPoints).toBe(
        nearStats.controller!.selection.targetPoints,
      );
    });

    it(`reselects when the projection mode changes under it: ${cloud.name}`, async () => {
      const page = session!;
      const before = (await page.settle()).controller!.selection.generation;

      await page.setProjection("perspective");
      const perspective = await page.settle();
      expect(perspective.controller!.selection.generation).toBeGreaterThan(
        before,
      );

      await page.setProjection("orthographic");
      const orthographic = await page.settle();
      expect(orthographic.controller!.selection.generation).toBeGreaterThan(
        perspective.controller!.selection.generation,
      );
    });

    it(`reports no uncaught error while switching projections: ${cloud.name}`, () => {
      expect(session!.failures).toEqual([]);
    });
  }
});
