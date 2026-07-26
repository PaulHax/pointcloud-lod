/**
 * Packet 3.2's browser check: an orthographic camera selects detail from how
 * far it is zoomed, never from how close it sits.
 *
 * The pure math is unit-tested. This exists because the original defect was
 * not in the math — it was that a parallel camera reported no field of view
 * and the bridge substituted 30 degrees, so selection tracked eye distance
 * through vtk.js's real composite projection matrix. Only a real camera
 * driving a real projection can show that.
 */

import { afterAll, beforeAll, describe, expect, it } from "vitest";

import {
  closeBrowser,
  openExample,
  type ExampleSession,
} from "./harness";

const spacing = (session: ExampleSession) => async (): Promise<number> => {
  const stats = await session.settle();
  const value =
    stats.controller?.selection.readyTerminalFrontier.projectedSpacingCssPx
      .p50 ?? null;
  expect(value, "the frontier reported no projected spacing").not.toBeNull();
  return value as number;
};

describe("an orthographic camera in a real projection", () => {
  let session: ExampleSession;
  let projectedSpacing: () => Promise<number>;

  beforeAll(async () => {
    session = await openExample();
    projectedSpacing = spacing(session);
    await session.setBudgetMode("fixed");
    await session.setProjection("orthographic");
    await session.settle();
  });

  afterAll(async () => {
    await session?.close();
    await closeBrowser();
  });

  it("refines when the view zooms in, with the camera left where it is", async () => {
    const before = await session.readCamera();
    const coarse = await projectedSpacing();
    const coarseStats = await session.stats();

    // Halving the parallel scale halves the world height the viewport spans,
    // so every spacing projects to twice as many pixels.
    await session.place({ parallelScale: before.parallelScale / 2 });
    const fine = await projectedSpacing();
    const fineStats = await session.stats();

    const after = await session.readCamera();
    expect(after.position).toEqual(before.position);
    expect(fine / coarse).toBeGreaterThan(1.6);
    expect(fine / coarse).toBeLessThan(2.4);

    // Finer projected spacing is what makes selection descend the octree —
    // but only a cloud holding more detail than the coarse view selected has
    // anywhere to descend to. The committed fixture is small enough to be
    // selected whole, so it can prove the projection arithmetic and not this;
    // a cloud supplied through POINTCLOUD_LOD_BROWSER_CLOUDS proves both.
    const coarsePoints = coarseStats.controller!.selection.targetPoints;
    const finePoints = fineStats.controller!.selection.targetPoints;
    expect(finePoints).toBeGreaterThanOrEqual(coarsePoints);
    if (coarsePoints < coarseStats.sourcePoints) {
      expect(finePoints).toBeGreaterThan(coarsePoints);
    }
  });

  it("selects the same detail however far along its own axis it sits", async () => {
    const start = await session.readCamera();
    const near = await projectedSpacing();
    const nearStats = await session.stats();

    // Dolly far along the view direction with the parallel scale untouched.
    // Under perspective this would coarsen selection; under parallel
    // projection there is no eye distance in the arithmetic at all.
    const axis = start.position.map(
      (value, index) => value - start.focalPoint[index]!,
    );
    await session.place({
      position: start.focalPoint.map(
        (value, index) => value + axis[index]! * 8,
      ),
    });

    const far = await projectedSpacing();
    const farStats = await session.stats();
    const moved = await session.readCamera();
    expect(moved.parallelScale).toBeCloseTo(start.parallelScale, 6);
    expect(far).toBeCloseTo(near, 6);
    expect(farStats.controller!.selection.targetTiles).toBe(
      nearStats.controller!.selection.targetTiles,
    );
    expect(farStats.controller!.selection.targetPoints).toBe(
      nearStats.controller!.selection.targetPoints,
    );
  });

  it("reselects when the projection mode changes under it", async () => {
    const before = (await session.settle()).controller!.selection.generation;

    await session.setProjection("perspective");
    const perspective = await session.settle();
    expect(perspective.controller!.selection.generation).toBeGreaterThan(before);

    await session.setProjection("orthographic");
    const orthographic = await session.settle();
    expect(orthographic.controller!.selection.generation).toBeGreaterThan(
      perspective.controller!.selection.generation,
    );
  });

  it("reports no uncaught error while switching projections", () => {
    expect(session.failures).toEqual([]);
  });
});
