/**
 * The two view inputs a host changes underneath a running selection: the size
 * of the canvas, and the projection the camera declares.
 *
 * Neither is a drawing-only property. Detail is chosen against the viewport's
 * CSS height and against the projection mode, so a height that never reaches
 * selection is invisible in every settled state — the view goes on choosing
 * detail for a viewport that no longer exists and looks perfectly converged
 * doing it. The comparisons here therefore hold everything else still: two
 * viewports of the same aspect at one camera differ in nothing but how many
 * pixels a world unit projects to, and a projection round trip returns the
 * camera it started from.
 */

import { afterAll, describe, expect, it } from "vitest";

import {
  closeBrowser,
  cloudsUnderTest,
  openExample,
  type CloudUnderTest,
  type ExampleSession,
  type ExampleStats,
  type Viewport,
} from "./harness";
import { settleAndAssert, watching } from "./invariants";

/** How often the live invariants are read while a gesture runs. */
const SAMPLE_INTERVAL_MS = 25;
/** Per-convergence bound, small enough that several fit in one check. */
const SETTLE_MS = 60_000;

/**
 * The root node's projected spacing. It is the one selection number fixed by
 * the camera and the viewport alone — no node the traversal chose enters it —
 * so it states the viewport-height law without the octree's shape in the way.
 * The harness's shape does not name it; the example reports the controller's
 * stats whole.
 */
type SelectionReport = NonNullable<ExampleStats["controller"]>["selection"] & {
  readonly projectedImportance: number;
};

const rootProjectedSpacing = (stats: ExampleStats): number =>
  (stats.controller!.selection as SelectionReport).projectedImportance;

type AdapterReport = NonNullable<ExampleStats["adapter"]> & {
  readonly devicePixelRatio: number;
};

const drawingRatio = (stats: ExampleStats): number =>
  (stats.adapter! as AdapterReport).devicePixelRatio;

const shown = (stats: ExampleStats): string => JSON.stringify(stats, null, 2);

/** Scale-free, because a cloud's coordinates may be metres or eastings. */
const relativeGap = (a: number, b: number): number =>
  Math.abs(a - b) / Math.max(1, Math.abs(a), Math.abs(b));

const openFixedBudget = async (
  cloud: CloudUnderTest,
): Promise<ExampleSession> => {
  const session = await openExample({ cloud: cloud.urlPath });
  // Fixed budget throughout: two measurements that are meant to differ only by
  // viewport or by projection cannot also have the adaptive loop moving the
  // point budget between them.
  await session.setBudgetMode("fixed");
  return session;
};

/** Small, large, smaller still: every step invalidates the last selection. */
const RESIZE_SWEEP: readonly Viewport[] = [
  { width: 1100, height: 820 },
  { width: 320, height: 240 },
  { width: 1400, height: 900 },
  { width: 240, height: 180 },
  { width: 900, height: 700 },
];

/** A range of ratios, and back, so the sweep has to undo as well as apply. */
const RATIO_SWEEP: readonly number[] = [1, 2, 3, 1];

/**
 * One aspect ratio, two heights. Equal aspect keeps the composite projection
 * matrix — and so the frustum, and so what is culled — identical between the
 * two, leaving the viewport height as the only difference selection can see.
 */
const TALL: Viewport = { width: 1200, height: 900 };
const SHORT: Viewport = { width: 400, height: 300 };

/**
 * Where to put the root node's projected spacing so that refinement is decided
 * by projected size: comfortably above the example's 0.75 px cutoff at the
 * tall height, and below it once a third of the pixels are left.
 */
const REFINEMENT_PROBE_CSS_PX = 1.5;

const resizeAndSettle = async (
  session: ExampleSession,
  size: Viewport,
): Promise<ExampleStats> => {
  const { result } = await watching(session, SAMPLE_INTERVAL_MS, async () => {
    await session.resize(size);
    expect(await session.viewport(), "the viewport ignored a resize").toEqual(
      size,
    );
    return settleAndAssert(session, SETTLE_MS);
  });
  return result;
};

/** How near the probe framing has to land to count as aimed. */
const ZOOM_TOLERANCE = 0.1;

const nearTarget = (value: number, target: number): boolean =>
  value > 0 && Math.abs(value / target - 1) <= ZOOM_TOLERANCE;

/**
 * Back the camera off until the root projects to `targetCssPx`. Stated as a
 * projected size read back from the view rather than as a distance, so it
 * lands in the same place whatever a cloud's extent happens to be.
 *
 * Read back and stepped rather than solved once: what the projection divides
 * by is the distance to the node's bounds, and while the camera sits within
 * the root node's own footprint that is nothing like proportional to the
 * distance the dolly scales.
 */
const zoomToRootSpacing = async (
  session: ExampleSession,
  targetCssPx: number,
): Promise<void> => {
  let spacing = rootProjectedSpacing(await session.stats());
  for (let step = 0; step < 6 && !nearTarget(spacing, targetCssPx); step += 1) {
    expect(spacing, "nothing is selected to zoom out from").toBeGreaterThan(0);
    await session.dolly(targetCssPx / spacing);
    await settleAndAssert(session, SETTLE_MS);
    spacing = rootProjectedSpacing(await session.stats());
  }
  expect(
    Math.abs(spacing / targetCssPx - 1),
    `the probe framing landed at ${spacing} css px, not ${targetCssPx}`,
  ).toBeLessThanOrEqual(ZOOM_TOLERANCE);
};

/**
 * The height law, measured at whatever the camera is doing now: the same
 * aspect at two heights, so the frusta match and only the pixels differ.
 */
const compareHeights = async (session: ExampleSession): Promise<void> => {
  const tall = await resizeAndSettle(session, TALL);
  const short = await resizeAndSettle(session, SHORT);

  // Projected spacing is linear in viewport height, so at one camera the ratio
  // of the two is the ratio of the heights and nothing else. This is the
  // assertion a stale height cannot survive: a resize that never reached
  // selection leaves it computing against the old height, and the two
  // measurements come back identical.
  expect(
    rootProjectedSpacing(tall),
    `the tall viewport selected nothing to measure\n${shown(tall)}`,
  ).toBeGreaterThan(0);
  expect(
    rootProjectedSpacing(short) / rootProjectedSpacing(tall),
    `projected spacing did not follow the viewport height\n${shown(short)}`,
  ).toBeCloseTo(SHORT.height / TALL.height, 6);

  // Fewer pixels per world unit is less reason to refine, and the two frusta
  // are the same, so the shorter viewport can only select the same detail or
  // less.
  expect(
    short.controller!.selection.targetPoints,
    `the shorter viewport selected more points\n${shown(short)}`,
  ).toBeLessThanOrEqual(tall.controller!.selection.targetPoints);
  expect(
    short.controller!.selection.targetTiles,
    `the shorter viewport selected more tiles\n${shown(short)}`,
  ).toBeLessThanOrEqual(tall.controller!.selection.targetTiles);
};

/** Alternating modes; the last one is where the churn lands. */
const MODE_CHURN = [
  "orthographic",
  "perspective",
  "orthographic",
  "perspective",
  "orthographic",
  "perspective",
  "orthographic",
] as const;

afterAll(async () => {
  await closeBrowser();
});

describe("a viewport that changes under a running selection", () => {
  for (const cloud of cloudsUnderTest()) {
    it(`reports every size it was resized to and converges: ${cloud.name}`, async () => {
      const session = await openFixedBudget(cloud);
      try {
        // The sweep starts before the first convergence on purpose: a resize
        // landing while the hierarchy is still being read is where a stale
        // height survives, because the selection it should invalidate has not
        // finished producing the tiles it asked for.
        const { samples } = await watching(
          session,
          SAMPLE_INTERVAL_MS,
          async () => {
            for (const size of RESIZE_SWEEP) {
              await session.resize(size);
              expect(
                await session.viewport(),
                `the viewport did not follow a resize to ${size.width}x${size.height}`,
              ).toEqual(size);
            }
          },
        );
        expect(
          samples,
          "the sampler never read the run it was watching",
        ).toBeGreaterThan(0);

        await settleAndAssert(session, SETTLE_MS);
        expect(session.failures).toEqual([]);
      } finally {
        await session.close();
      }
    });

    it(`never selects more detail from a shorter viewport than a taller one: ${cloud.name}`, async () => {
      const session = await openFixedBudget(cloud);
      try {
        // As framed, with the whole cloud in view.
        await compareHeights(session);

        // And again from far enough out that projected size, not the depth of
        // the octree, is what stops refinement. That is where a height which
        // reached the reported numbers but not the traversal shows up: as a
        // different count of tiles, rather than only as a different spacing.
        await resizeAndSettle(session, TALL);
        await zoomToRootSpacing(session, REFINEMENT_PROBE_CSS_PX);
        await compareHeights(session);

        await settleAndAssert(session, SETTLE_MS);
        expect(session.failures).toEqual([]);
      } finally {
        await session.close();
      }
    });

    it(`draws at each device pixel ratio it is given without growing GPU residency: ${cloud.name}`, async () => {
      const session = await openFixedBudget(cloud);
      try {
        const baseline = await settleAndAssert(session, SETTLE_MS);
        const baselineKeys = await session.keys();
        const held = baseline.adapter!;

        // The ratio changes how large a point is drawn, never which points are
        // held, so a sweep across it must not cost the GPU a byte. A ratio
        // applied by rebuilding actors instead of restating a scale factor
        // shows up here as residency climbing with each step.
        const { samples } = await watching(
          session,
          SAMPLE_INTERVAL_MS,
          async () => {
            for (const ratio of RATIO_SWEEP) {
              await session.setDevicePixelRatio(ratio);
              await session.frame();
              const stats = await session.stats();
              expect(
                drawingRatio(stats),
                `the adapter is not drawing at the ratio it was given\n${shown(stats)}`,
              ).toBe(ratio);
              expect(
                stats.adapter!.gpuResidentBytes,
                `GPU bytes grew at device pixel ratio ${ratio}\n${shown(stats)}`,
              ).toBeLessThanOrEqual(held.gpuResidentBytes);
              expect(
                stats.adapter!.gpuResidentTiles,
                `GPU tiles grew at device pixel ratio ${ratio}\n${shown(stats)}`,
              ).toBeLessThanOrEqual(held.gpuResidentTiles);
            }
          },
        );
        expect(
          samples,
          "the sampler never read the run it was watching",
        ).toBeGreaterThan(0);

        const settled = await settleAndAssert(session, SETTLE_MS);
        expect(drawingRatio(settled)).toBe(1);
        expect(
          (await session.keys()).adapter?.submitted,
          `the ratio sweep changed which tiles are held\n${shown(settled)}`,
        ).toEqual(baselineKeys.adapter?.submitted);
        expect(session.failures).toEqual([]);
      } finally {
        await session.close();
      }
    });
  }
});

describe("a projection toggled under a moving camera", () => {
  for (const cloud of cloudsUnderTest()) {
    it(`reselects on every flip and converges in whichever mode it lands: ${cloud.name}`, async () => {
      const session = await openFixedBudget(cloud);
      try {
        let generation = (await settleAndAssert(session, SETTLE_MS)).controller!
          .selection.generation;

        // Orbit and zoom on the same frame as the flip, so the mode changes
        // under a selection that is already outstanding rather than between
        // two quiet ones.
        const { samples } = await watching(
          session,
          SAMPLE_INTERVAL_MS,
          async () => {
            for (const [step, mode] of MODE_CHURN.entries()) {
              await session.azimuth(11);
              await session.dolly(step % 2 === 0 ? 1.15 : 1 / 1.15);
              await session.setProjection(mode);
              const seen = await session.until(
                `selection to rerun after switching to ${mode}`,
                (stats) =>
                  (stats.controller?.selection.generation ?? 0) > generation,
                SETTLE_MS,
              );
              generation = seen.controller!.selection.generation;
              expect(
                (await session.readCamera()).parallelProjection,
                `the camera did not take the ${mode} mode`,
              ).toBe(mode === "orthographic");
            }
          },
        );
        expect(
          samples,
          "the sampler never read the run it was watching",
        ).toBeGreaterThan(0);

        // Both landings, because the two modes converge by different laws.
        const orthographic = await settleAndAssert(session, SETTLE_MS);
        expect(
          orthographic.controller!.selection.generation,
          "the settled generation fell behind what the churn already saw",
        ).toBeGreaterThanOrEqual(generation);
        await session.setProjection("perspective");
        const perspective = await settleAndAssert(session, SETTLE_MS);
        expect(perspective.controller!.selection.generation).toBeGreaterThan(
          orthographic.controller!.selection.generation,
        );
        expect(session.failures).toEqual([]);
      } finally {
        await session.close();
      }
    });

    it(`returns to the same tiles after a round trip through orthographic: ${cloud.name}`, async () => {
      const session = await openFixedBudget(cloud);
      try {
        await session.setProjection("perspective");
        const before = await settleAndAssert(session, SETTLE_MS);
        const beforeKeys = (await session.keys()).controller!.submitted;
        const beforeCamera = await session.readCamera();
        expect(
          beforeKeys.length,
          `nothing was selected to round-trip\n${shown(before)}`,
        ).toBeGreaterThan(0);

        // Switching preserves the world height the viewport covers, so the
        // camera that comes back is the camera that left. A selection that
        // does not come back with it is holding something the round trip
        // should have undone.
        const { result: middle } = await watching(
          session,
          SAMPLE_INTERVAL_MS,
          async () => {
            await session.setProjection("orthographic");
            const settled = await settleAndAssert(session, SETTLE_MS);
            await session.setProjection("perspective");
            return settled;
          },
        );
        expect(middle.controller!.selection.generation).toBeGreaterThan(
          before.controller!.selection.generation,
        );

        const after = await settleAndAssert(session, SETTLE_MS);
        const afterCamera = await session.readCamera();
        expect(afterCamera.parallelProjection).toBe(false);
        for (const [axis, value] of beforeCamera.position.entries()) {
          expect(
            relativeGap(afterCamera.position[axis]!, value),
            "the camera did not round-trip, so its selection cannot be asked to",
          ).toBeLessThan(1e-9);
        }

        // The committed fixture is small enough that every framing selects it
        // whole, so here this proves the churn lost, duplicated and stranded
        // nothing; a cloud supplied through POINTCLOUD_LOD_BROWSER_CLOUDS has
        // a frontier that can come back different, and proves the law.
        expect(
          (await session.keys()).controller!.submitted,
          `the same camera selected different tiles after a round trip\n${shown(after)}`,
        ).toEqual(beforeKeys);
        expect(session.failures).toEqual([]);
      } finally {
        await session.close();
      }
    });
  }
});
