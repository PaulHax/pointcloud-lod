/**
 * The three view inputs a host changes underneath a running selection: the size
 * of the canvas, the projection the camera declares, and the ratio of
 * framebuffer pixels to CSS pixels it draws at.
 *
 * The first two are not drawing-only properties. Detail is chosen against the
 * viewport's CSS height and against the projection mode, so a height that never
 * reaches selection is invisible in every settled state — the view goes on
 * choosing detail for a viewport that no longer exists and looks perfectly
 * converged doing it. The comparisons here therefore hold everything else
 * still: two viewports of the same aspect at one camera differ in nothing but
 * how many pixels a world unit projects to, and a projection round trip returns
 * the camera it started from.
 *
 * The ratio is the one that is drawing-only, and so the one with nowhere to
 * show up except the scene. The adapter leaves the point diameter on the actor
 * property in CSS pixels and hands the ratio to the mapper as its scale factor;
 * the fork's point-gaussian shader multiplies the two into `gl_PointSize`. A
 * ratio that reached the adapter's field but neither actor nor mapper is a
 * change nothing draws, so the checks below read it off the scene the renderer
 * will draw with rather than off the field it was stored in.
 */

import { afterAll, describe, expect, it } from "vitest";

import {
  closeBrowser,
  cloudsUnderTest,
  openExample,
  SETTLE_QUIET_MS,
  shown,
  type CloudUnderTest,
  type ExampleSession,
  type ExampleStats,
  type SceneReading,
  type Viewport,
} from "./harness";
import {
  assertRendererHoldsOnlyItsOwn,
  settleAndAssert,
  watching,
  withSession,
} from "./invariants";

/** How often the live invariants are read while a gesture runs. */
const SAMPLE_INTERVAL_MS = 25;
/** Per-convergence bound, small enough that several fit in one check. */
const SETTLE_MS = 60_000;

/**
 * The root node's projected spacing. It is the one selection number fixed by
 * the camera and the viewport alone — no node the traversal chose enters it —
 * so it states the viewport-height law without the octree's shape in the way.
 */
const rootProjectedSpacing = (stats: ExampleStats): number =>
  stats.controller!.selection.projectedImportance;

const drawingRatio = (stats: ExampleStats): number =>
  stats.adapter!.devicePixelRatio;

/** The uniform diameter the controller's presentation policy asked for. */
const drawingDiameterCssPx = (stats: ExampleStats): number =>
  stats.adapter!.diameterCssPx;

/**
 * What one point will be drawn at, in framebuffer pixels, read off the actor
 * and the mapper the renderer holds.
 *
 * The two halves are not interchangeable. The adapter leaves the actor property
 * in CSS pixels (`setPointSize(diameterCssPx)`) and gives the ratio to the
 * mapper (`setScaleFactor(devicePixelRatio)`), and the shader assigns
 * `gl_PointSize = getPointSize() * getScaleFactor()`. This product is the whole
 * of what a ratio change may move, and the CSS half is the whole of what it
 * must leave alone — an adapter that premultiplied the ratio into the actor
 * property as well would draw every point ratio-squared too large.
 */
const drawnDiameterDevicePx = (scene: SceneReading): number =>
  scene.pointSizeDevicePx! * scene.mapperScaleFactor!;

/**
 * The scene's own account of the ratio, and the renderer's own count of what
 * holds it.
 *
 * `scene()` reports the renderer's first actor, which is the oldest one it
 * still holds — actors are appended and removal preserves order. The reuse pool
 * holds actors the renderer owns but never draws and which a ratio change is
 * not required to reach, so this states that the pool is empty before reading:
 * otherwise an undrawn actor's stale numbers would be reported as the scene's.
 */
const assertSceneDrawsAtRatio = async (
  session: ExampleSession,
  stats: ExampleStats,
  ratio: number,
): Promise<number> => {
  const scene = await session.scene();
  assertRendererHoldsOnlyItsOwn(scene, stats, "at the ratio reading");
  expect(
    scene.actors,
    `nothing is on screen to be drawn at a ratio\n${shown(stats)}`,
  ).toBeGreaterThan(0);
  expect(
    stats.adapter!.pooledTiles,
    `an undrawn pooled actor may be the one the scene reports\n${shown(stats)}`,
  ).toBe(0);

  expect(
    scene.mapperScaleFactor,
    `the mapper the renderer draws with is not scaling to ${ratio}\n${shown(stats)}`,
  ).toBe(ratio);
  expect(
    scene.pointSizeDevicePx,
    `the actor's point size is no longer the CSS diameter\n${shown(stats)}`,
  ).toBe(drawingDiameterCssPx(stats));
  return drawnDiameterDevicePx(scene);
};

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
 * The same journey for the drawn size, with the fractional ratios real displays
 * actually report — a ratio that only ever arrives as a small integer can be
 * carried by a rounding step and never noticed.
 */
const DRAWN_RATIO_SWEEP: readonly number[] = [1.5, 2, 3, 1.25, 1];

/**
 * Two more ratios for the ordering checks, neither of them 1 and neither of
 * them each other, so no step can pass by leaving the ratio where it was.
 */
const FRESH_ACTOR_RATIO = 3;
const ARRIVING_ACTOR_RATIO = 2;

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

/** Loads to spend looking for the window before calling it unreachable. */
const MID_STREAM_ATTEMPTS = 4;

/**
 * Reload the cloud and return the first state with tiles already on screen and
 * reads still outstanding — the one moment a ratio change has both populations
 * of actor to reach at once.
 *
 * Polled with no pause between reads, and reloaded rather than waited on when a
 * load finishes without showing the window. The committed fixture is nine tiles
 * over a loopback socket: its whole arrival can pass inside a single 50 ms
 * polling step, and once a load has converged the window does not come back.
 */
const loadUntilMidStream = async (
  session: ExampleSession,
  cloud: CloudUnderTest,
): Promise<ExampleStats> => {
  for (let attempt = 0; attempt < MID_STREAM_ATTEMPTS; attempt += 1) {
    await session.load(cloud.urlPath);
    const deadline = Date.now() + SETTLE_MS;
    let quietSince: number | null = null;
    while (Date.now() < deadline) {
      const stats = await session.stats();
      const held = stats.controller!;
      // Hierarchy reads count: a page still on its way is more tiles still on
      // their way, and on a multi-page cloud that is most of the stream.
      const outstanding =
        held.queuedTiles +
        held.inFlight +
        held.physicalTileOperations +
        held.queuedPages +
        held.physicalHierarchyOperations;
      const onScreen = stats.adapter?.submittedTiles ?? 0;
      if (onScreen > 0 && outstanding > 0) return stats;
      if (outstanding > 0) {
        quietSince = null;
        continue;
      }
      quietSince ??= Date.now();
      // Tiles on screen and nothing outstanding for a whole quiet window: this
      // load has finished arriving, and the moment it never showed will not
      // turn up now. The window is the harness's own, borrowed rather than
      // restated: selection is debounced there for the same reason it is
      // debounced here, so an idle instant is not the end of the stream.
      if (onScreen > 0 && Date.now() - quietSince >= SETTLE_QUIET_MS) break;
    }
  }
  throw new Error(
    `no load of ${cloud.name} was caught with tiles on screen and more still arriving`,
  );
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
      await withSession(
        () => openFixedBudget(cloud),
        async (session) => {
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
        },
      );
    });

    it(`never selects more detail from a shorter viewport than a taller one: ${cloud.name}`, async () => {
      await withSession(
        () => openFixedBudget(cloud),
        async (session) => {
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
        },
      );
    });

    it(`draws at each device pixel ratio it is given without growing GPU residency: ${cloud.name}`, async () => {
      await withSession(
        () => openFixedBudget(cloud),
        async (session) => {
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
                // The adapter's own numbers cannot see an actor it stopped
                // tracking, and a rebuilt-actor implementation of the ratio is
                // exactly how one gets abandoned in the renderer. The renderer's
                // count can.
                assertRendererHoldsOnlyItsOwn(
                  await session.scene(),
                  stats,
                  `at device pixel ratio ${ratio}`,
                );
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
        },
      );
    });
  }
});

describe("a device pixel ratio that changes under a drawn scene", () => {
  for (const cloud of cloudsUnderTest()) {
    it(`scales what the renderer draws by the ratio and the CSS diameter by nothing: ${cloud.name}`, async () => {
      await withSession(
        () => openFixedBudget(cloud),
        async (session) => {
          const baseline = await settleAndAssert(session, SETTLE_MS);
          const baselineCssPx = drawingDiameterCssPx(baseline);
          const baselineRatio = drawingRatio(baseline);
          const baselineDrawn = await assertSceneDrawsAtRatio(
            session,
            baseline,
            baselineRatio,
          );

          const { samples } = await watching(
            session,
            SAMPLE_INTERVAL_MS,
            async () => {
              for (const ratio of DRAWN_RATIO_SWEEP) {
                await session.setDevicePixelRatio(ratio);
                await session.frame();
                const stats = await session.stats();
                const drawn = await assertSceneDrawsAtRatio(
                  session,
                  stats,
                  ratio,
                );

                // The camera never moves here, so the diameter the presentation
                // policy asked for is fixed for the whole sweep. The ratio is
                // only allowed to multiply it: a ratio that fed back into the
                // CSS diameter would change how large a point looks on a display
                // that did not change.
                expect(
                  drawingDiameterCssPx(stats),
                  `the device pixel ratio moved the CSS point diameter\n${shown(stats)}`,
                ).toBe(baselineCssPx);

                // The law itself, stated against the scene rather than against
                // the field the ratio was stored in: what reaches the shader is
                // proportional to the ratio, exactly.
                expect(
                  drawn / baselineDrawn,
                  `the drawn point size did not follow the ratio to ${ratio}\n${shown(stats)}`,
                ).toBeCloseTo(ratio / baselineRatio, 10);
              }
            },
          );
          expect(
            samples,
            "the sampler never read the run it was watching",
          ).toBeGreaterThan(0);

          // The sweep ends where it started, so the scene has to have undone
          // every step as well as applied it.
          const settled = await settleAndAssert(session, SETTLE_MS);
          expect(
            await assertSceneDrawsAtRatio(session, settled, 1),
            `the scene did not come back to the diameter it started at\n${shown(settled)}`,
          ).toBeCloseTo(baselineCssPx, 10);
        },
      );
    });

    it(`draws actors it already had and actors that arrive later at one ratio: ${cloud.name}`, async () => {
      await withSession(
        () => openFixedBudget(cloud),
        async (session) => {
          await settleAndAssert(session, SETTLE_MS);

          // First the ordering with nothing to update: deactivating hands this
          // cloud's share of the GPU pool back, which releases the reuse pool as
          // well as the drawn set, so the ratio is stated against an empty
          // renderer and every actor that comes back is built under it. A ratio
          // the adapter only ever applied to the actors it had at the time never
          // reaches these.
          await session.setActive(false);
          await session.until(
            "the renderer to hold nothing",
            (stats) => (stats.adapter?.gpuResidentTiles ?? 1) === 0,
            SETTLE_MS,
          );
          expect(
            (await session.scene()).actors,
            "an actor outlived the residency that owned it",
          ).toBe(0);
          await session.setDevicePixelRatio(FRESH_ACTOR_RATIO);
          await session.setActive(true);
          const rebuilt = await settleAndAssert(session, SETTLE_MS);
          expect(
            rebuilt.adapter!.submittedTiles,
            `the cloud never came back to be drawn\n${shown(rebuilt)}`,
          ).toBeGreaterThan(0);
          await assertSceneDrawsAtRatio(session, rebuilt, FRESH_ACTOR_RATIO);

          // Then the ordering that mixes both populations: a ratio changed while
          // tiles are still arriving has to reach the actors already on screen
          // and the actors the outstanding reads have yet to create. A fresh load
          // is what makes tiles physically arrive again — reactivation restores
          // them from the decoded cache in a single batch, with no window to land
          // in — and it builds a new adapter at the page's own ratio, so the
          // change below is a change for every actor either way.
          const { result: arrived } = await watching(
            session,
            SAMPLE_INTERVAL_MS,
            async () => {
              const midflight = await loadUntilMidStream(session, cloud);
              await session.setDevicePixelRatio(ARRIVING_ACTOR_RATIO);
              const settled = await settleAndAssert(session, SETTLE_MS);
              // Says the staging happened: tiles genuinely landed after the
              // change, so the settled scene is a mixture and not one population
              // asserted twice.
              expect(
                settled.adapter!.submittedTiles,
                `no tile arrived after the ratio changed, so nothing here was created under it\n${shown(settled)}`,
              ).toBeGreaterThan(midflight.adapter!.submittedTiles);
              return settled;
            },
          );

          // The renderer reports its oldest surviving actor, which is one that
          // predated the change; the count above says the rest were created after
          // it. Both populations answer with the same ratio, so neither the
          // update loop nor the actor the adapter builds has been left behind.
          await assertSceneDrawsAtRatio(session, arrived, ARRIVING_ACTOR_RATIO);
        },
      );
    });
  }
});

describe("a projection toggled under a moving camera", () => {
  for (const cloud of cloudsUnderTest()) {
    it(`reselects on every flip and converges in whichever mode it lands: ${cloud.name}`, async () => {
      await withSession(
        () => openFixedBudget(cloud),
        async (session) => {
          let generation = (await settleAndAssert(session, SETTLE_MS))
            .controller!.selection.generation;

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
        },
      );
    });

    it(`returns to the same tiles after a round trip through orthographic: ${cloud.name}`, async () => {
      await withSession(
        () => openFixedBudget(cloud),
        async (session) => {
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
        },
      );
    });
  }
});
