/**
 * Source lifecycle in a real browser: switching clouds faster than they load,
 * and tearing the anchor down in the middle of one.
 *
 * Both are the same failure from two directions — something outliving the load
 * that asked for it. A switch landing while the previous cloud still has reads
 * outstanding can leave the renderer holding actors no controller owns; a
 * dispose in that same window can leave a timer or a listener behind, which
 * shows up as drift only after several cycles. Neither is reachable from a
 * unit test with a fake source: both need a read that is genuinely still
 * running when the next call arrives.
 */

import { afterAll, describe, expect, it } from "vitest";

import {
  cloudPair,
  cloudsUnderTest,
  closeBrowser,
  openExample,
  type CloudUnderTest,
  type ExampleKeys,
  type ExampleSession,
  type ExampleStats,
  type SceneReading,
} from "./harness";
import { assertSettled, settleAndAssert, watching } from "./invariants";

/** Live-tier sampling cadence: short enough to land inside a single switch. */
const SAMPLE_MS = 25;

/** Long enough for an abandoned range read to come back and misbehave. */
const AFTER_DISPOSE_MS = 1_000;

/** A supplied cloud is a real one; convergence is not a few seconds. */
const CONVERGE_MS = 120_000;

/**
 * The page's driving handles, redeclared because the checks below reach them
 * from inside the page. Deciding in Node would decide about the past: a round
 * trip is longer than a range read of the fixture, so the "still loading" a
 * check acted on could already have converged.
 */
interface ExampleHandles {
  pointCloudExample: {
    stats(): ExampleStats;
    load(url: string): Promise<void>;
    dispose(): void;
  };
}

/** Physical work the controller had outstanding at the instant of the act. */
interface WorkInFlight {
  readonly tileReads: number;
  readonly pageReads: number;
  readonly residentTiles: number;
}

/**
 * Act at the first instant the controller has a physical read outstanding.
 *
 * The counters are read and acted on in the same task, so the report is about
 * the moment the act happened rather than a moment that has since passed.
 * `null` means the load drained before the poller ever caught it — a run that
 * tested nothing, which is a failure of the scenario, not a pass.
 */
const actWhileReading = (
  session: ExampleSession,
  plan: {
    /** Load this first, then wait for it to start reading. */
    readonly begin?: string;
    readonly act: "load" | "dispose";
    /** The source to switch to, when the act is a load. */
    readonly url?: string;
  },
  timeoutMs = 30_000,
): Promise<WorkInFlight | null> =>
  session.page.evaluate(
    async ({ begin, act, url, timeoutMs: limit }) => {
      const api = (window as unknown as ExampleHandles).pointCloudExample;
      const started = begin === undefined ? null : api.load(begin);
      const deadline = performance.now() + limit;
      let seen: WorkInFlight | null = null;
      while (performance.now() < deadline) {
        const cloud = api.stats().controller;
        if (
          cloud !== null &&
          cloud.physicalTileOperations + cloud.physicalHierarchyOperations > 0
        ) {
          seen = {
            tileReads: cloud.physicalTileOperations,
            pageReads: cloud.physicalHierarchyOperations,
            residentTiles: cloud.residentTiles,
          };
          if (act === "dispose") api.dispose();
          else if (url !== undefined) void api.load(url);
          break;
        }
        await new Promise((resume) => setTimeout(resume, 2));
      }
      if (started !== null) await started;
      return seen;
    },
    { begin: plan.begin, act: plan.act, url: plan.url, timeoutMs },
  );

/** Fail the scenario, rather than pass it, when the act missed its window. */
const caughtLoading = (seen: WorkInFlight | null, when: string): WorkInFlight => {
  expect(
    seen,
    `${when}: the load drained before the act could land on it, so nothing was tested`,
  ).not.toBeNull();
  const work = seen as WorkInFlight;
  expect(
    work.tileReads + work.pageReads,
    `${when}: acted with no read outstanding`,
  ).toBeGreaterThan(0);
  return work;
};

/**
 * Watch for a while that nothing rebuilds the cloud behind a teardown, and
 * that the renderer is left holding nothing.
 *
 * One sample taken straight after the call would miss both halves of what can
 * go wrong: a continuation of the abandoned load installing a controller after
 * the anchor was removed, and an abandoned read coming back to report into a
 * page that no longer owns anything.
 *
 * The actor count is the half no other reading can make. `stats()` and
 * `keys()` both come from the adapter, and a disposed adapter reports null —
 * so an actor it failed to remove from the renderer is counted by nothing it
 * reports, and every "the renderer holds nothing" assertion phrased through it
 * passes on an empty handle. `scene()` asks the renderer itself.
 */
const staysTornDown = async (
  session: ExampleSession,
  ms: number,
): Promise<ExampleStats> => {
  const deadline = Date.now() + ms;
  let last = await session.stats();
  while (Date.now() < deadline) {
    last = await session.stats();
    expect(last.controller, "the cloud came back after dispose").toBeNull();
    expect(last.adapter, "the renderer adapter came back after dispose").toBeNull();
    expect(
      (await session.scene()).actors,
      "the renderer still holds actors a disposed adapter abandoned there",
    ).toBe(0);
    await session.page.waitForTimeout(SAMPLE_MS);
  }
  return last;
};

/** The parts of a converged state that must not drift between teardowns. */
const shape = (stats: ExampleStats, keys: ExampleKeys) => ({
  source: stats.source,
  sourcePoints: stats.sourcePoints,
  residentTiles: stats.controller?.residentTiles,
  residentPoints: stats.controller?.residentPoints,
  targetTiles: stats.controller?.selection.targetTiles,
  targetPoints: stats.controller?.selection.targetPoints,
  submittedTiles: stats.adapter?.submittedTiles,
  tiles: keys.controller?.resident,
});

const settledShape = async (session: ExampleSession) => {
  const stats = await settleAndAssert(session, CONVERGE_MS);
  const keys = await session.keys();
  // Read at rest, so this is one consistent picture: nothing is loading, and
  // no batch can land between the adapter's numbers and the renderer's.
  const scene = await session.scene();
  return { stats, keys, scene, shape: shape(stats, keys) };
};

/**
 * What the renderer holding exactly the controller's tiles means numerically.
 *
 * `settleAndAssert` already compares the two key sets; these are the payload
 * counts each side arrived at independently, and the disjointness the
 * adapter's teardown relies on to release every actor exactly once.
 *
 * The scene reading is what makes the claim about the renderer rather than
 * about the adapter's opinion of it: an actor abandoned by a superseded
 * adapter is in no live adapter's accounting, so it is invisible to every
 * count above and shows up only in what the renderer itself holds.
 */
const assertRendererMatchesController = (
  stats: ExampleStats,
  keys: ExampleKeys,
  scene: SceneReading,
): void => {
  const cloud = stats.controller;
  const gpu = stats.adapter;
  expect(cloud, "the cloud is gone at the end of a switch").not.toBeNull();
  expect(gpu, "the renderer adapter is gone at the end of a switch").not.toBeNull();
  // An empty cloud would satisfy every comparison below without holding
  // anything, so state that there is something to compare.
  expect(stats.sourcePoints, "no asset behind the settled source").toBeGreaterThan(0);
  expect(cloud!.residentTiles, "converged holding no tiles at all").toBeGreaterThan(0);
  expect(gpu!.submittedTiles, "the renderer holds a different number of tiles").toBe(
    cloud!.residentTiles,
  );
  expect(gpu!.submittedPoints, "the two sides disagree on points held").toBe(
    cloud!.residentPoints,
  );
  expect(
    gpu!.submittedPoints,
    "the renderer holds more points than the loaded asset has",
  ).toBeLessThanOrEqual(stats.sourcePoints);
  const pooled = new Set(keys.adapter?.pooled ?? []);
  expect(
    (keys.adapter?.submitted ?? []).filter((key) => pooled.has(key)),
    "a tile is both submitted and pooled, so teardown would release it twice",
  ).toEqual([]);
  // Submitted *plus* pooled, not submitted alone: a retired actor stays in the
  // renderer, hidden, until the pool is trimmed to `resourceCeilingBytes`, so
  // that is legitimately what the renderer holds. The example adds no other
  // props, so this is an exact equality — anything above it is an actor no
  // live adapter would ever remove again.
  expect(
    scene.actors,
    `the renderer holds actors no live adapter accounts for\n${JSON.stringify({ scene, adapter: gpu }, null, 2)}`,
  ).toBe(gpu!.submittedTiles + gpu!.pooledTiles);
};

afterAll(async () => {
  await closeBrowser();
});

describe("switching sources faster than they load", () => {
  // The pair comes from the same list every other scenario runs over, so a
  // supplied cloud is switched against a second supplied one when the
  // environment provides them. With none, `cloudPair()` serves ONE file at two
  // URLs: two sources to the library — separate endpoint, full teardown, fresh
  // hierarchy — but not two different point sets, so this run cannot prove the
  // final cloud's points came from the last URL rather than the first.
  const [first, second] = cloudPair();

  it(`ends holding only the last cloud asked for: ${first.name} against ${second.name}`, async () => {
    const session = await openExample({ cloud: first.urlPath });
    try {
      await session.setBudgetMode("fixed");
      const alone = new Map([[first.urlPath, (await settledShape(session)).shape]]);
      await session.load(second.urlPath);
      alone.set(second.urlPath, (await settledShape(session)).shape);

      const { result: settled, samples } = await watching(session, SAMPLE_MS, async () => {
        // Each switch is fired at an instant the cloud it replaces is provably
        // still reading, so the controller being torn down always has work to
        // abandon rather than merely usually having some.
        const switches: readonly (readonly [CloudUnderTest, CloudUnderTest])[] = [
          [first, second],
          [second, first],
          [first, second],
        ];
        for (const [from, to] of switches) {
          caughtLoading(
            await actWhileReading(session, {
              begin: from.urlPath,
              act: "load",
              url: to.urlPath,
            }),
            `switching from ${from.name} to ${to.name}`,
          );
        }
        // And once with no gap at all: the second call lands while the first
        // has not even finished opening its source.
        const opening = session.load(first.urlPath);
        const overlapping = session.load(second.urlPath);
        await Promise.all([opening, overlapping]);
        return settleAndAssert(session, CONVERGE_MS);
      });
      expect(samples, "the live invariants were never sampled").toBeGreaterThan(0);

      const keys = await session.keys();
      expect(settled.source, "settled on a source nobody asked for last").toBe(
        second.urlPath,
      );
      expect(settled.sourcePoints, "the asset behind the last URL is not the one loaded").toBe(
        alone.get(second.urlPath)!.sourcePoints,
      );
      assertRendererMatchesController(settled, keys, await session.scene());
      // Loading that cloud alone is the whole answer; anything the storm added
      // or dropped shows as a different tile set for the same camera.
      expect(
        shape(settled, keys),
        "switching under load converged somewhere else than loading it alone",
      ).toEqual(alone.get(second.urlPath));
      expect(session.failures).toEqual([]);
    } finally {
      await session.close();
    }
  });

  it(`leaves the renderer holding no more than one cloud's actors, switch after switch: ${first.name} against ${second.name}`, async () => {
    const session = await openExample({ cloud: first.urlPath });
    try {
      await session.setBudgetMode("fixed");
      // What each cloud settles to on its own, in the renderer's own terms.
      // Every switch below ends on one of these two clouds, so a switch that
      // abandoned actors ends holding more than this — and by roughly a whole
      // cloud, since a superseded adapter is disposed with its actors still
      // added to the renderer.
      const alone = new Map<string, { actors: number; submitted: number }>();
      const record = async (cloud: CloudUnderTest) => {
        const { stats, keys, scene } = await settledShape(session);
        assertRendererMatchesController(stats, keys, scene);
        alone.set(cloud.urlPath, {
          actors: scene.actors,
          submitted: stats.adapter!.submittedTiles,
        });
        expect(
          scene.actors,
          `${cloud.name}: loading it alone put nothing in the renderer, so there is no baseline to grow past`,
        ).toBeGreaterThan(0);
      };
      await record(first);
      await session.load(second.urlPath);
      await record(second);

      const switches: readonly (readonly [CloudUnderTest, CloudUnderTest])[] = [
        [second, first],
        [first, second],
        [second, first],
        [first, second],
      ];
      for (const [cycle, [from, to]] of switches.entries()) {
        const when = `switch ${cycle + 1}, ${from.name} to ${to.name}`;
        // Fired while the cloud it replaces is provably still reading: that is
        // the window where the adapter being torn down still has a batch on
        // the way to it, and an actor created after its owner is gone is
        // exactly the actor nothing will ever remove.
        caughtLoading(
          await actWhileReading(session, {
            begin: from.urlPath,
            act: "load",
            url: to.urlPath,
          }),
          when,
        );
        const { stats, keys, scene } = await settledShape(session);
        // The renderer holds what the live adapter holds — nothing from the
        // adapters this switch and every switch before it disposed.
        assertRendererMatchesController(stats, keys, scene);
        expect(stats.source, `${when}: settled on the wrong source`).toBe(to.urlPath);
        expect(
          stats.adapter!.submittedTiles,
          `${when}: converged on a different tile count than loading it alone`,
        ).toBe(alone.get(to.urlPath)!.submitted);
        // Non-growth stated against the same cloud loaded clean, so a per-switch
        // leak of even one actor fails here however many switches it took.
        expect(
          scene.actors,
          `${when}: the renderer holds more actors than loading this cloud alone leaves it holding\n${JSON.stringify({ scene, baseline: alone.get(to.urlPath), adapter: stats.adapter }, null, 2)}`,
        ).toBeLessThanOrEqual(alone.get(to.urlPath)!.actors);
      }
      expect(session.failures).toEqual([]);
    } finally {
      await session.close();
    }
  });
});

describe("removing and recreating the anchor mid-load", () => {
  for (const cloud of cloudsUnderTest()) {
    it(`tears down and comes back, cycle after cycle: ${cloud.name}`, async () => {
      const session = await openExample({ cloud: cloud.urlPath });
      try {
        await session.setBudgetMode("fixed");
        const settledOnce = await settledShape(session);
        const reference = settledOnce.shape;
        const referenceActors = settledOnce.scene.actors;

        // One cycle proves the teardown; several prove nothing accumulates
        // across them, which is the only way a surviving listener or timer
        // becomes visible.
        for (const cycle of [1, 2, 3, 4]) {
          const { result: settled, samples } = await watching(
            session,
            SAMPLE_MS,
            async () => {
              // The renderer is holding this cloud's actors going in, so the
              // zero `staysTornDown` insists on afterwards is a teardown that
              // happened rather than a scene that was empty all along.
              expect(
                (await session.scene()).actors,
                `cycle ${cycle}: the renderer held nothing to tear down`,
              ).toBeGreaterThan(0);
              caughtLoading(
                await actWhileReading(session, {
                  begin: cloud.urlPath,
                  act: "dispose",
                }),
                `cycle ${cycle}`,
              );

              const down = await staysTornDown(session, AFTER_DISPOSE_MS);
              // With the controller gone this states what a torn-down anchor
              // must look like: nothing held, and nothing the renderer still
              // has to be told about.
              assertSettled(down, await session.keys());
              expect(
                session.failures,
                `cycle ${cycle}: teardown mid-load produced an error`,
              ).toEqual([]);

              await session.load(cloud.urlPath);
              return settleAndAssert(session, CONVERGE_MS);
            },
          );
          expect(samples, `cycle ${cycle}: the live invariants were never sampled`)
            .toBeGreaterThan(0);

          const keys = await session.keys();
          const scene = await session.scene();
          assertRendererMatchesController(settled, keys, scene);
          // A teardown that released everything but one actor per cycle would
          // still converge on the same tiles; only the renderer's own count
          // grows, and only against the first cycle's is that visible.
          expect(
            scene.actors,
            `cycle ${cycle}: the renderer holds more actors than the first load left it holding`,
          ).toBeLessThanOrEqual(referenceActors);
          expect(
            shape(settled, keys),
            `cycle ${cycle}: the recreated cloud converged somewhere else`,
          ).toEqual(reference);
          expect(session.failures, `cycle ${cycle}`).toEqual([]);
        }
      } finally {
        await session.close();
      }
    });
  }
});
