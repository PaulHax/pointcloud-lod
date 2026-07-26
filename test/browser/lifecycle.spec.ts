/**
 * Scenario 4 of the browser stress matrix, and the activation contract that
 * sits next to it: what hiding a streaming cloud must not do, and what
 * deactivating one must.
 *
 * `setVisible` is a draw switch and nothing else — the renderer keeps every
 * actor, additions keep arriving while hidden, and showing again is a pure
 * state restore. The failure it hides is a hide that quietly frees something,
 * which reads as a cloud that comes back blank or a tile short only in the
 * one gesture nobody repeats. `setActive(false)` is the other half: it is
 * what actually releases residency, and what it releases belongs in the CPU
 * cache so coming back does not re-read the file.
 *
 * Both switches only mean anything with reads genuinely in flight, which a
 * localhost fixture never gives you: a range read completes long before any
 * sampler can see it. Every session here therefore serves its cloud through a
 * stated per-request latency and reloads it, so the switch lands mid-decode.
 * Budgets are fixed for the same reason the other checks fix them — an
 * adaptive loop reacting to frames nobody drew would be a second variable.
 */

import { afterAll, describe, expect, it } from "vitest";

import {
  closeBrowser,
  cloudsUnderTest,
  openExample,
  type ExampleKeys,
  type ExampleSession,
  type ExampleStats,
} from "./harness";
import { settleAndAssert, watching } from "./invariants";

/**
 * Latency added to every range request. Stated, not measured: it exists to
 * hold reads open long enough for a switch to land while they run, and
 * nothing here asserts on how long anything took.
 */
const READ_LATENCY_MS = 25;

/** Node-side cadence for the live invariants. */
const SAMPLE_INTERVAL_MS = 25;

/** In-page cadence, which pays no round trip and so can be much tighter. */
const TRACE_INTERVAL_MS = 2;

const shown = (value: unknown): string => JSON.stringify(value, null, 2);

/** Serve this session's cloud slowly enough that a read can be caught running. */
const withReadLatency = async (session: ExampleSession): Promise<void> => {
  await session.page.route(
    (url) => url.pathname.endsWith(".laz"),
    async (route) => {
      await new Promise((done) => setTimeout(done, READ_LATENCY_MS));
      // A read cancelled by the switch under test tears its route down; that
      // is the scenario working, not a failure.
      await route.continue().catch(() => {});
    },
  );
};

/**
 * Reload the cloud and flip one switch at the first instant the controller is
 * both drawing tiles and still reading more.
 *
 * It runs in the page because the flip has to happen on the sample that saw
 * the reads: a round trip back to node between the two would leave the
 * scenario asserting against a state it never observed. The returned snapshot
 * is read immediately before the flip, so it is the evidence that reads were
 * in flight and tiles were on screen when it happened.
 */
const reloadAndInterrupt = (
  session: ExampleSession,
  options: { url: string; flip: "hide" | "deactivate"; pollMs: number },
): Promise<{ stats: ExampleStats; keys: ExampleKeys }> =>
  session.page.evaluate(async ({ url, flip, pollMs }) => {
    const api = (
      window as never as {
        pointCloudExample: {
          stats(): ExampleStats;
          keys(): ExampleKeys;
          load(target: string): Promise<void>;
          setVisible(visible: boolean): void;
          setActive(active: boolean): void;
        };
      }
    ).pointCloudExample;
    (window as never as { lifecycleLoad?: Promise<void> }).lifecycleLoad =
      api.load(url);

    const deadline = Date.now() + 60_000;
    while (Date.now() < deadline) {
      const stats = api.stats();
      const cloud = stats.controller;
      if (
        cloud !== null &&
        cloud.inFlight > 0 &&
        cloud.physicalTileOperations > 0 &&
        (stats.adapter?.submittedTiles ?? 0) > 0
      ) {
        const keys = api.keys();
        if (flip === "hide") api.setVisible(false);
        else api.setActive(false);
        return { stats, keys };
      }
      await new Promise((done) => setTimeout(done, pollMs));
    }
    throw new Error(
      "the cloud never had tiles on screen and a read still running at once",
    );
  }, options);

/** Resolve once the reload started by `reloadAndInterrupt` has finished. */
const awaitReload = (session: ExampleSession): Promise<void> =>
  session.page.evaluate(
    () => (window as never as { lifecycleLoad?: Promise<void> }).lifecycleLoad,
  ) as Promise<void>;

/** Poll until the renderer holds a tile it was not holding at `baseline`. */
const untilRendererGains = async (
  session: ExampleSession,
  baseline: readonly string[],
  timeoutMs = 60_000,
): Promise<{ gained: string[]; stats: ExampleStats }> => {
  const deadline = Date.now() + timeoutMs;
  let last: ExampleKeys | null = null;
  while (Date.now() < deadline) {
    const keys = await session.keys();
    const stats = await session.stats();
    last = keys;
    const gained = (keys.adapter?.submitted ?? []).filter(
      (key) => !baseline.includes(key),
    );
    if (gained.length > 0) return { gained, stats };
    await session.page.waitForTimeout(SAMPLE_INTERVAL_MS);
  }
  throw new Error(
    `no tile reached the renderer\nlast keys: ${shown(last)}`,
  );
};

/**
 * How much physical reading happened over an interval.
 *
 * `physicalTileOperations` is a gauge, not a total, so this is a sampled
 * observation and undercounts by construction — a read that starts and ends
 * between two samples is invisible. It is enough for the only claim made from
 * it: whether any read was running at all. The sampler lives in the page so
 * its cadence is not a round trip.
 */
interface ReadTrace {
  samples: number;
  /** Samples that caught at least one tile read running. */
  readSamples: number;
  peak: number;
}

const beginReadTrace = (session: ExampleSession): Promise<void> =>
  session.page.evaluate((intervalMs) => {
    const api = (
      window as never as { pointCloudExample: { stats(): ExampleStats } }
    ).pointCloudExample;
    const trace = { samples: 0, readSamples: 0, peak: 0, running: true };
    (window as never as { lifecycleTrace?: typeof trace }).lifecycleTrace =
      trace;
    const tick = (): void => {
      if (!trace.running) return;
      const reads = api.stats().controller?.physicalTileOperations ?? 0;
      trace.samples += 1;
      if (reads > 0) trace.readSamples += 1;
      if (reads > trace.peak) trace.peak = reads;
      setTimeout(tick, intervalMs);
    };
    tick();
  }, TRACE_INTERVAL_MS);

const endReadTrace = (session: ExampleSession): Promise<ReadTrace> =>
  session.page.evaluate(() => {
    const trace = (
      window as never as { lifecycleTrace: ReadTrace & { running: boolean } }
    ).lifecycleTrace;
    trace.running = false;
    return {
      samples: trace.samples,
      readSamples: trace.readSamples,
      peak: trace.peak,
    };
  });

/** Every session in this file streams slowly and draws to a fixed budget. */
const openSlowly = async (urlPath: string): Promise<ExampleSession> => {
  const session = await openExample({ cloud: urlPath });
  await withReadLatency(session);
  await session.setBudgetMode("fixed");
  return session;
};

describe("a cloud hidden while it is still decoding tiles", () => {
  afterAll(async () => {
    await closeBrowser();
  });

  for (const cloud of cloudsUnderTest()) {
    it(`keeps every tile it was handed and restores them on showing: ${cloud.name}`, async () => {
      const session = await openSlowly(cloud.urlPath);
      try {
        await settleAndAssert(session, 120_000);

        const atHide = await reloadAndInterrupt(session, {
          url: cloud.urlPath,
          flip: "hide",
          pollMs: TRACE_INTERVAL_MS,
        });
        // The snapshot predates the hide, so it is what the switch interrupted.
        expect(
          atHide.stats.controller!.inFlight,
          `hid with no read in flight\n${shown(atHide.stats)}`,
        ).toBeGreaterThan(0);
        expect(atHide.stats.controller!.physicalTileOperations).toBeGreaterThan(0);
        expect(atHide.stats.adapter!.visible).toBe(true);
        expect(atHide.stats.adapter!.drawnTiles).toBeGreaterThan(0);

        // Hiding is a draw switch: the renderer must still hold every actor it
        // held a moment ago, and only stop drawing them.
        const justHidden = await session.stats();
        const justHiddenKeys = await session.keys();
        expect(justHidden.adapter!.visible).toBe(false);
        expect(justHidden.adapter!.drawnTiles).toBe(0);
        expect(justHidden.adapter!.drawnPoints).toBe(0);
        expect(
          justHidden.adapter!.submittedTiles,
          `hiding shrank the submitted set\n${shown(justHidden)}`,
        ).toBeGreaterThanOrEqual(atHide.stats.adapter!.submittedTiles);
        expect(
          atHide.keys.adapter!.submitted.filter(
            (key) => !justHiddenKeys.adapter!.submitted.includes(key),
          ),
          `hiding released tiles the renderer was holding\n${shown(justHiddenKeys)}`,
        ).toEqual([]);

        const hidden = await watching(session, SAMPLE_INTERVAL_MS, async () => {
          // The reads the switch interrupted have to land in the renderer
          // anyway; a hidden cloud that stops accepting additions comes back
          // missing exactly the tiles that were in flight when it went away.
          const arrival = await untilRendererGains(
            session,
            atHide.keys.adapter!.submitted,
          );
          await awaitReload(session);
          // Selection has to keep running too, so a camera moved while the
          // layer was off screen is already correct when it comes back.
          await session.azimuth(24);
          await session.dolly(1.4);
          const settled = await settleAndAssert(session, 120_000);
          return { arrival, settled, keys: await session.keys() };
        });
        expect(hidden.samples).toBeGreaterThan(0);

        const { arrival, settled, keys: beforeShow } = hidden.result;
        expect(
          arrival.stats.adapter!.drawnTiles,
          `a hidden cloud drew tiles\n${shown(arrival.stats)}`,
        ).toBe(0);
        expect(arrival.stats.adapter!.drawnPoints).toBe(0);
        expect(
          arrival.stats.adapter!.submittedTiles,
          `a hidden cloud stopped accepting additions\n${shown(arrival.stats)}`,
        ).toBeGreaterThan(0);
        expect(settled.adapter!.drawnTiles).toBe(0);
        expect(settled.adapter!.drawnPoints).toBe(0);
        expect(settled.adapter!.submittedTiles).toBeGreaterThan(0);

        const arrivedWhileHidden = beforeShow.adapter!.submitted.filter(
          (key) => !atHide.keys.adapter!.submitted.includes(key),
        );
        expect(
          arrivedWhileHidden.length,
          `nothing arrived while hidden, so the restore proves nothing\n${shown(beforeShow)}`,
        ).toBeGreaterThan(0);

        await session.setVisible(true);
        await session.frame();
        const restored = await session.stats();
        const restoredKeys = await session.keys();
        expect(restored.adapter!.visible).toBe(true);
        expect(
          arrivedWhileHidden.filter(
            (key) => !restoredKeys.adapter!.submitted.includes(key),
          ),
          `tiles that arrived while hidden were dropped on showing\n${shown(restoredKeys)}`,
        ).toEqual([]);
        // Showing restores the whole submitted set, not the part of it that
        // predates the hide, and adds nothing of its own.
        expect(
          restoredKeys.adapter!.submitted,
          `showing changed what the renderer holds\n${shown(restoredKeys)}`,
        ).toEqual(beforeShow.adapter!.submitted);
        expect(
          restored.adapter!.drawnTiles,
          `showing drew less than the renderer holds\n${shown(restored)}`,
        ).toBe(restored.adapter!.submittedTiles);
        expect(restored.adapter!.drawnPoints).toBe(
          restored.adapter!.submittedPoints,
        );
        expect(restoredKeys.adapter!.submitted).toEqual(
          restoredKeys.controller!.submitted,
        );

        await settleAndAssert(session, 120_000);
        expect(session.failures).toEqual([]);
      } finally {
        await session.close();
      }
    });
  }
});

describe("a cloud deactivated while it is still decoding tiles", () => {
  afterAll(async () => {
    await closeBrowser();
  });

  for (const cloud of cloudsUnderTest()) {
    it(`sheds residency into the byte-bounded cache: ${cloud.name}`, async () => {
      const session = await openSlowly(cloud.urlPath);
      try {
        await settleAndAssert(session, 120_000);

        const atDeactivate = await reloadAndInterrupt(session, {
          url: cloud.urlPath,
          flip: "deactivate",
          pollMs: TRACE_INTERVAL_MS,
        });
        expect(
          atDeactivate.stats.controller!.inFlight,
          `deactivated with no read in flight\n${shown(atDeactivate.stats)}`,
        ).toBeGreaterThan(0);
        expect(
          atDeactivate.stats.controller!.residentTiles,
        ).toBeGreaterThan(0);

        const quiet = await watching(session, SAMPLE_INTERVAL_MS, async () => {
          // Deactivation cancels reads but not the promises behind them, and
          // the actors it removes wait in the adapter's reuse pool until the
          // host's next paint re-applies the ceiling: the shed state is only
          // complete once both have happened.
          const inactive = await session.until(
            "the cancelled reads to settle and the reuse pool to be trimmed",
            (value) =>
              value.controller !== null &&
              value.adapter !== null &&
              value.controller.physicalTileOperations === 0 &&
              value.controller.physicalHierarchyOperations === 0 &&
              value.adapter.gpuResidentTiles === 0,
            120_000,
          );
          await awaitReload(session);
          const keys = await session.keys();

          expect(inactive.controller!.active).toBe(false);
          expect(
            inactive.controller!.residentTiles,
            `a deactivated controller kept residency\n${shown(inactive)}`,
          ).toBe(0);
          expect(inactive.controller!.residentPoints).toBe(0);
          expect(inactive.controller!.residentBytes).toBe(0);
          expect(keys.controller!.resident).toEqual([]);
          expect(keys.controller!.submitted).toEqual([]);

          // The removals have to reach the renderer, or the memory the
          // controller stopped accounting for is still on the GPU.
          expect(
            keys.adapter!.submitted,
            `the renderer kept tiles a deactivated controller released\n${shown(inactive)}`,
          ).toEqual([]);
          expect(inactive.adapter!.submittedTiles).toBe(0);
          expect(inactive.adapter!.drawnTiles).toBe(0);
          expect(
            inactive.adapter!.gpuResidentTiles,
            `GPU resources outlived deactivation\n${shown(inactive)}`,
          ).toBe(0);

          // Released payloads are parked, not thrown away: that is the whole
          // reason coming back is cheap.
          expect(
            inactive.controller!.cachedTiles,
            `decoded payloads were discarded rather than cached\n${shown(inactive)}`,
          ).toBeGreaterThanOrEqual(atDeactivate.stats.controller!.residentTiles);
          expect(inactive.controller!.cachedBytes).toBeGreaterThanOrEqual(
            atDeactivate.stats.controller!.residentBytes,
          );
          expect(inactive.controller!.cachedTiles).toBeGreaterThan(
            atDeactivate.stats.controller!.cachedTiles,
          );
          expect(inactive.controller!.cachedBytes).toBeGreaterThan(
            atDeactivate.stats.controller!.cachedBytes,
          );
          // Nothing decoded is anywhere else now.
          expect(inactive.controller!.decodedTiles).toBe(
            inactive.controller!.cachedTiles,
          );
          expect(inactive.controller!.decodedBytes).toBe(
            inactive.controller!.cachedBytes,
          );
          return inactive;
        });
        expect(quiet.samples).toBeGreaterThan(0);

        // The cache the shed just filled is byte-bounded, and the bound has to
        // hold on a cloud nobody is drawing exactly as it does on one being
        // streamed. The bound is `cacheBytes`, the LRU's own ceiling — not
        // `memoryBudgetBytes`, which is this controller's share of the GPU
        // pool and is surrendered by the very call that fills the cache.
        // Comparing against that share asserted the cache must be empty
        // whenever the cloud is inactive, which is the opposite of the
        // retention deactivation exists to provide.
        expect(
          quiet.result.controller!.cachedBytes,
          `the decoded cache exceeded its byte ceiling\n${shown(quiet.result)}`,
        ).toBeLessThanOrEqual(quiet.result.controller!.cacheBytes);
        expect(
          quiet.result.controller!.memoryBudgetBytes,
          `an inactive cloud kept its share of the GPU pool\n${shown(quiet.result)}`,
        ).toBe(0);

        // Coming back is the other half of the switch: a cloud deactivated
        // mid-decode still owes the camera the selection it asked for.
        await session.setActive(true);
        await settleAndAssert(session, 120_000);
        expect(session.failures).toEqual([]);
      } finally {
        await session.close();
      }
    });

    it(`comes back from its own cache, not from the file: ${cloud.name}`, async () => {
      const session = await openSlowly(cloud.urlPath);
      try {
        // Deactivating at rest rather than mid-decode: what makes the cost
        // comparison mean anything is that the cache holds the whole selected
        // set, and a tile still queued when the switch flipped was never read
        // at all. Mid-decode deactivation is the check above.
        const loaded = await settleAndAssert(session, 120_000);
        const loadedKeys = await session.keys();
        expect(loaded.controller!.residentTiles).toBeGreaterThan(0);

        await session.setActive(false);
        // Removed actors go to the adapter's reuse pool, and the ceiling that
        // empties it is re-applied by the host on its next paint — so the
        // shed is only complete a frame later, not on the call.
        await session.frame();
        const inactive = await session.until(
          "residency to be shed and the reuse pool trimmed",
          (value) =>
            value.controller !== null &&
            value.adapter !== null &&
            value.controller.residentTiles === 0 &&
            value.controller.physicalTileOperations === 0 &&
            value.adapter.gpuResidentTiles === 0,
          120_000,
        );
        const inactiveKeys = await session.keys();
        expect(inactive.controller!.active).toBe(false);
        expect(inactive.controller!.residentPoints).toBe(0);
        expect(inactive.controller!.residentBytes).toBe(0);
        expect(inactiveKeys.controller!.resident).toEqual([]);
        expect(inactiveKeys.controller!.submitted).toEqual([]);
        expect(
          inactiveKeys.adapter!.submitted,
          `the renderer kept tiles a deactivated controller released\n${shown(inactive)}`,
        ).toEqual([]);
        expect(
          inactive.adapter!.gpuResidentTiles,
          `GPU resources outlived deactivation\n${shown(inactive)}`,
        ).toBe(0);
        expect(
          inactive.controller!.cachedTiles,
          `decoded payloads were discarded rather than cached\n${shown(inactive)}`,
        ).toBeGreaterThanOrEqual(loaded.controller!.residentTiles);
        expect(inactive.controller!.decodedTiles).toBe(
          inactive.controller!.cachedTiles,
        );

        // Whether the reactivation reads anything is decided here: a cache
        // that had to evict cannot serve what it dropped.
        const keptEverything =
          inactive.controller!.cachedTiles ===
          loaded.controller!.residentTiles + loaded.controller!.cachedTiles;

        // The trace starts before the switch so no read can predate it. The
        // inactive window itself is not sampled for the live invariants here;
        // the check above owns that window.
        await beginReadTrace(session);
        await session.setActive(true);
        const back = await watching(session, SAMPLE_INTERVAL_MS, () =>
          settleAndAssert(session, 120_000),
        );
        const warm = await endReadTrace(session);
        expect(back.samples).toBeGreaterThan(0);

        const backKeys = await session.keys();
        expect(back.result.controller!.active).toBe(true);
        expect(
          backKeys.controller!.resident,
          `reactivation converged on a different set\n${shown(back.result)}`,
        ).toEqual(loadedKeys.controller!.resident);
        expect(backKeys.adapter!.submitted).toEqual(
          loadedKeys.adapter!.submitted,
        );
        expect(back.result.controller!.residentPoints).toBe(
          loaded.controller!.residentPoints,
        );

        // The same instrument over a cold load of the same cloud, so the two
        // numbers are comparable. Routing is installed on this page, which
        // disables the browser's HTTP cache, so the only cache in the
        // comparison is the controller's own.
        await beginReadTrace(session);
        const cold = await watching(session, SAMPLE_INTERVAL_MS, async () => {
          await session.load(cloud.urlPath);
          return settleAndAssert(session, 120_000);
        });
        const coldTrace = await endReadTrace(session);
        expect(cold.samples).toBeGreaterThan(0);

        expect(
          coldTrace.readSamples,
          `the sampler never caught the cold load reading, so there is nothing to compare against\n${shown({ warm, coldTrace })}`,
        ).toBeGreaterThan(0);
        expect(
          warm.readSamples,
          `coming back read more than loading from cold\n${shown({ warm, coldTrace })}`,
        ).toBeLessThanOrEqual(coldTrace.readSamples);
        if (keptEverything) {
          expect(
            warm.readSamples,
            `reactivation re-read tiles the cache still held\n${shown({ warm, coldTrace, inactive })}`,
          ).toBe(0);
          expect(warm.peak).toBe(0);
        }

        await settleAndAssert(session, 120_000);
        expect(session.failures).toEqual([]);
      } finally {
        await session.close();
      }
    });
  }
});
