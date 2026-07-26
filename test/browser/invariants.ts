/**
 * The properties every stress run asserts, whatever it is stressing.
 *
 * Two tiers, because they are not true at the same times. Live invariants hold
 * at every instant, including halfway through a gesture with reads in flight —
 * they are the ones a sampler can check while the scenario runs. Settled
 * invariants describe what convergence means, and are false in the middle of
 * any transition by design: a budget that has just shrunk leaves more points on
 * screen than it allows until the next selection pass replaces them.
 *
 * A scenario that only checked the settled tier could pass while doing
 * something unbounded on the way there; one that only sampled would never
 * notice the system failing to converge at all.
 */

import { expect } from "vitest";

import type { ExampleKeys, ExampleSession, ExampleStats } from "./harness";

/** Both tiers report the whole sample, so a failure names the state it saw. */
const shown = (stats: ExampleStats): string => JSON.stringify(stats, null, 2);

/**
 * True at every instant, mid-gesture included.
 *
 * Physical operation counts are the ones worth watching: `inFlight` counts
 * what the controller still wants, which cancellation lowers immediately,
 * while the physical counts only fall when the abandoned promise actually
 * settles. A leak of read slots shows up here and nowhere else.
 */
export const assertLive = (stats: ExampleStats): void => {
  const cloud = stats.controller;
  if (cloud === null) return;

  expect(
    cloud.physicalTileOperations,
    `tile reads exceeded fetchConcurrency\n${shown(stats)}`,
  ).toBeLessThanOrEqual(cloud.fetchConcurrency);
  expect(
    cloud.physicalHierarchyOperations,
    `hierarchy reads exceeded hierarchyConcurrency\n${shown(stats)}`,
  ).toBeLessThanOrEqual(cloud.hierarchyConcurrency);

  // Wanted work is a subset of physical work in the steady state, but
  // cancellation makes it possible to want nothing while a read still runs.
  // Only the reverse — wanting a read nothing is running — is broken.
  expect(
    cloud.inFlight,
    `more reads wanted than are running\n${shown(stats)}`,
  ).toBeLessThanOrEqual(cloud.physicalTileOperations);
  expect(
    cloud.hierarchyInFlight,
    `more pages wanted than are running\n${shown(stats)}`,
  ).toBeLessThanOrEqual(cloud.physicalHierarchyOperations);

  // The LRU is the only decoded store with a byte bound; resident payloads are
  // bounded by the point budget instead, so they are not in this comparison.
  expect(
    cloud.cachedBytes,
    `the decoded cache exceeded its byte budget\n${shown(stats)}`,
  ).toBeLessThanOrEqual(cloud.memoryBudgetBytes);

  for (const [label, value] of [
    ["residentTiles", cloud.residentTiles],
    ["residentPoints", cloud.residentPoints],
    ["queuedTiles", cloud.queuedTiles],
    ["queuedPages", cloud.queuedPages],
    ["physicalTileOperations", cloud.physicalTileOperations],
    ["physicalHierarchyOperations", cloud.physicalHierarchyOperations],
  ] as const) {
    expect(value, `${label} went negative\n${shown(stats)}`).toBeGreaterThanOrEqual(0);
  }

  const gpu = stats.adapter;
  if (gpu !== null) {
    // Only the pool is trimmed to the ceiling, so submitted tiles may carry it
    // over — but then there is nothing left to trim.
    if (gpu.gpuResidentBytes > gpu.resourceCeilingBytes) {
      expect(
        gpu.pooledTiles,
        `over the GPU ceiling with a pool still holding actors\n${shown(stats)}`,
      ).toBe(0);
    }
  }

  const view = stats.governor;
  if (view !== null) {
    expect(
      Number.isFinite(view.trackBudget),
      `the governor's budget stopped being a finite number\n${shown(stats)}`,
    ).toBe(true);
    if (view.memoryCeilingPoints !== null) {
      expect(
        view.trackBudget,
        `the governor grew past the memory ceiling\n${shown(stats)}`,
      ).toBeLessThanOrEqual(view.memoryCeilingPoints);
    }
  }
};

/**
 * True once the scenario has converged, and only then.
 *
 * `settle()` has already established that no work is in flight; this states
 * what the converged state must look like.
 */
export const assertSettled = (stats: ExampleStats, keys: ExampleKeys): void => {
  assertLive(stats);
  const cloud = stats.controller;
  if (cloud === null) {
    expect(
      keys.adapter?.submitted ?? [],
      "the cloud is gone but the renderer still holds tiles",
    ).toEqual([]);
    return;
  }

  expect(cloud.queuedTiles, `tiles still queued at rest\n${shown(stats)}`).toBe(0);
  expect(cloud.queuedPages, `pages still queued at rest\n${shown(stats)}`).toBe(0);
  expect(
    cloud.physicalTileOperations,
    `a tile read outlived convergence\n${shown(stats)}`,
  ).toBe(0);
  expect(
    cloud.physicalHierarchyOperations,
    `a page read outlived convergence\n${shown(stats)}`,
  ).toBe(0);

  // A flush is the only reason the two controller sets differ, and a converged
  // controller has no flush pending.
  expect(
    keys.controller?.submitted,
    `the controller's screen set and handed-off set disagree\n${shown(stats)}`,
  ).toEqual(keys.controller?.resident);

  // The invariant the whole ownership design exists to keep: what the
  // controller believes the renderer holds is what the renderer holds.
  expect(
    keys.adapter?.submitted ?? [],
    `renderer and controller hold different tiles\n${shown(stats)}`,
  ).toEqual(keys.controller?.submitted ?? []);

  // Selection is what the budget bounds; residency follows selection, and at
  // rest cannot have outrun it.
  expect(
    cloud.selection.targetPoints,
    `selection exceeded the point budget\n${shown(stats)}`,
  ).toBeLessThanOrEqual(cloud.pointBudget);
  expect(
    cloud.residentPoints,
    `more points on screen than selection asked for\n${shown(stats)}`,
  ).toBeLessThanOrEqual(cloud.selection.targetPoints);
  expect(
    cloud.pointBudget,
    `the budget exceeded what memory allows\n${shown(stats)}`,
  ).toBeLessThanOrEqual(cloud.memoryCeilingPoints);

  if (stats.governor !== null) {
    expect(
      stats.governor.needsFrame,
      `converged but still asking for frames\n${shown(stats)}`,
    ).toBe(false);
  }
};

/**
 * Sample the live invariants while `body` runs.
 *
 * A scenario that only checked before and after would miss a bound that is
 * violated during the gesture and tidy again by the time it ends — which is
 * exactly the shape of every concurrency bug this suite is looking for.
 */
export const watching = async <T>(
  session: ExampleSession,
  intervalMs: number,
  body: () => Promise<T>,
): Promise<{ result: T; samples: number; peak: ExampleStats | null }> => {
  let running = true;
  let samples = 0;
  let peak: ExampleStats | null = null;
  let failure: unknown = null;

  const sampler = (async () => {
    while (running) {
      const stats = await session.stats().catch(() => null);
      if (stats !== null) {
        samples += 1;
        try {
          assertLive(stats);
        } catch (error) {
          failure ??= error;
          // Keep sampling: stopping here would race the body's own teardown.
        }
        const cloud = stats.controller;
        if (
          cloud !== null &&
          (peak === null || cloud.residentPoints > peak.controller!.residentPoints)
        ) {
          peak = stats;
        }
      }
      await session.page.waitForTimeout(intervalMs);
    }
  })();

  try {
    const result = await body();
    return { result, samples, peak };
  } finally {
    running = false;
    await sampler;
    if (failure !== null) throw failure;
  }
};

/** Converge, then assert everything convergence is supposed to mean. */
export const settleAndAssert = async (
  session: ExampleSession,
  timeoutMs?: number,
): Promise<ExampleStats> => {
  const stats = await session.settle(timeoutMs);
  assertSettled(stats, await session.keys());
  return stats;
};
