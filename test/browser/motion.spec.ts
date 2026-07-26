/**
 * Scenarios 1, 2 and 10 of the browser stress matrix: a camera that will not
 * stop moving.
 *
 * All three drive the camera and nothing else, because the failures they hunt
 * live in the seams the page assembles rather than in any one module: a read
 * slot the controller never gets back, a key set that drifts from what the
 * renderer holds, a motion regime that latches on and is never released. None
 * of those are visible to a check that moves the camera once and looks at what
 * came out — they need many selection generations, or a supersession landing
 * while a read is outstanding, or a gesture that stops.
 */

import { afterAll, describe, expect, it } from "vitest";

import {
  closeBrowser,
  cloudsUnderTest,
  openExample,
  type ExampleSession,
  type ExampleStats,
} from "./harness";
import { settleAndAssert, watching } from "./invariants";

/**
 * The frame duration every scenario here reports.
 *
 * SwiftShader's real timings swing with whatever else is on the machine and
 * the budget loop makes decisions from them, so a measured frame would make
 * the governor decide something different on every run. 30 fps is the cadence
 * a video-driven camera actually delivers, and stating it pins the loop's
 * behaviour: the governor reads it as over both its frame-time targets, so the
 * budget walks down to its floor and stays there.
 */
const VIDEO_FRAME_MS = 33;

/** Live sampling cadence during a burst. Fast enough to catch a transient. */
const SAMPLE_MS = 50;

const openCloud = async (urlPath: string): Promise<ExampleSession> => {
  const session = await openExample({ cloud: urlPath });
  await session.setSyntheticFrameMs(VIDEO_FRAME_MS);
  return session;
};

/**
 * Run `step` until `seconds` have passed, one step per painted frame.
 *
 * A driving call only queues a frame, and the camera does not reach the
 * controller until one paints — so a paint is the unit of camera motion, and
 * pacing to it is what "as fast as the page will take it" means here. Anything
 * faster only overwrites a camera nothing has read yet.
 */
const driveFrames = async (
  session: ExampleSession,
  seconds: number,
  step: (index: number) => Promise<void>,
): Promise<number> => {
  const deadline = Date.now() + seconds * 1000;
  let index = 0;
  while (Date.now() < deadline) {
    await step(index);
    await session.frame();
    index += 1;
  }
  return index;
};

/** The worst a burst reached, so an assertion can name the number it saw. */
interface Extremes {
  readonly samples: number;
  readonly physicalTiles: number;
  readonly inFlight: number;
  readonly fetchConcurrency: number;
}

const nothingSeen: Extremes = {
  samples: 0,
  physicalTiles: 0,
  inFlight: 0,
  // Read off the page rather than assumed: the example owns this number.
  fetchConcurrency: Number.POSITIVE_INFINITY,
};

const extend = (seen: Extremes, stats: ExampleStats): Extremes => {
  const cloud = stats.controller;
  if (cloud === null) return { ...seen, samples: seen.samples + 1 };
  return {
    samples: seen.samples + 1,
    physicalTiles: Math.max(seen.physicalTiles, cloud.physicalTileOperations),
    inFlight: Math.max(seen.inFlight, cloud.inFlight),
    fetchConcurrency: Math.min(seen.fetchConcurrency, cloud.fetchConcurrency),
  };
};

// ---------------------------------------------------------------------------
// Scenario 1 — rapid orbit and zoom, held for long enough to expose a leak
// ---------------------------------------------------------------------------

const ORBIT_SECONDS = 30;
const ORBIT_DEGREES_PER_FRAME = 2;
const ZOOM_STEP = 1.05;
/** Frames spent zooming one way before the direction flips; ~2.2x each way. */
const ZOOM_RUN_FRAMES = 16;
/**
 * Selections a 30-second drive must have produced for the run to have tested
 * duration at all. Measured on this fixture it is 233; sixty is two a second,
 * far under what a healthy page manages and far over anything a stalled or
 * latched selector could reach.
 */
const MIN_SELECTIONS = 60;

describe("a camera orbiting and zooming without pause", () => {
  afterAll(async () => {
    await closeBrowser();
  });

  for (const cloud of cloudsUnderTest()) {
    it(`holds every live bound across ${ORBIT_SECONDS}s of it, then converges: ${cloud.name}`, async () => {
      const session = await openCloud(cloud.urlPath);
      try {
        const start = await settleAndAssert(session, 120_000);
        const before = start.controller!.selection.generation;

        const { result: frames, samples } = await watching(
          session,
          SAMPLE_MS * 2,
          () =>
            driveFrames(session, ORBIT_SECONDS, async (index) => {
              await session.azimuth(ORBIT_DEGREES_PER_FRAME);
              // A zoom that only ever approached would leave a working set
              // that had only grown. Flipping the direction makes it shrink
              // again, which is the half that releases tiles and evicts.
              const approaching =
                index % (2 * ZOOM_RUN_FRAMES) < ZOOM_RUN_FRAMES;
              await session.dolly(approaching ? ZOOM_STEP : 1 / ZOOM_STEP);
            }),
        );

        expect(frames, "the drive never completed a frame").toBeGreaterThan(0);
        expect(
          samples,
          "the live invariants were not sampled during the drive",
        ).toBeGreaterThan(ORBIT_SECONDS);

        const settled = await settleAndAssert(session, 120_000);
        // Surviving 30 seconds proves nothing on its own: a page that stopped
        // reselecting would also survive it. The generations are the evidence
        // that the bounds above were re-established a few hundred times.
        expect(
          settled.controller!.selection.generation - before,
          "the drive produced too few selections to have tested anything",
        ).toBeGreaterThanOrEqual(MIN_SELECTIONS);
        expect(session.failures).toEqual([]);
      } finally {
        await session.close();
      }
    });
  }
});

// ---------------------------------------------------------------------------
// Scenario 2 — reversal, so selections supersede each other mid-read
// ---------------------------------------------------------------------------

const REVERSAL_SECONDS = 8;
/**
 * Large enough that the two poles do not want the same tiles: a reversal that
 * changed nothing on screen would cancel nothing either, and this scenario is
 * only interesting while reads are being abandoned.
 */
const REVERSAL_DEGREES = 30;

describe("an orbit reversing as fast as the page will take it", () => {
  afterAll(async () => {
    await closeBrowser();
  });

  for (const cloud of cloudsUnderTest()) {
    it(`never exceeds its read concurrency, and drains every slot afterwards: ${cloud.name}`, async () => {
      const session = await openCloud(cloud.urlPath);
      try {
        await settleAndAssert(session, 120_000);

        let seen = nothingSeen;
        // watching() re-checks both bounds on its own cadence — physical reads
        // against fetchConcurrency, and wanted reads against physical ones —
        // so the burst is covered at every instant, not only where it is
        // sampled here. These samples record how close it came.
        const { samples } = await watching(session, SAMPLE_MS, () =>
          driveFrames(session, REVERSAL_SECONDS, async (index) => {
            await session.azimuth(
              index % 2 === 0 ? REVERSAL_DEGREES : -REVERSAL_DEGREES,
            );
            seen = extend(seen, await session.stats());
          }),
        );

        expect(
          samples,
          "the live invariants were not sampled during the burst",
        ).toBeGreaterThan(0);
        // Cancellation is advisory: an abandoned read keeps running to
        // completion, so the physical count is the one a slot leak shows up in.
        // `inFlight` is what is still wanted and may sit far below it here —
        // that is the burst working, not a fault.
        expect(
          seen.physicalTiles,
          `reads outran the fetch limit during the burst: ${JSON.stringify(seen)}`,
        ).toBeLessThanOrEqual(seen.fetchConcurrency);
        expect(
          seen.inFlight,
          `more reads were wanted than were running: ${JSON.stringify(seen)}`,
        ).toBeLessThanOrEqual(seen.physicalTiles);

        // A cloud too small to stream is fully resident before the burst
        // starts and has no reads to cancel, so only a deep one can show the
        // bounds were load-bearing rather than vacuous.
        if (cloud.deep) {
          expect(
            seen.physicalTiles,
            "the reversal never put a read in flight, so it cancelled nothing",
          ).toBeGreaterThan(0);
        }

        // The drain is what this scenario hunts: a leaked slot is permanent,
        // so convergence — which requires every physical operation back at
        // zero — is exactly the assertion that catches it.
        await settleAndAssert(session, 120_000);
        expect(session.failures).toEqual([]);
      } finally {
        await session.close();
      }
    });
  }
});

// ---------------------------------------------------------------------------
// Scenario 10 — video-driven motion, then a pause
// ---------------------------------------------------------------------------

const MOTION_SECONDS = 6;
/** A video-driven camera never jumps; it never stops either. */
const MOTION_DEGREES_PER_FRAME = 0.15;
/**
 * The regime is not asserted over the first moments of the drive: the first
 * camera a view supplies is a baseline rather than a movement, so nothing can
 * be inferred until a second frame has painted against it.
 */
const REGIME_WARMUP_MS = 500;
/** Motion debounce plus the settle window, with room for a slow paint. */
const RELEASE_TIMEOUT_MS = 15_000;

describe("a camera fed small deltas at video cadence", () => {
  afterAll(async () => {
    await closeBrowser();
  });

  for (const cloud of cloudsUnderTest()) {
    it(`reports interaction throughout, then lets go once the deltas stop: ${cloud.name}`, async () => {
      const session = await openCloud(cloud.urlPath);
      try {
        const start = await settleAndAssert(session, 120_000);
        expect(
          start.governor,
          "no view governor, so there is no regime to assert about",
        ).not.toBeNull();
        expect(start.governor!.regime).toBe("stationary");

        const flowing = { asserted: 0, attributed: 0 };
        const began = Date.now();
        const { result: frames } = await watching(session, SAMPLE_MS, () =>
          driveFrames(session, MOTION_SECONDS, async () => {
            await session.azimuth(MOTION_DEGREES_PER_FRAME);
            const view = (await session.stats()).governor;
            if (Date.now() - began < REGIME_WARMUP_MS) return;
            flowing.asserted += 1;
            // Deltas are still arriving, so the governor has no grounds to
            // treat the view as settled — this is the sample a latch-free
            // regime must never miss.
            expect(
              view?.regime,
              `the governor left the interaction regime while deltas were still arriving: ${JSON.stringify(view)}`,
            ).toBe("interaction");
            if (view?.motion.source !== null) flowing.attributed += 1;
          }),
        );

        expect(frames, "the drive never completed a frame").toBeGreaterThan(
          MOTION_SECONDS * 5,
        );
        expect(
          flowing.asserted,
          "the regime was never asserted while deltas flowed",
        ).toBeGreaterThan(0);
        // The regime alone would also be held by the settle debounce winding
        // down. A named source is what says the camera itself is holding it.
        expect(
          flowing.attributed,
          "no motion source ever held the regime, so nothing attributed the camera's movement",
        ).toBeGreaterThan(0);

        // Nothing drives the camera from here. The regime must come back on
        // its own, and hand its reference back with it.
        await session.until(
          "the governor to release the motion it inferred",
          (value) =>
            value.governor !== null &&
            value.governor.regime === "stationary" &&
            value.governor.motion.source === null &&
            !value.governor.motion.settling,
          RELEASE_TIMEOUT_MS,
        );

        const settled = await settleAndAssert(session, 120_000);
        // Released and still released: a regime that re-latched off its own
        // refinement frames would look identical at the instant above.
        expect(settled.governor!.regime).toBe("stationary");
        expect(settled.governor!.motion.source).toBeNull();
        expect(settled.governor!.motion.settling).toBe(false);
        expect(session.failures).toEqual([]);
      } finally {
        await session.close();
      }
    });
  }
});
