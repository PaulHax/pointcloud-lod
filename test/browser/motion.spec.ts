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
 *
 * The last group drives a real pointer instead. Every programmatic handle moves
 * the camera behind the interactor's back, so the page can only ever *infer*
 * that motion happened; a pointer gesture is the one thing that fires the
 * interactor's animation callbacks, and those are what take an explicit motion
 * reference and tell the controller an interaction has begun. Same seams,
 * different path through them — and it is the path a user takes.
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
/**
 * How far the reversal walks per pair of frames.
 *
 * Oscillating between exactly two orientations reverses direction but never
 * looks anywhere new: after the first pass both poles are decoded, and the
 * cache answers every reselection without issuing a read. Measured on a
 * 9.1 M-point cloud, that burst produced zero physical reads in eight seconds
 * — the bounds held, and held vacuously. Letting the axis drift keeps the
 * reversal while giving each one fresh tiles to abandon.
 */
const REVERSAL_DRIFT_DEGREES = 7;

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
              index % 2 === 0
                ? REVERSAL_DEGREES
                : -REVERSAL_DEGREES + REVERSAL_DRIFT_DEGREES,
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

        // Not a frame rate. A software rasteriser painting a few million
        // points manages single digits per second, which says nothing about
        // this library — the load-bearing assertion is the per-sample regime
        // check above, and this only establishes that the drive drove.
        expect(frames, "the drive never completed a frame").toBeGreaterThan(0);
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

// ---------------------------------------------------------------------------
// A real pointer gesture — the path a user's drag takes, and no other
// ---------------------------------------------------------------------------

/**
 * The page reports more about motion than the shared stats type names.
 *
 * Every other scenario drives the camera programmatically, so the governor
 * only ever *infers* motion and one nullable source name says everything there
 * is to say. A gesture takes an explicit reference through the interactor's
 * animation callbacks and the two kinds compose, so this scenario needs the
 * per-kind counts and the controller's interaction depth — read here rather
 * than added to a type every other file shares.
 */
interface MotionDetail {
  readonly source: "explicit" | "inferred" | "both" | null;
  readonly explicitReferences: number;
  readonly inferredReferences: number;
  readonly settling: boolean;
}

interface MotionDetailReading {
  readonly motion: MotionDetail | null;
  readonly interactionDepth: number | null;
  readonly regime: string | null;
}

const detailOf = (stats: ExampleStats): MotionDetailReading => {
  const page = stats as unknown as {
    controller: { interactionDepth: number } | null;
    governor: { regime: string; motion: MotionDetail } | null;
  };
  return {
    motion: page.governor?.motion ?? null,
    interactionDepth: page.controller?.interactionDepth ?? null,
    regime: page.governor?.regime ?? null,
  };
};

/** What a gesture was seen doing, sampled while it was still running. */
interface GestureTrace {
  readonly samples: number;
  /** Samples where the governor held an explicit motion reference. */
  readonly explicit: number;
  /** Samples where it held an explicit and an inferred reference at once. */
  readonly both: number;
  readonly peakExplicitReferences: number;
  readonly peakInteractionDepth: number;
  /**
   * The first sample where an explicit reference was held without everything
   * that must accompany it: the interaction regime, a source that names it,
   * and a controller that was told the interaction began. Kept as text so a
   * failure reports the sample rather than a boolean about it.
   */
  readonly unpaired: string | null;
}

const nothingTraced: GestureTrace = {
  samples: 0,
  explicit: 0,
  both: 0,
  peakExplicitReferences: 0,
  peakInteractionDepth: 0,
  unpaired: null,
};

const traceGesture = (seen: GestureTrace, stats: ExampleStats): GestureTrace => {
  const { motion, interactionDepth, regime } = detailOf(stats);
  const samples = seen.samples + 1;
  if (motion === null || motion.explicitReferences === 0) {
    return { ...seen, samples };
  }
  const paired =
    regime === "interaction" &&
    (motion.source === "explicit" || motion.source === "both") &&
    (interactionDepth ?? 0) >= 1;
  return {
    samples,
    explicit: seen.explicit + 1,
    both: seen.both + (motion.source === "both" ? 1 : 0),
    peakExplicitReferences: Math.max(
      seen.peakExplicitReferences,
      motion.explicitReferences,
    ),
    peakInteractionDepth: Math.max(
      seen.peakInteractionDepth,
      interactionDepth ?? 0,
    ),
    unpaired:
      seen.unpaired ??
      (paired ? null : JSON.stringify({ regime, interactionDepth, motion })),
  };
};

/**
 * Record what the governor and the controller did while `body` ran.
 *
 * Nested inside `watching()` rather than folded into it: the live invariants
 * are true of every scenario and this is true of this one, and neither can be
 * recovered afterwards — mouse-up puts the references and the depth back, so a
 * check that only looked before and after would see two identical states.
 */
const tracingWhile = async (
  session: ExampleSession,
  intervalMs: number,
  body: () => Promise<void>,
): Promise<GestureTrace> => {
  let running = true;
  let seen = nothingTraced;
  const sampler = (async () => {
    while (running) {
      const stats = await session.stats().catch(() => null);
      if (stats !== null) seen = traceGesture(seen, stats);
      await session.page.waitForTimeout(intervalMs);
    }
  })();
  try {
    await body();
  } finally {
    running = false;
    await sampler;
  }
  return seen;
};

/** Everything the gesture took must be back before this holds. */
const handedBack = (stats: ExampleStats): boolean => {
  const { motion, interactionDepth, regime } = detailOf(stats);
  return (
    motion !== null &&
    regime === "stationary" &&
    motion.source === null &&
    motion.explicitReferences === 0 &&
    !motion.settling &&
    interactionDepth === 0
  );
};

/**
 * One gesture: a drag across the middle of the canvas, paced so a sampler sees
 * the middle of it and not only its two ends.
 */
const GESTURE_STEPS = Array.from({ length: 12 }, () => ({ dx: 14, dy: 5 }));
const GESTURE_PAUSE_MS = 40;
/** Faster than the live sampler: the state being traced is the gesture itself. */
const GESTURE_SAMPLE_MS = 20;
/**
 * A press-and-hold: the same gesture with the pointer standing still, so the
 * interactor is animating while the camera is not moving. Paced slowly and
 * held for seconds, because both phases of the overlap check have to happen
 * with the button still down.
 */
const HELD_STEPS = Array.from({ length: 40 }, () => ({ dx: 0, dy: 0 }));
const HOLD_PAUSE_MS = 120;
/**
 * How long the hold is given to reach the explicit source on its own.
 *
 * Not instant: the first pointer move of a gesture perturbs the view even at
 * zero delta — vtk.js re-orthogonalises the view up and resets the clipping
 * range on every move — so the page infers a motion of its own at the start of
 * the gesture and hands it back a debounce later. Comfortably over that, and
 * well inside the hold.
 */
const HOLD_TIMEOUT_MS = 5_000;
/** A host moving the camera underneath the gesture, and its cadence. */
const CONCURRENT_DEGREES = 1.5;
const CONCURRENT_NUDGE_MS = 60;

describe("a real pointer gesture on the canvas", () => {
  afterAll(async () => {
    await closeBrowser();
  });

  for (const cloud of cloudsUnderTest()) {
    it(`holds an explicit motion reference and an interaction depth for its duration, then hands both back: ${cloud.name}`, async () => {
      const session = await openCloud(cloud.urlPath);
      try {
        const start = await settleAndAssert(session, 120_000);
        expect(
          start.governor,
          "no view governor, so there is no regime to assert about",
        ).not.toBeNull();
        expect(start.governor!.regime).toBe("stationary");
        expect(
          detailOf(start).interactionDepth,
          "the controller was already interacting before anything touched it",
        ).toBe(0);

        // The control the rest of this check rests on: a programmatic move is
        // motion the governor *infers*, and it must reach neither the explicit
        // reference count nor the controller's interaction depth. Without this
        // an explicit assertion below would pass on any motion at all.
        await session.azimuth(ORBIT_DEGREES_PER_FRAME);
        await session.frame();
        const inferredOnly = detailOf(await session.stats());
        expect(
          inferredOnly.motion?.explicitReferences,
          `a programmatic camera move took an explicit motion reference: ${JSON.stringify(inferredOnly)}`,
        ).toBe(0);
        expect(
          inferredOnly.interactionDepth,
          `a programmatic camera move raised the controller's interaction depth: ${JSON.stringify(inferredOnly)}`,
        ).toBe(0);
        await session.until(
          "the inferred motion to be released before the gesture",
          handedBack,
          RELEASE_TIMEOUT_MS,
        );

        const before = await session.readCamera();
        const { result: seen } = await watching(session, SAMPLE_MS, () =>
          tracingWhile(session, GESTURE_SAMPLE_MS, () =>
            session.drag(GESTURE_STEPS, GESTURE_PAUSE_MS),
          ),
        );
        const after = await session.readCamera();

        // The gesture is evidence of nothing unless the interactor received
        // it. A camera that did not move means the pointer events never
        // reached the interactor style, and every assertion below is vacuous.
        expect(
          after.position,
          `the drag never reached the interactor: the camera did not move — ${JSON.stringify(seen)}`,
        ).not.toEqual(before.position);
        expect(seen.samples, "the gesture was never sampled").toBeGreaterThan(0);

        // `onStartAnimation` is the only thing in the page that takes an
        // explicit reference, so a zero here is that callback never firing.
        expect(
          seen.explicit,
          `the gesture never held an explicit motion reference: ${JSON.stringify(seen)}`,
        ).toBeGreaterThan(0);
        expect(
          seen.peakExplicitReferences,
          `the explicit reference count never reached one: ${JSON.stringify(seen)}`,
        ).toBeGreaterThanOrEqual(1);
        // The other half of the same event: the controller is told a gesture
        // has begun, which is what lets it skip its camera debounce.
        expect(
          seen.peakInteractionDepth,
          `the gesture never raised the controller's interaction depth: ${JSON.stringify(seen)}`,
        ).toBeGreaterThanOrEqual(1);
        expect(
          seen.unpaired,
          "an explicit reference was held while the regime, the source or the controller disagreed",
        ).toBeNull();

        // Mouse-up must put all of it back on its own. An unpaired
        // beginInteraction latches the controller into interaction for the
        // rest of the session, and a leaked reference latches the regime with
        // it — both of which look exactly like this check timing out.
        await session.until(
          "the gesture's references and interaction depth to be handed back",
          handedBack,
          RELEASE_TIMEOUT_MS,
        );

        const settled = await settleAndAssert(session, 120_000);
        const rest = detailOf(settled);
        expect(settled.governor!.regime).toBe("stationary");
        expect(
          rest.motion?.explicitReferences,
          `an explicit motion reference outlived the gesture: ${JSON.stringify(rest)}`,
        ).toBe(0);
        expect(
          rest.interactionDepth,
          `the controller stayed interacting after the gesture ended: ${JSON.stringify(rest)}`,
        ).toBe(0);
        expect(settled.governor!.motion.source).toBeNull();
        expect(settled.governor!.motion.settling).toBe(false);
        expect(session.failures).toEqual([]);
      } finally {
        await session.close();
      }
    });

    it(`attributes both sources when a programmatic move lands during a gesture: ${cloud.name}`, async () => {
      const session = await openCloud(cloud.urlPath);
      try {
        await settleAndAssert(session, 120_000);
        // Nothing carried in: an inferred reference left over from framing the
        // cloud would make the second phase's "both" mean nothing.
        await session.until(
          "the view to be holding no motion before the gesture",
          handedBack,
          RELEASE_TIMEOUT_MS,
        );

        const { result: together } = await watching(
          session,
          SAMPLE_MS,
          async () => {
            // Press and hold: the pointer goes down and stays where it is, so
            // the gesture holds the regime while the camera does not move.
            const gesture = session.drag(HELD_STEPS, HOLD_PAUSE_MS);
            // Waiting for the source to be *only* explicit is what makes the
            // second phase mean something: with nothing inferred and the
            // camera standing still, an inferred reference appearing next can
            // only have come from the programmatic move. It is also the state
            // an inferred-only page could never reach, since a held pointer
            // gives it no camera change to notice.
            await session.until(
              "the held gesture to hold the regime on the explicit source alone",
              (value) => {
                const { motion, interactionDepth, regime } = detailOf(value);
                return (
                  regime === "interaction" &&
                  motion?.source === "explicit" &&
                  interactionDepth === 1
                );
              },
              HOLD_TIMEOUT_MS,
            );

            let nudging = true;
            const nudges = (async () => {
              while (nudging) {
                await session.azimuth(CONCURRENT_DEGREES);
                await session.page.waitForTimeout(CONCURRENT_NUDGE_MS);
              }
            })();
            return tracingWhile(session, GESTURE_SAMPLE_MS, async () => {
              try {
                await gesture;
              } finally {
                nudging = false;
                await nudges;
              }
            });
          },
        );

        expect(
          together.explicit,
          `the gesture stopped holding its explicit reference once the camera moved under it: ${JSON.stringify(together)}`,
        ).toBeGreaterThan(0);
        // "both" is the governor's own word for holding the two kinds at
        // once. A page that let one overwrite the other would still report a
        // source and still hold the regime — it would just name one of them,
        // which is exactly what this number distinguishes.
        expect(
          together.both,
          `the governor lost a source while a gesture and a programmatic move overlapped: ${JSON.stringify(together)}`,
        ).toBeGreaterThan(0);
        expect(
          together.unpaired,
          "an explicit reference was held while the regime, the source or the controller disagreed",
        ).toBeNull();

        // The two release independently and the last one ends the regime, so
        // an inferred reference outliving the gesture must not keep the
        // controller interacting either.
        await session.until(
          "both motion sources to be released after the overlap",
          handedBack,
          RELEASE_TIMEOUT_MS,
        );
        const settled = await settleAndAssert(session, 120_000);
        expect(settled.governor!.regime).toBe("stationary");
        expect(settled.governor!.motion.source).toBeNull();
        expect(detailOf(settled).interactionDepth).toBe(0);
        expect(session.failures).toEqual([]);
      } finally {
        await session.close();
      }
    });
  }
});
