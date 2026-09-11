import { describe, expect, it, vi } from "vitest";

import { createSubmissionScheduler } from "./submissionScheduler";

describe("createSubmissionScheduler", () => {
  it("paces admission by estimated bytes and schedules every remaining slice", () => {
    const scheduleRender = vi.fn();
    const admitted: number[] = [];
    const scheduler = createSubmissionScheduler({
      scheduleRender,
      maxBytesPerFrame: 10,
      maxTimeMsPerFrame: 100,
      now: () => 0,
    });
    for (const bytes of [6, 6, 4]) {
      scheduler.enqueue({ bytes, run: () => admitted.push(bytes) });
    }
    expect(scheduleRender).toHaveBeenCalledTimes(1);
    expect(scheduler.stats()).toMatchObject({
      maxBytesPerFrame: 10,
      maxTimeMsPerFrame: 100,
    });
    scheduler.prepareFrame();
    expect(admitted).toEqual([6]);
    expect(scheduler.stats()).toMatchObject({ queuedJobs: 2, queuedBytes: 10 });
    expect(scheduleRender).toHaveBeenCalledTimes(2);
    scheduler.prepareFrame();
    expect(admitted).toEqual([6, 6, 4]);
    expect(scheduler.hasPending()).toBe(false);
  });

  it("rejects an oversized job with an actionable split requirement", () => {
    const run = vi.fn();
    const scheduler = createSubmissionScheduler({
      scheduleRender: vi.fn(),
      maxBytesPerFrame: 10,
      now: () => 0,
    });
    expect(() => scheduler.enqueue({ bytes: 100, run })).toThrow(
      /exceeding.*split it into smaller jobs/,
    );
    expect(run).not.toHaveBeenCalled();
  });

  it("isolates oversized atomic resources between ordinary slices", () => {
    const admitted: number[] = [];
    const scheduler = createSubmissionScheduler({
      scheduleRender: vi.fn(),
      maxBytesPerFrame: 10,
      now: () => 0,
    });
    scheduler.enqueue({ bytes: 4, run: () => admitted.push(4) });
    scheduler.enqueue({
      bytes: 20,
      atomic: true,
      run: () => admitted.push(20),
    });
    scheduler.enqueue({ bytes: 0, run: () => admitted.push(0) });
    scheduler.enqueue({ bytes: 6, run: () => admitted.push(6) });
    scheduler.prepareFrame();
    expect(admitted).toEqual([4]);
    scheduler.prepareFrame();
    expect(admitted).toEqual([4, 20]);
    expect(scheduler.stats()).toMatchObject({
      lastFrameAdmittedJobs: 1,
      lastFrameAdmittedBytes: 20,
      queuedBytes: 6,
    });
    scheduler.prepareFrame();
    expect(admitted).toEqual([4, 20, 0, 6]);
    expect(scheduler.hasPending()).toBe(false);
  });

  it("uses elapsed wall time as a second per-frame boundary", () => {
    let now = 0;
    const admitted: number[] = [];
    const scheduler = createSubmissionScheduler({
      scheduleRender: vi.fn(),
      maxBytesPerFrame: 100,
      maxTimeMsPerFrame: 4,
      now: () => now,
    });
    scheduler.enqueue({
      bytes: 1,
      run: () => {
        admitted.push(1);
        now = 5;
      },
    });
    scheduler.enqueue({ bytes: 1, run: () => admitted.push(2) });
    scheduler.prepareFrame();
    expect(admitted).toEqual([1]);
    expect(scheduler.stats()).toMatchObject({
      lastFrameAdmittedJobs: 1,
      lastFrameAdmittedBytes: 1,
      lastFrameElapsedMs: 5,
    });
    scheduler.prepareFrame();
    expect(scheduler.stats()).toMatchObject({
      lastFrameAdmittedJobs: 1,
      lastFrameAdmittedBytes: 1,
      lastFrameElapsedMs: 0,
    });
    scheduler.prepareFrame();
    expect(scheduler.stats()).toMatchObject({
      lastFrameAdmittedJobs: 0,
      lastFrameAdmittedBytes: 0,
      lastFrameElapsedMs: 0,
    });
  });

  it("cancels queued work and drops all jobs on dispose", () => {
    const run = vi.fn();
    const scheduler = createSubmissionScheduler({ scheduleRender: vi.fn() });
    const submission = scheduler.enqueue({ bytes: 3, run });
    expect(submission.cancel()).toBe(true);
    expect(submission.cancel()).toBe(false);
    expect(scheduler.stats().queuedBytes).toBe(0);
    scheduler.enqueue({ bytes: 4, run });
    scheduler.dispose();
    scheduler.prepareFrame();
    expect(run).not.toHaveBeenCalled();
  });

  it("contains a throwing error observer and continues later jobs", () => {
    const later = vi.fn();
    const scheduler = createSubmissionScheduler({
      scheduleRender: vi.fn(),
      maxBytesPerFrame: 10,
      now: () => 0,
    });
    scheduler.enqueue({
      bytes: 1,
      run: () => {
        throw new Error("work failed");
      },
      onError: () => {
        throw new Error("observer failed");
      },
    });
    scheduler.enqueue({ bytes: 1, run: later });
    expect(() => scheduler.prepareFrame()).not.toThrow();
    expect(later).toHaveBeenCalledOnce();
  });
});
