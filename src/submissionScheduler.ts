import { finiteAtLeast, wholeAtLeast } from "./numeric";

const MiB = 1024 * 1024;

/** Default main-thread/GPU admission limits applied before each paint. */
export const DEFAULT_SUBMISSION_BYTES_PER_FRAME = 16 * MiB;
export const DEFAULT_SUBMISSION_TIME_MS_PER_FRAME = 4;

export type SubmissionJob = {
  /** Estimated retained geometry/texture bytes admitted by this job. */
  readonly bytes: number;
  /** Synchronous VTK construction or GPU-admission work. */
  readonly run: () => void;
  readonly onError?: (error: unknown) => void;
};

export type Submission = {
  /** Cancel while queued. Returns true only for the first successful cancel. */
  cancel(): boolean;
};

export type SubmissionSchedulerOptions = {
  readonly scheduleRender: () => void;
  readonly maxBytesPerFrame?: number;
  readonly maxTimeMsPerFrame?: number;
  /** Injectable monotonic clock for deterministic tests. */
  readonly now?: () => number;
};

export type SubmissionSchedulerStats = {
  readonly maxBytesPerFrame: number;
  readonly maxTimeMsPerFrame: number;
  readonly queuedJobs: number;
  readonly queuedBytes: number;
  readonly admittedJobs: number;
  readonly admittedBytes: number;
  readonly lastFrameAdmittedJobs: number;
  readonly lastFrameAdmittedBytes: number;
  readonly lastFrameElapsedMs: number;
  readonly peakQueuedJobs: number;
  readonly peakQueuedBytes: number;
  readonly peakFrameAdmittedBytes: number;
  readonly peakFrameElapsedMs: number;
  readonly admissionFrames: number;
  readonly disposed: boolean;
};

export type SubmissionScheduler = {
  enqueue(job: SubmissionJob): Submission;
  /** Drain one bounded admission slice immediately before a view paint. */
  prepareFrame(): void;
  hasPending(): boolean;
  stats(): SubmissionSchedulerStats;
  dispose(): void;
};

type QueuedJob = SubmissionJob & { cancelled: boolean };

export const createSubmissionScheduler = (
  options: SubmissionSchedulerOptions,
): SubmissionScheduler => {
  const maxBytes = wholeAtLeast(
    "maxBytesPerFrame",
    options.maxBytesPerFrame ?? DEFAULT_SUBMISSION_BYTES_PER_FRAME,
    1,
  );
  const maxTimeMs = finiteAtLeast(
    "maxTimeMsPerFrame",
    options.maxTimeMsPerFrame ?? DEFAULT_SUBMISSION_TIME_MS_PER_FRAME,
    0,
  );
  const now = options.now ?? (() => performance.now());
  const queue: QueuedJob[] = [];
  let queuedBytes = 0;
  let admittedJobs = 0;
  let admittedBytes = 0;
  let lastFrameAdmittedJobs = 0;
  let lastFrameAdmittedBytes = 0;
  let lastFrameElapsedMs = 0;
  let peakQueuedJobs = 0;
  let peakQueuedBytes = 0;
  let peakFrameAdmittedBytes = 0;
  let peakFrameElapsedMs = 0;
  let admissionFrames = 0;
  let disposed = false;

  const discardCancelledHead = (): void => {
    while (queue[0]?.cancelled) queue.shift();
  };

  return {
    enqueue(job) {
      const bytes = wholeAtLeast("submission bytes", job.bytes, 0);
      if (bytes > maxBytes) {
        throw new Error(
          `submission job is ${bytes} bytes, exceeding the ${maxBytes}-byte per-frame cap; split it into smaller jobs`,
        );
      }
      const queued: QueuedJob = { ...job, bytes, cancelled: false };
      if (!disposed) {
        const wasEmpty = queue.length === 0;
        queue.push(queued);
        queuedBytes += bytes;
        peakQueuedJobs = Math.max(peakQueuedJobs, queue.length);
        peakQueuedBytes = Math.max(peakQueuedBytes, queuedBytes);
        if (wasEmpty) options.scheduleRender();
      } else {
        queued.cancelled = true;
      }
      return {
        cancel() {
          if (queued.cancelled || !queue.includes(queued)) return false;
          queued.cancelled = true;
          queuedBytes -= queued.bytes;
          return true;
        },
      };
    },

    prepareFrame() {
      lastFrameAdmittedJobs = 0;
      lastFrameAdmittedBytes = 0;
      lastFrameElapsedMs = 0;
      if (disposed) return;
      const started = now();
      let frameBytes = 0;
      let frameJobs = 0;
      discardCancelledHead();
      while (queue.length > 0) {
        const next = queue[0]!;
        // Enqueue already proved each job fits one byte slice. Every later job
        // waits for the next frame once either cap has been consumed.
        if (
          frameJobs > 0 &&
          (frameBytes + next.bytes > maxBytes || now() - started >= maxTimeMs)
        ) {
          break;
        }
        queue.shift();
        queuedBytes -= next.bytes;
        if (next.cancelled) {
          discardCancelledHead();
          continue;
        }
        try {
          next.run();
        } catch (error) {
          try {
            next.onError?.(error);
          } catch {
            // One consumer's observer must not escape the frame boundary or
            // strand unrelated queued work behind it.
          }
        }
        frameJobs += 1;
        frameBytes += next.bytes;
        admittedJobs += 1;
        admittedBytes += next.bytes;
        discardCancelledHead();
      }
      lastFrameAdmittedJobs = frameJobs;
      lastFrameAdmittedBytes = frameBytes;
      lastFrameElapsedMs = Math.max(0, now() - started);
      if (frameJobs > 0) admissionFrames += 1;
      peakFrameAdmittedBytes = Math.max(peakFrameAdmittedBytes, frameBytes);
      peakFrameElapsedMs = Math.max(peakFrameElapsedMs, lastFrameElapsedMs);
      if (queue.length > 0) options.scheduleRender();
    },

    hasPending: () => !disposed && queue.some((job) => !job.cancelled),

    stats: () => ({
      maxBytesPerFrame: maxBytes,
      maxTimeMsPerFrame: maxTimeMs,
      queuedJobs: queue.reduce(
        (count, job) => count + (job.cancelled ? 0 : 1),
        0,
      ),
      queuedBytes,
      admittedJobs,
      admittedBytes,
      lastFrameAdmittedJobs,
      lastFrameAdmittedBytes,
      lastFrameElapsedMs,
      peakQueuedJobs,
      peakQueuedBytes,
      peakFrameAdmittedBytes,
      peakFrameElapsedMs,
      admissionFrames,
      disposed,
    }),

    dispose() {
      if (disposed) return;
      disposed = true;
      queue.length = 0;
      queuedBytes = 0;
    },
  };
};
