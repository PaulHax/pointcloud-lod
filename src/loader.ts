/**
 * A bounded request queue with advisory cancellation and retry.
 *
 * The owner states, with every `want`, the complete set of keys it lacks and
 * wants, in priority order. The loader keeps at most one read per key, bounds
 * the reads physically running, lets a later `want` adopt a read it was
 * about to cancel, and retries failures on a timer, so a failed key comes
 * back without anything else asking for it. It forgets a key once it has
 * delivered it: payloads belong to the owner.
 *
 * Invariant: every wanted key is queued, being read, or has a retry armed,
 * either soon or for the end of its rest. So `counts` tells whether wanted
 * work is outstanding from the reads and armed retries alone, without
 * walking the wanted set; a resting key is deliberately not outstanding,
 * because nothing is going to fetch it now.
 */

import {
  isResting,
  recordFailure,
  retryDelayMs,
  type FailureRecord,
  type RetryPolicy,
} from "./retryPolicy";

export type LoaderCounts = {
  /** Keys waiting for a slot. */
  readonly queued: number;
  /**
   * Reads physically running, abandoned ones included: what the concurrency
   * bounds. Cancellation is advisory, so a read counts until it settles.
   */
  readonly physical: number;
  /** Wanted keys whose next attempt waits out a short retry delay. */
  readonly retrying: number;
  /** Reads started since the last reset that have not settled. */
  readonly reading: number;
  /** Of those, the reads whose result is still wanted. */
  readonly wantedReading: number;
};

export type Loader = {
  /**
   * Replace the wanted set. A read of a key it leaves out is cancelled once
   * the grace period passes; a read of a key it names is adopted. A resting
   * key waits for its rest to end; every other key is queued, in the order
   * given.
   */
  want(keys: readonly string[]): void;
  /**
   * Want nothing, and abort every read now. A read keeps its slot until it
   * settles, so a later `want` can still adopt it.
   */
  abandon(): void;
  /**
   * Start over: abort every read and forget failures and armed retries. A
   * read still running keeps its slot until it settles, and its result is
   * ignored.
   */
  reset(): void;
  /** Whether the key is inside a rest. */
  resting(key: string): boolean;
  /** Keys inside a rest; walks every failure record. */
  restingCount(): number;
  counts(): LoaderCounts;
};

type Read = {
  readonly abort: AbortController;
  wanted: boolean;
  graceTimer: ReturnType<typeof setTimeout> | null;
};

type Retry = {
  readonly timer: ReturnType<typeof setTimeout>;
  /** Armed for the end of a rest rather than for a prompt retry. */
  readonly rest: boolean;
};

const isAbortError = (error: unknown): boolean =>
  error instanceof Error && error.name === "AbortError";

export const createLoader = <T>(options: {
  readonly load: (key: string, signal: AbortSignal) => Promise<T>;
  readonly concurrency: number;
  readonly retry: RetryPolicy;
  /**
   * How long an unwanted read keeps running before it is aborted: 0 aborts
   * at once, Infinity never.
   */
  readonly cancelGraceMs: number;
  /** A read delivered; the owner decides where the payload goes. */
  readonly onLoaded: (key: string, value: T) => void;
  /** A read failed with anything but an abort. */
  readonly onFailed: (key: string, error: unknown) => void;
  /** A read started or settled. */
  readonly onChange: () => void;
}): Loader => {
  const { concurrency, retry, cancelGraceMs } = options;
  const reads = new Map<string, Read>();
  const failures = new Map<string, FailureRecord>();
  const retries = new Map<string, Retry>();
  let wanted = new Set<string>();
  let queue: string[] = [];
  let physical = 0;

  const clearGrace = (read: Read): void => {
    if (read.graceTimer === null) return;
    clearTimeout(read.graceTimer);
    read.graceTimer = null;
  };

  const cancel = (read: Read, immediately: boolean): void => {
    read.wanted = false;
    if (immediately || cancelGraceMs === 0) {
      clearGrace(read);
      read.abort.abort();
      return;
    }
    if (read.graceTimer !== null || cancelGraceMs === Infinity) return;
    read.graceTimer = setTimeout(() => {
      read.graceTimer = null;
      if (!read.wanted) read.abort.abort();
    }, cancelGraceMs);
  };

  const adopt = (read: Read): void => {
    read.wanted = true;
    clearGrace(read);
  };

  const resting = (key: string): boolean =>
    isResting(failures.get(key), retry, Date.now());

  const clearRetry = (key: string): void => {
    const armed = retries.get(key);
    if (armed === undefined) return;
    clearTimeout(armed.timer);
    retries.delete(key);
  };

  /** At most one armed retry per key, always at the earliest legal moment. */
  const armRetry = (key: string): void => {
    clearRetry(key);
    const record = failures.get(key);
    const rest = record !== undefined && record.count >= retry.attempts;
    const timer = setTimeout(
      () => {
        retries.delete(key);
        retryDue(key);
      },
      retryDelayMs(record, retry, Date.now()),
    );
    retries.set(key, { timer, rest });
  };

  const retryDue = (key: string): void => {
    // Unwanted, or already being read: nothing is missing.
    if (!wanted.has(key) || reads.has(key)) return;
    // A key inside its rest waits rather than spending an attempt early.
    if (resting(key)) {
      armRetry(key);
      return;
    }
    // At the head: a wanted key is a hole in the current view, not new detail.
    queue.unshift(key);
    pump();
  };

  const start = (key: string): void => {
    const read: Read = {
      abort: new AbortController(),
      wanted: true,
      graceTimer: null,
    };
    reads.set(key, read);
    physical += 1;
    options.onChange();
    options.load(key, read.abort.signal).then(
      (value) => {
        physical -= 1;
        options.onChange();
        // A reset forgot this read; its result belongs to nobody now.
        if (reads.get(key) !== read) {
          pump();
          return;
        }
        clearGrace(read);
        reads.delete(key);
        failures.delete(key);
        wanted.delete(key);
        options.onLoaded(key, value);
        pump();
      },
      (error) => {
        physical -= 1;
        options.onChange();
        if (reads.get(key) !== read) {
          pump();
          return;
        }
        clearGrace(read);
        reads.delete(key);
        if (!isAbortError(error)) {
          failures.set(
            key,
            recordFailure(failures.get(key), retry, Date.now()),
          );
          options.onFailed(key, error);
          armRetry(key);
        } else if (wanted.has(key)) {
          // A load that honours its signal really stopped, and the key was
          // wanted again while it was cancelled: it needs a fresh read.
          queue.unshift(key);
        }
        pump();
      },
    );
  };

  const pump = (): void => {
    while (physical < concurrency && queue.length > 0) {
      const key = queue.shift()!;
      if (!wanted.has(key) || reads.has(key)) continue;
      start(key);
    }
  };

  return {
    want(keys) {
      wanted = new Set(keys);
      for (const [key, read] of reads) {
        if (wanted.has(key)) adopt(read);
        else cancel(read, false);
      }
      const next: string[] = [];
      for (const key of keys) {
        if (reads.has(key)) continue;
        if (resting(key)) {
          // Its rest ends on the clock, not on a want: a key that spent its
          // last attempt while unwanted may have no retry armed.
          if (!retries.has(key)) armRetry(key);
          continue;
        }
        next.push(key);
      }
      queue = next;
      pump();
    },

    abandon() {
      wanted = new Set();
      queue = [];
      for (const read of reads.values()) cancel(read, true);
    },

    reset() {
      for (const read of reads.values()) cancel(read, true);
      reads.clear();
      for (const armed of retries.values()) clearTimeout(armed.timer);
      retries.clear();
      failures.clear();
      wanted = new Set();
      queue = [];
    },

    resting,

    restingCount() {
      let count = 0;
      for (const key of failures.keys()) if (resting(key)) count += 1;
      return count;
    },

    counts() {
      let wantedReading = 0;
      for (const read of reads.values()) if (read.wanted) wantedReading += 1;
      let retrying = 0;
      for (const [key, armed] of retries) {
        if (!armed.rest && wanted.has(key)) retrying += 1;
      }
      return {
        queued: queue.length,
        physical,
        retrying,
        reading: reads.size,
        wantedReading,
      };
    },
  };
};
