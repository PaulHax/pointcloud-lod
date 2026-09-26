import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createLoader } from "./loader";
import type { RetryPolicy } from "./retryPolicy";

const POLICY: RetryPolicy = {
  attempts: 3,
  retryDelayMs: () => 1_000,
  restMs: (round) => 30_000 * 2 ** round,
};

type Load = {
  readonly key: string;
  readonly signal: AbortSignal;
  resolve(value: string): void;
  reject(error: Error): void;
};

const abortError = (): Error => {
  const error = new Error("aborted");
  error.name = "AbortError";
  return error;
};

const makeLoader = (
  options: {
    readonly concurrency?: number;
    readonly cancelGraceMs?: number;
    /** Reject as soon as the signal fires, like a real fetch. */
    readonly honoursAbort?: boolean;
  } = {},
) => {
  const loads: Load[] = [];
  const loaded: string[] = [];
  const failed: string[] = [];
  const loader = createLoader<string>({
    load: (key, signal) =>
      new Promise<string>((resolve, reject) => {
        loads.push({ key, signal, resolve, reject });
        if (options.honoursAbort) {
          signal.addEventListener("abort", () => reject(abortError()));
        }
      }),
    concurrency: options.concurrency ?? 2,
    retry: POLICY,
    cancelGraceMs: options.cancelGraceMs ?? 0,
    onLoaded: (key, value) => loaded.push(`${key}=${value}`),
    onFailed: (key) => failed.push(key),
    onChange: () => {},
  });
  const calls = (): string[] => loads.map((load) => load.key);
  const last = (key: string): Load =>
    loads.filter((load) => load.key === key).at(-1)!;
  return { loader, loads, loaded, failed, calls, last };
};

const settle = async (): Promise<void> => {
  await Promise.resolve();
  await Promise.resolve();
};

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(0);
});
afterEach(() => {
  vi.useRealTimers();
});

describe("createLoader", () => {
  it("reads wanted keys in the order given, never more at once than its concurrency", async () => {
    const { loader, calls, last, loaded } = makeLoader();
    loader.want(["a", "b", "c", "d"]);
    expect(calls()).toEqual(["a", "b"]);
    expect(loader.counts()).toMatchObject({ queued: 2, physical: 2 });

    last("b").resolve("B");
    await settle();
    expect(calls()).toEqual(["a", "b", "c"]);
    expect(loaded).toEqual(["b=B"]);
  });

  it("rebuilds the queue from every want and never reads a key twice at once", async () => {
    const { loader, calls, last } = makeLoader({ concurrency: 1 });
    loader.want(["a", "b", "c"]);
    loader.want(["c", "a", "b"]);
    loader.want(["c", "b"]);
    // "a" is still being read; "c" now leads the queue.
    expect(calls()).toEqual(["a"]);
    expect(loader.counts()).toMatchObject({
      queued: 2,
      reading: 1,
      wantedReading: 0,
    });
    last("a").resolve("A");
    await settle();
    expect(calls()).toEqual(["a", "c"]);
  });

  it("adopts a read wanted again within its grace, and aborts it after", async () => {
    const { loader, last, calls } = makeLoader({ cancelGraceMs: 150 });
    loader.want(["a"]);
    loader.want([]);
    await vi.advanceTimersByTimeAsync(100);
    expect(last("a").signal.aborted).toBe(false);
    loader.want(["a"]);
    await vi.advanceTimersByTimeAsync(1_000);
    expect(last("a").signal.aborted).toBe(false);
    expect(calls()).toEqual(["a"]);

    loader.want([]);
    await vi.advanceTimersByTimeAsync(149);
    expect(last("a").signal.aborted).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    expect(last("a").signal.aborted).toBe(true);
  });

  it("aborts at once with no grace, and never with an infinite one", () => {
    const immediate = makeLoader({ cancelGraceMs: 0 });
    immediate.loader.want(["a"]);
    immediate.loader.want([]);
    expect(immediate.last("a").signal.aborted).toBe(true);

    const never = makeLoader({ cancelGraceMs: Number.POSITIVE_INFINITY });
    never.loader.want(["a"]);
    never.loader.want([]);
    expect(vi.getTimerCount()).toBe(0);
    expect(never.last("a").signal.aborted).toBe(false);
  });

  it("holds a slot until an abandoned read settles, and lets a want adopt it", async () => {
    const { loader, calls, last, loaded } = makeLoader({ concurrency: 1 });
    loader.want(["a"]);
    loader.abandon();
    expect(last("a").signal.aborted).toBe(true);
    loader.want(["b"]);
    expect(calls()).toEqual(["a"]);
    expect(loader.counts()).toMatchObject({ physical: 1, queued: 1 });

    // Cancellation is advisory: the read runs on and can still be wanted.
    loader.want(["a", "b"]);
    last("a").resolve("A");
    await settle();
    expect(loaded).toEqual(["a=A"]);
    expect(calls()).toEqual(["a", "b"]);
  });

  it("reads a wanted key again when its read really stopped", async () => {
    const { loader, calls } = makeLoader({ honoursAbort: true });
    loader.want(["a"]);
    loader.abandon();
    loader.want(["a"]);
    await settle();
    expect(calls()).toEqual(["a", "a"]);

    loader.abandon();
    await settle();
    expect(calls()).toEqual(["a", "a"]);
  });

  it("retries a failure soon, then rests the key once its attempts are spent", async () => {
    const { loader, calls, last, failed } = makeLoader();
    loader.want(["a"]);
    for (let attempt = 1; attempt <= 3; attempt += 1) {
      last("a").reject(new Error("500"));
      await settle();
      expect(failed).toHaveLength(attempt);
      if (attempt < 3) {
        expect(loader.counts().retrying).toBe(1);
        await vi.advanceTimersByTimeAsync(1_000);
        expect(calls()).toHaveLength(attempt + 1);
      }
    }
    // Resting is not outstanding work: nothing will fetch it until it ends.
    expect(loader.resting("a")).toBe(true);
    expect(loader.restingCount()).toBe(1);
    expect(loader.counts()).toMatchObject({ retrying: 0, queued: 0 });
    loader.want(["a"]);
    expect(calls()).toHaveLength(3);

    await vi.advanceTimersByTimeAsync(29_999);
    expect(calls()).toHaveLength(3);
    await vi.advanceTimersByTimeAsync(1);
    expect(calls()).toHaveLength(4);
  });

  it("fetches a key whose last attempt failed while unwanted once its rest ends", async () => {
    const { loader, calls, last } = makeLoader();
    loader.want(["a"]);
    for (let attempt = 1; attempt < 3; attempt += 1) {
      last("a").reject(new Error("500"));
      await settle();
      await vi.advanceTimersByTimeAsync(1_000);
    }
    loader.want([]);
    last("a").reject(new Error("500"));
    await settle();
    // Wanted again inside its rest: no attempt spent early, none forgotten.
    await vi.advanceTimersByTimeAsync(200);
    loader.want(["a"]);
    expect(calls()).toHaveLength(3);
    await vi.advanceTimersByTimeAsync(60_000);
    expect(calls()).toHaveLength(4);
  });

  it("drops a retry for a key that is no longer wanted", async () => {
    const { loader, calls, last } = makeLoader();
    loader.want(["a"]);
    last("a").reject(new Error("500"));
    await settle();
    loader.want([]);
    expect(loader.counts().retrying).toBe(0);
    await vi.advanceTimersByTimeAsync(60_000);
    expect(calls()).toEqual(["a"]);
  });

  it("forgets a key it delivered", async () => {
    const { loader, calls, last, loaded } = makeLoader();
    loader.want(["a"]);
    last("a").resolve("A");
    await settle();
    expect(loaded).toEqual(["a=A"]);
    expect(loader.counts()).toEqual({
      queued: 0,
      physical: 0,
      retrying: 0,
      reading: 0,
      wantedReading: 0,
    });
    // The owner holds the payload; asking again is a new read.
    loader.want(["a"]);
    expect(calls()).toEqual(["a", "a"]);
  });

  it("ignores results from before a reset but keeps their slots until they settle", async () => {
    const { loader, calls, loads, loaded, failed } = makeLoader({
      concurrency: 1,
    });
    loader.want(["a"]);
    loads[0]!.reject(new Error("500"));
    await settle();
    await vi.advanceTimersByTimeAsync(1_000);
    expect(calls()).toEqual(["a", "a"]);

    loader.reset();
    expect(loads[1]!.signal.aborted).toBe(true);
    expect(vi.getTimerCount()).toBe(0);
    loader.want(["a"]);
    expect(calls()).toEqual(["a", "a"]);

    loads[1]!.resolve("stale");
    await settle();
    expect(loaded).toEqual([]);
    expect(calls()).toEqual(["a", "a", "a"]);
    // A reset forgets failures too: this is a first attempt again.
    loads[2]!.reject(new Error("500"));
    await settle();
    expect(failed).toEqual(["a", "a"]);
    expect(loader.resting("a")).toBe(false);
  });
});
