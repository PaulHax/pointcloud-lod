import { describe, expect, it, vi } from "vitest";

import {
  createContentQueue,
  type ContentQueueClock,
  type TileContentRequest,
} from "./contentQueue";
import { TileUnsupportedExtensionError } from "./decode";

type Decoded = { readonly id: string; readonly bytes: number };

const request = (id: string): TileContentRequest => ({
  id,
  url: `/tiles/${id}.glb`,
});

const deferred = <T>() => {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};

const flush = async () => {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
};

const ok = (bytes = new Uint8Array([1])) => ({
  ok: true,
  status: 200,
  statusText: "OK",
  arrayBuffer: async () =>
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
});

const makeQueue = (
  overrides: Partial<Parameters<typeof createContentQueue<Decoded>>[0]> = {},
) => {
  const decoded: string[] = [];
  const queue = createContentQueue<Decoded>({
    revision: "r1",
    configGeneration: 1,
    maxConcurrency: 2,
    maxDecodedBytes: 100,
    fetch: vi.fn(async (url) => ok(new TextEncoder().encode(String(url)))),
    decode: async (_bytes, item) => ({ id: item.id, bytes: 10 }),
    decodedByteLength: (value) => value.bytes,
    onContent: (item) => decoded.push(item.id),
    ...overrides,
  });
  return { queue, decoded };
};

describe("content queue", () => {
  it("bounds fetch/decode concurrency and preserves selected order", async () => {
    const gates = [
      deferred<ReturnType<typeof ok>>(),
      deferred<ReturnType<typeof ok>>(),
      deferred<ReturnType<typeof ok>>(),
    ];
    let active = 0;
    let peak = 0;
    const fetcher = vi.fn(async () => {
      const gate = gates[fetcher.mock.calls.length - 1]!;
      active += 1;
      peak = Math.max(peak, active);
      return gate.promise.finally(() => {
        active -= 1;
      });
    });
    const { queue, decoded } = makeQueue({ fetch: fetcher, maxConcurrency: 2 });
    queue.setSelection([request("a"), request("b"), request("c")]);
    expect(fetcher).toHaveBeenCalledTimes(2);
    gates[1]!.resolve(ok());
    await vi.waitFor(() => expect(fetcher).toHaveBeenCalledTimes(3));
    gates[0]!.resolve(ok());
    gates[2]!.resolve(ok());
    await vi.waitFor(() => expect(queue.snapshot().workPending).toBe(false));
    expect(peak).toBe(2);
    expect(decoded).toEqual(["b", "a", "c"]);
    expect(queue.snapshot()).toMatchObject({
      workPending: false,
      active: 0,
      queued: 0,
      cacheMisses: 3,
      cacheHits: 0,
    });
    expect(queue.snapshot().entries.map((entry) => entry.id)).toEqual([
      "a",
      "b",
      "c",
    ]);
  });

  it("aborts active deselected work and never reports it to the consumer", async () => {
    let signal: AbortSignal | undefined;
    const gate = deferred<ReturnType<typeof ok>>();
    const fetcher = vi.fn((_url: string, init?: { signal?: AbortSignal }) => {
      signal = init?.signal;
      return gate.promise;
    });
    const onContent = vi.fn();
    const { queue } = makeQueue({ fetch: fetcher, onContent });
    queue.setSelection([request("a")]);
    queue.setSelection([]);
    expect(signal?.aborted).toBe(true);
    gate.resolve(ok());
    await flush();
    expect(onContent).not.toHaveBeenCalled();
    expect(queue.snapshot().workPending).toBe(false);
  });

  it("keeps canceled operations in the physical concurrency bound until they settle", async () => {
    const gates = Array.from({ length: 4 }, () =>
      deferred<ReturnType<typeof ok>>(),
    );
    const signals: AbortSignal[] = [];
    const urls: string[] = [];
    let physical = 0;
    let peak = 0;
    const fetcher = vi.fn(
      (url: string, init: { readonly signal: AbortSignal }) => {
        const gate = gates[fetcher.mock.calls.length - 1]!;
        urls.push(url);
        signals.push(init.signal);
        physical += 1;
        peak = Math.max(peak, physical);
        return gate.promise.finally(() => {
          physical -= 1;
        });
      },
    );
    const { queue } = makeQueue({ fetch: fetcher, maxConcurrency: 2 });

    queue.setSelection([request("a")]);
    queue.setSelection([]);
    expect(signals[0]?.aborted).toBe(true);

    // One physical slot remains occupied by canceled `a`. The spare slot may
    // start `b`, but the newly selected `a` must not replace its active entry.
    queue.setSelection([request("a"), request("b")]);
    expect(urls).toEqual(["/tiles/a.glb", "/tiles/b.glb"]);
    expect(queue.snapshot().active).toBe(2);

    // A generation change cancels both operations but cannot release their
    // physical slots until their uncancellable promises settle.
    queue.configure({ revision: "r2", configGeneration: 2 });
    expect(signals.every((signal) => signal.aborted)).toBe(true);
    expect(fetcher).toHaveBeenCalledTimes(2);

    gates[0]!.resolve(ok());
    await vi.waitFor(() => expect(fetcher).toHaveBeenCalledTimes(3));
    expect(urls).toEqual(["/tiles/a.glb", "/tiles/b.glb", "/tiles/a.glb"]);
    gates[1]!.resolve(ok());
    await vi.waitFor(() => expect(fetcher).toHaveBeenCalledTimes(4));
    expect(urls).toEqual([
      "/tiles/a.glb",
      "/tiles/b.glb",
      "/tiles/a.glb",
      "/tiles/b.glb",
    ]);
    gates[2]!.resolve(ok());
    gates[3]!.resolve(ok());
    await vi.waitFor(() => expect(queue.snapshot().workPending).toBe(false));
    expect(peak).toBe(2);
  });

  it("deduplicates repeated selection and conflicting IDs fail deterministically", async () => {
    const fetcher = vi.fn(async () => ok());
    const { queue } = makeQueue({ fetch: fetcher });
    queue.setSelection([request("a"), request("a")]);
    await flush();
    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(() =>
      queue.setSelection([request("a"), { id: "a", url: "/other.glb" }]),
    ).toThrow(/conflicting.*a/i);
  });

  it("prevalidates retained URL conflicts before canceling any work", () => {
    const signals = new Map<string, AbortSignal>();
    const fetcher = vi.fn(
      (url: string, init: { readonly signal: AbortSignal }) => {
        signals.set(url, init.signal);
        return new Promise<ReturnType<typeof ok>>(() => {});
      },
    );
    const { queue } = makeQueue({ fetch: fetcher, maxConcurrency: 2 });
    queue.setSelection([request("a"), request("b")]);
    const before = queue.snapshot();

    expect(() =>
      queue.setSelection([{ id: "b", url: "/different/b.glb" }]),
    ).toThrow(/conflicting.*b/i);
    expect(signals.get("/tiles/a.glb")?.aborted).toBe(false);
    expect(signals.get("/tiles/b.glb")?.aborted).toBe(false);
    expect(queue.snapshot()).toEqual(before);
  });

  it("retries through an injectable clock and counts backoff as pending work", async () => {
    const timers: Array<{
      callback: () => void;
      delay: number;
      cancelled: boolean;
    }> = [];
    const clock: ContentQueueClock = {
      now: () => 10,
      setTimeout(callback, delay) {
        const timer = { callback, delay, cancelled: false };
        timers.push(timer);
        return timer;
      },
      clearTimeout(timer) {
        (timer as (typeof timers)[number]).cancelled = true;
      },
    };
    const fetcher = vi
      .fn()
      .mockRejectedValueOnce(new Error("temporary"))
      .mockResolvedValueOnce(ok());
    const { queue } = makeQueue({
      fetch: fetcher,
      clock,
      maxAttempts: 2,
      retryBackoffMs: (failedAttempt) => failedAttempt * 25,
    });
    queue.setSelection([request("a")]);
    await flush();
    expect(timers).toMatchObject([{ delay: 25, cancelled: false }]);
    expect(queue.snapshot()).toMatchObject({ workPending: true, retrying: 1 });
    timers[0]!.callback();
    await flush();
    expect(fetcher).toHaveBeenCalledTimes(2);
    expect(queue.snapshot().workPending).toBe(false);
  });

  it("cancels retry timers on deselection", async () => {
    const clearTimeout = vi.fn();
    let retry!: () => void;
    const clock: ContentQueueClock = {
      now: () => 0,
      setTimeout(callback) {
        retry = callback;
        return 7;
      },
      clearTimeout,
    };
    const { queue } = makeQueue({
      fetch: vi.fn().mockRejectedValue(new Error("no")),
      clock,
    });
    queue.setSelection([request("a")]);
    await flush();
    queue.setSelection([]);
    expect(clearTimeout).toHaveBeenCalledWith(7);
    retry();
    await flush();
    expect(queue.snapshot().workPending).toBe(false);
  });

  it.each([
    [
      "throwing backoff",
      () => () => {
        throw new Error("backoff failed");
      },
      undefined,
    ],
    ["invalid backoff", () => () => Number.NaN, undefined],
    ["timer scheduling failure", () => () => 5, new Error("timer failed")],
  ])(
    "contains %s as a terminal typed retry error",
    async (_name, createBackoff, timerFailure) => {
      const onError = vi.fn();
      const clock: ContentQueueClock = {
        now: () => 0,
        setTimeout() {
          if (timerFailure) throw timerFailure;
          return 1;
        },
        clearTimeout: vi.fn(),
      };
      const { queue } = makeQueue({
        fetch: vi.fn().mockRejectedValue(new Error("temporary")),
        maxAttempts: 2,
        retryBackoffMs: createBackoff(),
        clock,
        onError,
      });
      queue.setSelection([request("a")]);
      await vi.waitFor(() => expect(onError).toHaveBeenCalledTimes(1));
      expect(onError.mock.calls[0]?.[1]).toMatchObject({
        name: "ContentQueueRetryError",
        id: "a",
      });
      expect(queue.snapshot()).toMatchObject({
        workPending: false,
        retrying: 0,
        failed: 1,
      });
    },
  );

  it("has terminal typed failures after the configured attempt count", async () => {
    const onError = vi.fn();
    const { queue } = makeQueue({
      fetch: vi.fn().mockResolvedValue({
        ...ok(),
        ok: false,
        status: 404,
        statusText: "Not Found",
      }),
      maxAttempts: 1,
      onError,
    });
    queue.setSelection([request("missing")]);
    await flush();
    expect(queue.snapshot()).toMatchObject({ workPending: false, failed: 1 });
    expect(onError.mock.calls[0]?.[0]).toMatchObject({ id: "missing" });
    expect(onError.mock.calls[0]?.[1]).toMatchObject({
      name: "ContentQueueFetchError",
      status: 404,
    });
  });

  it("retries a failed decode rather than failing it terminally", async () => {
    const timers: Array<() => void> = [];
    const clock: ContentQueueClock = {
      now: () => 0,
      setTimeout: (callback) => timers.push(callback),
      clearTimeout: () => {},
    };
    const onError = vi.fn();
    const decode = vi
      .fn()
      .mockRejectedValueOnce(new Error("malformed"))
      .mockResolvedValueOnce({ id: "a", bytes: 10 });
    const { queue } = makeQueue({ clock, decode, maxAttempts: 2, onError });

    queue.setSelection([request("a")]);
    await flush();
    expect(queue.snapshot()).toMatchObject({ retrying: 1, failed: 0 });

    timers[0]!();
    await flush();
    expect(onError).not.toHaveBeenCalled();
    expect(queue.snapshot()).toMatchObject({ ready: 1, cached: 1 });
  });

  it("classifies invalid decoded byte accounting as a typed decode failure", async () => {
    const onError = vi.fn();
    const { queue } = makeQueue({
      decodedByteLength: () => Number.NaN,
      maxAttempts: 1,
      onError,
    });
    queue.setSelection([request("invalid")]);
    await flush();
    expect(onError.mock.calls[0]?.[1]).toMatchObject({
      name: "ContentQueueDecodeError",
      id: "invalid",
    });
  });

  it("wraps typed tile decode failures without flattening their cause", async () => {
    const onError = vi.fn();
    const typed = new TileUnsupportedExtensionError(
      "/tiles/meshopt.glb",
      "EXT_meshopt_compression",
    );
    const { queue } = makeQueue({
      decode: async () => {
        throw typed;
      },
      maxAttempts: 1,
      onError,
    });
    queue.setSelection([request("meshopt")]);
    await flush();

    const error = onError.mock.calls[0]?.[1];
    expect(error).toMatchObject({ name: "ContentQueueDecodeError" });
    expect(error.cause).toBe(typed);
    expect(error.cause).toMatchObject({
      name: "TileUnsupportedExtensionError",
      tileUri: "/tiles/meshopt.glb",
      stage: "profile",
      extension: "EXT_meshopt_compression",
    });
  });

  it("drops obsolete uncancellable completions across revision and generation changes", async () => {
    const first = deferred<ReturnType<typeof ok>>();
    const second = deferred<ReturnType<typeof ok>>();
    const fetcher = vi
      .fn()
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    const onContent = vi.fn();
    const { queue } = makeQueue({ fetch: fetcher, onContent });
    queue.setSelection([request("a")]);
    queue.configure({ revision: "r2", configGeneration: 2 });
    queue.setSelection([request("a")]);
    first.resolve(ok());
    await flush();
    expect(onContent).not.toHaveBeenCalled();
    second.resolve(ok());
    await flush();
    expect(onContent).toHaveBeenCalledTimes(1);
    expect(queue.snapshot()).toMatchObject({
      revision: "r2",
      configGeneration: 2,
    });
  });

  it("accounts decoded bytes and evicts least-recently-used content deterministically", async () => {
    const evicted: string[] = [];
    const { queue } = makeQueue({
      maxDecodedBytes: 20,
      onEvict: (item) => evicted.push(item.id),
    });
    queue.setSelection([request("a"), request("b")]);
    await flush();
    expect(queue.snapshot()).toMatchObject({ decodedBytes: 20, cached: 2 });
    expect(queue.get("a")?.id).toBe("a");
    queue.setSelection([request("a"), request("b"), request("c")]);
    await flush();
    expect(queue.snapshot()).toMatchObject({ decodedBytes: 20, cached: 2 });
    expect(evicted).toEqual(["b"]);
    expect([...queue.contents().keys()]).toEqual(["a", "c"]);
    expect(queue.get("b")).toBeUndefined();
    expect(queue.get("a")?.id).toBe("a");
    expect(queue.get("c")?.id).toBe("c");
  });

  it("delivers oversize content without caching or evicting retained entries", async () => {
    const events: string[] = [];
    const { queue } = makeQueue({
      maxDecodedBytes: 10,
      decode: async (_bytes, item) => ({
        id: item.id,
        bytes: item.id === "oversize" ? 20 : 5,
      }),
      onContent: (item) => events.push(`content:${item.id}`),
      onEvict: (item) => events.push(`evict:${item.id}`),
    });
    queue.setSelection([request("small")]);
    await flush();
    queue.setSelection([request("small"), request("oversize")]);
    await flush();

    expect(events).toEqual(["content:small", "content:oversize"]);
    expect(queue.snapshot()).toMatchObject({ cached: 1, decodedBytes: 5 });
    expect(queue.get("small")?.id).toBe("small");
    expect(queue.get("oversize")).toBeUndefined();
  });

  it("reconfigures concurrency and cache bytes explicitly", async () => {
    const { queue } = makeQueue();
    queue.setSelection([request("a"), request("b")]);
    await flush();
    queue.configure({ maxConcurrency: 1, maxDecodedBytes: 10 });
    expect(queue.snapshot()).toMatchObject({
      maxConcurrency: 1,
      maxDecodedBytes: 10,
      decodedBytes: 10,
    });
  });

  it("contains consumer callback exceptions and exposes deterministic snapshots", async () => {
    const { queue } = makeQueue({
      onContent: () => {
        throw new Error("consumer failed");
      },
      onStateChange: () => {
        throw new Error("observer failed");
      },
    });
    queue.setSelection([request("a")]);
    await flush();
    expect(queue.snapshot().entries).toEqual([
      { id: "a", url: "/tiles/a.glb", status: "ready", attempt: 1 },
    ]);
  });

  it("disposes without leaving pending work", () => {
    const { queue } = makeQueue();
    queue.setSelection([request("a")]);
    queue.dispose();
    expect(queue.snapshot()).toMatchObject({
      workPending: false,
      disposed: true,
      selected: 0,
    });
    expect(() => queue.setSelection([request("b")])).toThrow(/disposed/i);
  });
});
