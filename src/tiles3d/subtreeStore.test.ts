import { describe, expect, it, vi } from "vitest";

import {
  makeSubtreeFixture,
  SUBTREE_METADATA_SCHEMA,
} from "../../test/fixtures/subtreeFixture";
import {
  SubtreeParseError,
  createSubtreeStore,
  type SubtreeFetch,
} from "./subtreeStore";

const settle = async (): Promise<void> => {
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
};

const response = (bytes: ArrayBuffer, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  statusText: status === 200 ? "OK" : "Gone",
  arrayBuffer: vi.fn().mockResolvedValue(bytes),
});

describe("subtree store", () => {
  it("deduplicates selected requests and reuses parsed cache entries", async () => {
    const fetcher = vi
      .fn<SubtreeFetch>()
      .mockResolvedValue(response(makeSubtreeFixture({ subtreeLevels: 1 })));
    const arrived = vi.fn();
    const store = createSubtreeStore({
      revision: "r1",
      configGeneration: 1,
      subtreeLevels: 1,
      metadataSchema: SUBTREE_METADATA_SCHEMA,
      maxConcurrency: 1,
      maxBytes: 1_000_000,
      fetch: fetcher,
      onSubtree: arrived,
    });
    const request = { id: "subtree/0/0/0", url: "/subtrees/0/0/0.subtree" };

    store.setSelection([request, request]);
    await settle();
    expect(fetcher).toHaveBeenCalledOnce();
    expect(store.snapshot()).toMatchObject({ ready: 1, cached: 1 });

    store.setSelection([]);
    store.setSelection([request]);
    expect(fetcher).toHaveBeenCalledOnce();
    expect(arrived).toHaveBeenCalledTimes(2);
    expect(store.snapshot()).toMatchObject({
      cacheHits: 1,
      workPending: false,
    });
  });

  it("aborts a fetch when camera selection retires its subtree", async () => {
    let signal: AbortSignal | undefined;
    const fetcher: SubtreeFetch = async (_url, init) => {
      signal = init.signal;
      return await new Promise(() => {});
    };
    const store = createSubtreeStore({
      revision: "r1",
      configGeneration: 1,
      subtreeLevels: 1,
      metadataSchema: SUBTREE_METADATA_SCHEMA,
      maxConcurrency: 1,
      maxBytes: 1024,
      fetch: fetcher,
    });

    store.setSelection([{ id: "a", url: "/a.subtree" }]);
    expect(signal?.aborted).toBe(false);
    store.setSelection([]);
    expect(signal?.aborted).toBe(true);
    expect(store.snapshot()).toMatchObject({ selected: 0, workPending: false });
  });

  it("surfaces malformed metadata as a typed terminal parse failure", async () => {
    const errors: Error[] = [];
    const store = createSubtreeStore({
      revision: "r1",
      configGeneration: 1,
      subtreeLevels: 1,
      metadataSchema: SUBTREE_METADATA_SCHEMA,
      maxConcurrency: 1,
      maxBytes: 1024,
      fetch: vi.fn().mockResolvedValue(response(new ArrayBuffer(24))),
      onError: (_request, error) => errors.push(error),
    });

    store.setSelection([{ id: "bad", url: "/bad.subtree" }]);
    await settle();
    expect(errors[0]).toBeInstanceOf(SubtreeParseError);
    expect(store.snapshot()).toMatchObject({ failed: 1, workPending: false });
  });

  it("retires a revision by aborting work and ignoring late completion", async () => {
    let resolve!: (value: ReturnType<typeof response>) => void;
    const fetcher = vi.fn<SubtreeFetch>().mockImplementation(
      () =>
        new Promise((resolvePromise) => {
          resolve = resolvePromise;
        }),
    );
    const arrived = vi.fn();
    const store = createSubtreeStore({
      revision: "old",
      configGeneration: 1,
      subtreeLevels: 1,
      metadataSchema: SUBTREE_METADATA_SCHEMA,
      maxConcurrency: 1,
      maxBytes: 1024,
      fetch: fetcher,
      onSubtree: arrived,
    });
    store.setSelection([{ id: "old", url: "/old.subtree" }]);
    store.dispose();
    resolve(response(makeSubtreeFixture({ subtreeLevels: 1 })));
    await settle();
    expect(arrived).not.toHaveBeenCalled();
    expect(store.snapshot()).toMatchObject({ disposed: true, cached: 0 });
  });

  it("reconfigures hierarchy concurrency for later queued work", () => {
    const fetcher: SubtreeFetch = async () => await new Promise(() => {});
    const store = createSubtreeStore({
      revision: "r1",
      configGeneration: 1,
      subtreeLevels: 1,
      metadataSchema: SUBTREE_METADATA_SCHEMA,
      maxConcurrency: 1,
      maxBytes: 1024,
      fetch: fetcher,
    });
    store.setSelection([
      { id: "a", url: "/a.subtree" },
      { id: "b", url: "/b.subtree" },
    ]);
    expect(store.snapshot()).toMatchObject({ active: 1, queued: 1 });

    store.configure({ maxConcurrency: 2 });

    expect(store.snapshot()).toMatchObject({ active: 2, queued: 0 });
    store.dispose();
  });
});
