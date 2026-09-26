import { describe, expect, it, vi } from "vitest";

import type { CopcDecodeState, CopcNodeEntry } from "./copcTileSource";
import { createCopcWorkerTileSource } from "./copcWorkerTileSource";
import type {
  CopcWorkerRequest,
  CopcWorkerResponse,
} from "./copcWorkerProtocol";
import { ROOT_KEY, keyFromString } from "./octree";

class FakeWorker {
  readonly sent: CopcWorkerRequest[] = [];
  terminated = false;
  private readonly listeners = new Map<string, Set<(event: unknown) => void>>();

  postMessage(message: CopcWorkerRequest): void {
    this.sent.push(message);
  }

  addEventListener(type: string, listener: (event: unknown) => void): void {
    const listeners = this.listeners.get(type) ?? new Set();
    listeners.add(listener);
    this.listeners.set(type, listeners);
  }

  removeEventListener(type: string, listener: (event: unknown) => void): void {
    this.listeners.get(type)?.delete(listener);
  }

  terminate(): void {
    this.terminated = true;
  }

  crash(message: string): void {
    for (const listener of this.listeners.get("error") ?? []) {
      listener({ message });
    }
  }

  respond(message: CopcWorkerResponse): void {
    for (const listener of this.listeners.get("message") ?? []) {
      listener({ data: message });
    }
  }
}

/**
 * A timer turn runs only once the microtask queue is empty, so any settlement
 * chained off the abort — however many `then` hops deep — has landed by then.
 */
const drainMicrotasks = (): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, 0);
  });

/** The header and shift a hierarchy worker hands its decoders. */
const STATE: CopcDecodeState = {
  copc: { info: { cube: [0, 0, 0, 8, 8, 8] } } as CopcDecodeState["copc"],
  rgbShift: 8,
};

const entryAt = (offset: number): CopcNodeEntry => ({
  pointCount: 2,
  pointDataOffset: offset,
  pointDataLength: 10,
});

/** Opens a source over `workers`, the first of which reads the hierarchy. */
const openSource = async (...workers: FakeWorker[]) => {
  const unused = [...workers];
  const sourcePromise = createCopcWorkerTileSource({
    source: "https://example.test/cloud.copc.laz",
    lazPerfWasmUrl: "https://example.test/laz-perf.wasm",
    createWorker: () => unused.shift() as unknown as Worker,
    workers: workers.length,
  });
  const hierarchy = workers[0]!;
  expect(hierarchy.sent[0]).toMatchObject({
    type: "open",
    source: "https://example.test/cloud.copc.laz",
    lazPerfWasmUrl: "https://example.test/laz-perf.wasm",
  });
  hierarchy.respond({
    type: "opened",
    id: hierarchy.sent[0]!.id,
    metadata: { pointCount: 42 },
    state: STATE,
  });
  return sourcePromise;
};

/** Answers every decoder's open request, then lets the source see it. */
const openDecoders = async (...decoders: FakeWorker[]) => {
  for (const decoder of decoders) {
    expect(decoder.sent[0]).toMatchObject({
      type: "open-decoder",
      source: "https://example.test/cloud.copc.laz",
      state: STATE,
    });
    decoder.respond({ type: "decoder-opened", id: decoder.sent[0]!.id });
  }
  await drainMicrotasks();
};

/** A request left in flight; disposing the source rejects it. */
const leaveInFlight = (request: Promise<unknown>): void => {
  request.catch(() => undefined);
};

const CHILDREN = ["1-0-0-0", "1-1-0-0", "1-0-1-0"];

/** Reads the root page, whose nodes are the root and `CHILDREN`. */
const readRootPage = async (
  source: Awaited<ReturnType<typeof openSource>>,
  hierarchy: FakeWorker,
) => {
  const nodesPromise = source.nodes(ROOT_KEY);
  const request = hierarchy.sent.at(-1)!;
  hierarchy.respond({
    type: "nodes",
    id: request.id,
    nodes: ["0-0-0-0", ...CHILDREN].map((key) => ({
      key: keyFromString(key),
      pointCount: 2,
      bounds: { min: [0, 0, 0], max: [1, 1, 1] },
      spacing: 1,
    })),
    entries: ["0-0-0-0", ...CHILDREN].map(
      (key, index) => [key, entryAt(index * 10)] as const,
    ),
  });
  await nodesPromise;
};

describe("COPC worker tile source", () => {
  it("proxies hierarchy and transferable tile payloads", async () => {
    const worker = new FakeWorker();
    const source = await openSource(worker);
    expect(source.metadata()).toEqual({ pointCount: 42 });

    const nodesPromise = source.nodes(ROOT_KEY);
    const nodesRequest = worker.sent.at(-1)!;
    expect(nodesRequest).toMatchObject({ type: "nodes", key: ROOT_KEY });
    worker.respond({
      type: "nodes",
      id: nodesRequest.id,
      nodes: [
        {
          key: ROOT_KEY,
          pointCount: 2,
          bounds: { min: [0, 0, 0], max: [1, 1, 1] },
          spacing: 1,
        },
      ],
      entries: [["0-0-0-0", entryAt(0)]],
    });
    expect(await nodesPromise).toHaveLength(1);

    const tilePromise = source.loadTile(ROOT_KEY);
    const tileRequest = worker.sent.at(-1)!;
    const positions = new Float32Array([0, 0, 0, 1, 1, 1]);
    worker.respond({
      type: "tile",
      id: tileRequest.id,
      tile: { origin: [2, 3, 4], positions, pointCount: 2 },
    });
    expect(await tilePromise).toEqual({
      origin: [2, 3, 4],
      positions,
      pointCount: 2,
    });

    source.dispose?.();
    expect(worker.terminated).toBe(true);
  });

  it("keeps an aborted request pending until physical worker work ends", async () => {
    const worker = new FakeWorker();
    const source = await openSource(worker);
    const abort = new AbortController();
    const tilePromise = source.loadTile(ROOT_KEY, { signal: abort.signal });
    const tileRequest = worker.sent.at(-1)!;
    let settled = false;
    void tilePromise.then(
      () => {
        settled = true;
      },
      () => {
        settled = true;
      },
    );

    abort.abort();
    await drainMicrotasks();
    expect(settled).toBe(false);
    expect(worker.sent.at(-1)).toEqual({
      type: "cancel",
      id: tileRequest.id,
    });

    worker.respond({
      type: "error",
      id: tileRequest.id,
      error: { name: "AbortError", message: "cancelled" },
    });
    await expect(tilePromise).rejects.toMatchObject({ name: "AbortError" });
    source.dispose?.();
  });

  it("terminates the worker when opening fails", async () => {
    const worker = new FakeWorker();
    const createWorker = vi.fn(() => worker as unknown as Worker);
    const sourcePromise = createCopcWorkerTileSource({
      source: "bad.copc.laz",
      createWorker,
      workers: 3,
    });
    const open = worker.sent[0]!;
    worker.respond({
      type: "error",
      id: open.id,
      error: { name: "DataError", message: "not COPC" },
    });

    await expect(sourcePromise).rejects.toMatchObject({
      name: "DataError",
      message: "not COPC",
    });
    expect(worker.terminated).toBe(true);
    expect(createWorker).toHaveBeenCalledOnce();
  });

  it("reads the hierarchy on one worker and spreads tiles over decoders", async () => {
    const workers = [new FakeWorker(), new FakeWorker(), new FakeWorker()];
    const [hierarchy, first, second] = workers as [
      FakeWorker,
      FakeWorker,
      FakeWorker,
    ];
    const source = await openSource(...workers);
    await openDecoders(first, second);
    await readRootPage(source, hierarchy);
    expect(first.sent).toHaveLength(1);
    expect(second.sent).toHaveLength(1);

    for (const key of CHILDREN)
      leaveInFlight(source.loadTile(keyFromString(key)));
    expect(hierarchy.sent.at(-1)).toMatchObject({
      type: "load-tile",
      key: keyFromString("1-0-0-0"),
    });
    expect(hierarchy.sent.at(-1)).not.toHaveProperty("entry");
    expect(first.sent.at(-1)).toMatchObject({
      type: "load-tile",
      key: keyFromString("1-1-0-0"),
      entry: entryAt(20),
    });
    expect(second.sent.at(-1)).toMatchObject({
      type: "load-tile",
      key: keyFromString("1-0-1-0"),
      entry: entryAt(30),
    });
    source.dispose?.();
    expect(workers.every((worker) => worker.terminated)).toBe(true);
  });

  it("keeps the root tile with the worker holding its sampled points", async () => {
    const workers = [new FakeWorker(), new FakeWorker()];
    const [hierarchy, decoder] = workers as [FakeWorker, FakeWorker];
    const source = await openSource(...workers);
    await openDecoders(decoder);
    await readRootPage(source, hierarchy);
    leaveInFlight(source.loadTile(keyFromString(CHILDREN[0]!)));
    leaveInFlight(source.loadTile(ROOT_KEY));
    expect(hierarchy.sent.at(-1)).toEqual({
      type: "load-tile",
      id: expect.any(Number),
      key: ROOT_KEY,
    });
    source.dispose?.();
  });

  it("gives tiles only to decoders that have finished opening", async () => {
    const workers = [new FakeWorker(), new FakeWorker()];
    const [hierarchy, decoder] = workers as [FakeWorker, FakeWorker];
    const source = await openSource(...workers);
    await readRootPage(source, hierarchy);
    for (const key of CHILDREN)
      leaveInFlight(source.loadTile(keyFromString(key)));
    expect(
      hierarchy.sent.filter((request) => request.type === "load-tile"),
    ).toHaveLength(3);
    expect(decoder.sent).toHaveLength(1);
    source.dispose?.();
  });

  it("keeps loading on the hierarchy worker when a decoder cannot open", async () => {
    const workers = [new FakeWorker(), new FakeWorker()];
    const [hierarchy, decoder] = workers as [FakeWorker, FakeWorker];
    const source = await openSource(...workers);
    decoder.respond({
      type: "error",
      id: decoder.sent[0]!.id,
      error: { name: "RuntimeError", message: "no wasm" },
    });
    await drainMicrotasks();
    expect(decoder.terminated).toBe(true);
    expect(hierarchy.terminated).toBe(false);
    await readRootPage(source, hierarchy);
    leaveInFlight(source.loadTile(keyFromString(CHILDREN[0]!)));
    expect(hierarchy.sent.at(-1)).toMatchObject({ type: "load-tile" });
    source.dispose?.();
  });

  it("fails only a crashed decoder's tiles and sends the next to the rest", async () => {
    const workers = [new FakeWorker(), new FakeWorker()];
    const [hierarchy, decoder] = workers as [FakeWorker, FakeWorker];
    const source = await openSource(...workers);
    await openDecoders(decoder);
    await readRootPage(source, hierarchy);
    const onHierarchy = source.loadTile(keyFromString(CHILDREN[0]!));
    const onDecoder = source.loadTile(keyFromString(CHILDREN[1]!));
    expect(decoder.sent.at(-1)).toMatchObject({ type: "load-tile" });
    decoder.crash("out of memory");
    await expect(onDecoder).rejects.toThrow("out of memory");
    expect(decoder.terminated).toBe(true);
    expect(hierarchy.terminated).toBe(false);
    leaveInFlight(onHierarchy);
    leaveInFlight(source.loadTile(keyFromString(CHILDREN[2]!)));
    expect(
      hierarchy.sent.filter((request) => request.type === "load-tile"),
    ).toHaveLength(2);
    source.dispose?.();
  });
});
