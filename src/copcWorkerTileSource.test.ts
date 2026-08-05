import { describe, expect, it } from "vitest";

import { createCopcWorkerTileSource } from "./copcWorkerTileSource";
import type {
  CopcWorkerRequest,
  CopcWorkerResponse,
} from "./copcWorkerProtocol";
import { ROOT_KEY } from "./octree";

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

const openSource = async (worker: FakeWorker) => {
  const sourcePromise = createCopcWorkerTileSource({
    source: "https://example.test/cloud.copc.laz",
    lazPerfWasmUrl: "https://example.test/laz-perf.wasm",
    createWorker: () => worker as unknown as Worker,
  });
  expect(worker.sent[0]).toMatchObject({
    type: "open",
    source: "https://example.test/cloud.copc.laz",
    lazPerfWasmUrl: "https://example.test/laz-perf.wasm",
  });
  const open = worker.sent[0]!;
  worker.respond({
    type: "opened",
    id: open.id,
    metadata: { pointCount: 42 },
  });
  return sourcePromise;
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
    const sourcePromise = createCopcWorkerTileSource({
      source: "bad.copc.laz",
      createWorker: () => worker as unknown as Worker,
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
  });
});
