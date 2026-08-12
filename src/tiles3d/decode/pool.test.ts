import { describe, expect, it, vi } from "vitest";

import {
  DecodeWorkerPool,
  DecodeWorkerPoolDisposedError,
  type DecodeTileRequest,
  type DecodeWorkerLike,
} from ".";

class FakeWorker implements DecodeWorkerLike {
  readonly messages: unknown[] = [];
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  terminated = false;

  postMessage(message: unknown): void {
    this.messages.push(message);
  }

  terminate(): void {
    this.terminated = true;
  }

  complete(index: number, result: unknown): void {
    const request = this.messages[index] as {
      jobId: number;
      generation: number;
    };
    this.onmessage?.({
      data: {
        kind: "complete",
        jobId: request.jobId,
        generation: request.generation,
        result,
      },
    } as MessageEvent);
  }

  respond(value: unknown): void {
    this.onmessage?.({ data: value } as MessageEvent);
  }
}

const decodeRequest = (): DecodeTileRequest => ({
  content: new ArrayBuffer(8),
  contentUrl: "https://fixture.invalid/tile.glb",
  revision: "r1",
  accumulatedTransform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
  ecefToScene: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
  textureCapabilities: { capabilityKey: "rgba", compressedFormats: [] },
});

describe("DecodeWorkerPool", () => {
  it("uses an injectable worker count and dispatches deterministically", async () => {
    const workers = [new FakeWorker(), new FakeWorker()];
    const pool = new DecodeWorkerPool({
      size: 2,
      workerFactory: (index) => workers[index]!,
    });
    const first = pool.decode(decodeRequest());
    const second = pool.decode(decodeRequest());
    const third = pool.decode(decodeRequest());
    expect(workers.map((worker) => worker.messages.length)).toEqual([1, 1]);

    workers[0]!.complete(0, {
      primitives: [],
      origin: [0, 0, 0],
      byteEstimate: { geometry: 0, textures: 0 },
    });
    await first.promise;
    expect(workers.map((worker) => worker.messages.length)).toEqual([2, 1]);
    workers[1]!.complete(0, {
      primitives: [],
      origin: [0, 0, 0],
      byteEstimate: { geometry: 0, textures: 0 },
    });
    workers[0]!.complete(1, {
      primitives: [],
      origin: [0, 0, 0],
      byteEstimate: { geometry: 0, textures: 0 },
    });
    await Promise.all([second.promise, third.promise]);
    expect(pool.stats()).toMatchObject({
      size: 2,
      queuedJobs: 0,
      activeJobs: 0,
      completedJobs: 3,
      failedJobs: 0,
      cancelledJobs: 0,
    });
  });

  it("discards completion after cancellation and continues queued work", async () => {
    const worker = new FakeWorker();
    const replacement = new FakeWorker();
    const pool = new DecodeWorkerPool({
      size: 1,
      workerFactory: vi
        .fn()
        .mockReturnValueOnce(worker)
        .mockReturnValueOnce(replacement),
    });
    const cancelled = pool.decode(decodeRequest());
    const next = pool.decode(decodeRequest());
    cancelled.cancel();
    await expect(cancelled.promise).rejects.toMatchObject({
      name: "AbortError",
    });
    worker.complete(0, { stale: true });
    expect(worker.terminated).toBe(true);
    expect(replacement.messages).toHaveLength(1);
    replacement.complete(0, {
      primitives: [],
      origin: [0, 0, 0],
      byteEstimate: { geometry: 0, textures: 0 },
    });
    await expect(next.promise).resolves.toMatchObject({ primitives: [] });
    expect(pool.stats().cancelledJobs).toBe(1);
  });

  it("invalidates an old generation and contains worker errors", async () => {
    const firstWorker = new FakeWorker();
    const secondWorker = new FakeWorker();
    const factory = vi
      .fn()
      .mockReturnValueOnce(firstWorker)
      .mockReturnValueOnce(secondWorker);
    const pool = new DecodeWorkerPool({ size: 1, workerFactory: factory });
    const stale = pool.decode(decodeRequest());
    pool.invalidate();
    await expect(stale.promise).rejects.toMatchObject({ name: "AbortError" });
    const live = pool.decode(decodeRequest());
    firstWorker.complete(0, { stale: true });
    firstWorker.onerror?.({ message: "old failure" } as ErrorEvent);
    expect(secondWorker.messages).toHaveLength(1);
    secondWorker.complete(0, {
      primitives: [],
      origin: [0, 0, 0],
      byteEstimate: { geometry: 0, textures: 0 },
    });
    await live.promise;
  });

  it("rejects all work and discards late completions after disposal", async () => {
    const worker = new FakeWorker();
    const pool = new DecodeWorkerPool({ size: 1, workerFactory: () => worker });
    const active = pool.decode(decodeRequest());
    pool.dispose();
    await expect(active.promise).rejects.toBeInstanceOf(
      DecodeWorkerPoolDisposedError,
    );
    worker.complete(0, { stale: true });
    expect(worker.terminated).toBe(true);
    expect(() => pool.decode(decodeRequest())).toThrow(
      DecodeWorkerPoolDisposedError,
    );
  });

  it("contains a malformed response by replacing the worker", async () => {
    const worker = new FakeWorker();
    const replacement = new FakeWorker();
    const pool = new DecodeWorkerPool({
      size: 1,
      workerFactory: vi
        .fn()
        .mockReturnValueOnce(worker)
        .mockReturnValueOnce(replacement),
    });
    const active = pool.decode(decodeRequest());
    worker.respond({ kind: "complete" });
    await expect(active.promise).rejects.toThrow(/malformed response/i);
    expect(worker.terminated).toBe(true);
    const next = pool.decode(decodeRequest());
    replacement.complete(0, {
      primitives: [],
      origin: [0, 0, 0],
      byteEstimate: { geometry: 0, textures: 0 },
    });
    await expect(next.promise).resolves.toMatchObject({ primitives: [] });
  });

  it("aggregates runtime cost and retains bounded per-target transcode samples", async () => {
    const worker = new FakeWorker();
    const pool = new DecodeWorkerPool({ size: 1, workerFactory: () => worker });
    const active = pool.decode(decodeRequest());
    worker.complete(0, {
      primitives: [],
      origin: [0, 0, 0],
      byteEstimate: { geometry: 0, textures: 0 },
      diagnostics: {
        totalDecodeMs: 90,
        basisRuntimeInitializationMs: 5,
        basisTranscodeMs: 2_485,
        basisTranscodeSamplesMs: Array.from(
          { length: 70 },
          (_value, index) => index + 1,
        ),
        basisTextures: 70,
        basisTarget: "bc7",
      },
    });
    await active.promise;

    expect(pool.stats()).toMatchObject({
      basisRuntimeInitializationMs: 5,
      basisTranscodeMs: 2_485,
      basisTextures: 70,
      basisTargets: { bc7: 70 },
      basisTargetTimings: {
        bc7: {
          count: 70,
          totalMs: 2_485,
          samplesMs: Array.from({ length: 64 }, (_value, index) => index + 7),
        },
      },
    });
    pool.dispose();
  });
});
