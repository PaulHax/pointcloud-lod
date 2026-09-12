import type {
  DecodeJob,
  DecodeTileRequest,
  DecodedTileContent,
  DecodeWorkerPoolHandle,
} from "./types";
import {
  TileDecodeError,
  TileUnsupportedExtensionError,
  type TileDecodeStage,
} from "./types";

type DecodeWorkerRequestMessage = {
  kind: "decode";
  jobId: number;
  generation: number;
  request: DecodeTileRequest;
};

type DecodeWorkerCompleteMessage = {
  kind: "complete";
  jobId: number;
  generation: number;
  result: DecodedTileContent;
};

type DecodeWorkerErrorMessage = {
  kind: "error";
  jobId: number;
  generation: number;
  error: {
    readonly name: string;
    readonly message: string;
    readonly tileUri?: string;
    readonly stage?: TileDecodeStage;
    readonly reason?: string;
    readonly extension?: string;
  };
};

type DecodeWorkerResponseMessage =
  | DecodeWorkerCompleteMessage
  | DecodeWorkerErrorMessage;

export type DecodeWorkerLike = {
  onmessage: ((event: MessageEvent<unknown>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  postMessage(
    message: DecodeWorkerRequestMessage,
    transfer?: Transferable[],
  ): void;
  terminate(): void;
};

export type DecodeWorkerPoolOptions = {
  size?: number;
  workerFactory?: (index: number) => DecodeWorkerLike;
  workerUrl?: string | URL;
};

type JobRecord = {
  jobId: number;
  generation: number;
  request: DecodeTileRequest;
  state: "queued" | "active" | "cancelled" | "settled";
  startedAt: number | null;
  resolve(value: DecodedTileContent): void;
  reject(reason: unknown): void;
};

type WorkerSlot = {
  worker: DecodeWorkerLike;
  active: JobRecord | null;
};

const BASIS_TIMING_SAMPLE_LIMIT = 64;

type MutableBasisTargetTiming = {
  count: number;
  totalMs: number;
  samplesMs: number[];
};

const abortError = (): DOMException =>
  new DOMException("Decode cancelled", "AbortError");

const defaultPoolSize = (): number => {
  const cores = globalThis.navigator?.hardwareConcurrency ?? 2;
  return Math.max(1, Math.min(4, cores - 1));
};

const defaultWorkerFactory =
  (workerUrl: string | URL) => (): DecodeWorkerLike =>
    new Worker(workerUrl, {
      // The artifact is deliberately classic because the Emscripten wrapper
      // dependencies are installed through importScripts.
      type: "classic",
      name: "tiles3d-decode",
    });

export class DecodeWorkerPoolDisposedError extends Error {
  constructor() {
    super("Decode worker pool has been disposed");
    this.name = "DecodeWorkerPoolDisposedError";
  }
}

export class DecodeWorkerError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DecodeWorkerError";
  }
}

export class DecodeWorkerPool implements DecodeWorkerPoolHandle {
  readonly #size: number;
  readonly #factory: (index: number) => DecodeWorkerLike;
  #slots: WorkerSlot[];
  #queued: JobRecord[] = [];
  #generation = 1;
  #nextJobId = 1;
  #disposed = false;
  #completedJobs = 0;
  #failedJobs = 0;
  #cancelledJobs = 0;
  #workerElapsedMs = 0;
  #decodedGeometryBytes = 0;
  #decodedTextureBytes = 0;
  #basisRuntimeInitializationMs = 0;
  #basisTranscodeMs = 0;
  #basisTextures = 0;
  #basisTargets = new Map<string, number>();
  #basisTargetTimings = new Map<string, MutableBasisTargetTiming>();

  constructor(options: DecodeWorkerPoolOptions = {}) {
    this.#size = options.size ?? defaultPoolSize();
    if (!Number.isInteger(this.#size) || this.#size <= 0) {
      throw new Error("Decode worker pool size must be a positive integer");
    }
    const workerUrl =
      options.workerUrl ??
      new URL(
        /* @vite-ignore */ "./tiles3dDecodeWorker.classic.js",
        import.meta.url,
      );
    this.#factory = options.workerFactory ?? defaultWorkerFactory(workerUrl);
    this.#slots = Array.from({ length: this.#size }, (_value, index) =>
      this.#makeSlot(index),
    );
  }

  get size(): number {
    return this.#size;
  }

  decode(request: DecodeTileRequest): DecodeJob {
    if (this.#disposed) throw new DecodeWorkerPoolDisposedError();
    let resolve!: (value: DecodedTileContent) => void;
    let reject!: (reason: unknown) => void;
    const promise = new Promise<DecodedTileContent>(
      (resolvePromise, rejectPromise) => {
        resolve = resolvePromise;
        reject = rejectPromise;
      },
    );
    const record: JobRecord = {
      jobId: this.#nextJobId,
      generation: this.#generation,
      request,
      state: "queued",
      startedAt: null,
      resolve,
      reject,
    };
    this.#nextJobId += 1;
    this.#queued.push(record);
    this.#dispatch();
    return {
      promise,
      cancel: () => this.#cancel(record),
    };
  }

  stats() {
    return {
      size: this.#size,
      queuedJobs: this.#queued.length,
      activeJobs: this.#slots.filter((slot) => slot.active !== null).length,
      completedJobs: this.#completedJobs,
      failedJobs: this.#failedJobs,
      cancelledJobs: this.#cancelledJobs,
      workerElapsedMs: this.#workerElapsedMs,
      decodedGeometryBytes: this.#decodedGeometryBytes,
      decodedTextureBytes: this.#decodedTextureBytes,
      basisRuntimeInitializationMs: this.#basisRuntimeInitializationMs,
      basisTranscodeMs: this.#basisTranscodeMs,
      basisTextures: this.#basisTextures,
      basisTargets: Object.fromEntries(this.#basisTargets),
      basisTargetTimings: Object.fromEntries(
        [...this.#basisTargetTimings].map(([target, timing]) => [
          target,
          { ...timing, samplesMs: [...timing.samplesMs] },
        ]),
      ),
    };
  }

  invalidate(): void {
    if (this.#disposed) return;
    this.#generation += 1;
    for (const record of this.#queued.splice(0)) {
      this.#reject(record, abortError());
    }
    this.#restartSlots(abortError());
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    const error = new DecodeWorkerPoolDisposedError();
    for (const record of this.#queued.splice(0)) this.#reject(record, error);
    for (const slot of this.#slots) {
      if (slot.active) this.#reject(slot.active, error);
      slot.active = null;
      slot.worker.onmessage = null;
      slot.worker.onerror = null;
      slot.worker.terminate();
    }
    this.#slots = [];
  }

  #makeSlot(index: number): WorkerSlot {
    const slot: WorkerSlot = {
      worker: this.#factory(index),
      active: null,
    };
    slot.worker.onmessage = (event) => this.#onMessage(slot, event.data);
    slot.worker.onerror = (event) =>
      this.#onWorkerFailure(
        slot,
        new DecodeWorkerError(event.message || "decode worker failed"),
      );
    return slot;
  }

  #dispatch(): void {
    if (this.#disposed) return;
    for (const slot of this.#slots) {
      if (slot.active) continue;
      const record = this.#queued.shift();
      if (!record) break;
      if (record.state !== "queued") continue;
      record.state = "active";
      record.startedAt = globalThis.performance?.now?.() ?? Date.now();
      slot.active = record;
      const message: DecodeWorkerRequestMessage = {
        kind: "decode",
        jobId: record.jobId,
        generation: record.generation,
        request: record.request,
      };
      slot.worker.postMessage(message, [record.request.content]);
    }
  }

  #onMessage(slot: WorkerSlot, value: unknown): void {
    if (
      !value ||
      typeof value !== "object" ||
      !("kind" in value) ||
      !("jobId" in value) ||
      !("generation" in value) ||
      (value.kind !== "complete" && value.kind !== "error") ||
      typeof value.jobId !== "number" ||
      typeof value.generation !== "number"
    ) {
      this.#onWorkerFailure(
        slot,
        new DecodeWorkerError("decode worker returned a malformed response"),
      );
      return;
    }
    const message = value as DecodeWorkerResponseMessage;
    const active = slot.active;
    if (
      !active ||
      message.jobId !== active.jobId ||
      message.generation !== active.generation ||
      message.generation !== this.#generation
    ) {
      return;
    }
    slot.active = null;
    if (active.state === "active") {
      if (message.kind === "complete") {
        active.state = "settled";
        const now = globalThis.performance?.now?.() ?? Date.now();
        this.#workerElapsedMs += Math.max(0, now - (active.startedAt ?? now));
        this.#completedJobs += 1;
        this.#decodedGeometryBytes += message.result.byteEstimate.geometry;
        this.#decodedTextureBytes += message.result.byteEstimate.textures;
        const diagnostics = message.result.diagnostics;
        if (diagnostics) {
          this.#basisRuntimeInitializationMs +=
            diagnostics.basisRuntimeInitializationMs;
          this.#basisTranscodeMs += diagnostics.basisTranscodeMs;
          this.#basisTextures += diagnostics.basisTextures;
          if (diagnostics.basisTarget) {
            const target = diagnostics.basisTarget;
            this.#basisTargets.set(
              target,
              (this.#basisTargets.get(target) ?? 0) + diagnostics.basisTextures,
            );
            const timing = this.#basisTargetTimings.get(target) ?? {
              count: 0,
              totalMs: 0,
              samplesMs: [],
            };
            timing.count += diagnostics.basisTextures;
            timing.totalMs += diagnostics.basisTranscodeMs;
            for (const sample of diagnostics.basisTranscodeSamplesMs) {
              if (Number.isFinite(sample) && sample >= 0) {
                timing.samplesMs.push(sample);
              }
            }
            if (timing.samplesMs.length > BASIS_TIMING_SAMPLE_LIMIT) {
              timing.samplesMs.splice(
                0,
                timing.samplesMs.length - BASIS_TIMING_SAMPLE_LIMIT,
              );
            }
            this.#basisTargetTimings.set(target, timing);
          }
        }
        active.resolve(message.result);
      } else {
        this.#failedJobs += 1;
        const payload = message.error;
        const decodeError =
          payload?.name === "TileUnsupportedExtensionError" &&
          typeof payload.tileUri === "string" &&
          typeof payload.extension === "string"
            ? new TileUnsupportedExtensionError(
                payload.tileUri,
                payload.extension,
              )
            : payload?.name === "TileDecodeError" &&
                typeof payload.tileUri === "string" &&
                (payload.stage === "profile" || payload.stage === "decode") &&
                typeof payload.reason === "string"
              ? new TileDecodeError(
                  payload.tileUri,
                  payload.stage,
                  payload.reason,
                )
              : new DecodeWorkerError(
                  typeof payload?.message === "string"
                    ? payload.message
                    : "decode worker returned an invalid error",
                );
        this.#reject(active, decodeError);
      }
    }
    this.#dispatch();
  }

  #onWorkerFailure(slot: WorkerSlot, error: DecodeWorkerError): void {
    const active = slot.active;
    if (active) {
      this.#failedJobs += 1;
      this.#reject(active, error);
    }
    const index = this.#slots.indexOf(slot);
    slot.worker.onmessage = null;
    slot.worker.onerror = null;
    slot.worker.terminate();
    if (index >= 0 && !this.#disposed)
      this.#slots[index] = this.#makeSlot(index);
    this.#dispatch();
  }

  #cancel(record: JobRecord): void {
    if (record.state === "queued") {
      this.#cancelledJobs += 1;
      const index = this.#queued.indexOf(record);
      if (index >= 0) this.#queued.splice(index, 1);
      this.#reject(record, abortError());
      return;
    }
    if (record.state === "active") {
      this.#cancelledJobs += 1;
      const slotIndex = this.#slots.findIndex((slot) => slot.active === record);
      this.#reject(record, abortError());
      if (slotIndex >= 0) {
        const slot = this.#slots[slotIndex]!;
        slot.worker.onmessage = null;
        slot.worker.onerror = null;
        slot.worker.terminate();
        this.#slots[slotIndex] = this.#makeSlot(slotIndex);
        this.#dispatch();
      }
    }
  }

  #restartSlots(reason: unknown): void {
    const previous = this.#slots;
    this.#slots = previous.map((slot, index) => {
      if (slot.active) this.#reject(slot.active, reason);
      slot.worker.onmessage = null;
      slot.worker.onerror = null;
      slot.worker.terminate();
      return this.#makeSlot(index);
    });
    this.#dispatch();
  }

  #reject(record: JobRecord, reason: unknown): void {
    if (record.state === "settled") return;
    record.state = "settled";
    record.reject(reason);
  }
}

export type {
  DecodeWorkerCompleteMessage,
  DecodeWorkerErrorMessage,
  DecodeWorkerRequestMessage,
};
