import type {
  CopcWorkerError,
  CopcWorkerRequest,
  CopcWorkerResponse,
} from "./copcWorkerProtocol";
import type { LoadOptions, NodeInfo, TileData, TileSource } from "./tileSource";
import type { VoxelKey } from "./octree";

export type CopcWorkerTileSourceOptions = {
  /** Remote COPC URL or a local file/blob sent to the source worker. */
  readonly source: string | Blob;
  /** A fresh module worker owned and terminated by the returned source. */
  readonly createWorker: () => Worker;
  /** URL of laz-perf.wasm. Defaults to `laz-perf.wasm` beside the document. */
  readonly lazPerfWasmUrl?: string;
};

type PendingRequest = {
  readonly resolve: (response: CopcWorkerResponse) => void;
  readonly reject: (error: unknown) => void;
  readonly signal: AbortSignal | undefined;
  readonly onAbort: (() => void) | undefined;
  aborted: boolean;
};

const abortReason = (signal: AbortSignal | undefined): unknown =>
  signal?.reason ?? new DOMException("The operation was aborted", "AbortError");

const remoteError = (input: CopcWorkerError): Error => {
  const error = new Error(input.message);
  error.name = input.name;
  return error;
};

const defaultWasmUrl = (): string => {
  if (typeof document === "undefined") return "laz-perf.wasm";
  return new URL("laz-perf.wasm", document.baseURI).href;
};

/**
 * Run COPC range reads, LAZ decoding, and point extraction in one worker.
 *
 * The proxy intentionally settles an aborted request only after the worker
 * reports that its physical operation ended. `LodController` can therefore
 * continue to use promise lifetime as its real concurrency ceiling even when
 * an abort arrives during synchronous WASM decoding.
 */
export const createCopcWorkerTileSource = async (
  options: CopcWorkerTileSourceOptions,
): Promise<TileSource> => {
  const worker = options.createWorker();
  const pending = new Map<number, PendingRequest>();
  let nextId = 0;
  let disposed = false;

  const detach = (request: PendingRequest): void => {
    if (request.onAbort !== undefined) {
      request.signal?.removeEventListener("abort", request.onAbort);
    }
  };

  const rejectAll = (error: unknown): void => {
    for (const request of pending.values()) {
      detach(request);
      request.reject(error);
    }
    pending.clear();
  };

  const onMessage = (event: MessageEvent<CopcWorkerResponse>): void => {
    const request = pending.get(event.data.id);
    if (request === undefined) return;
    pending.delete(event.data.id);
    detach(request);
    if (request.aborted) {
      request.reject(abortReason(request.signal));
    } else if (event.data.type === "error") {
      request.reject(remoteError(event.data.error));
    } else {
      request.resolve(event.data);
    }
  };

  const onError = (event: ErrorEvent): void =>
    teardown(new Error(event.message || "COPC worker failed"));
  const onMessageError = (): void =>
    teardown(new Error("COPC worker returned an unreadable message"));
  worker.addEventListener("message", onMessage);
  worker.addEventListener("error", onError);
  worker.addEventListener("messageerror", onMessageError);

  /** Idempotent: detach, fail everything still in flight, and stop the worker. */
  const teardown = (error: unknown): void => {
    if (disposed) return;
    disposed = true;
    worker.removeEventListener("message", onMessage);
    worker.removeEventListener("error", onError);
    worker.removeEventListener("messageerror", onMessageError);
    rejectAll(error);
    worker.terminate();
  };

  const send = (
    request: CopcWorkerRequest,
    signal?: AbortSignal,
  ): Promise<CopcWorkerResponse> => {
    if (disposed) return Promise.reject(new Error("COPC source is disposed"));
    if (signal?.aborted) return Promise.reject(abortReason(signal));
    return new Promise((resolve, reject) => {
      const onAbort =
        signal === undefined
          ? undefined
          : (): void => {
              const current = pending.get(request.id);
              if (current === undefined || current.aborted) return;
              current.aborted = true;
              worker.postMessage({ type: "cancel", id: request.id });
            };
      pending.set(request.id, {
        resolve,
        reject,
        signal,
        onAbort,
        aborted: false,
      });
      if (signal !== undefined && onAbort !== undefined) {
        signal.addEventListener("abort", onAbort, { once: true });
      }
      worker.postMessage(request);
    });
  };

  const openId = ++nextId;
  let opened: CopcWorkerResponse;
  try {
    opened = await send({
      type: "open",
      id: openId,
      source: options.source,
      lazPerfWasmUrl: options.lazPerfWasmUrl ?? defaultWasmUrl(),
    });
  } catch (error) {
    teardown(error);
    throw error;
  }
  if (opened.type !== "opened") {
    const error = new Error(
      `COPC worker returned ${opened.type} while opening`,
    );
    teardown(error);
    throw error;
  }
  const metadata = opened.metadata;

  return {
    metadata: () => metadata,

    async nodes(key: VoxelKey, load?: LoadOptions): Promise<NodeInfo[]> {
      const response = await send(
        { type: "nodes", id: ++nextId, key },
        load?.signal,
      );
      if (response.type !== "nodes") {
        throw new Error(`COPC worker returned ${response.type} for hierarchy`);
      }
      return response.nodes;
    },

    async loadTile(key: VoxelKey, load?: LoadOptions): Promise<TileData> {
      const response = await send(
        { type: "load-tile", id: ++nextId, key },
        load?.signal,
      );
      if (response.type !== "tile") {
        throw new Error(`COPC worker returned ${response.type} for tile`);
      }
      return response.tile;
    },

    dispose() {
      teardown(new Error("COPC source is disposed"));
    },
  };
};
