import type { CopcNodeEntry } from "./copcTileSource";
import type {
  CopcWorkerError,
  CopcWorkerRequest,
  CopcWorkerResponse,
} from "./copcWorkerProtocol";
import { ROOT_KEY, keyToString, type VoxelKey } from "./octree";
import type { LoadOptions, NodeInfo, TileData, TileSource } from "./tileSource";

export type CopcWorkerTileSourceOptions = {
  /** Remote COPC URL or a local file/blob sent to the source workers. */
  readonly source: string | Blob;
  /** A fresh module worker, owned and terminated by the returned source. */
  readonly createWorker: () => Worker;
  /** URL of laz-perf.wasm. Defaults to `laz-perf.wasm` beside the document. */
  readonly lazPerfWasmUrl?: string;
  /**
   * Workers decoding tiles; the first also reads the hierarchy. Defaults to
   * one fewer than the logical cores, from one to three.
   */
  readonly workers?: number;
};

type PendingRequest = {
  readonly resolve: (response: CopcWorkerResponse) => void;
  readonly reject: (error: unknown) => void;
  readonly signal: AbortSignal | undefined;
  readonly onAbort: () => void;
  aborted: boolean;
};

/** One worker and the requests it has not answered yet. */
type Channel = {
  readonly send: (
    request: CopcWorkerRequest,
    signal?: AbortSignal,
  ) => Promise<CopcWorkerResponse>;
  readonly inFlight: () => number;
  /** Idempotent: fail everything in flight and stop the worker. */
  readonly close: (error: unknown) => void;
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

const defaultWorkerCount = (): number => {
  const cores = globalThis.navigator?.hardwareConcurrency ?? 2;
  return Math.max(1, Math.min(3, cores - 1));
};

/**
 * A channel settles an aborted request only after its worker reports that the
 * physical operation ended. `LodController` can therefore keep using promise
 * lifetime as its real concurrency ceiling even when an abort arrives during
 * synchronous WASM decoding.
 */
const openChannel = (worker: Worker, fail: (error: Error) => void): Channel => {
  const pending = new Map<number, PendingRequest>();
  let closed = false;

  const detach = (request: PendingRequest): void =>
    request.signal?.removeEventListener("abort", request.onAbort);

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
    fail(new Error(event.message || "COPC worker failed"));
  const onMessageError = (): void =>
    fail(new Error("COPC worker returned an unreadable message"));
  worker.addEventListener("message", onMessage);
  worker.addEventListener("error", onError);
  worker.addEventListener("messageerror", onMessageError);

  return {
    send: (request, signal) => {
      if (closed) return Promise.reject(new Error("COPC source is disposed"));
      if (signal?.aborted) return Promise.reject(abortReason(signal));
      return new Promise((resolve, reject) => {
        const onAbort = (): void => {
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
        signal?.addEventListener("abort", onAbort, { once: true });
        worker.postMessage(request);
      });
    },
    inFlight: () => pending.size,
    close: (error) => {
      if (closed) return;
      closed = true;
      worker.removeEventListener("message", onMessage);
      worker.removeEventListener("error", onError);
      worker.removeEventListener("messageerror", onMessageError);
      for (const request of pending.values()) {
        detach(request);
        request.reject(error);
      }
      pending.clear();
      worker.terminate();
    },
  };
};

/**
 * Run COPC range reads, LAZ decoding, and point extraction in workers.
 *
 * The first worker opens the file and reads the hierarchy. The others are
 * decoders, opened from the header and colour shift the first one sampled;
 * each tile goes to the least busy worker, with its file location attached
 * for a decoder. Decoding is what bounds a COPC stream once its ranges are in
 * flight, so this multiplies how fast a view fills in. The hierarchy worker
 * failing fails the whole source, as it would with one worker; a decoder only
 * adds capacity, so one that fails is dropped and the tiles it held fail,
 * to be retried on the workers left.
 */
export const createCopcWorkerTileSource = async (
  options: CopcWorkerTileSourceOptions,
): Promise<TileSource> => {
  const workerCount = Math.max(
    1,
    Math.floor(options.workers ?? defaultWorkerCount()),
  );
  const lazPerfWasmUrl = options.lazPerfWasmUrl ?? defaultWasmUrl();
  const channels = new Set<Channel>();
  let disposed = false;
  let nextId = 0;

  const teardown = (error: unknown): void => {
    if (disposed) return;
    disposed = true;
    for (const channel of channels) channel.close(error);
  };

  const hierarchy = openChannel(options.createWorker(), teardown);
  channels.add(hierarchy);
  let opened: CopcWorkerResponse;
  try {
    opened = await hierarchy.send({
      type: "open",
      id: ++nextId,
      source: options.source,
      lazPerfWasmUrl,
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
  const { metadata, state } = opened;

  /** Decoders that have finished opening; only these are given tiles. */
  const decoders: Channel[] = [];
  const drop = (decoder: Channel, error: unknown): void => {
    decoder.close(error);
    channels.delete(decoder);
    const index = decoders.indexOf(decoder);
    if (index >= 0) decoders.splice(index, 1);
  };
  for (let index = 1; index < workerCount; index += 1) {
    const decoder: Channel = openChannel(options.createWorker(), (error) =>
      drop(decoder, error),
    );
    channels.add(decoder);
    decoder
      .send({
        type: "open-decoder",
        id: ++nextId,
        source: options.source,
        lazPerfWasmUrl,
        state,
      })
      .then(
        () => {
          if (channels.has(decoder)) decoders.push(decoder);
        },
        (error: unknown) => drop(decoder, error),
      );
  }

  /** The file location of each node the hierarchy reads have described. */
  const entries = new Map<string, CopcNodeEntry>();
  const rootKey = keyToString(ROOT_KEY);

  const tileRequest = (
    key: VoxelKey,
  ): { readonly channel: Channel; readonly request: CopcWorkerRequest } => {
    const keyString = keyToString(key);
    // The root stays with the hierarchy worker, which kept its decoded points
    // from the colour sample.
    const entry = keyString === rootKey ? undefined : entries.get(keyString);
    let channel = hierarchy;
    if (entry !== undefined) {
      for (const decoder of decoders) {
        if (decoder.inFlight() < channel.inFlight()) channel = decoder;
      }
    }
    const id = ++nextId;
    return channel === hierarchy
      ? { channel, request: { type: "load-tile", id, key } }
      : { channel, request: { type: "load-tile", id, key, entry } };
  };

  return {
    metadata: () => metadata,

    async nodes(key: VoxelKey, load?: LoadOptions): Promise<NodeInfo[]> {
      const response = await hierarchy.send(
        { type: "nodes", id: ++nextId, key },
        load?.signal,
      );
      if (response.type !== "nodes") {
        throw new Error(`COPC worker returned ${response.type} for hierarchy`);
      }
      for (const [keyString, entry] of response.entries) {
        entries.set(keyString, entry);
      }
      return response.nodes;
    },

    async loadTile(key: VoxelKey, load?: LoadOptions): Promise<TileData> {
      const { channel, request } = tileRequest(key);
      const response = await channel.send(request, load?.signal);
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
