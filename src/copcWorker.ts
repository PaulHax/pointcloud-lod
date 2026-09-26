import { createLazPerf } from "laz-perf/lib/worker";

import {
  createCopcTileDecoder,
  createCopcTileSource,
  type CopcNodeEntry,
  type CopcTileDecoder,
  type CopcTileSource,
  type RangeGetter,
} from "./copcTileSource";
import type {
  CopcWorkerError,
  CopcWorkerRequest,
  CopcWorkerResponse,
} from "./copcWorkerProtocol";
import { keyToString } from "./octree";
import type { LoadOptions, TileData } from "./tileSource";

type WorkerScope = {
  addEventListener(
    type: "message",
    listener: (event: MessageEvent<CopcWorkerRequest>) => void,
  ): void;
  postMessage(message: CopcWorkerResponse, transfer?: Transferable[]): void;
};

const serializedError = (error: unknown): CopcWorkerError =>
  error instanceof Error
    ? { name: error.name, message: error.message }
    : { name: "Error", message: String(error) };

const blobGetter =
  (blob: Blob): RangeGetter =>
  async (begin, end, signal) => {
    signal?.throwIfAborted();
    const bytes = new Uint8Array(await blob.slice(begin, end).arrayBuffer());
    signal?.throwIfAborted();
    return bytes;
  };

const rangeSource = (source: string | Blob): string | RangeGetter =>
  typeof source === "string" ? source : blobGetter(source);

/**
 * Start the worker endpoint used by `createCopcWorkerTileSource`.
 *
 * One worker of a source opens the file and reads its hierarchy; the rest are
 * decoders, opened from the state that one hands over, and read tiles from the
 * entries sent with each request.
 */
export const serveCopcTileSourceWorker = (
  scope: WorkerScope = globalThis as unknown as WorkerScope,
): void => {
  let source: CopcTileSource | null = null;
  let decoder: CopcTileDecoder | null = null;
  const operations = new Map<number, AbortController>();

  const respondError = (id: number, error: unknown): void =>
    scope.postMessage({ type: "error", id, error: serializedError(error) });

  const loadTile = (
    request: Extract<CopcWorkerRequest, { type: "load-tile" }>,
    options: LoadOptions,
  ): Promise<TileData> => {
    if (source !== null) return source.loadTile(request.key, options);
    if (decoder !== null && request.entry !== undefined) {
      return decoder.loadTile(request.key, request.entry, options);
    }
    throw new Error("COPC worker is not open");
  };

  const run = async (request: CopcWorkerRequest): Promise<void> => {
    if (request.type === "cancel") {
      operations.get(request.id)?.abort();
      return;
    }

    if (request.type === "open" || request.type === "open-decoder") {
      try {
        const lazPerf = await createLazPerf({
          locateFile: () => request.lazPerfWasmUrl,
        });
        if (request.type === "open") {
          source = await createCopcTileSource({
            source: rangeSource(request.source),
            lazPerf,
          });
          scope.postMessage({
            type: "opened",
            id: request.id,
            metadata: source.metadata(),
            state: source.decodeState,
          });
        } else {
          decoder = createCopcTileDecoder({
            source: rangeSource(request.source),
            lazPerf,
            state: request.state,
          });
          scope.postMessage({ type: "decoder-opened", id: request.id });
        }
      } catch (error) {
        respondError(request.id, error);
      }
      return;
    }

    const operation = new AbortController();
    operations.set(request.id, operation);
    try {
      if (request.type === "nodes") {
        if (source === null) throw new Error("COPC worker holds no hierarchy");
        const nodes = await source.nodes(request.key, {
          signal: operation.signal,
        });
        operation.signal.throwIfAborted();
        const entries: (readonly [string, CopcNodeEntry])[] = [];
        for (const node of nodes) {
          const entry = node.pageRef ? undefined : source.entry(node.key);
          if (entry !== undefined) entries.push([keyToString(node.key), entry]);
        }
        scope.postMessage({ type: "nodes", id: request.id, nodes, entries });
      } else {
        const tile = await loadTile(request, { signal: operation.signal });
        operation.signal.throwIfAborted();
        const transfer: Transferable[] = [tile.positions.buffer];
        if (tile.rgb !== undefined) transfer.push(tile.rgb.buffer);
        scope.postMessage({ type: "tile", id: request.id, tile }, transfer);
      }
    } catch (error) {
      respondError(request.id, error);
    } finally {
      operations.delete(request.id);
    }
  };

  scope.addEventListener("message", (event) => {
    void run(event.data);
  });
};
