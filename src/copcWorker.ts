import { createLazPerf } from "laz-perf/lib/worker";

import { createCopcTileSource } from "./copcTileSource";
import type { RangeGetter } from "./copcTileSource";
import type {
  CopcWorkerError,
  CopcWorkerRequest,
  CopcWorkerResponse,
} from "./copcWorkerProtocol";
import type { TileSource } from "./tileSource";

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

/** Start the worker endpoint used by `createCopcWorkerTileSource`. */
export const serveCopcTileSourceWorker = (
  scope: WorkerScope = globalThis as unknown as WorkerScope,
): void => {
  let source: TileSource | null = null;
  const operations = new Map<number, AbortController>();

  const respondError = (id: number, error: unknown): void =>
    scope.postMessage({ type: "error", id, error: serializedError(error) });

  const run = async (request: CopcWorkerRequest): Promise<void> => {
    if (request.type === "cancel") {
      operations.get(request.id)?.abort();
      return;
    }

    if (request.type === "open") {
      try {
        const lazPerf = await createLazPerf({
          locateFile: () => request.lazPerfWasmUrl,
        });
        source = await createCopcTileSource({
          source:
            typeof request.source === "string"
              ? request.source
              : blobGetter(request.source),
          lazPerf,
        });
        scope.postMessage({
          type: "opened",
          id: request.id,
          metadata: source.metadata(),
        });
      } catch (error) {
        respondError(request.id, error);
      }
      return;
    }

    const operation = new AbortController();
    operations.set(request.id, operation);
    try {
      if (source === null) throw new Error("COPC worker is not open");
      if (request.type === "nodes") {
        const nodes = await source.nodes(request.key, {
          signal: operation.signal,
        });
        operation.signal.throwIfAborted();
        scope.postMessage({ type: "nodes", id: request.id, nodes });
      } else {
        const tile = await source.loadTile(request.key, {
          signal: operation.signal,
        });
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
