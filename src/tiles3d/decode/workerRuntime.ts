import { decodeTileContent } from "./decoder";
import { TileDecodeError, TileUnsupportedExtensionError } from "./types";
import { buildTransferList } from "./transfer";
import type {
  DecodeWorkerErrorMessage,
  DecodeWorkerRequestMessage,
} from "./pool";

interface ClassicWorkerScope {
  onmessage: ((event: MessageEvent<unknown>) => void) | null;
  postMessage(message: unknown, transfer?: Transferable[]): void;
}

const errorMessage = (
  message: DecodeWorkerRequestMessage,
  error: unknown,
): DecodeWorkerErrorMessage => ({
  kind: "error",
  jobId: message.jobId,
  generation: message.generation,
  error:
    error instanceof TileDecodeError
      ? {
          name: error.name,
          message: error.message,
          tileUri: error.tileUri,
          stage: error.stage,
          reason: error.reason,
          ...(error instanceof TileUnsupportedExtensionError
            ? { extension: error.extension }
            : {}),
        }
      : {
          name: error instanceof Error ? error.name : "Error",
          message: error instanceof Error ? error.message : String(error),
        },
});

export const installDecodeWorker = (scope: ClassicWorkerScope): void => {
  scope.onmessage = (event) => {
    const value = event.data;
    if (
      !value ||
      typeof value !== "object" ||
      !("kind" in value) ||
      value.kind !== "decode" ||
      !("jobId" in value) ||
      typeof value.jobId !== "number" ||
      !("generation" in value) ||
      typeof value.generation !== "number" ||
      !("request" in value)
    ) {
      return;
    }
    const message = value as DecodeWorkerRequestMessage;
    void decodeTileContent(message.request).then(
      (result) => {
        scope.postMessage(
          {
            kind: "complete",
            jobId: message.jobId,
            generation: message.generation,
            result,
          },
          buildTransferList(result),
        );
      },
      (error: unknown) => scope.postMessage(errorMessage(message, error)),
    );
  };
};
