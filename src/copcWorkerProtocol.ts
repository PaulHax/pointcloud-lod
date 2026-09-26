import type { CopcDecodeState, CopcNodeEntry } from "./copcTileSource";
import type { NodeInfo, TileData, TileSourceMetadata } from "./tileSource";
import type { VoxelKey } from "./octree";

export type CopcWorkerRequest =
  | {
      readonly type: "open";
      readonly id: number;
      readonly source: string | Blob;
      readonly lazPerfWasmUrl: string;
    }
  | {
      readonly type: "open-decoder";
      readonly id: number;
      readonly source: string | Blob;
      readonly lazPerfWasmUrl: string;
      readonly state: CopcDecodeState;
    }
  | { readonly type: "nodes"; readonly id: number; readonly key: VoxelKey }
  | {
      readonly type: "load-tile";
      readonly id: number;
      readonly key: VoxelKey;
      /** Present when the worker is a decoder, which holds no hierarchy. */
      readonly entry?: CopcNodeEntry;
    }
  | { readonly type: "cancel"; readonly id: number };

export type CopcWorkerError = {
  readonly name: string;
  readonly message: string;
};

export type CopcWorkerResponse =
  | {
      readonly type: "opened";
      readonly id: number;
      readonly metadata: TileSourceMetadata;
      readonly state: CopcDecodeState;
    }
  | { readonly type: "decoder-opened"; readonly id: number }
  | {
      readonly type: "nodes";
      readonly id: number;
      readonly nodes: NodeInfo[];
      /** The file location of every node, as opposed to page, in `nodes`. */
      readonly entries: readonly (readonly [string, CopcNodeEntry])[];
    }
  | { readonly type: "tile"; readonly id: number; readonly tile: TileData }
  | {
      readonly type: "error";
      readonly id: number;
      readonly error: CopcWorkerError;
    };
