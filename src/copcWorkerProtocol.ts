import type { NodeInfo, TileData, TileSourceMetadata } from "./tileSource";
import type { VoxelKey } from "./octree";

export type CopcWorkerRequest =
  | {
      readonly type: "open";
      readonly id: number;
      readonly source: string | Blob;
      readonly lazPerfWasmUrl: string;
    }
  | { readonly type: "nodes"; readonly id: number; readonly key: VoxelKey }
  | { readonly type: "load-tile"; readonly id: number; readonly key: VoxelKey }
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
    }
  | {
      readonly type: "nodes";
      readonly id: number;
      readonly nodes: NodeInfo[];
    }
  | { readonly type: "tile"; readonly id: number; readonly tile: TileData }
  | {
      readonly type: "error";
      readonly id: number;
      readonly error: CopcWorkerError;
    };
