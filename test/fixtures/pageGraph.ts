/**
 * Synthetic multipage COPC-shaped hierarchy for the deterministic suite.
 *
 * Every node below the root is its own hierarchy page, so walking the tree
 * costs one `nodes()` call per node: a page graph both deeper and far wider
 * than any sane concurrency limit, which is exactly what the bounded hierarchy
 * scheduler has to survive. Nothing settles on its own — the test lands each
 * request — so "how much work is physically running right now" is an exact
 * number instead of a race.
 *
 * Cancellation defaults to advisory, the way a COPC range read behaves: the
 * signal is recorded, the operation keeps running, and the promise stays
 * pending until the test lands it. That is the case physical-concurrency
 * accounting exists for.
 *
 * A genuinely multipage COPC *file* is still required for the browser
 * acceptance test; this covers unit-level scheduling only.
 */

import {
  ROOT_KEY,
  childKeys,
  keyFromString,
  keyToString,
  nodeBounds,
  pointSpacing,
  type Cube,
  type VoxelKey,
} from "../../src/octree";
import type {
  LoadNodesOptions,
  LoadTileOptions,
  NodeInfo,
  TileData,
  TileSource,
  TileSourceMetadata,
} from "../../src/tileSource";

/** Unit cube centred on the origin: every node lands inside the NDC cube. */
export const ROOT_CUBE: Cube = { center: [0, 0, 0], halfSize: 0.5 };
const ROOT_SPACING = 0.1;

export interface PageGraphOptions {
  /** Levels below the root. Depth 3 gives 85 pages at branching 4. */
  readonly depth?: number;
  /** Children per interior node, taken from the eight octants in order. */
  readonly branching?: number;
  /** Points carried by every node. */
  readonly pointsPerNode?: number;
  /**
   * `"advisory"` (default) models the COPC getter: `abort()` is recorded but
   * the promise stays pending until landed. `"immediate"` models a real fetch
   * signal and rejects as soon as the signal fires.
   */
  readonly cancellation?: "advisory" | "immediate";
}

export interface PendingRequest {
  /** Key of the page or tile this operation is reading. */
  readonly key: string;
  /** Whether the controller cancelled it; advisory work runs on regardless. */
  readonly aborted: boolean;
  /** Whether the promise has settled — what physical concurrency counts. */
  readonly settled: boolean;
  /** Deliver this operation's payload. */
  land(): void;
  /** Fail this operation. */
  fail(error: Error): void;
}

export interface PageGraphSource {
  readonly source: TileSource;
  /** Every `nodes()` call, in the order the scheduler made them. */
  readonly pageCalls: string[];
  /** Every `loadTile()` call, in the order the scheduler made them. */
  readonly tileCalls: string[];
  /** Page operations that have not settled: physically running work. */
  activePages(): PendingRequest[];
  /** Tile operations that have not settled: physically running work. */
  activeTiles(): PendingRequest[];
  /** Land every outstanding page operation. */
  landPages(): void;
  /** Land every outstanding tile operation. */
  landTiles(): void;
  /** Page keys in the graph, breadth first (the root page included). */
  pageKeys(): string[];
}

interface MutableRequest extends PendingRequest {
  aborted: boolean;
  settled: boolean;
}

export const createPageGraphSource = (
  options: PageGraphOptions = {},
): PageGraphSource => {
  const depth = options.depth ?? 3;
  const branching = options.branching ?? 4;
  const pointsPerNode = options.pointsPerNode ?? 10;
  const advisory = (options.cancellation ?? "advisory") === "advisory";

  const childrenOf = (key: VoxelKey): VoxelKey[] =>
    key.level >= depth ? [] : childKeys(key).slice(0, branching);

  const entry = (key: VoxelKey, pageRef: boolean): NodeInfo => ({
    key,
    // A page reference stands in for a node nobody has read yet, so it claims
    // no points; the page itself replaces it with the real entry.
    pointCount: pageRef ? 0 : pointsPerNode,
    bounds: nodeBounds(ROOT_CUBE, key),
    spacing: pointSpacing(ROOT_SPACING, key.level),
    ...(pageRef ? { pageRef: true } : { children: childrenOf(key) }),
  });

  /** One page: the node itself, plus a reference to each child's page. */
  const pagePayload = (key: VoxelKey): NodeInfo[] => [
    entry(key, false),
    ...childrenOf(key).map((child) => entry(child, true)),
  ];

  const pageCalls: string[] = [];
  const tileCalls: string[] = [];
  const pageRequests: MutableRequest[] = [];
  const tileRequests: MutableRequest[] = [];

  const pending = <T>(
    key: string,
    signal: AbortSignal | undefined,
    into: MutableRequest[],
    payload: () => T,
  ): Promise<T> =>
    new Promise<T>((resolve, reject) => {
      const abortError = (): Error => {
        const error = new Error("aborted");
        error.name = "AbortError";
        return error;
      };
      const request: MutableRequest = {
        key,
        aborted: false,
        settled: false,
        land() {
          if (request.settled) return;
          request.settled = true;
          resolve(payload());
        },
        fail(error) {
          if (request.settled) return;
          request.settled = true;
          reject(error);
        },
      };
      into.push(request);
      if (signal?.aborted === true) {
        request.aborted = true;
        if (!advisory) request.fail(abortError());
        return;
      }
      signal?.addEventListener("abort", () => {
        request.aborted = true;
        if (!advisory) request.fail(abortError());
      });
    });

  const source: TileSource = {
    metadata: (): TileSourceMetadata => ({ pointCount: pointsPerNode }),

    nodes(key: VoxelKey, opts?: LoadNodesOptions): Promise<NodeInfo[]> {
      const keyString = keyToString(key);
      pageCalls.push(keyString);
      return pending(keyString, opts?.signal, pageRequests, () =>
        pagePayload(key),
      );
    },

    loadTile(key: VoxelKey, opts?: LoadTileOptions): Promise<TileData> {
      const keyString = keyToString(key);
      tileCalls.push(keyString);
      return pending(keyString, opts?.signal, tileRequests, () => ({
        origin: nodeBounds(ROOT_CUBE, key).min,
        positions: new Float32Array(pointsPerNode * 3),
        pointCount: pointsPerNode,
      }));
    },
  };

  const outstanding = (requests: MutableRequest[]): PendingRequest[] =>
    requests.filter((request) => !request.settled);

  return {
    source,
    pageCalls,
    tileCalls,
    activePages: () => outstanding(pageRequests),
    activeTiles: () => outstanding(tileRequests),
    landPages: () => {
      for (const request of outstanding(pageRequests)) request.land();
    },
    landTiles: () => {
      for (const request of outstanding(tileRequests)) request.land();
    },
    pageKeys: () => {
      const keys: string[] = [];
      let level: VoxelKey[] = [ROOT_KEY];
      while (level.length > 0) {
        keys.push(...level.map(keyToString));
        level = level.flatMap(childrenOf);
      }
      return keys;
    },
  };
};

/** Level of a key string, for asserting the scheduler's coarse-first order. */
export const levelOf = (keyString: string): number =>
  keyFromString(keyString).level;
