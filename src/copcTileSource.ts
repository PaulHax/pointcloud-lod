/**
 * TileSource reading COPC (Cloud-Optimized Point Cloud) files directly via
 * the 'copc' package — typically over HTTP Range requests, so any static
 * file host serves LOD point clouds with no tile server at all.
 *
 * Positions are delivered tile-local against each node's cube center (the
 * per-tile Float32 precision pattern); `TileData.origin` restores world
 * coordinates. Coordinates stay in whatever CRS the file uses — reprojection
 * is out of scope here.
 */

import { Copc, Getter, type Hierarchy } from "copc";

import {
  ROOT_KEY,
  keyFromString,
  keyToString,
  nodeBounds,
  nodeCube,
  pointSpacing,
  type VoxelKey,
} from "./octree";
import type {
  LoadNodesOptions,
  LoadTileOptions,
  NodeInfo,
  TileData,
  TileSource,
  TileSourceMetadata,
} from "./tileSource";

export type RangeGetter = (begin: number, end: number) => Promise<Uint8Array>;

export interface CopcTileSourceOptions {
  /** URL fetched via HTTP Range requests, or a custom byte-range getter. */
  source: string | RangeGetter;
}

const ABORT_CHECK_STRIDE = 4096;

/** LAS point-data record formats carrying Red/Green/Blue channels. */
const RGB_POINT_FORMATS = new Set([2, 3, 5, 7, 8, 10]);

const abortError = (): Error => {
  const error = new Error("The operation was aborted");
  error.name = "AbortError";
  return error;
};

/**
 * `Copc.loadPointDataView` and `Copc.loadHierarchyPage` take no AbortSignal,
 * so cancellation here is advisory: it is checked around each copc call and,
 * for tiles, every `ABORT_CHECK_STRIDE` decoded points. The byte-range getter
 * is ours though, and it is the one place a cancellation can reach physical
 * I/O — a canceled operation issues no further range reads. Reads already in
 * flight still run to completion, which is exactly why the controller counts
 * physical operations rather than logical requests. Should copc accept a
 * signal, hand it to the two calls below and this wrapper becomes redundant.
 */
const abortableGetter = (
  getter: RangeGetter,
  signal: AbortSignal | undefined,
): RangeGetter =>
  signal === undefined
    ? getter
    : (begin, end) =>
        signal.aborted ? Promise.reject(abortError()) : getter(begin, end);

/**
 * Bits to drop from every RGB channel of this asset, sampled once at open.
 *
 * LAS keeps RGB in 16-bit fields whatever the real depth and no header field
 * states which, so the depth has to be observed. The root node is the
 * cloud-wide coarse sample: any channel above 255 there means the file really
 * does use the full 16-bit range, so all of its tiles shift by 8. Sampling per
 * node instead would leave a dark node unshifted beside a shifted bright
 * neighbour and band the render, which is the whole reason this is decided
 * once. A root that cannot be read, or that holds no points, falls back to no
 * shift: shifting a genuinely 8-bit cloud would crush every channel to zero,
 * while leaving a 16-bit cloud unshifted only mangles colors that are still
 * visible. The `reg-ui` HTTP tile service samples by the same rule, so both
 * transports hand the renderer identical bytes.
 */
const sampleRgbShift = async (
  copc: Copc,
  getter: RangeGetter,
): Promise<number> => {
  try {
    // The root node's byte range lives in the root hierarchy page, and nothing
    // has read that page yet at open time, so the sample loads it itself.
    const subtree = await Copc.loadHierarchyPage(
      getter,
      copc.info.rootHierarchyPage,
    );
    const root = subtree.nodes[keyToString(ROOT_KEY)];
    if (root === undefined || root.pointCount === 0) return 0;

    const view = await Copc.loadPointDataView(getter, copc, root);
    const getR = view.getter("Red");
    const getG = view.getter("Green");
    const getB = view.getter("Blue");
    for (let i = 0; i < view.pointCount; i += 1) {
      if (getR(i) > 255 || getG(i) > 255 || getB(i) > 255) return 8;
    }
    return 0;
  } catch (error) {
    console.warn(
      "pointcloud-lod: could not sample root-node RGB, assuming 8-bit channels",
      error,
    );
    return 0;
  }
};

/** COPC sources resolve their metadata asynchronously (header + info VLR). */
export const createCopcTileSource = async (
  options: CopcTileSourceOptions,
): Promise<TileSource> => {
  const getter: RangeGetter =
    typeof options.source === "string"
      ? Getter.http(options.source)
      : options.source;

  const copc = await Copc.create(getter);
  const [minX, minY, minZ, maxX] = copc.info.cube;
  const halfSize = (maxX - minX) / 2;
  const metadata: TileSourceMetadata = {
    pointCount: copc.header.pointCount,
  };
  const rootCube = {
    center: [minX + halfSize, minY + halfSize, minZ + halfSize] as const,
    halfSize,
  };

  const hasRgb = RGB_POINT_FORMATS.has(copc.header.pointDataRecordFormat);
  const rgbShift = hasRgb ? await sampleRgbShift(copc, getter) : 0;

  const nodeMap = new Map<string, Hierarchy.Node>();
  const pageMap = new Map<string, Hierarchy.Page>([
    ["0-0-0-0", copc.info.rootHierarchyPage],
  ]);

  return {
    metadata: () => metadata,

    async nodes(key: VoxelKey, opts?: LoadNodesOptions): Promise<NodeInfo[]> {
      const keyString = keyToString(key);
      const page = pageMap.get(keyString);
      if (page === undefined) {
        throw new Error(`No hierarchy page rooted at ${keyString}`);
      }
      const signal = opts?.signal;
      if (signal?.aborted) throw abortError();

      const subtree = await Copc.loadHierarchyPage(
        abortableGetter(getter, signal),
        page,
      );
      if (signal?.aborted) throw abortError();

      const infos: NodeInfo[] = [];
      for (const [nodeKey, node] of Object.entries(subtree.nodes)) {
        if (node === undefined) continue;
        nodeMap.set(nodeKey, node);
        const nodeVoxelKey = keyFromString(nodeKey);
        infos.push({
          key: nodeVoxelKey,
          pointCount: node.pointCount,
          bounds: nodeBounds(rootCube, nodeVoxelKey),
          spacing: pointSpacing(copc.info.spacing, nodeVoxelKey.level),
        });
      }
      for (const [pageKey, subPage] of Object.entries(subtree.pages)) {
        if (subPage === undefined) continue;
        pageMap.set(pageKey, subPage);
        const pageVoxelKey = keyFromString(pageKey);
        infos.push({
          key: pageVoxelKey,
          pointCount: 0,
          bounds: nodeBounds(rootCube, pageVoxelKey),
          spacing: pointSpacing(copc.info.spacing, pageVoxelKey.level),
          pageRef: true,
        });
      }
      return infos;
    },

    async loadTile(key: VoxelKey, opts?: LoadTileOptions): Promise<TileData> {
      const keyString = keyToString(key);
      const node = nodeMap.get(keyString);
      if (node === undefined) {
        throw new Error(`Hierarchy not loaded for node ${keyString}`);
      }
      const signal = opts?.signal;
      if (signal?.aborted) throw abortError();

      const view = await Copc.loadPointDataView(
        abortableGetter(getter, signal),
        copc,
        node,
      );
      if (signal?.aborted) throw abortError();

      const pointCount = view.pointCount;
      const { center: origin } = nodeCube(rootCube, key);
      const getX = view.getter("X");
      const getY = view.getter("Y");
      const getZ = view.getter("Z");

      const positions = new Float32Array(pointCount * 3);
      // The asset-wide shift is already known, so channels land in their final
      // 8-bit form here — no intermediate 16-bit copy of the whole tile.
      const rgb = hasRgb ? new Uint8Array(pointCount * 3) : undefined;
      const getR = hasRgb ? view.getter("Red") : null;
      const getG = hasRgb ? view.getter("Green") : null;
      const getB = hasRgb ? view.getter("Blue") : null;

      for (let i = 0; i < pointCount; i += 1) {
        if (
          signal !== undefined &&
          i % ABORT_CHECK_STRIDE === 0 &&
          signal.aborted
        ) {
          throw abortError();
        }
        positions[i * 3] = getX(i) - origin[0];
        positions[i * 3 + 1] = getY(i) - origin[1];
        positions[i * 3 + 2] = getZ(i) - origin[2];
        if (rgb !== undefined) {
          rgb[i * 3] = getR!(i) >> rgbShift;
          rgb[i * 3 + 1] = getG!(i) >> rgbShift;
          rgb[i * 3 + 2] = getB!(i) >> rgbShift;
        }
      }

      return { origin, positions, rgb, pointCount };
    },
  };
};
