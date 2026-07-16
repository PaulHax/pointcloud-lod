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

import { Copc, Getter, type Hierarchy } from 'copc';

import { keyFromString, keyToString, nodeCube, type VoxelKey } from './octree';
import type {
  LoadTileOptions,
  NodeInfo,
  TileData,
  TileSource,
  TileSourceMetadata,
} from './tileSource';

export type RangeGetter = (begin: number, end: number) => Promise<Uint8Array>;

export interface CopcTileSourceOptions {
  /** URL fetched via HTTP Range requests, or a custom byte-range getter. */
  source: string | RangeGetter;
}

const ABORT_CHECK_STRIDE = 4096;

const abortError = (): Error => {
  const error = new Error('The operation was aborted');
  error.name = 'AbortError';
  return error;
};

/** COPC sources resolve their metadata asynchronously (header + info VLR). */
export const createCopcTileSource = async (
  options: CopcTileSourceOptions,
): Promise<TileSource> => {
  const getter: RangeGetter =
    typeof options.source === 'string'
      ? Getter.http(options.source)
      : options.source;

  const copc = await Copc.create(getter);
  const [minX, minY, minZ, maxX] = copc.info.cube;
  const halfSize = (maxX - minX) / 2;
  const metadata: TileSourceMetadata = {
    pointCount: copc.header.pointCount,
    cube: {
      center: [minX + halfSize, minY + halfSize, minZ + halfSize],
      halfSize,
    },
    spacing: copc.info.spacing,
  };

  const nodeMap = new Map<string, Hierarchy.Node>();
  const pageMap = new Map<string, Hierarchy.Page>([
    ['0-0-0-0', copc.info.rootHierarchyPage],
  ]);

  return {
    metadata: () => metadata,

    async nodes(key: VoxelKey): Promise<NodeInfo[]> {
      const keyString = keyToString(key);
      const page = pageMap.get(keyString);
      if (page === undefined) {
        throw new Error(`No hierarchy page rooted at ${keyString}`);
      }
      const subtree = await Copc.loadHierarchyPage(getter, page);
      const infos: NodeInfo[] = [];
      for (const [nodeKey, node] of Object.entries(subtree.nodes)) {
        if (node === undefined) continue;
        nodeMap.set(nodeKey, node);
        infos.push({
          key: keyFromString(nodeKey),
          pointCount: node.pointCount,
        });
      }
      for (const [pageKey, subPage] of Object.entries(subtree.pages)) {
        if (subPage === undefined) continue;
        pageMap.set(pageKey, subPage);
        infos.push({
          key: keyFromString(pageKey),
          pointCount: 0,
          pageRef: true,
        });
      }
      return infos;
    },

    async loadTile(
      key: VoxelKey,
      opts?: LoadTileOptions,
    ): Promise<TileData> {
      const keyString = keyToString(key);
      const node = nodeMap.get(keyString);
      if (node === undefined) {
        throw new Error(`Hierarchy not loaded for node ${keyString}`);
      }
      const signal = opts?.signal;
      if (signal?.aborted) throw abortError();

      const view = await Copc.loadPointDataView(getter, copc, node);
      if (signal?.aborted) throw abortError();

      const pointCount = view.pointCount;
      const { center: origin } = nodeCube(metadata.cube, key);
      const getX = view.getter('X');
      const getY = view.getter('Y');
      const getZ = view.getter('Z');
      const hasRgb =
        view.dimensions['Red'] !== undefined &&
        view.dimensions['Green'] !== undefined &&
        view.dimensions['Blue'] !== undefined;

      const positions = new Float32Array(pointCount * 3);
      const rawRgb = hasRgb ? new Uint16Array(pointCount * 3) : null;
      const getR = hasRgb ? view.getter('Red') : null;
      const getG = hasRgb ? view.getter('Green') : null;
      const getB = hasRgb ? view.getter('Blue') : null;

      let maxChannel = 0;
      for (let i = 0; i < pointCount; i += 1) {
        if (signal !== undefined && i % ABORT_CHECK_STRIDE === 0 && signal.aborted) {
          throw abortError();
        }
        positions[i * 3] = getX(i) - origin[0];
        positions[i * 3 + 1] = getY(i) - origin[1];
        positions[i * 3 + 2] = getZ(i) - origin[2];
        if (rawRgb !== null) {
          const r = getR!(i);
          const g = getG!(i);
          const b = getB!(i);
          rawRgb[i * 3] = r;
          rawRgb[i * 3 + 1] = g;
          rawRgb[i * 3 + 2] = b;
          if (r > maxChannel) maxChannel = r;
          if (g > maxChannel) maxChannel = g;
          if (b > maxChannel) maxChannel = b;
        }
      }

      let rgb: Uint8Array | undefined;
      if (rawRgb !== null) {
        // LAS RGB is nominally 16-bit but often stored 8-bit; scale only
        // when any channel actually exceeds 8 bits.
        const shift = maxChannel > 255 ? 8 : 0;
        rgb = new Uint8Array(pointCount * 3);
        for (let i = 0; i < rgb.length; i += 1) {
          rgb[i] = rawRgb[i]! >> shift;
        }
      }

      return { origin, positions, rgb, pointCount };
    },
  };
};
