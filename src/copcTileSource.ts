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
  type Bounds,
  type VoxelKey,
} from "./octree";
import type {
  LoadOptions,
  NodeInfo,
  TileData,
  TileSource,
  TileSourceMetadata,
} from "./tileSource";

export type RangeGetter = (begin: number, end: number) => Promise<Uint8Array>;

export type CopcTileSourceOptions = {
  /** URL fetched via HTTP Range requests, or a custom byte-range getter. */
  source: string | RangeGetter;
};

const ABORT_CHECK_STRIDE = 4096;

/** One node's decoded points, as the copc reader hands them back. */
type PointDataView = Awaited<ReturnType<typeof Copc.loadPointDataView>>;

/** LAS point-data record formats carrying Red/Green/Blue channels. */
const RGB_POINT_FORMATS = new Set([2, 3, 5, 7, 8, 10]);

/**
 * The data extent from the LAS header, or nothing when the header does not
 * carry a usable one. An unset extent reads as zeros and an inverted one as a
 * negative box; either would frame a host's camera on empty space, which is
 * worse than leaving it with the octree cube it would have used anyway.
 */
const headerBounds = ({ min, max }: Copc["header"]): Bounds | undefined => {
  const usable = min.every((low, axis) => {
    const high = max[axis]!;
    return Number.isFinite(low) && Number.isFinite(high) && high >= low;
  });
  if (!usable) return undefined;
  if (min.every((value, axis) => value === max[axis])) return undefined;
  return {
    min: [min[0], min[1], min[2]],
    max: [max[0], max[1], max[2]],
  };
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
        signal.aborted ? Promise.reject(signal.reason) : getter(begin, end);

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
 *
 * The root node comes from the page the source already read at open, and the
 * points it decodes to are kept for the first read of the root tile, so the
 * sample costs no hierarchy read and no second decode of the node.
 */
const rgbShiftOf = (view: PointDataView): number => {
  const getR = view.getter("Red");
  const getG = view.getter("Green");
  const getB = view.getter("Blue");
  for (let i = 0; i < view.pointCount; i += 1) {
    if (getR(i) > 255 || getG(i) > 255 || getB(i) > 255) return 8;
  }
  return 0;
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
    // The LAS header's own extent, which is the data's — unlike `info.cube`,
    // the octree's enclosing cube, whose centre for a wide, flat survey floats
    // well above anything in the file. Omitted rather than invented when the
    // header does not carry a usable one.
    bounds: headerBounds(copc.header),
  };
  const rootCube = {
    center: [minX + halfSize, minY + halfSize, minZ + halfSize] as const,
    halfSize,
  };

  const nodeMap = new Map<string, Hierarchy.Node>();
  const pageMap = new Map<string, Hierarchy.Page>();

  /** Where a key sits in the octree — the same derivation for nodes and pages. */
  const geometry = (key: VoxelKey) => ({
    bounds: nodeBounds(rootCube, key),
    spacing: pointSpacing(copc.info.spacing, key.level),
  });

  const describeSubtree = (subtree: Hierarchy.Subtree): NodeInfo[] => {
    const infos: NodeInfo[] = [];
    for (const [nodeKey, node] of Object.entries(subtree.nodes)) {
      if (node === undefined) continue;
      nodeMap.set(nodeKey, node);
      const key = keyFromString(nodeKey);
      infos.push({ key, pointCount: node.pointCount, ...geometry(key) });
    }
    for (const [pageKey, subPage] of Object.entries(subtree.pages)) {
      if (subPage === undefined) continue;
      pageMap.set(pageKey, subPage);
      const key = keyFromString(pageKey);
      infos.push({ key, pointCount: 0, ...geometry(key), pageRef: true });
    }
    return infos;
  };

  /**
   * The root page's entries, read at open and kept for the source's life.
   *
   * Only the root page, and only because it is the one page with several
   * readers at open: the RGB sample below needs the root node's byte range, a
   * host framing the scene reads the hierarchy, and the controller bootstraps
   * from it. Without this each of them paid for the same range read. Deeper
   * pages are read once each by the controller, which tracks what it holds, so
   * caching them would grow with the octree and buy nothing.
   */
  const rootKeyString = keyToString(ROOT_KEY);
  const rootPageInfos = describeSubtree(
    await Copc.loadHierarchyPage(getter, copc.info.rootHierarchyPage),
  );

  const hasRgb = RGB_POINT_FORMATS.has(copc.header.pointDataRecordFormat);
  /**
   * The root node's decoded points, held from the RGB sample until the first
   * read of the root tile takes them.
   *
   * The sample has to decode that node, and the controller's very first tile
   * request is for the same one, so without this a coloured cloud decoded its
   * root twice before drawing anything. Released on use; only a source whose
   * root is never drawn keeps it, and that node is the smallest in the file.
   */
  let rootView: PointDataView | null = null;
  let rgbShift = 0;
  const rootNode = nodeMap.get(rootKeyString);
  if (hasRgb && rootNode !== undefined && rootNode.pointCount > 0) {
    try {
      rootView = await Copc.loadPointDataView(getter, copc, rootNode);
      rgbShift = rgbShiftOf(rootView);
    } catch (error) {
      // Fall back to no shift, for the reason `rgbShiftOf` gives above.
      console.warn(
        "pointcloud-lod: could not sample root-node RGB, assuming 8-bit channels",
        error,
      );
      rootView = null;
    }
  }

  return {
    metadata: () => metadata,

    async nodes(key: VoxelKey, opts?: LoadOptions): Promise<NodeInfo[]> {
      const keyString = keyToString(key);
      const signal = opts?.signal;
      signal?.throwIfAborted();

      // A copy, so every caller owns the array it was handed even though the
      // root page's entries are shared.
      if (keyString === rootKeyString) return [...rootPageInfos];

      const page = pageMap.get(keyString);
      if (page === undefined) {
        throw new Error(`No hierarchy page rooted at ${keyString}`);
      }

      const subtree = await Copc.loadHierarchyPage(
        abortableGetter(getter, signal),
        page,
      );
      signal?.throwIfAborted();

      return describeSubtree(subtree);
    },

    async loadTile(key: VoxelKey, opts?: LoadOptions): Promise<TileData> {
      const keyString = keyToString(key);
      const node = nodeMap.get(keyString);
      if (node === undefined) {
        throw new Error(`Hierarchy not loaded for node ${keyString}`);
      }
      const signal = opts?.signal;
      signal?.throwIfAborted();

      let view: PointDataView;
      if (keyString === rootKeyString && rootView !== null) {
        view = rootView;
        rootView = null;
      } else {
        view = await Copc.loadPointDataView(
          abortableGetter(getter, signal),
          copc,
          node,
        );
      }
      signal?.throwIfAborted();

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

      const [ox, oy, oz] = origin;
      const shift = rgbShift;
      for (let i = 0, o = 0; i < pointCount; i += 1, o += 3) {
        if (signal !== undefined && i % ABORT_CHECK_STRIDE === 0) {
          signal.throwIfAborted();
        }
        positions[o] = getX(i) - ox;
        positions[o + 1] = getY(i) - oy;
        positions[o + 2] = getZ(i) - oz;
        if (rgb !== undefined) {
          rgb[o] = getR!(i) >> shift;
          rgb[o + 1] = getG!(i) >> shift;
          rgb[o + 2] = getB!(i) >> shift;
        }
      }

      return { origin, positions, rgb, pointCount };
    },
  };
};
