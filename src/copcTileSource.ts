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

import { Copc, type Hierarchy } from "copc";

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
import { orderTileForProgressiveDrawing } from "./progressiveOrder";
import type {
  LoadOptions,
  NodeInfo,
  TileData,
  TileSource,
  TileSourceMetadata,
} from "./tileSource";

export type RangeGetter = (
  begin: number,
  end: number,
  signal?: AbortSignal,
) => Promise<Uint8Array>;

export type CopcTileSourceOptions = {
  /** URL fetched via HTTP Range requests, or a custom byte-range getter. */
  source: string | RangeGetter;
  /** An initialized laz-perf module, used by worker-hosted sources. */
  lazPerf?: NonNullable<
    Parameters<typeof Copc.loadPointDataView>[3]
  >["lazPerf"];
};

const ABORT_CHECK_STRIDE = 4096;
const HTTP_RANGE_ATTEMPTS = 3;
const HTTP_RETRY_BASE_MS = 100;

const delay = (
  milliseconds: number,
  signal: AbortSignal | undefined,
): Promise<void> =>
  new Promise((resolve, reject) => {
    signal?.throwIfAborted();
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, milliseconds);
    const onAbort = (): void => {
      clearTimeout(timer);
      reject(signal!.reason);
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });

/**
 * Length of a shorter response that is proven to end at the file's EOF.
 *
 * Range servers clamp an oversized final range to the resource length. That
 * is complete data, not a truncated transfer, but only `Content-Range` can
 * distinguish it from a connection that ended early. Unknown totals and
 * partial subranges therefore remain failures.
 */
const eofClampedLength = (
  contentRange: string | null,
  requestedBegin: number,
  requestedEnd: number,
): number | null => {
  const match = /^bytes\s+(\d+)-(\d+)\/(\d+)$/i.exec(
    contentRange?.trim() ?? "",
  );
  if (match === null) return null;
  const first = Number(match[1]);
  const last = Number(match[2]);
  const total = Number(match[3]);
  if (![first, last, total].every(Number.isSafeInteger)) return null;
  if (
    first !== requestedBegin ||
    last < first ||
    last + 1 !== total ||
    last >= requestedEnd - 1
  ) {
    return null;
  }
  return last - first + 1;
};

/**
 * HTTP byte ranges with bounded recovery from transport and server failures.
 *
 * `copc`'s HTTP getter makes one unchecked fetch. A transient S3 connection
 * failure therefore tears down an otherwise healthy tile and reaches the UI
 * as the browser's context-free "TypeError: Failed to fetch". Range reads are
 * idempotent, so retry network failures, truncated bodies, throttling and 5xx
 * responses here. Permanent HTTP responses fail immediately, and every final
 * error names the byte range and source that could not be read.
 */
const httpRangeGetter =
  (url: string): RangeGetter =>
  async (begin, end, signal) => {
    signal?.throwIfAborted();
    if (begin < 0 || end < 0 || begin > end) {
      throw new Error(`Invalid byte range ${begin}-${end}`);
    }
    const expectedLength = end - begin;
    if (expectedLength === 0) return new Uint8Array();
    let lastError: unknown = null;
    let attemptsMade = 0;

    for (let attempt = 1; attempt <= HTTP_RANGE_ATTEMPTS; attempt += 1) {
      attemptsMade = attempt;
      let retryable = true;
      try {
        const response = await fetch(url, {
          headers: { Range: `bytes=${begin}-${end - 1}` },
          ...(signal === undefined ? {} : { signal }),
        });
        if (response.status !== 206) {
          retryable =
            response.status === 408 ||
            response.status === 429 ||
            response.status >= 500;
          throw new Error(
            `HTTP ${response.status} ${response.statusText || "response"}`,
          );
        }
        const bytes = new Uint8Array(await response.arrayBuffer());
        const clampedLength = eofClampedLength(
          response.headers.get("content-range"),
          begin,
          end,
        );
        if (
          bytes.byteLength !== expectedLength &&
          bytes.byteLength !== clampedLength
        ) {
          throw new Error(
            `Expected ${expectedLength} bytes, received ${bytes.byteLength}`,
          );
        }
        signal?.throwIfAborted();
        return bytes;
      } catch (error) {
        signal?.throwIfAborted();
        if (error instanceof Error && error.name === "AbortError") throw error;
        lastError = error;
        if (!retryable || attempt === HTTP_RANGE_ATTEMPTS) break;
        await delay(HTTP_RETRY_BASE_MS * 2 ** (attempt - 1), signal);
      }
    }

    const detail =
      lastError instanceof Error ? lastError.message : String(lastError);
    throw new Error(
      `Could not fetch bytes ${begin}-${end - 1} from ${url} after ` +
        `${attemptsMade} ${attemptsMade === 1 ? "attempt" : "attempts"}: ${detail}`,
      { cause: lastError },
    );
  };

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
 * so this wrapper closes over one and hands it to every byte-range read. The
 * built-in HTTP getter cancels both an in-flight fetch and its retry delay; a
 * custom getter can honor the optional third argument too. Cancellation is
 * also checked around each copc call and, for tiles, every
 * `ABORT_CHECK_STRIDE` decoded points. The controller still counts physical
 * operations rather than logical requests because custom getters may treat
 * cancellation as advisory and decoding already under way remains synchronous.
 */
const abortableGetter = (
  getter: RangeGetter,
  signal: AbortSignal | undefined,
): RangeGetter =>
  signal === undefined
    ? getter
    : (begin, end) => {
        signal.throwIfAborted();
        return getter(begin, end, signal);
      };

/**
 * Bits to drop from every RGB channel of this asset, sampled once at open.
 *
 * LAS keeps RGB in 16-bit fields whatever the real depth and no header field
 * states which, so the depth has to be observed. The root node is the
 * cloud-wide coarse sample: any channel above 255 there means the file really
 * does use the full 16-bit range, so all of its tiles shift by 8. Sampling per
 * node instead would leave a dark node unshifted beside a shifted bright
 * neighbour and band the render, which is the whole reason this is decided
 * once. A root that cannot be read, that holds no points, or that is simply
 * darker than the rest of the cloud falls back to no shift: shifting a
 * genuinely 8-bit cloud would crush every channel to zero, while leaving a
 * 16-bit cloud unshifted clips its channels to white (see `clipToByte`) — a
 * washed-out cloud, still recognisably the survey. The `reg-ui` HTTP tile
 * service samples by the same rule, so both transports hand the renderer
 * identical bytes wherever that sample is right, which is every cloud whose
 * root carries its brightest decade.
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

/**
 * A shifted channel narrowed to what a Uint8Array can hold.
 *
 * The asset-wide shift is a sample, and a sample can miss — a dark root over a
 * 16-bit cloud decides no shift, and then a deeper node's real channels arrive
 * far above 255. Storing those straight into the byte array would wrap them mod
 * 256, so a smooth 16-bit gradient lands as pseudo-random noise that reads as a
 * decoder bug rather than a too-bright cloud. Clipping keeps the miss legible
 * and the gradient monotonic.
 */
const clipToByte = (channel: number): number => (channel > 255 ? 255 : channel);

/** COPC sources resolve their metadata asynchronously (header + info VLR). */
export const createCopcTileSource = async (
  options: CopcTileSourceOptions,
): Promise<TileSource> => {
  const getter: RangeGetter =
    typeof options.source === "string"
      ? httpRangeGetter(options.source)
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
      rootView = await Copc.loadPointDataView(getter, copc, rootNode, {
        lazPerf: options.lazPerf,
      });
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
          { lazPerf: options.lazPerf },
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
          rgb[o] = clipToByte(getR!(i) >> shift);
          rgb[o + 1] = clipToByte(getG!(i) >> shift);
          rgb[o + 2] = clipToByte(getB!(i) >> shift);
        }
      }

      return orderTileForProgressiveDrawing(
        { origin, positions, rgb, pointCount },
        keyString,
      );
    },
  };
};
