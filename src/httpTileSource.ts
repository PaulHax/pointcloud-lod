/**
 * TileSource over the revision-scoped HTTP tile protocol:
 *
 *   GET <endpoint>/hierarchy/<l-x-y-z>.json
 *   GET <endpoint>/tile/<l-x-y-z>.bin        (PCT1 payload)
 *
 * The endpoint embeds the asset revision, so responses are immutable and the
 * browser cache is the only cache needed. A 410 response means the revision
 * is gone (asset replaced or re-anchored); it surfaces as `RevisionGoneError`
 * so callers can drop state and wait for a fresh anchor.
 */

import { keyFromString, keyToString, type VoxelKey } from './octree';
import type {
  LoadTileOptions,
  NodeInfo,
  TileData,
  TileSource,
  TileSourceMetadata,
} from './tileSource';

/** The asset revision behind this source no longer exists on the server. */
export class RevisionGoneError extends Error {
  constructor(url: string) {
    super(`Revision gone: ${url}`);
    this.name = 'RevisionGoneError';
  }
}

/** PCT1 header size in bytes; Float32 positions start here. */
export const PCT1_HEADER_BYTES = 40;
const PCT1_MAGIC = 0x31544350; // 'PCT1' little-endian
const PCT1_FLAG_RGB = 1;

/**
 * Parse one PCT1 payload. Typed-array views are constructed directly over
 * `buffer` (the format keeps offsets aligned by design), so the returned
 * tile shares storage with the input.
 */
export const parsePct1 = (buffer: ArrayBuffer): TileData => {
  if (buffer.byteLength < PCT1_HEADER_BYTES) {
    throw new Error(`PCT1 payload too short: ${buffer.byteLength} bytes`);
  }
  const header = new DataView(buffer);
  if (header.getUint32(0, true) !== PCT1_MAGIC) {
    throw new Error('PCT1 payload has wrong magic');
  }
  const pointCount = header.getUint32(4, true);
  const flags = header.getUint32(8, true);
  const hasRgb = (flags & PCT1_FLAG_RGB) !== 0;
  const positionBytes = pointCount * 3 * 4;
  const rgbBytes = hasRgb ? pointCount * 3 : 0;
  const expected = PCT1_HEADER_BYTES + positionBytes + rgbBytes;
  if (buffer.byteLength < expected) {
    throw new Error(
      `PCT1 payload truncated: expected ${expected} bytes, got ${buffer.byteLength}`,
    );
  }
  const origin: [number, number, number] = [
    header.getFloat64(16, true),
    header.getFloat64(24, true),
    header.getFloat64(32, true),
  ];
  const positions = new Float32Array(buffer, PCT1_HEADER_BYTES, pointCount * 3);
  const rgb = hasRgb
    ? new Uint8Array(buffer, PCT1_HEADER_BYTES + positionBytes, pointCount * 3)
    : undefined;
  return { origin, positions, rgb, pointCount };
};

interface HierarchyEntryJson {
  readonly pointCount: number;
  readonly children: readonly string[];
  readonly page: string | null;
}

export interface HttpTileSourceOptions {
  /**
   * Base URL of one asset revision, absolute or page-relative, without a
   * trailing slash — e.g. `/pointcloud/<asset>/<revision>`.
   */
  endpoint: string;
  /** Dataset metadata (the scene anchor carries it; there is no endpoint). */
  metadata: TileSourceMetadata;
  /** Injectable fetch for tests; defaults to the global. */
  fetchImpl?: typeof fetch;
}

export const createHttpTileSource = (
  options: HttpTileSourceOptions,
): TileSource => {
  const { endpoint, metadata } = options;
  const fetchImpl = options.fetchImpl ?? fetch;

  const request = async (
    url: string,
    signal?: AbortSignal,
  ): Promise<Response> => {
    const response = await fetchImpl(url, signal ? { signal } : undefined);
    if (response.status === 410) throw new RevisionGoneError(url);
    if (!response.ok) {
      throw new Error(`Tile request failed (${response.status}): ${url}`);
    }
    return response;
  };

  return {
    metadata: () => metadata,

    async nodes(key: VoxelKey): Promise<NodeInfo[]> {
      const response = await request(
        `${endpoint}/hierarchy/${keyToString(key)}.json`,
      );
      const body = (await response.json()) as {
        nodes: Record<string, HierarchyEntryJson>;
      };
      return Object.entries(body.nodes).map(([keyString, entry]) => ({
        key: keyFromString(keyString),
        pointCount: entry.pointCount,
        children: entry.children.map(keyFromString),
        pageRef: entry.page !== null && entry.page === keyString,
      }));
    },

    async loadTile(
      key: VoxelKey,
      opts?: LoadTileOptions,
    ): Promise<TileData> {
      const response = await request(
        `${endpoint}/tile/${keyToString(key)}.bin`,
        opts?.signal,
      );
      return parsePct1(await response.arrayBuffer());
    },
  };
};
