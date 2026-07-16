import { describe, expect, it, vi } from 'vitest';

import {
  PCT1_HEADER_BYTES,
  RevisionGoneError,
  createHttpTileSource,
  parsePct1,
} from './httpTileSource';
import type { TileSourceMetadata } from './tileSource';

interface Pct1Spec {
  origin: [number, number, number];
  positions: number[];
  rgb?: number[];
  magic?: string;
}

const makePct1 = ({
  origin,
  positions,
  rgb,
  magic = 'PCT1',
}: Pct1Spec): ArrayBuffer => {
  const pointCount = positions.length / 3;
  const bytes =
    PCT1_HEADER_BYTES + positions.length * 4 + (rgb ? rgb.length : 0);
  const buffer = new ArrayBuffer(bytes);
  const view = new DataView(buffer);
  for (let i = 0; i < 4; i += 1) view.setUint8(i, magic.charCodeAt(i));
  view.setUint32(4, pointCount, true);
  view.setUint32(8, rgb ? 1 : 0, true);
  view.setUint32(12, 0, true);
  view.setFloat64(16, origin[0], true);
  view.setFloat64(24, origin[1], true);
  view.setFloat64(32, origin[2], true);
  new Float32Array(buffer, PCT1_HEADER_BYTES, positions.length).set(positions);
  if (rgb) {
    new Uint8Array(
      buffer,
      PCT1_HEADER_BYTES + positions.length * 4,
      rgb.length,
    ).set(rgb);
  }
  return buffer;
};

const METADATA: TileSourceMetadata = {
  pointCount: 12,
  cube: { center: [0, 0, 0], halfSize: 8 },
  spacing: 1,
};

describe('parsePct1', () => {
  it('parses a golden payload', () => {
    const tile = parsePct1(
      makePct1({
        origin: [100.5, -7.25, 3],
        positions: [1, 2, 3, -4, 5, 6.5],
        rgb: [255, 200, 3, 0, 127, 128],
      }),
    );
    expect(tile.pointCount).toBe(2);
    expect(tile.origin).toEqual([100.5, -7.25, 3]);
    expect(Array.from(tile.positions)).toEqual([1, 2, 3, -4, 5, 6.5]);
    expect(Array.from(tile.rgb!)).toEqual([255, 200, 3, 0, 127, 128]);
  });

  it('parses a payload without RGB', () => {
    const tile = parsePct1(
      makePct1({ origin: [0, 0, 0], positions: [1, 2, 3] }),
    );
    expect(tile.pointCount).toBe(1);
    expect(tile.rgb).toBeUndefined();
  });

  it('rejects a wrong magic', () => {
    expect(() =>
      parsePct1(makePct1({ origin: [0, 0, 0], positions: [], magic: 'NOPE' })),
    ).toThrow(/magic/);
  });

  it('rejects truncated payloads', () => {
    const full = makePct1({
      origin: [0, 0, 0],
      positions: [1, 2, 3],
      rgb: [1, 2, 3],
    });
    expect(() => parsePct1(full.slice(0, 20))).toThrow(/too short/);
    expect(() => parsePct1(full.slice(0, full.byteLength - 2))).toThrow(
      /truncated/,
    );
  });
});

describe('createHttpTileSource', () => {
  it('fetches and parses hierarchy pages', async () => {
    const fetchImpl = vi.fn(async () =>
      new Response(
        JSON.stringify({
          nodes: {
            '0-0-0-0': {
              pointCount: 10,
              children: ['1-0-0-0'],
              page: null,
            },
            '1-0-0-0': { pointCount: 0, children: [], page: '1-0-0-0' },
          },
        }),
        { status: 200 },
      ),
    );
    const source = createHttpTileSource({
      endpoint: '/pointcloud/a/rev1',
      metadata: METADATA,
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });

    const nodes = await source.nodes({ level: 0, x: 0, y: 0, z: 0 });
    expect(fetchImpl).toHaveBeenCalledWith(
      '/pointcloud/a/rev1/hierarchy/0-0-0-0.json',
      undefined,
    );
    expect(nodes).toHaveLength(2);
    const root = nodes.find((n) => n.key.level === 0)!;
    expect(root.pointCount).toBe(10);
    expect(root.children).toEqual([{ level: 1, x: 0, y: 0, z: 0 }]);
    expect(root.pageRef).toBe(false);
    const page = nodes.find((n) => n.key.level === 1)!;
    expect(page.pageRef).toBe(true);
  });

  it('fetches and parses tiles, passing the abort signal', async () => {
    const payload = makePct1({
      origin: [5, 6, 7],
      positions: [0.5, -0.5, 0],
      rgb: [200, 5, 255],
    });
    const fetchImpl = vi.fn(async () => new Response(payload, { status: 200 }));
    const source = createHttpTileSource({
      endpoint: 'http://host/pc/a/rev1',
      metadata: METADATA,
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });

    const controller = new AbortController();
    const tile = await source.loadTile(
      { level: 1, x: 1, y: 0, z: 0 },
      { signal: controller.signal },
    );
    expect(fetchImpl).toHaveBeenCalledWith('http://host/pc/a/rev1/tile/1-1-0-0.bin', {
      signal: controller.signal,
    });
    expect(tile.origin).toEqual([5, 6, 7]);
    expect(Array.from(tile.rgb!)).toEqual([200, 5, 255]);
  });

  it('maps HTTP 410 to RevisionGoneError', async () => {
    const fetchImpl = vi.fn(async () => new Response(null, { status: 410 }));
    const source = createHttpTileSource({
      endpoint: '/pc/a/dead',
      metadata: METADATA,
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    await expect(
      source.loadTile({ level: 0, x: 0, y: 0, z: 0 }),
    ).rejects.toBeInstanceOf(RevisionGoneError);
  });

  it('rejects other HTTP errors with the status', async () => {
    const fetchImpl = vi.fn(async () => new Response(null, { status: 404 }));
    const source = createHttpTileSource({
      endpoint: '/pc/a/rev1',
      metadata: METADATA,
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    await expect(
      source.nodes({ level: 0, x: 0, y: 0, z: 0 }),
    ).rejects.toThrow(/404/);
  });
});
