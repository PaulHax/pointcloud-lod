/**
 * End-to-end CopcTileSource tests against the committed fixture
 * (test/fixtures/fixture.copc.laz — 2000 points, full-range 16-bit RGB, two
 * hierarchy levels; regenerate with test/fixtures/generate_fixture.py).
 * Decoding runs through the real laz-perf WASM in node.
 */

import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { Getter } from 'copc';

import { createCopcTileSource } from './copcTileSource';
import { ROOT_KEY, keyToString, nodeCube } from './octree';

const FIXTURE = fileURLToPath(
  new URL('../test/fixtures/fixture.copc.laz', import.meta.url),
);

const makeSource = () =>
  createCopcTileSource({ source: Getter.file(FIXTURE) });

describe('createCopcTileSource', () => {
  it('reads metadata from the COPC info VLR', async () => {
    const source = await makeSource();
    const meta = source.metadata();
    expect(meta.pointCount).toBe(2000);
    // generate_fixture.py: ~40 m extent → cubic root cube anchored at the
    // actual data minima (random sampling shaves ~0.02 m off each side).
    expect(meta.cube.halfSize).toBeCloseTo(20, 1);
    expect(meta.cube.center[0]).toBeCloseTo(120, 1);
    expect(meta.cube.center[1]).toBeCloseTo(220, 1);
    expect(meta.cube.center[2]).toBeCloseTo(70, 1);
    expect(meta.spacing).toBeGreaterThan(0);
  });

  it('enumerates the hierarchy with point counts summing to the total', async () => {
    const source = await makeSource();
    const nodes = await source.nodes(ROOT_KEY);
    expect(nodes.length).toBeGreaterThanOrEqual(2); // capacity forces a split
    const total = nodes.reduce((sum, n) => sum + n.pointCount, 0);
    expect(total).toBe(2000);
    const root = nodes.find((n) => keyToString(n.key) === '0-0-0-0');
    expect(root).toBeDefined();
    expect(root!.pointCount).toBeGreaterThan(0);
  });

  it('loads tiles with tile-local Float32 positions and 8-bit RGB', async () => {
    const source = await makeSource();
    const nodes = await source.nodes(ROOT_KEY);
    const meta = source.metadata();

    for (const info of nodes.filter((n) => n.pointCount > 0).slice(0, 3)) {
      const tile = await source.loadTile(info.key);
      expect(tile.pointCount).toBe(info.pointCount);
      expect(tile.positions).toHaveLength(info.pointCount * 3);

      // Origin is the node cube center; locals stay inside the node cube.
      const cube = nodeCube(meta.cube, info.key);
      expect(tile.origin).toEqual(cube.center);
      for (let i = 0; i < tile.positions.length; i += 1) {
        expect(Math.abs(tile.positions[i]!)).toBeLessThanOrEqual(
          cube.halfSize + 0.01,
        );
      }

      // Absolute reconstruction lands in the generated extent.
      for (let i = 0; i < Math.min(tile.pointCount, 50); i += 1) {
        const x = tile.origin[0] + tile.positions[i * 3]!;
        const y = tile.origin[1] + tile.positions[i * 3 + 1]!;
        const z = tile.origin[2] + tile.positions[i * 3 + 2]!;
        expect(x).toBeGreaterThanOrEqual(99.9);
        expect(x).toBeLessThanOrEqual(140.1);
        expect(y).toBeGreaterThanOrEqual(199.9);
        expect(y).toBeLessThanOrEqual(240.1);
        expect(z).toBeGreaterThanOrEqual(49.9);
        expect(z).toBeLessThanOrEqual(90.1);
      }

      // Full-range 16-bit fixture colors arrive scaled to full-range 8-bit.
      expect(tile.rgb).toHaveLength(info.pointCount * 3);
      expect(Math.max(...tile.rgb!)).toBeGreaterThan(127);
    }
  });

  it('rejects when aborted before decode', async () => {
    const source = await makeSource();
    const nodes = await source.nodes(ROOT_KEY);
    const first = nodes.find((n) => n.pointCount > 0)!;
    const controller = new AbortController();
    controller.abort();
    await expect(
      source.loadTile(first.key, { signal: controller.signal }),
    ).rejects.toMatchObject({ name: 'AbortError' });
  });

  it('refuses tiles whose hierarchy page was never loaded', async () => {
    const source = await makeSource();
    await expect(
      source.loadTile({ level: 9, x: 0, y: 0, z: 0 }),
    ).rejects.toThrow(/Hierarchy not loaded/);
  });
});
