import { beforeEach, describe, expect, it, vi } from 'vitest';

import { createRendererAdapter } from './rendererAdapter';
import type { TileData } from './tileSource';
import {
  actorInstances,
  mapperInstances,
  polyDataInstances,
  resetStubs,
} from '../test/stubs/vtkStub';

const tile = (
  origin: [number, number, number],
  pointCount = 2,
  withRgb = true,
): TileData => ({
  origin,
  positions: new Float32Array(pointCount * 3),
  rgb: withRgb ? new Uint8Array(pointCount * 3).fill(200) : undefined,
  pointCount,
});

const KEY_A = { level: 1, x: 0, y: 0, z: 0 };
const KEY_B = { level: 1, x: 1, y: 0, z: 0 };

const makeAdapter = (options?: { pointSize?: number; visible?: boolean }) => {
  const renderer = { addActor: vi.fn(), removeActor: vi.fn() };
  const scheduleRender = vi.fn();
  const adapter = createRendererAdapter({
    renderer,
    scheduleRender,
    ...options,
  });
  return { adapter, renderer, scheduleRender };
};

beforeEach(resetStubs);

describe('createRendererAdapter', () => {
  it('creates one polydata/mapper/actor per added tile', () => {
    const { adapter, renderer, scheduleRender } = makeAdapter({
      pointSize: 3,
    });
    const data = tile([10, 20, 30]);
    adapter.applyBatch({ added: [{ key: KEY_A, tile: data }], removed: [] });

    expect(adapter.tileCount()).toBe(1);
    expect(renderer.addActor).toHaveBeenCalledTimes(1);
    expect(scheduleRender).toHaveBeenCalledTimes(1);

    expect(polyDataInstances[0]!.points).toBe(data.positions);
    expect((polyDataInstances[0]!.scalars as { values: unknown }).values).toBe(
      data.rgb,
    );
    expect(mapperInstances[0]!.inputData).toBe(polyDataInstances[0]);
    expect(mapperInstances[0]!.static).toBe(true);
    expect(actorInstances[0]!.pointSize).toBe(3);
    // Identity base: the tile matrix is a plain translation to the origin.
    expect(actorInstances[0]!.userMatrix!.slice(12, 15)).toEqual([10, 20, 30]);
  });

  it('skips scalars for tiles without RGB', () => {
    const { adapter } = makeAdapter();
    adapter.applyBatch({
      added: [{ key: KEY_A, tile: tile([0, 0, 0], 2, false) }],
      removed: [],
    });
    expect(polyDataInstances[0]!.scalars).toBeNull();
  });

  it('removes and releases tile resources', () => {
    const { adapter, renderer } = makeAdapter();
    adapter.applyBatch({ added: [{ key: KEY_A, tile: tile([0, 0, 0]) }], removed: [] });
    adapter.applyBatch({ added: [], removed: [KEY_A] });

    expect(adapter.tileCount()).toBe(0);
    expect(renderer.removeActor).toHaveBeenCalledTimes(1);
    expect(actorInstances[0]!.deleted).toBe(true);
    expect(mapperInstances[0]!.deleted).toBe(true);
    expect(polyDataInstances[0]!.deleted).toBe(true);
  });

  it('composes the base matrix with each tile origin', () => {
    const { adapter } = makeAdapter();
    adapter.applyBatch({ added: [{ key: KEY_A, tile: tile([1, 2, 3]) }], removed: [] });

    // base = translation by (10, 20, 30): composed translation adds up.
    adapter.setBaseMatrix([
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 20, 30, 1,
    ]);
    expect(actorInstances[0]!.userMatrix!.slice(12, 15)).toEqual([11, 22, 33]);

    // base with a scale of 2: rotation/scale part multiplies the origin.
    adapter.setBaseMatrix([
      2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1,
    ]);
    expect(actorInstances[0]!.userMatrix!.slice(12, 15)).toEqual([2, 4, 6]);
    expect(actorInstances[0]!.userMatrix![0]).toBe(2);
  });

  it('fans out point size and visibility to every tile actor', () => {
    const { adapter, scheduleRender } = makeAdapter();
    adapter.applyBatch({
      added: [
        { key: KEY_A, tile: tile([0, 0, 0]) },
        { key: KEY_B, tile: tile([1, 0, 0]) },
      ],
      removed: [],
    });

    scheduleRender.mockClear();
    adapter.setPointSize(7);
    adapter.setVisible(false);
    expect(actorInstances.map((a) => a.pointSize)).toEqual([7, 7]);
    expect(actorInstances.map((a) => a.visibility)).toEqual([false, false]);
    expect(scheduleRender).toHaveBeenCalledTimes(2);

    // No-op updates do not schedule renders.
    scheduleRender.mockClear();
    adapter.setPointSize(7);
    adapter.setVisible(false);
    expect(scheduleRender).not.toHaveBeenCalled();
  });

  it('new tiles inherit the current base matrix, size, and visibility', () => {
    const { adapter } = makeAdapter();
    adapter.setBaseMatrix([
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 100, 0, 0, 1,
    ]);
    adapter.setPointSize(5);
    adapter.setVisible(false);
    adapter.applyBatch({ added: [{ key: KEY_A, tile: tile([1, 0, 0]) }], removed: [] });

    expect(actorInstances[0]!.userMatrix!.slice(12, 15)).toEqual([101, 0, 0]);
    expect(actorInstances[0]!.pointSize).toBe(5);
    expect(actorInstances[0]!.visibility).toBe(false);
  });

  it('dispose releases everything and is idempotent', () => {
    const { adapter, renderer } = makeAdapter();
    adapter.applyBatch({
      added: [
        { key: KEY_A, tile: tile([0, 0, 0]) },
        { key: KEY_B, tile: tile([1, 0, 0]) },
      ],
      removed: [],
    });
    adapter.dispose();
    adapter.dispose();

    expect(adapter.tileCount()).toBe(0);
    expect(renderer.removeActor).toHaveBeenCalledTimes(2);
    expect(actorInstances.every((a) => a.deleted)).toBe(true);

    // Batches after dispose are ignored.
    adapter.applyBatch({ added: [{ key: KEY_A, tile: tile([0, 0, 0]) }], removed: [] });
    expect(adapter.tileCount()).toBe(0);
  });
});
