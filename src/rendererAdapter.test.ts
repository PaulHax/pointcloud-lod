import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createRendererAdapter } from "./rendererAdapter";
import type { TileData } from "./tileSource";
import {
  actorInstances,
  mapperInstances,
  polyDataInstances,
  resetStubs,
} from "../test/stubs/vtkStub";

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

const makeAdapter = (options?: {
  diameterCssPx?: number;
  devicePixelRatio?: number;
  visible?: boolean;
}) => {
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
afterEach(() => vi.useRealTimers());

describe("createRendererAdapter", () => {
  it("creates one polydata/mapper/actor per added tile", () => {
    const { adapter, renderer, scheduleRender } = makeAdapter({
      diameterCssPx: 3,
      devicePixelRatio: 2,
    });
    const data = tile([10, 20, 30]);
    adapter.applyBatch({ added: [{ key: KEY_A, tile: data }], removed: [] });

    expect(adapter.stats()).toEqual({
      gpuResidentTiles: 1,
      gpuResidentPoints: 2,
      gpuResidentBytes: 94,
      activeDrawTiles: 1,
      activeDrawPoints: 2,
      diameterCssPx: 3,
      devicePixelRatio: 2,
    });
    expect(renderer.addActor).toHaveBeenCalledTimes(1);
    expect(scheduleRender).toHaveBeenCalledTimes(1);

    expect(polyDataInstances[0]!.points).toBe(data.positions);
    expect((polyDataInstances[0]!.scalars as { values: unknown }).values).toBe(
      data.rgb,
    );
    expect(mapperInstances[0]!.inputData).toBe(polyDataInstances[0]);
    expect(mapperInstances[0]!.static).toBe(true);
    expect(mapperInstances[0]!.scaleFactor).toBe(2);
    expect(actorInstances[0]!.pointSize).toBe(3);
    // Identity base: the tile matrix is a plain translation to the origin.
    expect(actorInstances[0]!.userMatrix!.slice(12, 15)).toEqual([10, 20, 30]);
  });

  it("skips scalars for tiles without RGB", () => {
    const { adapter } = makeAdapter();
    adapter.applyBatch({
      added: [{ key: KEY_A, tile: tile([0, 0, 0], 2, false) }],
      removed: [],
    });
    expect(polyDataInstances[0]!.scalars).toBeNull();
  });

  it("pools removed resources until memory pressure or teardown", () => {
    const { adapter, renderer } = makeAdapter();
    adapter.applyBatch({
      added: [{ key: KEY_A, tile: tile([0, 0, 0]) }],
      removed: [],
    });
    adapter.applyBatch({ added: [], removed: [KEY_A] });

    expect(actorInstances[0]!.visibility).toBe(false);
    expect(adapter.stats()).toMatchObject({
      gpuResidentTiles: 1,
      activeDrawTiles: 0,
      activeDrawPoints: 0,
    });
    expect(renderer.removeActor).not.toHaveBeenCalled();

    adapter.dispose();
    expect(renderer.removeActor).toHaveBeenCalledTimes(1);
    expect(actorInstances[0]!.deleted).toBe(true);
    expect(mapperInstances[0]!.deleted).toBe(true);
    expect(polyDataInstances[0]!.deleted).toBe(true);
    expect(adapter.stats().gpuResidentTiles).toBe(0);
  });

  it("evicts retired resources at the shared-memory ceiling", () => {
    const { adapter, renderer } = makeAdapter();
    const dataA = tile([0, 0, 0]);
    const dataB = tile([1, 0, 0]);
    adapter.setResourceCeilingBytes(94);
    adapter.applyBatch({
      added: [{ key: KEY_A, tile: dataA }],
      removed: [],
    });
    adapter.applyBatch({ added: [], removed: [KEY_A] });
    adapter.applyBatch({
      added: [{ key: KEY_B, tile: dataB }],
      removed: [],
    });

    expect(renderer.removeActor).toHaveBeenCalledTimes(1);
    expect(actorInstances[0]!.deleted).toBe(true);
    expect(adapter.stats()).toMatchObject({
      gpuResidentTiles: 1,
      gpuResidentBytes: 94,
      activeDrawTiles: 1,
    });
  });

  it("reuses hidden tiles when stationary detail returns", () => {
    const { adapter, renderer } = makeAdapter();
    const data = tile([0, 0, 0]);
    adapter.applyBatch({
      added: [{ key: KEY_A, tile: data }],
      removed: [],
    });

    adapter.applyBatch({ added: [], removed: [KEY_A] });
    expect(renderer.removeActor).not.toHaveBeenCalled();
    expect(adapter.stats()).toMatchObject({
      gpuResidentTiles: 1,
      activeDrawTiles: 0,
    });

    adapter.applyBatch({
      added: [{ key: KEY_A, tile: data }],
      removed: [],
    });
    expect(renderer.addActor).toHaveBeenCalledTimes(1);
    expect(renderer.removeActor).not.toHaveBeenCalled();
    expect(actorInstances).toHaveLength(1);
    expect(adapter.stats()).toMatchObject({
      gpuResidentTiles: 1,
      activeDrawTiles: 1,
    });
  });

  it("composes the base matrix with each tile origin", () => {
    const { adapter } = makeAdapter();
    adapter.applyBatch({
      added: [{ key: KEY_A, tile: tile([1, 2, 3]) }],
      removed: [],
    });

    // base = translation by (10, 20, 30): composed translation adds up.
    adapter.setBaseMatrix([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 20, 30, 1]);
    expect(actorInstances[0]!.userMatrix!.slice(12, 15)).toEqual([11, 22, 33]);

    // base with a scale of 2: rotation/scale part multiplies the origin.
    adapter.setBaseMatrix([2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1]);
    expect(actorInstances[0]!.userMatrix!.slice(12, 15)).toEqual([2, 4, 6]);
    expect(actorInstances[0]!.userMatrix![0]).toBe(2);
  });

  it("does not repaint for an unchanged base matrix", () => {
    const { adapter, scheduleRender } = makeAdapter();
    adapter.applyBatch({
      added: [{ key: KEY_A, tile: tile([1, 2, 3]) }],
      removed: [],
    });
    const matrix = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 20, 30, 1];
    adapter.setBaseMatrix(matrix);

    scheduleRender.mockClear();
    adapter.setBaseMatrix([...matrix]);
    expect(scheduleRender).not.toHaveBeenCalled();

    matrix[12] = 20;
    adapter.setBaseMatrix(matrix);
    expect(scheduleRender).toHaveBeenCalledTimes(1);
    expect(actorInstances[0]!.userMatrix!.slice(12, 15)).toEqual([21, 22, 33]);
  });

  it("fans out CSS diameter, DPR, and visibility to every tile actor", () => {
    const { adapter, scheduleRender } = makeAdapter();
    adapter.applyBatch({
      added: [
        { key: KEY_A, tile: tile([0, 0, 0]) },
        { key: KEY_B, tile: tile([1, 0, 0]) },
      ],
      removed: [],
    });

    scheduleRender.mockClear();
    adapter.setPointDiameterCssPx(7);
    adapter.setDevicePixelRatio(2);
    adapter.setVisible(false);
    expect(actorInstances.map((a) => a.pointSize)).toEqual([7, 7]);
    expect(mapperInstances.map((m) => m.scaleFactor)).toEqual([2, 2]);
    expect(actorInstances.map((a) => a.visibility)).toEqual([false, false]);
    expect(adapter.stats()).toMatchObject({
      gpuResidentTiles: 0,
      gpuResidentPoints: 0,
      activeDrawTiles: 0,
      activeDrawPoints: 0,
    });
    expect(scheduleRender).toHaveBeenCalledTimes(3);

    // No-op updates do not schedule renders.
    scheduleRender.mockClear();
    adapter.setPointDiameterCssPx(7);
    adapter.setDevicePixelRatio(2);
    adapter.setVisible(false);
    expect(scheduleRender).not.toHaveBeenCalled();
  });

  it("hidden adapters reject new renderer resources", () => {
    const { adapter } = makeAdapter();
    adapter.setBaseMatrix([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 100, 0, 0, 1]);
    adapter.setPointDiameterCssPx(5);
    adapter.setDevicePixelRatio(2);
    adapter.setVisible(false);
    adapter.applyBatch({
      added: [{ key: KEY_A, tile: tile([1, 0, 0]) }],
      removed: [],
    });

    expect(adapter.stats().gpuResidentTiles).toBe(0);
    expect(actorInstances).toHaveLength(0);
  });

  it("keeps apparent CSS diameter invariant across DPR", () => {
    const { adapter, scheduleRender } = makeAdapter({
      diameterCssPx: 2.5,
      devicePixelRatio: 1,
    });
    adapter.applyBatch({
      added: [{ key: KEY_A, tile: tile([0, 0, 0]) }],
      removed: [],
    });
    expect(actorInstances[0]!.pointSize).toBe(2.5);
    expect(mapperInstances[0]!.scaleFactor).toBe(1);

    scheduleRender.mockClear();
    adapter.setDevicePixelRatio(2);
    expect(actorInstances[0]!.pointSize).toBe(2.5);
    expect(mapperInstances[0]!.scaleFactor).toBe(2);
    expect(scheduleRender).toHaveBeenCalledTimes(1);
  });

  it("dispose releases everything and is idempotent", () => {
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

    expect(adapter.stats().gpuResidentTiles).toBe(0);
    expect(renderer.removeActor).toHaveBeenCalledTimes(2);
    expect(actorInstances.every((a) => a.deleted)).toBe(true);

    // Batches after dispose are ignored.
    adapter.applyBatch({
      added: [{ key: KEY_A, tile: tile([0, 0, 0]) }],
      removed: [],
    });
    expect(adapter.stats().gpuResidentTiles).toBe(0);
  });
});
