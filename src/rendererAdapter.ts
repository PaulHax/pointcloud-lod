/**
 * vtk.js renderer adapter: consumes controller tile batches and manages one
 * vtkPolyData + vtkPointGaussianMapper + vtkActor per resident tile.
 *
 * This is the only module in the library that imports '@kitware/vtk.js'.
 * The mapper renders each point as one gl.POINTS vertex (no cell topology),
 * so a tile's cost is exactly its point payload.
 *
 * The adapter never calls `renderWindow.render()`; every visual change goes
 * through the injected `scheduleRender`, which must coalesce (the host owns
 * render pacing — e.g. a shared-context integration).
 */

import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkPointGaussianMapper from '@kitware/vtk.js/Rendering/Core/PointGaussianMapper';

import type { TileBatch } from './controller';
import { keyToString, type Vec3 } from './octree';
import type { TileData } from './tileSource';

/** Column-major 4x4 multiply: out = a · b. */
const multiplyMat4 = (
  a: ArrayLike<number>,
  b: ArrayLike<number>,
): number[] => {
  const out = new Array<number>(16);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      let sum = 0;
      for (let k = 0; k < 4; k += 1) {
        sum += a[k * 4 + row]! * b[column * 4 + k]!;
      }
      out[column * 4 + row] = sum;
    }
  }
  return out;
};

const IDENTITY: readonly number[] = [
  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
];

const translation = (origin: Vec3): number[] => [
  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, origin[0], origin[1], origin[2], 1,
];

export interface RendererAdapterOptions {
  /** vtk.js renderer the tile actors are added to. */
  renderer: { addActor(actor: unknown): void; removeActor(actor: unknown): void };
  /** Coalescing render request (see module doc). */
  scheduleRender: () => void;
  /** Initial point size in pixels. Default 2. */
  pointSize?: number;
  /** Initial visibility. Default true. */
  visible?: boolean;
}

export interface RendererAdapter {
  /** Apply one controller batch (typically wired as `onTiles`). */
  applyBatch(batch: TileBatch): void;
  /**
   * Anchor transform (column-major 16 floats, e.g. the scene actor's
   * UserMatrix); each tile renders with `base · translate(tile.origin)`.
   */
  setBaseMatrix(matrix: ArrayLike<number> | null): void;
  setPointSize(pixels: number): void;
  setVisible(visible: boolean): void;
  tileCount(): number;
  /** Remove and release every tile actor. Idempotent. */
  dispose(): void;
}

interface TileActors {
  actor: any;
  mapper: any;
  polyData: any;
  origin: Vec3;
}

export const createRendererAdapter = (
  options: RendererAdapterOptions,
): RendererAdapter => {
  const { renderer, scheduleRender } = options;
  let pointSize = options.pointSize ?? 2;
  let visible = options.visible ?? true;
  let baseMatrix: ArrayLike<number> = IDENTITY;
  const tiles = new Map<string, TileActors>();
  let disposed = false;

  const tileMatrix = (origin: Vec3): number[] =>
    multiplyMat4(baseMatrix, translation(origin));

  const createTile = (tile: TileData): TileActors => {
    const polyData = vtkPolyData.newInstance();
    polyData.getPoints().setData(tile.positions, 3);
    if (tile.rgb !== undefined) {
      polyData.getPointData().setScalars(
        vtkDataArray.newInstance({
          name: 'RGB',
          values: tile.rgb,
          numberOfComponents: 3,
        }),
      );
    }
    const mapper = vtkPointGaussianMapper.newInstance();
    mapper.setInputData(polyData);
    mapper.setStatic?.(true);
    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);
    actor.getProperty().setPointSize(pointSize);
    actor.setVisibility(visible);
    actor.setUserMatrix(tileMatrix(tile.origin));
    return { actor, mapper, polyData, origin: tile.origin };
  };

  const releaseTile = (entry: TileActors): void => {
    renderer.removeActor(entry.actor);
    entry.actor.delete?.();
    entry.mapper.delete?.();
    entry.polyData.delete?.();
  };

  return {
    applyBatch(batch) {
      if (disposed) return;
      let changed = false;
      for (const key of batch.removed) {
        const entry = tiles.get(keyToString(key));
        if (entry === undefined) continue;
        tiles.delete(keyToString(key));
        releaseTile(entry);
        changed = true;
      }
      for (const { key, tile } of batch.added) {
        const keyString = keyToString(key);
        if (tiles.has(keyString)) continue;
        const entry = createTile(tile);
        tiles.set(keyString, entry);
        renderer.addActor(entry.actor);
        changed = true;
      }
      if (changed) scheduleRender();
    },

    setBaseMatrix(matrix) {
      if (disposed) return;
      baseMatrix = matrix ?? IDENTITY;
      for (const entry of tiles.values()) {
        entry.actor.setUserMatrix(tileMatrix(entry.origin));
      }
      scheduleRender();
    },

    setPointSize(pixels) {
      if (disposed || pixels === pointSize) return;
      pointSize = pixels;
      for (const entry of tiles.values()) {
        entry.actor.getProperty().setPointSize(pixels);
      }
      scheduleRender();
    },

    setVisible(nextVisible) {
      if (disposed || nextVisible === visible) return;
      visible = nextVisible;
      for (const entry of tiles.values()) {
        entry.actor.setVisibility(nextVisible);
      }
      scheduleRender();
    },

    tileCount() {
      return tiles.size;
    },

    dispose() {
      if (disposed) return;
      disposed = true;
      for (const entry of tiles.values()) releaseTile(entry);
      tiles.clear();
    },
  };
};
