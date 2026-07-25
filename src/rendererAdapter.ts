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

import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkPointGaussianMapper from "@kitware/vtk.js/Rendering/Core/PointGaussianMapper";

import type { TileBatch } from "./controller";
import { keyToString, type Vec3 } from "./octree";
import type { TileData } from "./tileSource";

/** Column-major 4x4 multiply: out = a · b. */
const multiplyMat4 = (a: ArrayLike<number>, b: ArrayLike<number>): number[] => {
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

const sameMatrix = (
  left: ArrayLike<number>,
  right: ArrayLike<number>,
): boolean => {
  if (left.length !== right.length) return false;
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) return false;
  }
  return true;
};

const translation = (origin: Vec3): number[] => [
  1,
  0,
  0,
  0,
  0,
  1,
  0,
  0,
  0,
  0,
  1,
  0,
  origin[0],
  origin[1],
  origin[2],
  1,
];

export interface RendererAdapterOptions {
  /** vtk.js renderer the tile actors are added to. */
  renderer: {
    addActor(actor: unknown): void;
    removeActor(actor: unknown): void;
  };
  /** Coalescing render request (see module doc). */
  scheduleRender: () => void;
  /** Initial uniform point diameter in CSS pixels. Default 2. */
  diameterCssPx?: number;
  /** Initial CSS-to-framebuffer scale. Default 1. */
  devicePixelRatio?: number;
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
  setPointDiameterCssPx(diameterCssPx: number): void;
  setDevicePixelRatio(devicePixelRatio: number): void;
  setVisible(visible: boolean): void;
  /** Set this adapter's allocation from the shared GPU-memory pool. */
  setResourceCeilingBytes(bytes: number): void;
  tileCount(): number;
  /** Renderer-owned resource and active-draw accounting. */
  stats(): RendererAdapterStats;
  /** Remove and release every tile actor. Idempotent. */
  dispose(): void;
}

export interface RendererAdapterStats {
  /** Tiles with live polydata, mapper, and actor resources. */
  readonly gpuResidentTiles: number;
  readonly gpuResidentPoints: number;
  /**
   * Tracked vertex/color buffer bytes plus a small per-tile object estimate;
   * driver allocation overhead is not observable through WebGL.
   */
  readonly gpuResidentBytes: number;
  /** Tiles and points whose actors currently participate in drawing. */
  readonly activeDrawTiles: number;
  readonly activeDrawPoints: number;
  readonly diameterCssPx: number;
  readonly devicePixelRatio: number;
  /** Upper bound before clipping/depth: circular submitted splat area. */
  readonly submittedSplatAreaDevicePx2: number;
}

interface TileActors {
  actor: any;
  mapper: any;
  polyData: any;
  origin: Vec3;
  pointCount: number;
  resourceBytes: number;
  positions: Float32Array;
  rgb: Uint8Array | undefined;
}

export const createRendererAdapter = (
  options: RendererAdapterOptions,
): RendererAdapter => {
  const { renderer, scheduleRender } = options;
  let diameterCssPx = options.diameterCssPx ?? 2;
  let devicePixelRatio = options.devicePixelRatio ?? 1;
  if (!Number.isFinite(diameterCssPx) || diameterCssPx <= 0) {
    throw new Error("diameterCssPx must be finite and > 0");
  }
  if (!Number.isFinite(devicePixelRatio) || devicePixelRatio <= 0) {
    throw new Error("devicePixelRatio must be finite and > 0");
  }
  let visible = options.visible ?? true;
  let baseMatrix: ArrayLike<number> = IDENTITY;
  const tiles = new Map<string, TileActors>();
  const pendingRelease = new Map<string, TileActors>();
  let resourceCeilingBytes = 256 * 1024 * 1024;
  let disposed = false;

  const tileMatrix = (origin: Vec3): number[] =>
    multiplyMat4(baseMatrix, translation(origin));

  const createTile = (tile: TileData): TileActors => {
    const polyData = vtkPolyData.newInstance();
    polyData.getPoints().setData(tile.positions, 3);
    if (tile.rgb !== undefined) {
      polyData.getPointData().setScalars(
        vtkDataArray.newInstance({
          name: "RGB",
          values: tile.rgb,
          numberOfComponents: 3,
        }),
      );
    }
    const mapper = vtkPointGaussianMapper.newInstance();
    mapper.setInputData(polyData);
    mapper.setStatic?.(true);
    // The actor property remains in CSS pixels. The custom dense-point mapper
    // multiplies it by scaleFactor before assigning physical gl_PointSize.
    mapper.setScaleFactor(devicePixelRatio);
    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);
    actor.getProperty().setPointSize(diameterCssPx);
    actor.setVisibility(visible);
    actor.setUserMatrix(tileMatrix(tile.origin));
    return {
      actor,
      mapper,
      polyData,
      origin: tile.origin,
      pointCount: tile.pointCount,
      resourceBytes:
        tile.positions.byteLength + (tile.rgb?.byteLength ?? 0) + 64,
      positions: tile.positions,
      rgb: tile.rgb,
    };
  };

  const releaseTile = (entry: TileActors): void => {
    renderer.removeActor(entry.actor);
    entry.actor.delete?.();
    entry.mapper.delete?.();
    entry.polyData.delete?.();
  };

  const resourceBytes = (): number => {
    let bytes = 0;
    for (const entry of tiles.values()) bytes += entry.resourceBytes;
    for (const entry of pendingRelease.values()) bytes += entry.resourceBytes;
    return bytes;
  };

  const trimPool = (additionalBytes = 0): void => {
    while (
      pendingRelease.size > 0 &&
      resourceBytes() + additionalBytes > resourceCeilingBytes
    ) {
      const [key, entry] = pendingRelease.entries().next().value as [
        string,
        TileActors,
      ];
      pendingRelease.delete(key);
      releaseTile(entry);
    }
  };

  return {
    applyBatch(batch) {
      if (disposed) return;
      let changed = false;
      for (const key of batch.removed) {
        const keyString = keyToString(key);
        const entry = tiles.get(keyString);
        if (entry === undefined) continue;
        tiles.delete(keyString);
        // Stop drawing the stale selection synchronously, but retain its
        // resources for bounded reuse. Releasing and reallocating a refined
        // cloud's actors at every interaction boundary creates input long tasks.
        entry.actor.setVisibility(false);
        pendingRelease.set(keyString, entry);
        changed = true;
      }
      for (const { key, tile } of batch.added) {
        if (!visible) continue;
        const keyString = keyToString(key);
        if (tiles.has(keyString)) continue;
        const stale = pendingRelease.get(keyString);
        if (stale !== undefined) {
          pendingRelease.delete(keyString);
          if (stale.positions === tile.positions && stale.rgb === tile.rgb) {
            stale.actor.getProperty().setPointSize(diameterCssPx);
            stale.mapper.setScaleFactor(devicePixelRatio);
            stale.actor.setUserMatrix(tileMatrix(stale.origin));
            stale.actor.setVisibility(visible);
            tiles.set(keyString, stale);
            changed = true;
            continue;
          }
          releaseTile(stale);
        }
        const tileResourceBytes =
          tile.positions.byteLength + (tile.rgb?.byteLength ?? 0) + 64;
        trimPool(tileResourceBytes);
        const entry = createTile(tile);
        tiles.set(keyString, entry);
        renderer.addActor(entry.actor);
        changed = true;
      }
      if (changed) scheduleRender();
    },

    setBaseMatrix(matrix) {
      if (disposed) return;
      const next = matrix ?? IDENTITY;
      if (sameMatrix(baseMatrix, next)) return;
      // Keep a value snapshot: vtk.js may mutate and reuse the same UserMatrix
      // object, and the next update still needs to detect that visual change.
      baseMatrix = Array.from(next);
      for (const entry of tiles.values()) {
        entry.actor.setUserMatrix(tileMatrix(entry.origin));
      }
      scheduleRender();
    },

    setPointDiameterCssPx(nextDiameterCssPx) {
      if (
        disposed ||
        !Number.isFinite(nextDiameterCssPx) ||
        nextDiameterCssPx <= 0 ||
        nextDiameterCssPx === diameterCssPx
      ) {
        return;
      }
      diameterCssPx = nextDiameterCssPx;
      for (const entry of tiles.values()) {
        entry.actor.getProperty().setPointSize(nextDiameterCssPx);
      }
      scheduleRender();
    },

    setDevicePixelRatio(nextDevicePixelRatio) {
      if (
        disposed ||
        !Number.isFinite(nextDevicePixelRatio) ||
        nextDevicePixelRatio <= 0 ||
        nextDevicePixelRatio === devicePixelRatio
      ) {
        return;
      }
      devicePixelRatio = nextDevicePixelRatio;
      for (const entry of tiles.values()) {
        entry.mapper.setScaleFactor(nextDevicePixelRatio);
      }
      scheduleRender();
    },

    setVisible(nextVisible) {
      if (disposed || nextVisible === visible) return;
      visible = nextVisible;
      if (!visible) {
        // Hidden clouds own no GPU resources. Controller removal batches that
        // follow this state change are harmless no-ops in the adapter.
        for (const entry of tiles.values()) {
          entry.actor.setVisibility(false);
          releaseTile(entry);
        }
        for (const entry of pendingRelease.values()) releaseTile(entry);
        tiles.clear();
        pendingRelease.clear();
      } else {
        for (const entry of tiles.values()) entry.actor.setVisibility(true);
      }
      scheduleRender();
    },

    setResourceCeilingBytes(bytes) {
      if (disposed || !Number.isFinite(bytes) || bytes < 0) return;
      resourceCeilingBytes = bytes;
      trimPool();
    },

    tileCount() {
      return tiles.size;
    },

    stats() {
      let gpuResidentPoints = 0;
      let gpuResidentBytes = 0;
      for (const entry of [...tiles.values(), ...pendingRelease.values()]) {
        gpuResidentPoints += entry.pointCount;
        gpuResidentBytes += entry.resourceBytes;
      }
      let activeDrawPoints = 0;
      for (const entry of tiles.values()) activeDrawPoints += entry.pointCount;
      return {
        gpuResidentTiles: tiles.size + pendingRelease.size,
        gpuResidentPoints,
        gpuResidentBytes,
        activeDrawTiles: visible ? tiles.size : 0,
        activeDrawPoints: visible ? activeDrawPoints : 0,
        diameterCssPx,
        devicePixelRatio,
        submittedSplatAreaDevicePx2: visible
          ? activeDrawPoints *
            (Math.PI / 4) *
            (diameterCssPx * devicePixelRatio) ** 2
          : 0,
      };
    },

    dispose() {
      if (disposed) return;
      disposed = true;
      for (const entry of tiles.values()) releaseTile(entry);
      for (const entry of pendingRelease.values()) releaseTile(entry);
      tiles.clear();
      pendingRelease.clear();
    },
  };
};
