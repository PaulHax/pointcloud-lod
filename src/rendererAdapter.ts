/**
 * vtk.js renderer adapter: consumes controller tile batches and manages one
 * vtkPolyData + vtkPointGaussianMapper + vtkActor per submitted tile.
 *
 * This is the only module in the library that imports '@kitware/vtk.js'.
 * The mapper renders each point as one gl.POINTS vertex (no cell topology),
 * so a tile's cost is exactly its point payload.
 *
 * Lifecycle contract — the controller owns residency, the adapter owns actors:
 *
 * - `applyBatch` is the only entry point that changes which tiles exist. A
 *   batch is a disjoint delta against what the controller last submitted, so
 *   an addition for a key already on screen means the payload was replaced.
 * - `setVisible` is a draw switch and nothing else. Hiding keeps every actor
 *   and its GPU resources so showing again is a pure state restore, and
 *   batches keep applying while hidden (new actors are created invisible).
 *   Freeing a hidden cloud is the controller's decision — `setActive(false)`
 *   arrives here as removals — and `setResourceCeilingBytes` bounds whatever
 *   the reuse pool still holds.
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
import { finiteAbove, finiteNonNegative } from "./numeric";
import { keyToString, type Vec3 } from "./octree";
import { tileBytes, type TileData } from "./tileSource";

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

export type RendererAdapterOptions = {
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
  /** Initial visibility. Default true. Tiles added while hidden stay hidden. */
  visible?: boolean;
};

export type RendererAdapter = {
  /**
   * Apply one controller batch (typically wired as `onTiles`). Batches apply
   * whether or not the adapter is visible; an addition for a key already on
   * screen replaces that tile's payload.
   */
  applyBatch(batch: TileBatch): void;
  /**
   * Anchor transform (column-major 16 floats, e.g. the scene actor's
   * UserMatrix); each tile renders with `base · translate(tile.origin)`.
   */
  setBaseMatrix(matrix: ArrayLike<number> | null): void;
  setPointDiameterCssPx(diameterCssPx: number): void;
  setDevicePixelRatio(devicePixelRatio: number): void;
  /**
   * Draw switch only: hiding keeps every actor and its GPU resources, so
   * showing again restores exactly the submitted set — including tiles that
   * arrived while hidden. Releasing a hidden cloud's residency is the
   * controller's call (`setActive(false)`, which arrives as removals).
   */
  setVisible(visible: boolean): void;
  /** Set this adapter's allocation from the shared GPU-memory pool. */
  setResourceCeilingBytes(bytes: number): void;
  /** Renderer-owned resource and draw accounting. */
  stats(): RendererAdapterStats;
  /**
   * Diagnostics: the keys this adapter holds actors for, and the keys whose
   * actors are pooled awaiting reuse or release. `submitted` must match the
   * controller's submitted set exactly; anything in `pooled` is off screen and
   * owned by nothing but the pool.
   */
  activeKeys(): { readonly submitted: string[]; readonly pooled: string[] };
  /**
   * Remove and release every tile actor, submitted and pooled. Idempotent.
   *
   * Disposing the controller is NOT enough on its own. Its teardown hands back
   * the submitted set as removals, which this adapter answers by moving those
   * actors into the reuse pool — so a host that disposes only the controller
   * leaves up to `resourceCeilingBytes` of GPU resources alive with nothing
   * left that could ever ask for them again. Both objects must be disposed,
   * and the pair is what releases a cloud.
   */
  dispose(): void;
};

/**
 * Three disjoint questions, three sets of fields: what the controller has
 * submitted, what the reuse pool still holds, and what actually draws.
 * `gpuResident*` is the sum of the first two — the resources this adapter
 * owns, which is what a memory ceiling has to work from.
 */
export type RendererAdapterStats = {
  /** Tiles matching the controller's submitted set; drawn while visible. */
  readonly submittedTiles: number;
  readonly submittedPoints: number;
  readonly submittedBytes: number;
  /** Retired tiles retained for same-key reuse; never drawn. */
  readonly pooledTiles: number;
  readonly pooledPoints: number;
  readonly pooledBytes: number;
  /** Submitted plus pooled: everything holding polydata/mapper/actor state. */
  readonly gpuResidentTiles: number;
  readonly gpuResidentPoints: number;
  /**
   * Tracked vertex/color buffer bytes plus a small per-tile object estimate;
   * driver allocation overhead is not observable through WebGL.
   *
   * These are the same ArrayBuffers the controller reports under
   * `decodedBytes`, so adding the two double-counts the submitted set. The
   * only part of this that the controller does not already count is
   * `pooledBytes` — retired tiles it has released and this adapter still
   * holds.
   */
  readonly gpuResidentBytes: number;
  /**
   * The pool ceiling. Only the pool is trimmed to it, so submitted tiles alone
   * may exceed it — but then the pool is empty.
   */
  readonly resourceCeilingBytes: number;
  /** What participates in drawing: the submitted set, or nothing while hidden. */
  readonly drawnTiles: number;
  readonly drawnPoints: number;
  readonly visible: boolean;
  readonly diameterCssPx: number;
  readonly devicePixelRatio: number;
};

type TileActors = {
  actor: any;
  mapper: any;
  polyData: any;
  tile: TileData;
  resourceBytes: number;
};

export const createRendererAdapter = (
  options: RendererAdapterOptions,
): RendererAdapter => {
  const { renderer, scheduleRender } = options;
  let diameterCssPx = finiteAbove(
    "diameterCssPx",
    options.diameterCssPx ?? 2,
    0,
  );
  let devicePixelRatio = finiteAbove(
    "devicePixelRatio",
    options.devicePixelRatio ?? 1,
    0,
  );
  let visible = options.visible ?? true;
  let baseMatrix: ArrayLike<number> = IDENTITY;
  // A key lives in at most one of the two: removal moves its entry from
  // `tiles` to `pendingRelease`, and a re-addition either takes that entry
  // back or releases it. Nothing else may hold a reference to an entry, so
  // teardown releases each actor exactly once.
  const tiles = new Map<string, TileActors>();
  const pendingRelease = new Map<string, TileActors>();
  let resourceCeilingBytes = 256 * 1024 * 1024;
  let disposed = false;

  // Running totals rather than a walk per question. Trimming asks how many
  // bytes are held once per evicted entry, so re-summing both maps there made
  // shedding a large pool quadratic in its size — and the pool is largest
  // exactly when a cloud has just been deactivated and the ceiling dropped to
  // nothing, which is when the walk is longest and the answer needed soonest.
  // Every mutation of either map goes through the four helpers below.
  let submittedPoints = 0;
  let submittedBytes = 0;
  let pooledPoints = 0;
  let pooledBytes = 0;

  const holdSubmitted = (keyString: string, entry: TileActors): void => {
    tiles.set(keyString, entry);
    submittedPoints += entry.tile.pointCount;
    submittedBytes += entry.resourceBytes;
  };

  const dropSubmitted = (keyString: string, entry: TileActors): void => {
    tiles.delete(keyString);
    submittedPoints -= entry.tile.pointCount;
    submittedBytes -= entry.resourceBytes;
  };

  const holdPooled = (keyString: string, entry: TileActors): void => {
    pendingRelease.set(keyString, entry);
    pooledPoints += entry.tile.pointCount;
    pooledBytes += entry.resourceBytes;
  };

  const dropPooled = (keyString: string, entry: TileActors): void => {
    pendingRelease.delete(keyString);
    pooledPoints -= entry.tile.pointCount;
    pooledBytes -= entry.resourceBytes;
  };

  /** A live scalar setter accepts only a usable, actually-different value. */
  const acceptsScalar = (next: number, current: number): boolean =>
    !disposed && Number.isFinite(next) && next > 0 && next !== current;

  /** base · translate(origin): only the last column differs from base. */
  const tileMatrix = (origin: Vec3): number[] => {
    const out = Array.from(baseMatrix);
    for (let row = 0; row < 4; row += 1) {
      out[12 + row] =
        baseMatrix[row]! * origin[0] +
        baseMatrix[4 + row]! * origin[1] +
        baseMatrix[8 + row]! * origin[2] +
        baseMatrix[12 + row]!;
    }
    return out;
  };

  /** Push the adapter's current visual state onto one tile's actor/mapper. */
  const applyTileState = (entry: TileActors): void => {
    // The actor property remains in CSS pixels. The custom dense-point mapper
    // multiplies it by scaleFactor before assigning physical gl_PointSize.
    entry.mapper.setScaleFactor(devicePixelRatio);
    entry.actor.getProperty().setPointSize(diameterCssPx);
    entry.actor.setVisibility(visible);
    entry.actor.setUserMatrix(tileMatrix(entry.tile.origin));
  };

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
    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);
    const entry: TileActors = {
      actor,
      mapper,
      polyData,
      tile,
      resourceBytes: tileBytes(tile),
    };
    applyTileState(entry);
    return entry;
  };

  const releaseTile = (entry: TileActors): void => {
    renderer.removeActor(entry.actor);
    entry.actor.delete?.();
    entry.mapper.delete?.();
    entry.polyData.delete?.();
  };

  /** Reuse is only sound while an entry still holds the payload being added. */
  const holdsPayload = (entry: TileActors, tile: TileData): boolean =>
    entry.tile.positions === tile.positions && entry.tile.rgb === tile.rgb;

  // Only the pool is trimmable: the submitted set is what the controller
  // decided fits its own memory ceiling, and dropping an actor from it would
  // punch a hole nothing would ever refill (the controller sees no change).
  const trimPool = (additionalBytes = 0): void => {
    // Deleting the entry just yielded is well-defined for a Map iterator, and
    // iteration stays in insertion order — so this sheds oldest-first without
    // rebuilding an iterator per eviction.
    for (const [key, entry] of pendingRelease) {
      if (
        submittedBytes + pooledBytes + additionalBytes <=
        resourceCeilingBytes
      ) {
        return;
      }
      dropPooled(key, entry);
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
        dropSubmitted(keyString, entry);
        // Stop drawing the stale selection synchronously, but retain its
        // resources for bounded reuse. Releasing and reallocating a refined
        // cloud's actors at every interaction boundary creates input long tasks.
        entry.actor.setVisibility(false);
        holdPooled(keyString, entry);
        changed = true;
      }
      for (const { key, tile } of batch.added) {
        const keyString = keyToString(key);
        const current = tiles.get(keyString);
        if (current !== undefined) {
          // Batches are disjoint deltas, so an addition for a key already on
          // screen is a payload replacement — never a duplicate. Keeping the
          // old actor would draw superseded points for the rest of the
          // session, and its payload can never be reused, so it goes now.
          if (holdsPayload(current, tile)) continue;
          dropSubmitted(keyString, current);
          releaseTile(current);
        }
        const stale = pendingRelease.get(keyString);
        if (stale !== undefined) {
          dropPooled(keyString, stale);
          if (holdsPayload(stale, tile)) {
            stale.tile = tile;
            applyTileState(stale);
            holdSubmitted(keyString, stale);
            changed = true;
            continue;
          }
          releaseTile(stale);
        }
        trimPool(tileBytes(tile));
        const entry = createTile(tile);
        holdSubmitted(keyString, entry);
        renderer.addActor(entry.actor);
        changed = true;
      }
      // Removals alone must still bring the pool back under its ceiling. A
      // deactivated cloud sends nothing but removals, and trimming only on the
      // addition path left its actors on the GPU until something else happened
      // to add a tile — which, for a cloud the host just switched off, is
      // never.
      trimPool();
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
        entry.actor.setUserMatrix(tileMatrix(entry.tile.origin));
      }
      scheduleRender();
    },

    setPointDiameterCssPx(nextDiameterCssPx) {
      if (!acceptsScalar(nextDiameterCssPx, diameterCssPx)) return;
      diameterCssPx = nextDiameterCssPx;
      for (const entry of tiles.values()) {
        entry.actor.getProperty().setPointSize(nextDiameterCssPx);
      }
      scheduleRender();
    },

    setDevicePixelRatio(nextDevicePixelRatio) {
      if (!acceptsScalar(nextDevicePixelRatio, devicePixelRatio)) return;
      devicePixelRatio = nextDevicePixelRatio;
      for (const entry of tiles.values()) {
        entry.mapper.setScaleFactor(nextDevicePixelRatio);
      }
      scheduleRender();
    },

    setVisible(nextVisible) {
      if (disposed || nextVisible === visible) return;
      visible = nextVisible;
      // Visibility is reversible on its own terms: hiding must not destroy
      // state that only new controller batches could rebuild, or a show with
      // no selection change behind it would leave the cloud blank forever.
      // Pooled entries stay hidden either way — they are not submitted.
      for (const entry of tiles.values()) entry.actor.setVisibility(visible);
      scheduleRender();
    },

    setResourceCeilingBytes(bytes) {
      if (disposed || !finiteNonNegative(bytes)) return;
      resourceCeilingBytes = bytes;
      trimPool();
    },

    stats() {
      return {
        submittedTiles: tiles.size,
        submittedPoints,
        submittedBytes,
        pooledTiles: pendingRelease.size,
        pooledPoints,
        pooledBytes,
        gpuResidentTiles: tiles.size + pendingRelease.size,
        gpuResidentPoints: submittedPoints + pooledPoints,
        gpuResidentBytes: submittedBytes + pooledBytes,
        resourceCeilingBytes,
        drawnTiles: visible ? tiles.size : 0,
        drawnPoints: visible ? submittedPoints : 0,
        visible,
        diameterCssPx,
        devicePixelRatio,
      };
    },

    activeKeys: () => ({
      submitted: [...tiles.keys()].sort(),
      pooled: [...pendingRelease.keys()].sort(),
    }),

    dispose() {
      if (disposed) return;
      disposed = true;
      // Submitted and pooled entries are disjoint, so this releases every
      // actor exactly once.
      for (const entry of tiles.values()) releaseTile(entry);
      for (const entry of pendingRelease.values()) releaseTile(entry);
      tiles.clear();
      pendingRelease.clear();
      submittedPoints = 0;
      submittedBytes = 0;
      pooledPoints = 0;
      pooledBytes = 0;
    },
  };
};
