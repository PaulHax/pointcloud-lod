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

import type { TileDrawPlan } from "./controller";
import type { TileBatch } from "./payloadResidency";
import type { SubmissionScheduler, Submission } from "./submissionScheduler";
import { IDENTITY, sameMatrix, translatedMatrix } from "./mat4";
import { finiteAbove, finiteNonNegative } from "./numeric";
import { keyToString, type VoxelKey } from "./octree";
import { setActorPointSize, setMapperPointCount } from "./drawState";
import { tileBytes, type TileData } from "./tileSource";

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
  /** Shared frame budget; without it, prefixes are admitted immediately. */
  submissions?: SubmissionScheduler;
  onWorkChange?: () => void;
};

export type RendererAdapter = {
  /** Queue one upload slice per pending tile for this frame. */
  prepareFrame(): void;
  drawnPointCount(key: VoxelKey): number;
  pendingUploads(): number;
  /**
   * Apply one controller batch (typically wired as `onTiles`). Batches apply
   * whether or not the adapter is visible; an addition for a key already on
   * screen replaces that tile's payload.
   */
  applyBatch(batch: TileBatch): void;
  /** Apply per-tile progressive prefixes without replacing actors or VBOs. */
  applyDrawPlan(plan: TileDrawPlan): void;
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
  /** Held counters for coordinator refreshes, without scanning the drawn set. */
  workState(): Pick<RendererAdapterStats, "workRevision" | "gpuResidentBytes">;
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
  /**
   * Monotonic revision of submitted resource changes. The first render after
   * this changes may allocate or upload buffers and is not a capacity sample.
   */
  readonly workRevision: number;
  /** Tiles matching the controller's submitted set; drawn while visible. */
  readonly submittedTiles: number;
  readonly submittedPoints: number;
  readonly submittedBytes: number;
  /** Retired tiles retained for same-key reuse; never drawn. */
  readonly pooledTiles: number;
  readonly pooledPoints: number;
  readonly pooledBytes: number;
  /**
   * Whether the pool is earning its residency: session totals for tiles that
   * came back to the same key holding the same payload, against tiles built
   * from nothing. A pool that never hits is memory spent on nothing.
   */
  readonly reusedTiles: number;
  readonly builtTiles: number;
  /**
   * Pool hits bucketed by how many tiles were pooled after the one that hit:
   * `[0], [1], [2,3], [4,7], ...` with the last bucket open.
   *
   * The count is insertions since the entry was pooled, not the entry's rank
   * in the pool at the moment it hit: insertions that were themselves evicted
   * before the hit still count. So it reads as an upper bound on how deep a
   * newest-N policy would have had to hold the entry, and the hits in the
   * buckets up to N are the ones such a policy is guaranteed to have earned —
   * it may earn some of the deeper ones too.
   */
  readonly reuseDepths: readonly number[];
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
  readonly drawnFraction: number;
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
  drawnPointCount: number;
  uploadedPointCount: number;
  ranges: { min: number; max: number }[];
  submission?: Submission;
  /** Pool insertions before this one, so a hit's depth in the pool is known. */
  pooledSerial: number;
};

/** `[0], [1], [2,3], [4,7], ...`; the last bucket takes everything deeper. */
const REUSE_DEPTH_BUCKETS = 12;

const depthBucket = (depth: number): number =>
  depth <= 0
    ? 0
    : Math.min(REUSE_DEPTH_BUCKETS - 1, Math.floor(Math.log2(depth)) + 1);

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
  /**
   * The prefix each submitted tile draws, as the controller last planned it.
   * Null until the first plan arrives: a tile submitted before any plan draws
   * whole, while a tile absent from a plan that did arrive draws nothing.
   */
  let pointPrefixes: ReadonlyMap<string, number> | null = null;
  let baseMatrix: ArrayLike<number> = IDENTITY;
  // A key lives in at most one of the two: removal moves its entry from
  // `tiles` to `pendingRelease`, and a re-addition either takes that entry
  // back or releases it. Nothing else may hold a reference to an entry, so
  // teardown releases each actor exactly once.
  const tiles = new Map<string, TileActors>();
  const pendingRelease = new Map<string, TileActors>();
  const pendingUploads = new Map<string, TileActors>();
  let resourceCeilingBytes = 256 * 1024 * 1024;
  let disposed = false;

  // Running totals rather than a walk per question. Trimming asks how many
  // bytes are held once per evicted entry, so re-summing both maps there would
  // make shedding a large pool quadratic in its size — and the pool is largest
  // exactly when a cloud has just been deactivated and the ceiling dropped to
  // nothing, which is when the walk is longest and the answer needed soonest.
  // Every mutation of either map goes through the four helpers below.
  let submittedPoints = 0;
  let submittedBytes = 0;
  let pooledPoints = 0;
  let pooledBytes = 0;
  let reusedTiles = 0;
  let builtTiles = 0;
  let workRevision = 0;
  let pooledSerial = 0;
  /**
   * Pool hits by how many tiles were pooled after the one that hit, in
   * power-of-two buckets. The pool is retained by bytes, but what it costs to
   * leave attached is counted in props, and those are not the same question.
   */
  const reuseDepths = Array.from({ length: REUSE_DEPTH_BUCKETS }, () => 0);

  const holdSubmitted = (keyString: string, entry: TileActors): void => {
    tiles.set(keyString, entry);
    submittedPoints += entry.tile.pointCount;
    submittedBytes += entry.resourceBytes;
    renderer.addActor(entry.actor);
  };

  const dropSubmitted = (keyString: string, entry: TileActors): void => {
    tiles.delete(keyString);
    pendingUploads.delete(keyString);
    entry.submission?.cancel();
    entry.submission = undefined;
    submittedPoints -= entry.tile.pointCount;
    submittedBytes -= entry.resourceBytes;
  };

  const holdPooled = (keyString: string, entry: TileActors): void => {
    pendingRelease.set(keyString, entry);
    pooledPoints += entry.tile.pointCount;
    pooledBytes += entry.resourceBytes;
    // Retained attached as well as resident. Detaching would let the renderer
    // delete the prop's view node and release its buffers, so the two tiles in
    // three that come back to the same key (`reusedTiles` against
    // `builtTiles`) would each pay a full upload to be drawn again.
    entry.actor.setVisibility(false);
    entry.pooledSerial = pooledSerial++;
  };

  const dropPooled = (keyString: string, entry: TileActors): void => {
    pendingRelease.delete(keyString);
    pooledPoints -= entry.tile.pointCount;
    pooledBytes -= entry.resourceBytes;
  };

  /** A live scalar setter accepts only a usable, actually-different value. */
  const acceptsScalar = (next: number, current: number): boolean =>
    !disposed && Number.isFinite(next) && next > 0 && next !== current;

  /** The tile's planned draw count; the whole tile when no plan is set. */
  const prefixFor = (keyString: string, entry: TileActors): number =>
    pointPrefixes === null
      ? entry.tile.pointCount
      : Math.min(
          entry.tile.pointCount,
          Math.max(0, Math.floor(pointPrefixes.get(keyString) ?? 0)),
        );

  const admit = (keyString: string, entry: TileActors, count: number): void => {
    if (count > entry.uploadedPointCount) {
      const points = entry.polyData.getPoints();
      points.resize(count);
      // Stable full-tile ranges keep the GPU transform unchanged as prefixes grow.
      entry.ranges.forEach((range, component) =>
        points.setRange(range, component),
      );
      entry.polyData.getPointData().getScalars()?.resize(count);
      entry.uploadedPointCount = count;
      workRevision += 1;
      options.onWorkChange?.();
    }
    entry.drawnPointCount = count;
    setMapperPointCount(entry.mapper, count);
    entry.actor.setVisibility(visible && count > 0);
    if (count < prefixFor(keyString, entry))
      pendingUploads.set(keyString, entry);
    else pendingUploads.delete(keyString);
  };

  const refreshPrefix = (keyString: string, entry: TileActors): boolean => {
    const target = prefixFor(keyString, entry);
    const count = options.submissions
      ? Math.min(target, entry.uploadedPointCount)
      : target;
    if (count < target) pendingUploads.set(keyString, entry);
    else {
      pendingUploads.delete(keyString);
      entry.submission?.cancel();
      entry.submission = undefined;
    }
    if (count === entry.drawnPointCount) return false;
    admit(keyString, entry, count);
    return true;
  };

  const refreshPrefixes = (): boolean => {
    let changed = false;
    for (const [keyString, entry] of tiles) {
      changed = refreshPrefix(keyString, entry) || changed;
    }
    return changed;
  };

  /** Push the adapter's current visual state onto one tile's actor/mapper. */
  const applyTileState = (keyString: string, entry: TileActors): void => {
    // The actor property remains in CSS pixels. The custom dense-point mapper
    // multiplies it by pointSizeScale before assigning physical gl_PointSize.
    entry.mapper.setPointSizeScale(devicePixelRatio);
    refreshPrefix(keyString, entry);
    setActorPointSize(entry.actor, diameterCssPx);
    entry.actor.setVisibility(visible && entry.drawnPointCount > 0);
    entry.actor.setUserMatrix(translatedMatrix(baseMatrix, entry.tile.origin));
  };

  const createTile = (keyString: string, tile: TileData): TileActors => {
    builtTiles += 1;
    const polyData = vtkPolyData.newInstance();
    const points = polyData.getPoints();
    points.setData(tile.positions, 3);
    const ranges = [0, 1, 2].map((component) => {
      const [min, max] = points.getRange(component);
      return { min, max };
    });
    points.resize(0);
    if (tile.rgb !== undefined) {
      polyData.getPointData().setScalars(
        vtkDataArray.newInstance({
          name: "RGB",
          values: tile.rgb,
          numberOfComponents: 3,
          size: 0,
        }),
      );
    }
    const mapper = vtkPointGaussianMapper.newInstance({ scaleFactor: 0 });
    mapper.setInputData(polyData);
    mapper.setStatic?.(true);
    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);
    // Points carry no alpha. Saying so up front spares vtk.js asking the
    // mapper, which looks its scalars up again, in several passes a frame.
    actor.setForceOpaque(true);
    const entry: TileActors = {
      actor,
      mapper,
      polyData,
      tile,
      resourceBytes: tileBytes(tile),
      drawnPointCount: 0,
      uploadedPointCount: 0,
      ranges,
      pooledSerial: 0,
    };
    setMapperPointCount(mapper, 0);
    applyTileState(keyString, entry);
    return entry;
  };

  const releaseTile = (entry: TileActors): void => {
    entry.submission?.cancel();
    entry.submission = undefined;
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
    prepareFrame() {
      if (disposed || !visible || !options.submissions) return;
      const maxBytes = options.submissions.stats().maxBytesPerFrame;
      const maxPoints = Math.max(1, Math.min(8192, Math.floor(maxBytes / 16)));
      for (const [keyString, entry] of pendingUploads) {
        if (entry.submission) continue;
        const count = Math.min(
          prefixFor(keyString, entry),
          entry.uploadedPointCount + maxPoints,
        );
        const bytes =
          Math.max(0, count - entry.uploadedPointCount) *
          (entry.tile.rgb ? 16 : 12);
        entry.submission = options.submissions.enqueue({
          bytes,
          atomic: bytes > maxBytes,
          run() {
            entry.submission = undefined;
            if (tiles.get(keyString) !== entry) return;
            admit(
              keyString,
              entry,
              Math.min(count, prefixFor(keyString, entry)),
            );
            scheduleRender();
          },
        });
      }
    },

    drawnPointCount(key) {
      return visible ? (tiles.get(keyToString(key))?.drawnPointCount ?? 0) : 0;
    },

    pendingUploads: () => pendingUploads.size,

    applyDrawPlan(plan) {
      if (disposed) return;
      const next = new Map(
        plan.entries.map(({ key, pointCount }) => [
          keyToString(key),
          finiteNonNegative(pointCount) ? Math.floor(pointCount) : 0,
        ]),
      );
      pointPrefixes = next;
      if (refreshPrefixes() || pendingUploads.size > 0) scheduleRender();
    },

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
          if (holdsPayload(current, tile)) {
            if (
              current.tile.pointCount === tile.pointCount &&
              current.tile.origin.every(
                (value, index) => value === tile.origin[index],
              )
            )
              continue;
            submittedPoints += tile.pointCount - current.tile.pointCount;
            current.tile = tile;
            applyTileState(keyString, current);
            changed = true;
            continue;
          }
          dropSubmitted(keyString, current);
          releaseTile(current);
        }
        const stale = pendingRelease.get(keyString);
        if (stale !== undefined) {
          dropPooled(keyString, stale);
          if (holdsPayload(stale, tile)) {
            stale.tile = tile;
            applyTileState(keyString, stale);
            holdSubmitted(keyString, stale);
            reusedTiles += 1;
            const bucket = depthBucket(pooledSerial - 1 - stale.pooledSerial);
            reuseDepths[bucket] = (reuseDepths[bucket] ?? 0) + 1;
            changed = true;
            continue;
          }
          releaseTile(stale);
        }
        trimPool(tileBytes(tile));
        const entry = createTile(keyString, tile);
        holdSubmitted(keyString, entry);
        changed = true;
      }
      // Removals alone must still bring the pool back under its ceiling. A
      // deactivated cloud sends nothing but removals, and trimming only on the
      // addition path would leave its actors on the GPU until something else
      // happened to add a tile — which, for a cloud the host just switched
      // off, is never.
      trimPool();
      if (changed) {
        workRevision += 1;
        scheduleRender();
      }
    },

    setBaseMatrix(matrix) {
      if (disposed) return;
      const next = matrix ?? IDENTITY;
      if (sameMatrix(baseMatrix, next)) return;
      // Keep a value snapshot: vtk.js may mutate and reuse the same UserMatrix
      // object, and the next update still needs to detect that visual change.
      baseMatrix = Array.from(next);
      for (const entry of tiles.values()) {
        entry.actor.setUserMatrix(
          translatedMatrix(baseMatrix, entry.tile.origin),
        );
      }
      scheduleRender();
    },

    setPointDiameterCssPx(nextDiameterCssPx) {
      if (!acceptsScalar(nextDiameterCssPx, diameterCssPx)) return;
      diameterCssPx = nextDiameterCssPx;
      for (const entry of tiles.values()) {
        setActorPointSize(entry.actor, nextDiameterCssPx);
      }
      scheduleRender();
    },

    setDevicePixelRatio(nextDevicePixelRatio) {
      if (!acceptsScalar(nextDevicePixelRatio, devicePixelRatio)) return;
      devicePixelRatio = nextDevicePixelRatio;
      for (const entry of tiles.values()) {
        entry.mapper.setPointSizeScale(nextDevicePixelRatio);
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
      for (const entry of tiles.values()) {
        if (!visible) {
          entry.submission?.cancel();
          entry.submission = undefined;
        }
        entry.actor.setVisibility(visible && entry.drawnPointCount > 0);
      }
      scheduleRender();
    },

    setResourceCeilingBytes(bytes) {
      if (disposed || !finiteNonNegative(bytes)) return;
      resourceCeilingBytes = bytes;
      trimPool();
    },

    workState() {
      return { workRevision, gpuResidentBytes: submittedBytes + pooledBytes };
    },

    stats() {
      let drawnPoints = 0;
      let drawnTiles = 0;
      if (visible) {
        for (const entry of tiles.values()) {
          drawnPoints += entry.drawnPointCount;
          if (entry.drawnPointCount > 0) drawnTiles += 1;
        }
      }
      return {
        workRevision,
        submittedTiles: tiles.size,
        submittedPoints,
        submittedBytes,
        pooledTiles: pendingRelease.size,
        pooledPoints,
        pooledBytes,
        reusedTiles,
        builtTiles,
        reuseDepths: [...reuseDepths],
        gpuResidentTiles: tiles.size + pendingRelease.size,
        gpuResidentPoints: submittedPoints + pooledPoints,
        gpuResidentBytes: submittedBytes + pooledBytes,
        resourceCeilingBytes,
        drawnTiles,
        drawnPoints,
        drawnFraction:
          visible && submittedPoints > 0 ? drawnPoints / submittedPoints : 0,
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
      pendingUploads.clear();
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
