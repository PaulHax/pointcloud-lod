/**
 * Tile-source contract.
 *
 * Two reference implementations ship with the library: `createHttpTileSource`
 * (a revision-scoped hierarchy/tile HTTP protocol with PCT1 binary tiles) and
 * `createCopcTileSource` (direct COPC reads, e.g. over HTTP Range requests).
 * Anything exposing the same octree shape can implement it.
 */

import type { Bounds, Vec3, VoxelKey } from "./octree";

export type TileSourceMetadata = {
  /** Total points in the dataset. */
  readonly pointCount: number;
  /**
   * The data's own extent, when the source knows it.
   *
   * Deliberately not the octree's root cube: that cube encloses the data, so
   * for a survey far wider than it is tall its centre sits high in empty air —
   * a host framing the scene from it orbits about a point in the sky. A source
   * that cannot state the real extent omits this, and a host then has nothing
   * to frame from but the root node's bounds.
   */
  readonly bounds?: Bounds;
};

/** One hierarchy entry, as delivered by a hierarchy page. */
export type NodeInfo = {
  readonly key: VoxelKey;
  /** Points stored in this node (0 is legal: structural node). */
  readonly pointCount: number;
  /** Conservative render-space AABB, before any actor registration transform. */
  readonly bounds: Bounds;
  /** Effective metric point spacing for this node, in render-space units. */
  readonly spacing: number;
  /**
   * Children known to exist. When absent, consumers derive children from the
   * presence of sibling entries in the same (or previously loaded) pages.
   */
  readonly children?: readonly VoxelKey[];
  /**
   * True when this entry only points at a further hierarchy page rooted at
   * `key`: call `nodes(key)` to materialize the subtree before using it.
   */
  readonly pageRef?: boolean;
};

/** Decoded payload of one octree node. */
export type TileData = {
  /** World-space origin the tile-local positions are relative to. */
  readonly origin: Vec3;
  /**
   * Tile-local xyz triplets, `3 * pointCount` floats, in progressive order:
   * every prefix has to be a spatially representative sample of the node,
   * because that is exactly what progressive drawing puts on the screen. LAS
   * and COPC record order promises no such thing, so a source that does not
   * already have one applies {@link orderTileForProgressiveDrawing} to the
   * payload it decoded — see {@link TileSource.loadTile}.
   */
  readonly positions: Float32Array;
  /** Optional per-point color, `3 * pointCount` bytes (RGB). */
  readonly rgb?: Uint8Array;
  readonly pointCount: number;
};

/**
 * Decoded bytes a tile occupies, plus a per-tile object estimate. Lives here
 * so the controller's `decodedBytes` and the adapter's `gpuResidentBytes`
 * cannot disagree about what a tile costs — their docs promise they agree.
 */
export const tileBytes = (tile: TileData): number =>
  tile.positions.byteLength + (tile.rgb?.byteLength ?? 0) + 64;

export type LoadOptions = {
  /**
   * Abort in-flight I/O and decoding; the promise rejects on abort. Hierarchy
   * requests are scheduled and cancelled exactly like tile requests, so both
   * take the same options.
   */
  readonly signal?: AbortSignal;
};

/** A source of octree point-cloud tiles. */
export type TileSource = {
  /** Dataset metadata, resolved when the source was created. */
  metadata(): TileSourceMetadata;
  /**
   * Hierarchy entries for the page rooted at `key` (the root page for the
   * root key). May return entries for several levels at once.
   */
  nodes(key: VoxelKey, opts?: LoadOptions): Promise<NodeInfo[]>;
  /**
   * Fetch and decode one node's points, in the progressive order
   * {@link TileData.positions} describes. Ordering belongs to the source
   * because that is where the payload is already being touched — for a
   * worker-backed source, off the main thread and before transfer.
   */
  loadTile(key: VoxelKey, opts?: LoadOptions): Promise<TileData>;
  /** Release source-owned resources such as a decoding worker. The host owns this lifetime. */
  dispose?(): void;
};
