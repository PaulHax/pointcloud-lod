/**
 * PROVISIONAL tile-source contract.
 *
 * This interface is a working draft; it finalizes after the COPC reader
 * spike. Expect field additions (attribute selection, density hints) and
 * possible signature changes before 0.1.
 */

import type { Cube, Vec3, VoxelKey } from './octree';

export interface TileSourceMetadata {
  /** Total points in the dataset. */
  readonly pointCount: number;
  /** Root octree cube in world coordinates (COPC cube). */
  readonly cube: Cube;
  /** Point spacing at the root level; halves per level. */
  readonly spacing: number;
  /**
   * Optional world offset applied to tile origins (e.g. a large UTM offset
   * kept out of Float32 tile-local coordinates).
   */
  readonly offset?: Vec3;
}

/** One hierarchy entry, as delivered by a hierarchy page. */
export interface NodeInfo {
  readonly key: VoxelKey;
  /** Points stored in this node (0 is legal: structural node). */
  readonly pointCount: number;
}

/** Decoded payload of one octree node. */
export interface TileData {
  /** World-space origin the tile-local positions are relative to. */
  readonly origin: Vec3;
  /** Tile-local xyz triplets, `3 * pointCount` floats. */
  readonly positions: Float32Array;
  /** Optional per-point color, `3 * pointCount` bytes (RGB). */
  readonly rgb?: Uint8Array;
  readonly pointCount: number;
}

export interface LoadTileOptions {
  /** Abort in-flight I/O and decoding; the promise rejects on abort. */
  readonly signal?: AbortSignal;
}

/**
 * A source of octree point-cloud tiles.
 *
 * The reference implementation reads COPC files over HTTP Range requests;
 * anything exposing the same hierarchy shape (Potree-style stores, in-memory
 * fixtures for tests) can implement it too.
 */
export interface TileSource {
  /** Dataset metadata, resolved when the source was created. */
  metadata(): TileSourceMetadata;
  /**
   * Hierarchy entries for the page rooted at `key` (the root page for the
   * root key). May return entries for several levels at once.
   */
  nodes(key: VoxelKey): Promise<NodeInfo[]>;
  /** Fetch and decode one node's points. */
  loadTile(key: VoxelKey, opts?: LoadTileOptions): Promise<TileData>;
}
