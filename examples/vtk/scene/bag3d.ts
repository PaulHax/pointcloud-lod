/**
 * Host-side reader for the published 3DBAG tileset.
 *
 * 3DBAG is 3D Tiles 1.1 with `box` bounds, `REPLACE` refinement and `.glb`
 * content — the profile `loadTileset` accepts — except that its index defers
 * the finer levels to 474 external `tileset-*.json` documents, and external
 * tilesets are outside that profile. Resolving them is a host's job, not the
 * parser's, so this runs on the `fetchTileset` seam: it fetches the index,
 * drops every tile outside the place being shown, splices the surviving
 * externals into their parents, and hands back one explicit document. The
 * library is unchanged and unaware.
 *
 * Pruning is what keeps that affordable. The whole country inlines to 13k
 * tiles behind 474 requests; one city needs a handful.
 */

import type { ContentQueueFetch } from "../../../src/tiles3d/contentQueue";
import type {
  TilesetFetch,
  TilesetFetchResponse,
} from "../../../src/tiles3d/tilesetSource";
import { repairMeshoptFallbackOffsets } from "./meshoptFallback";

type RawTile = {
  readonly boundingVolume?: { readonly box?: readonly number[] };
  readonly content?: { readonly uri?: string };
  readonly children?: readonly RawTile[];
  readonly [key: string]: unknown;
};

type RawTileset = {
  readonly root?: RawTile;
  readonly [key: string]: unknown;
};

export type Bag3dTilesetOptions = {
  readonly endpoint: string;
  /** Scene origin in ECEF metres — tiles beyond `radiusMeters` are dropped. */
  readonly center: readonly [number, number, number];
  readonly radiusMeters: number;
  /** Reports each external tileset as it lands, for a loading message. */
  readonly onExternalLoaded?: (loaded: number) => void;
};

const isExternalTileset = (uri: string | undefined): boolean =>
  uri !== undefined && uri.toLowerCase().endsWith(".json");

/**
 * Conservative reject test. The half axes span the box, so their summed
 * lengths bound its extent from the centre without decomposing the basis.
 */
const withinRadius = (
  tile: RawTile,
  center: readonly [number, number, number],
  radiusMeters: number,
): boolean => {
  const box = tile.boundingVolume?.box;
  if (!Array.isArray(box) || box.length !== 12) return true;
  const distance = Math.hypot(
    box[0]! - center[0],
    box[1]! - center[1],
    box[2]! - center[2],
  );
  const extent =
    Math.hypot(box[3]!, box[4]!, box[5]!) +
    Math.hypot(box[6]!, box[7]!, box[8]!) +
    Math.hypot(box[9]!, box[10]!, box[11]!);
  return distance <= radiusMeters + extent;
};

const fetchJson = async (
  url: string,
  signal?: AbortSignal,
): Promise<unknown> => {
  const response = await fetch(url, signal ? { signal } : {});
  if (!response.ok) {
    throw new Error(
      `${url} responded ${response.status} ${response.statusText}`,
    );
  }
  return response.json();
};

/**
 * A pruned tile keeps its own content and whichever children survive. An
 * external reference is replaced by the root of the document it names, which
 * may in turn reference further externals, so resolution recurses.
 */
const resolveTile = async (
  tile: RawTile,
  options: Bag3dTilesetOptions,
  state: { loaded: number },
  signal: AbortSignal | undefined,
): Promise<RawTile> => {
  const uri = tile.content?.uri;
  if (isExternalTileset(uri)) {
    const document = (await fetchJson(
      `${options.endpoint}/${uri}`,
      signal,
    )) as RawTileset;
    state.loaded += 1;
    options.onExternalLoaded?.(state.loaded);
    const { content: _external, ...withoutContent } = tile;
    const external = document.root;
    // The external root repeats its parent's bounds, so it becomes the
    // parent's child rather than replacing the parent's place in the tree.
    const children = external
      ? [...(tile.children ?? []), external]
      : (tile.children ?? []);
    return resolveChildren(
      { ...withoutContent, children },
      options,
      state,
      signal,
    );
  }
  return resolveChildren(tile, options, state, signal);
};

const resolveChildren = async (
  tile: RawTile,
  options: Bag3dTilesetOptions,
  state: { loaded: number },
  signal: AbortSignal | undefined,
): Promise<RawTile> => {
  const children = tile.children ?? [];
  if (children.length === 0) return tile;
  const kept = children.filter((child) =>
    withinRadius(child, options.center, options.radiusMeters),
  );
  const resolved = await Promise.all(
    kept.map((child) => resolveTile(child, options, state, signal)),
  );
  return { ...tile, children: resolved };
};

/**
 * A `TilesetFetch` the tiles3d member can be handed directly. It answers the
 * one `tileset.json` request the member makes with the resolved document, so
 * every content URL still resolves against the real 3DBAG endpoint.
 */
export const createBag3dTilesetFetch = (
  options: Bag3dTilesetOptions,
): TilesetFetch => {
  return async (url, init): Promise<TilesetFetchResponse> => {
    const state = { loaded: 0 };
    const index = (await fetchJson(url, init.signal)) as RawTileset;
    const root = index.root;
    const resolved = root
      ? await resolveTile(root, options, state, init.signal)
      : undefined;
    const document = resolved ? { ...index, root: resolved } : index;
    return {
      ok: true,
      status: 200,
      statusText: "OK",
      json: async () => document,
    };
  };
};

/**
 * Tile content, repaired on the way past. 3DBAG's GLBs stack both of their
 * meshopt buffer views at offset zero of one fallback buffer; see
 * `meshoptFallback.ts` for what that does to the triangles.
 */
export const bag3dContentFetch: ContentQueueFetch = async (url, init) => {
  const response = await fetch(url, { signal: init.signal });
  return {
    ok: response.ok,
    status: response.status,
    statusText: response.statusText,
    arrayBuffer: async () =>
      repairMeshoptFallbackOffsets(await response.arrayBuffer()),
  };
};
