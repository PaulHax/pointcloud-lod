import { afterEach, describe, expect, it, vi } from "vitest";

import { loadTileset } from "../../../src/tiles3d/tilesetSource";
import { createBag3dTilesetFetch } from "./bag3d";

const ENDPOINT = "https://data.3dbag.nl/v20250903/cesium3dtiles/lod22";

/** ECEF centre near Rotterdam, and a box shaped the way 3DBAG publishes them. */
const NEAR: [number, number, number] = [3_926_000, 307_600, 4_997_000];
const FAR: [number, number, number] = [3_800_000, 900_000, 4_900_000];

const box = (
  centre: readonly [number, number, number],
  half: number,
): number[] => [...centre, half, 0, 0, 0, half, 0, 0, 0, half];

/**
 * The published shape: an index whose leaves defer to external tilesets, with
 * coarse `.glb` content at the intermediate levels.
 */
const INDEX = {
  asset: { version: "1.1" },
  geometricError: 2000,
  root: {
    boundingVolume: { box: box(NEAR, 20_000) },
    geometricError: 2000,
    refine: "REPLACE",
    children: [
      {
        boundingVolume: { box: box(NEAR, 1000) },
        geometricError: 500,
        refine: "REPLACE",
        content: { uri: "t/4/64/128.glb" },
        children: [
          {
            boundingVolume: { box: box(NEAR, 400) },
            geometricError: 100,
            refine: "REPLACE",
            content: { uri: "tileset-5-64-192.json" },
          },
        ],
      },
      {
        boundingVolume: { box: box(FAR, 1000) },
        geometricError: 500,
        refine: "REPLACE",
        content: { uri: "tileset-5-999-999.json" },
      },
    ],
  },
};

const EXTERNAL = {
  asset: { version: "1.1" },
  geometricError: 100,
  root: {
    boundingVolume: { box: box(NEAR, 400) },
    geometricError: 100,
    refine: "REPLACE",
    children: [
      {
        boundingVolume: { box: box(NEAR, 200) },
        geometricError: 0,
        refine: "REPLACE",
        content: { uri: "t/6/64/192.glb" },
      },
    ],
  },
};

const stubFetch = (): string[] => {
  const requested: string[] = [];
  vi.stubGlobal("fetch", async (url: string) => {
    requested.push(url);
    const body = url.endsWith("tileset.json") ? INDEX : EXTERNAL;
    return { ok: true, status: 200, statusText: "OK", json: async () => body };
  });
  return requested;
};

afterEach(() => vi.unstubAllGlobals());

describe("3DBAG resolution", () => {
  it("is rejected by the library before resolution", async () => {
    stubFetch();
    const error = await loadTileset({ endpoint: ENDPOINT }).catch(
      (cause) => cause,
    );
    expect(error.name).toBe("TilesetUnsupportedError");
    expect(error.path).toContain("content.uri");
  });

  it("splices in externals within the radius and drops the rest", async () => {
    const requested = stubFetch();
    const source = await loadTileset({
      endpoint: ENDPOINT,
      fetch: createBag3dTilesetFetch({
        endpoint: ENDPOINT,
        center: NEAR,
        radiusMeters: 1500,
      }),
    });

    // Only the near external was fetched; the far one never became a request.
    expect(requested.filter((url) => url.endsWith(".json"))).toEqual([
      `${ENDPOINT}/tileset.json`,
      `${ENDPOINT}/tileset-5-64-192.json`,
    ]);
    const contentUrls = source.tiles
      .filter((tile) => tile.contentUrl)
      .map((tile) => tile.contentUrl);
    expect(contentUrls).toEqual([
      `${ENDPOINT}/t/4/64/128.glb`,
      `${ENDPOINT}/t/6/64/192.glb`,
    ]);
  });

  it("keeps the refinement chain the external documents carry", async () => {
    stubFetch();
    const source = await loadTileset({
      endpoint: ENDPOINT,
      fetch: createBag3dTilesetFetch({
        endpoint: ENDPOINT,
        center: NEAR,
        radiusMeters: 1500,
      }),
    });
    const errors = source.tiles.map((tile) => tile.geometricError);
    expect(errors).toEqual([2000, 500, 100, 100, 0]);
  });

  it("reports each external as it lands", async () => {
    stubFetch();
    const loaded: number[] = [];
    await loadTileset({
      endpoint: ENDPOINT,
      fetch: createBag3dTilesetFetch({
        endpoint: ENDPOINT,
        center: NEAR,
        radiusMeters: 1500,
        onExternalLoaded: (count) => loaded.push(count),
      }),
    });
    expect(loaded).toEqual([1]);
  });
});
