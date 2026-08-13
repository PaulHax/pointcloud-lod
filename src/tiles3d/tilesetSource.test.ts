import { describe, expect, it, vi } from "vitest";

import {
  TilesetFetchError,
  TilesetUnsupportedError,
  TilesetValidationError,
  loadTileset,
  resolveTilesetContentUri,
} from "./tilesetSource";

const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const box = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1];

const tile = (uri = "content/root.glb", children: unknown[] = []) => ({
  boundingVolume: { box },
  geometricError: children.length === 0 ? 0 : 8,
  refine: "REPLACE",
  transform: identity,
  content: { uri },
  ...(children.length > 0 ? { children } : {}),
});

const document = () => ({
  asset: { version: "1.1" },
  geometricError: 16,
  root: tile("content/root.glb", [
    {
      ...tile("content/west.glb"),
      transform: [...identity.slice(0, 12), -1, 0, 0, 1],
    },
    {
      ...tile("content/east.gltf"),
      transform: [...identity.slice(0, 12), 1, 0, 0, 1],
    },
  ]),
});

const documentWithRootTransform = (transform: number[]) => {
  const value = document();
  value.root.transform = transform;
  return value;
};

const response = (body: unknown, status = 200) => ({
  ok: status >= 200 && status < 300,
  status,
  statusText: status === 200 ? "OK" : "Gone",
  json: vi.fn().mockResolvedValue(body),
});

describe("loadTileset", () => {
  it("inherits REPLACE into children while still rejecting explicit ADD", async () => {
    const inherited = document();
    delete (inherited.root.children![0] as { refine?: string }).refine;
    const source = await loadTileset({
      endpoint: "/tiles",
      fetch: vi.fn().mockResolvedValue(response(inherited)),
    });
    expect(source.root.children[0]?.id).toBe("root/0");

    const add = document();
    (add.root.children![0] as { refine?: string }).refine = "ADD";
    await expect(
      loadTileset({
        endpoint: "/tiles",
        fetch: vi.fn().mockResolvedValue(response(add)),
      }),
    ).rejects.toThrow(/root\.children\[0\]\.refine/);
  });

  it("uses the shared affine tolerance for fixed matrix entries", async () => {
    const tolerated = documentWithRootTransform([
      1,
      0,
      0,
      0.5e-12,
      ...identity.slice(4, 15),
      1 + 0.5e-12,
    ]);
    await expect(
      loadTileset({
        endpoint: "/tiles",
        fetch: vi.fn().mockResolvedValue(response(tolerated)),
      }),
    ).resolves.toBeDefined();
  });

  it("fetches and validates the constrained profile with stable IDs and URLs", async () => {
    const fetcher = vi.fn().mockResolvedValue(response(document()));
    const source = await loadTileset({
      endpoint: "/tileset/asset/revision/",
      fetch: fetcher,
    });

    expect(fetcher).toHaveBeenCalledWith(
      "/tileset/asset/revision/tileset.json",
      expect.objectContaining({ signal: undefined }),
    );
    expect(source.endpoint).toBe("/tileset/asset/revision");
    expect(source.root.id).toBe("root");
    expect(source.tiles.map((entry) => entry.id)).toEqual([
      "root",
      "root/0",
      "root/1",
    ]);
    expect(source.tiles.map((entry) => entry.contentUrl)).toEqual([
      "/tileset/asset/revision/content/root.glb",
      "/tileset/asset/revision/content/west.glb",
      "/tileset/asset/revision/content/east.gltf",
    ]);
    expect(source.root.children.map((entry) => entry.id)).toEqual([
      "root/0",
      "root/1",
    ]);
    expect(source.root.children[0]?.worldTransform[12]).toBe(-1);
    expect(source.root.children[1]?.worldTransform[12]).toBe(1);
  });

  it("accepts contentless explicit ancestors while preserving their descendants", async () => {
    const contentless = document();
    delete (contentless.root as Partial<typeof contentless.root>).content;
    const children = contentless.root.children!;
    delete (children[0] as Record<string, unknown>).content;
    (children[0] as Record<string, unknown>).children = [
      tile("content/west-detail.glb"),
    ];

    const source = await loadTileset({
      endpoint: "/tiles",
      fetch: vi.fn().mockResolvedValue(response(contentless)),
    });

    expect(source.tiles.map(({ id, contentUrl }) => [id, contentUrl])).toEqual([
      ["root", undefined],
      ["root/0", undefined],
      ["root/0/0", "/tiles/content/west-detail.glb"],
      ["root/1", "/tiles/content/east.gltf"],
    ]);
  });

  it("reports transport, status, and malformed JSON as typed fetch errors", async () => {
    await expect(
      loadTileset({
        endpoint: "/tiles",
        fetch: vi.fn().mockRejectedValue(new TypeError("offline")),
      }),
    ).rejects.toMatchObject({
      name: "TilesetFetchError",
      url: "/tiles/tileset.json",
    });

    await expect(
      loadTileset({
        endpoint: "/tiles",
        fetch: vi.fn().mockResolvedValue(response({}, 410)),
      }),
    ).rejects.toMatchObject({ name: "TilesetFetchError", status: 410 });

    const badJson = response({});
    badJson.json.mockRejectedValue(new SyntaxError("bad json"));
    await expect(
      loadTileset({
        endpoint: "/tiles",
        fetch: vi.fn().mockResolvedValue(badJson),
      }),
    ).rejects.toBeInstanceOf(TilesetFetchError);
  });

  it.each([
    ["missing asset", { ...document(), asset: undefined }, "asset"],
    [
      "wrong version",
      { ...document(), asset: { version: "1.0" } },
      "asset.version",
    ],
    [
      "missing tileset error",
      { ...document(), geometricError: undefined },
      "geometricError",
    ],
    ["missing root", { ...document(), root: undefined }, "root"],
    [
      "null content",
      { ...document(), root: { ...tile(), content: null } },
      "root.content",
    ],
    [
      "missing URI",
      { ...document(), root: { ...tile(), content: {} } },
      "root.content.uri",
    ],
    [
      "missing refinement",
      { ...document(), root: { ...tile(), refine: undefined } },
      "root.refine",
    ],
    [
      "wrong refinement",
      { ...document(), root: { ...tile(), refine: "ADD" } },
      "root.refine",
    ],
    [
      "negative error",
      { ...document(), root: { ...tile(), geometricError: -1 } },
      "root.geometricError",
    ],
    [
      "nonnumeric error",
      { ...document(), root: { ...tile(), geometricError: "8" } },
      "root.geometricError",
    ],
    [
      "nonfinite transform",
      {
        ...document(),
        root: { ...tile(), transform: [...identity.slice(0, 15), Infinity] },
      },
      "root.transform",
    ],
    [
      "non-affine transform",
      documentWithRootTransform([1, 0, 0, 1, ...identity.slice(4)]),
      "root.transform",
    ],
    [
      "short box",
      {
        ...document(),
        root: { ...tile(), boundingVolume: { box: box.slice(0, 11) } },
      },
      "root.boundingVolume.box",
    ],
    [
      "degenerate box",
      {
        ...document(),
        root: {
          ...tile(),
          boundingVolume: { box: [...box.slice(0, 9), 0, 0, 0] },
        },
      },
      "root.boundingVolume.box",
    ],
    [
      "bad children",
      { ...document(), root: { ...tile(), children: {} } },
      "root.children",
    ],
  ])("rejects invalid documents: %s", async (_name, body, path) => {
    await expect(
      loadTileset({
        endpoint: "/tiles",
        fetch: vi.fn().mockResolvedValue(response(body)),
      }),
    ).rejects.toMatchObject({ name: "TilesetValidationError", path });
  });

  it.each([
    ["implicit tiling", { ...tile(), implicitTiling: {} }, "implicitTiling"],
    [
      "multiple contents",
      { ...tile(), contents: [{ uri: "a.glb" }] },
      "contents",
    ],
    [
      "sphere volume",
      { ...tile(), boundingVolume: { sphere: [0, 0, 0, 1] } },
      "boundingVolume",
    ],
    [
      "region volume",
      { ...tile(), boundingVolume: { region: [0, 0, 0, 0, 0, 0] } },
      "boundingVolume",
    ],
    ["legacy url", { ...tile(), content: { uri: "root.b3dm" } }, "content.uri"],
    [
      "legacy content URL key",
      { ...tile(), content: { url: "root.glb" } },
      "content.url",
    ],
    [
      "point content",
      { ...tile(), content: { uri: "root.pnts" } },
      "content.uri",
    ],
    [
      "nested tileset",
      { ...tile(), content: { uri: "nested.json" } },
      "content.uri",
    ],
  ])(
    "rejects unsupported profile features: %s",
    async (_name, root, feature) => {
      await expect(
        loadTileset({
          endpoint: "/tiles",
          fetch: vi.fn().mockResolvedValue(response({ ...document(), root })),
        }),
      ).rejects.toMatchObject({ name: "TilesetUnsupportedError", feature });
    },
  );

  it("preserves the typed error hierarchy", () => {
    expect(new TilesetUnsupportedError("x", "root.x")).toBeInstanceOf(Error);
    expect(new TilesetValidationError("root.x", "bad")).toBeInstanceOf(Error);
  });
});

describe("resolveTilesetContentUri", () => {
  it.each([
    ["content/a.glb", "/tiles/rev/content/a.glb"],
    ["safe%20name.glb", "/tiles/rev/safe%20name.glb"],
    ["nested/safe%20name.glb", "/tiles/rev/nested/safe%20name.glb"],
    ["nested/./a.gltf", "/tiles/rev/nested/a.gltf"],
  ])("resolves safe same-root references", (uri, expected) => {
    expect(resolveTilesetContentUri("/tiles/rev/", uri)).toBe(expected);
  });

  it.each([
    "https://example.test/a.glb",
    "file:a.glb",
    "//example.test/a.glb",
    "/absolute/a.glb",
    "C:/absolute/a.glb",
    "../escape.glb",
    "nested/../../escape.glb",
    "%2e%2e/escape.glb",
    "%252e%252e/escape.glb",
    "nested%2f..%2fescape.glb",
    "https%3a%2f%2fexample.test%2fa.glb",
    "nested\\escape.glb",
    "a.glb?remote=https://example.test",
    "a.glb#fragment",
  ])("rejects unsafe URI %s", (uri) => {
    expect(() => resolveTilesetContentUri("/tiles/rev", uri)).toThrow(
      TilesetValidationError,
    );
  });
});
