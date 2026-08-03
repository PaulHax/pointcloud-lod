/**
 * End-to-end CopcTileSource tests against the committed fixture
 * (test/fixtures/fixture.copc.laz — 2000 points, full-range 16-bit RGB, two
 * hierarchy levels; regenerate with test/fixtures/generate_fixture.py).
 * Decoding runs through the real laz-perf WASM in node.
 */

import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Copc, Getter, type Hierarchy } from "copc";

import { createCopcTileSource } from "./copcTileSource";
import { ROOT_KEY, keyFromString, keyToString } from "./octree";
import type { TileSource } from "./tileSource";

const FIXTURE = fileURLToPath(
  new URL("../test/fixtures/fixture.copc.laz", import.meta.url),
);

const makeSource = () => createCopcTileSource({ source: Getter.file(FIXTURE) });

/** One synthetic node: interleaved RGB, absent for a format without color. */
type StubNode = {
  pointCount: number;
  channels?: number[];
  /** Simulates a node whose point data cannot be read. */
  unreadable?: boolean;
  /** Make this node consume one byte range while loading. */
  readSourceOnLoad?: { begin: number; end: number };
};

type StubAsset = {
  pointDataRecordFormat: number;
  /** Keyed by "level-x-y-z"; "0-0-0-0" is the node the shift is sampled from. */
  nodes: Record<string, StubNode>;
  /** Make the stub consume one byte range while opening. */
  readSourceAtOpen?: { begin: number; end: number };
};

/**
 * RGB-depth cases no committed fixture can express — dark and bright
 * neighbours in one cloud, 8-bit channels, an unreadable root, a colorless
 * point format — need a synthetic asset, so an installed stub stands in for
 * the copc reader. With nothing installed every call reaches the real package
 * and the fixture still decodes through laz-perf.
 */
const stub = vi.hoisted(() => ({
  asset: null as StubAsset | null,
  viewLoads: 0,
  pageLoads: 0,
}));

vi.mock("copc", async (importOriginal) => {
  const actual = await importOriginal<typeof import("copc")>();
  const asset = () => stub.asset;
  const nodeKeys = (a: StubAsset) => Object.keys(a.nodes);

  return {
    ...actual,
    Copc: {
      ...actual.Copc,
      create: async (source: never) => {
        const a = asset();
        if (a === null) return actual.Copc.create(source);
        if (a.readSourceAtOpen !== undefined) {
          await (
            source as unknown as (
              begin: number,
              end: number,
            ) => Promise<Uint8Array>
          )(a.readSourceAtOpen.begin, a.readSourceAtOpen.end);
        }
        const nodes = Object.values(a.nodes);
        return {
          header: {
            pointCount: nodes.reduce((sum, n) => sum + n.pointCount, 0),
            pointDataRecordFormat: a.pointDataRecordFormat,
            // A data extent well inside the octree cube below, which is what a
            // real survey looks like and what makes the two distinguishable.
            min: [1, 2, 3],
            max: [21, 12, 7],
          },
          info: {
            cube: [0, 0, 0, 64, 64, 64],
            spacing: 1,
            rootHierarchyPage: { pageOffset: 0, pageLength: 0 },
          },
        };
      },
      loadHierarchyPage: async (source: never, page: never) => {
        stub.pageLoads += 1;
        const a = asset();
        if (a === null) return actual.Copc.loadHierarchyPage(source, page);
        const nodes = Object.fromEntries(
          nodeKeys(a).map((key, index) => [
            key,
            // The index doubles as the node's identity for view loads below.
            {
              pointCount: a.nodes[key]!.pointCount,
              pointDataOffset: index,
              pointDataLength: 0,
            },
          ]),
        );
        return { nodes, pages: {} };
      },
      loadPointDataView: async (
        source: never,
        copc: never,
        node: Hierarchy.Node,
      ) => {
        const a = asset();
        if (a === null) {
          return actual.Copc.loadPointDataView(source, copc, node);
        }
        stub.viewLoads += 1;
        const spec = a.nodes[nodeKeys(a)[node.pointDataOffset]!]!;
        if (spec.unreadable) throw new Error("point data unavailable");
        if (spec.readSourceOnLoad !== undefined) {
          await (
            source as unknown as (
              begin: number,
              end: number,
            ) => Promise<Uint8Array>
          )(spec.readSourceOnLoad.begin, spec.readSourceOnLoad.end);
        }
        const channelOffset: Record<string, number> = {
          Red: 0,
          Green: 1,
          Blue: 2,
        };
        return {
          pointCount: spec.pointCount,
          dimensions: {},
          getter: (name: string) => {
            const offset = channelOffset[name];
            if (offset === undefined) return () => 0; // X/Y/Z
            if (spec.channels === undefined) {
              throw new Error(`No dimension ${name}`);
            }
            return (index: number) => spec.channels![index * 3 + offset]!;
          },
        };
      },
    },
  };
});

/** Opens a stubbed asset with its hierarchy already enumerated. */
const openStub = async (asset: StubAsset): Promise<TileSource> => {
  stub.asset = asset;
  const source = await createCopcTileSource({
    source: async () => new Uint8Array(),
  });
  await source.nodes(ROOT_KEY);
  return source;
};

const loadRgb = async (source: TileSource, key: string) =>
  (await source.loadTile(keyFromString(key))).rgb;

/**
 * The `reg-ui` HTTP tile service rule, transcribed from
 * `app/telesculptor_web/app/pointcloud_tiles/service.py`: `_asset_rgb_shift`
 * samples the root node once — 8 when any channel there exceeds 255 — and
 * `_rgb_to_uint8` applies `(rgb16 >> shift).astype(np.uint8)` to every tile of
 * the asset. Every asset asserted against this is one the sample decides
 * correctly, so each shifted channel already fits a byte and the clipping this
 * source adds cannot move it — agreement holds byte for byte.
 */
const serviceRgb = (rootChannels: number[], channels: number[]): Uint8Array => {
  const shift = Math.max(...rootChannels) > 255 ? 8 : 0;
  return Uint8Array.from(channels, (c) => (c >> shift) & 0xff);
};

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
  stub.asset = null;
  stub.viewLoads = 0;
  stub.pageLoads = 0;
});

describe("createCopcTileSource", () => {
  it("retries a transient HTTP range failure before opening the source", async () => {
    vi.useFakeTimers();
    const url = "https://example.test/cloud.copc.laz";
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(new Response(new Uint8Array(10), { status: 206 }));
    stub.asset = {
      pointDataRecordFormat: 0,
      nodes: { "0-0-0-0": { pointCount: 0 } },
      readSourceAtOpen: { begin: 10, end: 20 },
    };

    const opening = createCopcTileSource({ source: url });
    await vi.runAllTimersAsync();
    await expect(opening).resolves.toBeDefined();
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(fetchMock).toHaveBeenLastCalledWith(url, {
      headers: { Range: "bytes=10-19" },
    });
  });

  it("accepts a shorter final range when Content-Range proves EOF", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(new Uint8Array(5), {
        status: 206,
        headers: { "Content-Range": "bytes 10-14/15" },
      }),
    );
    stub.asset = {
      pointDataRecordFormat: 0,
      nodes: { "0-0-0-0": { pointCount: 0 } },
      readSourceAtOpen: { begin: 10, end: 20 },
    };

    await expect(
      createCopcTileSource({ source: "https://example.test/small.copc.laz" }),
    ).resolves.toBeDefined();
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it("retries and rejects a short range that does not reach EOF", async () => {
    vi.useFakeTimers();
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(() =>
      Promise.resolve(
        new Response(new Uint8Array(5), {
          status: 206,
          headers: { "Content-Range": "bytes 10-14/20" },
        }),
      ),
    );
    stub.asset = {
      pointDataRecordFormat: 0,
      nodes: { "0-0-0-0": { pointCount: 0 } },
      readSourceAtOpen: { begin: 10, end: 20 },
    };

    const opening = createCopcTileSource({
      source: "https://example.test/truncated.copc.laz",
    });
    const rejected = expect(opening).rejects.toThrow(
      "Expected 10 bytes, received 5",
    );
    await vi.runAllTimersAsync();
    await rejected;
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("cancels a pending HTTP retry when the tile is no longer needed", async () => {
    vi.useFakeTimers();
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockRejectedValue(new TypeError("Failed to fetch"));
    stub.asset = {
      pointDataRecordFormat: 0,
      nodes: {
        "0-0-0-0": {
          pointCount: 1,
          readSourceOnLoad: { begin: 20, end: 30 },
        },
      },
    };
    const source = await createCopcTileSource({
      source: "https://example.test/cloud.copc.laz",
    });
    const controller = new AbortController();
    const loading = source.loadTile(ROOT_KEY, { signal: controller.signal });
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));

    controller.abort();
    await expect(loading).rejects.toMatchObject({ name: "AbortError" });
    await vi.runAllTimersAsync();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0]![1]).toMatchObject({
      signal: controller.signal,
    });
  });

  it("reads metadata from the COPC info VLR", async () => {
    const source = await makeSource();
    const meta = source.metadata();
    expect(meta.pointCount).toBe(2000);
  });

  it("states the data's own extent, not the octree cube enclosing it", async () => {
    const source = await makeSource();
    const bounds = source.metadata().bounds;
    expect(bounds, "the fixture's header carries an extent").toBeDefined();

    const root = (await source.nodes(ROOT_KEY)).find(
      (n) => keyToString(n.key) === "0-0-0-0",
    )!;
    for (let axis = 0; axis < 3; axis += 1) {
      // Inside the cube, and — on a fixture that is not a perfect cube — some
      // axis is strictly inside it. That difference is the whole reason a host
      // frames from this rather than from the node bounds.
      expect(bounds!.min[axis]).toBeGreaterThanOrEqual(root.bounds.min[axis]!);
      expect(bounds!.max[axis]).toBeLessThanOrEqual(root.bounds.max[axis]!);
    }
    const cubeSide = root.bounds.max[0]! - root.bounds.min[0]!;
    const spans = [0, 1, 2].map(
      (axis) => bounds!.max[axis]! - bounds!.min[axis]!,
    );
    expect(Math.min(...spans)).toBeLessThan(cubeSide);
  });

  it("reads the root hierarchy page once, however early it is wanted", async () => {
    // Opening samples RGB from the root node, a host frames the scene from the
    // hierarchy, and the controller bootstraps from it. The page is immutable,
    // so all three must come out of one range read.
    const source = await makeSource();
    const before = stub.pageLoads;
    expect(await source.nodes(ROOT_KEY)).toEqual(await source.nodes(ROOT_KEY));
    expect(stub.pageLoads).toBe(before);
    expect(before).toBe(1);
  });

  it("enumerates the hierarchy with point counts summing to the total", async () => {
    const source = await makeSource();
    const nodes = await source.nodes(ROOT_KEY);
    expect(nodes.length).toBeGreaterThanOrEqual(2); // capacity forces a split
    const total = nodes.reduce((sum, n) => sum + n.pointCount, 0);
    expect(total).toBe(2000);
    const root = nodes.find((n) => keyToString(n.key) === "0-0-0-0");
    expect(root).toBeDefined();
    expect(root!.pointCount).toBeGreaterThan(0);
    expect(root!.bounds.min[0]).toBeCloseTo(100, 1);
    expect(root!.bounds.max[0]).toBeCloseTo(140, 1);
    expect(root!.spacing).toBeGreaterThan(0);
  });

  it("loads tiles with tile-local Float32 positions and 8-bit RGB", async () => {
    const source = await makeSource();
    const nodes = await source.nodes(ROOT_KEY);
    for (const info of nodes.filter((n) => n.pointCount > 0).slice(0, 3)) {
      const tile = await source.loadTile(info.key);
      expect(tile.pointCount).toBe(info.pointCount);
      expect(tile.positions).toHaveLength(info.pointCount * 3);

      // Absolute reconstruction stays inside the node geometry contract.
      for (let i = 0; i < Math.min(tile.pointCount, 50); i += 1) {
        const x = tile.origin[0] + tile.positions[i * 3]!;
        const y = tile.origin[1] + tile.positions[i * 3 + 1]!;
        const z = tile.origin[2] + tile.positions[i * 3 + 2]!;
        expect(x).toBeGreaterThanOrEqual(info.bounds.min[0] - 0.01);
        expect(x).toBeLessThanOrEqual(info.bounds.max[0] + 0.01);
        expect(y).toBeGreaterThanOrEqual(info.bounds.min[1] - 0.01);
        expect(y).toBeLessThanOrEqual(info.bounds.max[1] + 0.01);
        expect(z).toBeGreaterThanOrEqual(info.bounds.min[2] - 0.01);
        expect(z).toBeLessThanOrEqual(info.bounds.max[2] + 0.01);
      }

      // Full-range 16-bit fixture colors arrive scaled to full-range 8-bit.
      expect(tile.rgb).toHaveLength(info.pointCount * 3);
      expect(Math.max(...tile.rgb!)).toBeGreaterThan(127);
    }
  });

  it("shifts every fixture node by the depth sampled at the root", async () => {
    const getter = Getter.file(FIXTURE);
    const copc = await Copc.create(getter);
    const subtree = await Copc.loadHierarchyPage(
      getter,
      copc.info.rootHierarchyPage,
    );
    const source = await createCopcTileSource({ source: getter });
    const infos = await source.nodes(ROOT_KEY);

    for (const info of infos.filter((n) => n.pointCount > 0)) {
      const tile = await source.loadTile(info.key);
      const view = await Copc.loadPointDataView(
        getter,
        copc,
        subtree.nodes[keyToString(info.key)]!,
      );
      const raw = [
        view.getter("Red"),
        view.getter("Green"),
        view.getter("Blue"),
      ];
      for (let i = 0; i < Math.min(tile.pointCount, 40); i += 1) {
        for (const [channel, get] of raw.entries()) {
          expect(tile.rgb![i * 3 + channel]).toBe(get(i) >> 8);
        }
      }
    }
  });

  it("rejects when aborted before decode", async () => {
    const source = await makeSource();
    const nodes = await source.nodes(ROOT_KEY);
    const first = nodes.find((n) => n.pointCount > 0)!;
    const controller = new AbortController();
    controller.abort();
    await expect(
      source.loadTile(first.key, { signal: controller.signal }),
    ).rejects.toMatchObject({ name: "AbortError" });
  });

  it("rejects a hierarchy page aborted before it starts, reading nothing", async () => {
    let reads = 0;
    const base = Getter.file(FIXTURE);
    const source = await createCopcTileSource({
      source: async (begin, end) => {
        reads += 1;
        return base(begin, end);
      },
    });
    const nodes = await source.nodes(ROOT_KEY);
    const first = nodes.find((n) => n.pointCount > 0)!;
    const before = reads;

    const controller = new AbortController();
    controller.abort();
    await expect(
      source.nodes(ROOT_KEY, { signal: controller.signal }),
    ).rejects.toMatchObject({ name: "AbortError" });
    await expect(
      source.loadTile(first.key, { signal: controller.signal }),
    ).rejects.toMatchObject({ name: "AbortError" });
    expect(reads).toBe(before);
  });

  it("issues no further range reads once the signal fires", async () => {
    const controller = new AbortController();
    let reads = 0;
    let armed = false;
    const base = Getter.file(FIXTURE);
    const source = await createCopcTileSource({
      source: async (begin, end) => {
        reads += 1;
        // Cancel from inside the operation's own first read.
        if (armed) controller.abort();
        return base(begin, end);
      },
    });
    const nodes = await source.nodes(ROOT_KEY);
    const first = nodes.find((n) => n.pointCount > 0)!;

    armed = true;
    const before = reads;
    await expect(
      source.loadTile(first.key, { signal: controller.signal }),
    ).rejects.toMatchObject({ name: "AbortError" });
    // The read already in flight finishes; nothing after it is issued.
    expect(reads).toBe(before + 1);
  });

  it("refuses tiles whose hierarchy page was never loaded", async () => {
    const source = await makeSource();
    await expect(
      source.loadTile({ level: 9, x: 0, y: 0, z: 0 }),
    ).rejects.toThrow(/Hierarchy not loaded/);
  });
});

describe("createCopcTileSource RGB depth", () => {
  const EIGHT_BIT: StubAsset = {
    pointDataRecordFormat: 2,
    nodes: {
      "0-0-0-0": { pointCount: 2, channels: [10, 20, 30, 250, 240, 230] },
      "1-0-0-0": { pointCount: 2, channels: [7, 8, 9, 128, 129, 130] },
    },
  };

  const FULL_RANGE: StubAsset = {
    pointDataRecordFormat: 2,
    nodes: {
      "0-0-0-0": {
        pointCount: 2,
        channels: [0, 32768, 65535, 4096, 8192, 16384],
      },
      "1-0-0-0": { pointCount: 2, channels: [65535, 32768, 256, 1, 2, 3] },
    },
  };

  // One 16-bit cloud whose root spans the full range, split into a dark node
  // and a bright node: sampling either child alone disagrees with the other.
  const DARK = [12, 24, 200, 255, 100, 3];
  const BRIGHT = [40000, 41000, 42000, 300, 400, 500];
  // Channels on both sides of 255, for the cases where the asset-wide sample
  // decides no shift: the small ones must survive untouched, the large ones
  // must clip rather than wrap round to near-black.
  const MIXED = [40000, 200, 42000, 30, 400, 50];
  const MIXED_CLIPPED = new Uint8Array([255, 200, 255, 30, 255, 50]);
  const NEIGHBOURS: StubAsset = {
    pointDataRecordFormat: 2,
    nodes: {
      "0-0-0-0": { pointCount: 2, channels: [40000, 50000, 60000, 20, 30, 40] },
      "1-0-0-0": { pointCount: 2, channels: DARK },
      "1-1-0-0": { pointCount: 2, channels: BRIGHT },
    },
  };

  it("leaves an 8-bit cloud unshifted", async () => {
    const source = await openStub(EIGHT_BIT);
    expect(await loadRgb(source, "0-0-0-0")).toEqual(
      new Uint8Array([10, 20, 30, 250, 240, 230]),
    );
    expect(await loadRgb(source, "1-0-0-0")).toEqual(
      new Uint8Array([7, 8, 9, 128, 129, 130]),
    );
  });

  it("shifts a full-range 16-bit cloud by eight bits", async () => {
    const source = await openStub(FULL_RANGE);
    expect(await loadRgb(source, "0-0-0-0")).toEqual(
      new Uint8Array([0, 128, 255, 16, 32, 64]),
    );
    expect(await loadRgb(source, "1-0-0-0")).toEqual(
      new Uint8Array([255, 128, 1, 0, 0, 0]),
    );
  });

  it("gives neighbouring dark and bright nodes the same shift", async () => {
    const source = await openStub(NEIGHBOURS);
    const shifted = (channels: number[]) =>
      Uint8Array.from(channels, (c) => c >> 8);
    const dark = await loadRgb(source, "1-0-0-0");
    expect(dark).toEqual(shifted(DARK));
    expect(await loadRgb(source, "1-1-0-0")).toEqual(shifted(BRIGHT));
    // Deciding from this node's own maximum would have left it unshifted and
    // banded it against its neighbour.
    expect(dark).not.toEqual(Uint8Array.from(DARK));
  });

  it("loads tiles from a point format without color, sampling nothing", async () => {
    const source = await openStub({
      pointDataRecordFormat: 1,
      nodes: { "0-0-0-0": { pointCount: 3 }, "1-0-0-0": { pointCount: 2 } },
    });
    const tile = await source.loadTile(keyFromString("1-0-0-0"));
    expect(tile.rgb).toBeUndefined();
    expect(tile.pointCount).toBe(2);
    expect(tile.positions).toHaveLength(6);
    // Only the tile itself was decoded: a colorless format needs no sample.
    expect(stub.viewLoads).toBe(1);
  });

  it("decodes the root node once, sampling and drawing from the same read", async () => {
    // The shift sample must decode the root, and the controller asks for that
    // same node as its first tile.
    const source = await openStub(FULL_RANGE);
    expect(stub.viewLoads).toBe(1);
    expect(await loadRgb(source, "0-0-0-0")).toEqual(
      new Uint8Array([0, 128, 255, 16, 32, 64]),
    );
    expect(stub.viewLoads).toBe(1);
    // Held for one read only; a second one goes back to the file.
    await loadRgb(source, "0-0-0-0");
    expect(stub.viewLoads).toBe(2);
  });

  it("falls back to no shift when the root node cannot be sampled", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const source = await openStub({
      pointDataRecordFormat: 2,
      nodes: {
        "0-0-0-0": { pointCount: 2, channels: BRIGHT, unreadable: true },
        "1-0-0-0": { pointCount: 2, channels: MIXED },
      },
    });
    expect(await loadRgb(source, "1-0-0-0")).toEqual(MIXED_CLIPPED);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it("clips a 16-bit channel the root sample missed instead of wrapping", async () => {
    // A root dark enough to read as 8-bit sitting over a node that is not: the
    // asset-wide decision misses, which is exactly when the store would wrap.
    const source = await openStub({
      pointDataRecordFormat: 2,
      nodes: {
        "0-0-0-0": { pointCount: 1, channels: [12, 24, 200] },
        "1-0-0-0": { pointCount: 2, channels: MIXED },
      },
    });
    // Wrapping would send 40000 to 64 and the brighter 42000 to 16: the top of
    // the cloud comes back darker than its own mid-tones, and out of order with
    // itself, which reads as a broken decoder rather than a washed-out sample.
    expect(await loadRgb(source, "1-0-0-0")).toEqual(MIXED_CLIPPED);
  });

  it("produces the RGB bytes the HTTP tile service would serve", async () => {
    for (const asset of [EIGHT_BIT, FULL_RANGE, NEIGHBOURS]) {
      const source = await openStub(asset);
      const rootChannels = asset.nodes["0-0-0-0"]!.channels!;
      for (const [key, node] of Object.entries(asset.nodes)) {
        expect(await loadRgb(source, key)).toEqual(
          serviceRgb(rootChannels, node.channels!),
        );
      }
    }
  });
});
