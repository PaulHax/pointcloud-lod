import { afterEach, expect, it, vi } from "vitest";
import {
  createFreemanSource,
  FREEMAN_URL,
  unwrapFreemanContent,
} from "./freeman";
import { parseTileset } from "../../../src/tiles3d/tilesetSource";

afterEach(() => vi.unstubAllGlobals());

const wrapped = (feature: object = { BATCH_LENGTH: 0 }): ArrayBuffer => {
  const json = new TextEncoder().encode(JSON.stringify(feature));
  const buffer = new ArrayBuffer(28 + json.length + 12);
  const view = new DataView(buffer);
  [0x6d643362, 1, buffer.byteLength, json.length, 0, 0, 0].forEach(
    (value, index) => view.setUint32(index * 4, value, true),
  );
  new Uint8Array(buffer, 28, json.length).set(json);
  const offset = 28 + json.length;
  view.setUint32(offset, 0x46546c67, true);
  view.setUint32(offset + 4, 2, true);
  view.setUint32(offset + 8, 12, true);
  return buffer;
};

it("preserves GLB bytes and rejects feature transforms it cannot preserve", () => {
  const source = wrapped();
  expect(unwrapFreemanContent(source)).toEqual(source.slice(-12));
  expect(() =>
    unwrapFreemanContent(wrapped({ RTC_CENTER: [1, 0, 0] })),
  ).toThrow();
  expect(() => unwrapFreemanContent(wrapped({ BATCH_LENGTH: 1 }))).toThrow();
  expect(() => unwrapFreemanContent(source.slice(0, 30))).toThrow();
});

it("expands external roots without losing placement and fetches their original content", async () => {
  const box = [0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 10];
  const transform = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 20, 30, 1];
  const fetcher = vi.fn(async (url: string) => {
    if (url.endsWith(".b3dm")) return new Response(wrapped());
    return Response.json({
      geometricError: 10,
      root: {
        boundingVolume: { box },
        geometricError: 10,
        ...(url === FREEMAN_URL
          ? { transform, content: { url: "part/index.json" } }
          : { content: { uri: "mesh.b3dm" } }),
      },
    });
  });
  vi.stubGlobal("fetch", fetcher);
  const source = createFreemanSource();
  const response = await source.fetchTileset("unused", {});
  const document = await response.json();
  const parsed = parseTileset(document, source.endpoint);
  expect(parsed.tiles.length).toBe(2);
  expect(document).toMatchObject({ root: { transform } });
  const signal = new AbortController().signal;
  const content = await source.fetchContent(
    `${source.endpoint}/part/mesh.glb`,
    { signal },
  );
  expect(await content.arrayBuffer()).toEqual(wrapped().slice(-12));
  expect(fetcher).toHaveBeenLastCalledWith(
    `${source.endpoint}/part/mesh.b3dm`,
    { signal },
  );
});
