import { createEcefToEnuTransform } from "../../../src/tiles3d/rtc";
/** NYT Freeman Alley example: expand external indexes and unwrap unbatched GLBs. */
import type { TilesetFetch } from "../../../src/tiles3d/tilesetSource";
import type { ContentQueueFetch } from "../../../src/tiles3d/contentQueue";

export const FREEMAN_URL =
  "https://int.nyt.com/data/3dscenes/ONA360/TILESET/0731_FREEMAN_ALLEY_10M_A_36x8K__10K-PN_50P_DB/tileset_tileset.json";
export const FREEMAN_LABEL = "Freeman Alley · textured mesh (NYT)";
const endpoint = new URL("./", FREEMAN_URL).href.replace(/\/$/, "");

type Tile = {
  content?: { uri?: string; url?: string };
  children?: Tile[];
  refine?: string;
  [key: string]: unknown;
};
type Document = { root: Tile; geometricError: number };

export const unwrapFreemanContent = (buffer: ArrayBuffer): ArrayBuffer => {
  const header = new DataView(buffer);
  if (
    buffer.byteLength < 28 ||
    header.getUint32(0, true) !== 0x6d643362 ||
    header.getUint32(4, true) !== 1 ||
    header.getUint32(8, true) !== buffer.byteLength
  ) {
    throw new Error("Invalid Freeman b3dm header");
  }
  const featureLength = header.getUint32(12, true);
  const offset =
    28 +
    [12, 16, 20, 24].reduce(
      (sum, position) => sum + header.getUint32(position, true),
      0,
    );
  if (offset + 12 > buffer.byteLength)
    throw new Error("Truncated Freeman b3dm");
  const feature = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buffer, 28, featureLength)),
  ) as { BATCH_LENGTH?: number; RTC_CENTER?: number[] };
  if (
    (feature.BATCH_LENGTH ?? 0) !== 0 ||
    feature.RTC_CENTER?.some((v) => v !== 0)
  ) {
    throw new Error(
      "Freeman content requires unsupported feature-table transforms",
    );
  }
  const glb = buffer.slice(offset);
  const view = new DataView(glb);
  if (
    view.getUint32(0, true) !== 0x46546c67 ||
    view.getUint32(8, true) !== glb.byteLength
  ) {
    throw new Error("Invalid embedded Freeman GLB");
  }
  return glb;
};

export const createFreemanSource = () => {
  const contents = new Map<string, string>();
  const fetchTileset: TilesetFetch = async (_url, { signal }) => {
    let active = 0;
    const waiting: (() => void)[] = [];
    const read = async (url: string): Promise<Document> => {
      if (active >= 4)
        await new Promise<void>((resolve) => waiting.push(resolve));
      else active += 1;
      try {
        signal?.throwIfAborted();
        const response = await fetch(url, { signal });
        if (!response.ok)
          throw new Error(`Freeman index: HTTP ${response.status}`);
        return (await response.json()) as Document;
      } finally {
        const next = waiting.shift();
        if (next) next();
        else active -= 1;
      }
    };
    const expand = async (node: Tile, base: string): Promise<Tile> => {
      const { content, children = [], ...rest } = node;
      // Resolve the hierarchy concurrently, with at most four index fetches.
      const result: Tile = {
        ...rest,
        children: await Promise.all(
          children.map((child) => expand(child, base)),
        ),
      };
      const uri = content?.uri ?? content?.url;
      if (uri) {
        const url = new URL(uri, base).href;
        if (!url.startsWith(`${endpoint}/`))
          throw new Error("Freeman content outside its root");
        if (url.endsWith(".json")) {
          result.children!.push(await expand((await read(url)).root, url));
        } else if (url.endsWith(".b3dm")) {
          const path = url
            .slice(endpoint.length + 1)
            .replace(/\.b3dm$/, ".glb");
          contents.set(`${endpoint}/${path}`, url);
          result.content = { uri: path };
        } else throw new Error(`Unsupported Freeman content: ${uri}`);
      }
      if (!result.children!.length) delete result.children;
      return result;
    };
    const document = await read(FREEMAN_URL);
    const root = await expand(document.root, FREEMAN_URL);
    root.refine = "REPLACE";
    return {
      ok: true,
      status: 200,
      statusText: "OK",
      json: async () => ({
        asset: { version: "1.1" },
        geometricError: document.geometricError,
        root,
      }),
    };
  };
  const fetchContent: ContentQueueFetch = async (url, init) => {
    const source = contents.get(url);
    if (!source) throw new Error("Unknown Freeman content");
    const response = await fetch(source, init);
    return {
      ok: response.ok,
      status: response.status,
      statusText: response.statusText,
      arrayBuffer: async () =>
        unwrapFreemanContent(await response.arrayBuffer()),
    };
  };
  return {
    label: FREEMAN_LABEL,
    endpoint,
    // Local ENU at the source's published root origin, not a relocation.
    tilesetToScene: createEcefToEnuTransform(
      16.7868877262,
      47.8752391091,
      1.29264,
    ),
    view: { center: [0, 0, 10] as [number, number, number], distance: 100 },
    fetchTileset,
    fetchContent,
    location: { kind: "tiles-url" as const, value: FREEMAN_URL },
  };
};
