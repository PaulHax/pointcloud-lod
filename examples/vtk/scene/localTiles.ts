import type { ContentQueueFetch } from "../../../src/tiles3d/contentQueue";
import type { TilesetFetch } from "../../../src/tiles3d/tilesetSource";

export type LocalTilesSource = {
  readonly endpoint: string;
  readonly label: string;
  readonly fetchTileset: TilesetFetch;
  readonly fetchContent: ContentQueueFetch;
};

const relativeName = (file: File): string =>
  (file.webkitRelativePath || file.name).replace(/^\/+/, "");

const withoutSharedRoot = (names: readonly string[]): string[] => {
  const firstSegments = names.map((name) => name.split("/")[0]);
  const shared =
    firstSegments.length > 0 &&
    firstSegments.every((segment) => segment === firstSegments[0]);
  return shared && names.every((name) => name.includes("/"))
    ? names.map((name) => name.slice(name.indexOf("/") + 1))
    : [...names];
};

/** Build fetch seams for a selected local 3D Tiles directory. */
export const createLocalTilesSource = (
  selected: readonly File[],
): LocalTilesSource => {
  if (selected.length === 0) throw new Error("Choose a 3D Tiles directory.");
  const names = withoutSharedRoot(selected.map(relativeName));
  const files = new Map(names.map((name, index) => [name, selected[index]!]));
  const tilesetName =
    names.find((name) => name.toLowerCase() === "tileset.json") ??
    names.find((name) => name.toLowerCase().endsWith("/tileset.json"));
  if (!tilesetName) {
    throw new Error("The selected directory does not contain tileset.json.");
  }

  const root = tilesetName.slice(0, tilesetName.lastIndexOf("/") + 1);
  const rooted = new Map<string, File>();
  for (const [name, file] of files) {
    if (name.startsWith(root)) rooted.set(name.slice(root.length), file);
  }
  const token = crypto.randomUUID();
  const endpoint = `https://local-tiles.invalid/${token}`;
  const fileFor = (url: string): File | undefined => {
    const prefix = `/${token}/`;
    const path = decodeURIComponent(new URL(url).pathname);
    return path.startsWith(prefix)
      ? rooted.get(path.slice(prefix.length))
      : undefined;
  };

  const fetchTileset: TilesetFetch = async (url) => {
    const file = fileFor(url);
    return file
      ? {
          ok: true,
          status: 200,
          statusText: "OK",
          json: async () => JSON.parse(await file.text()) as unknown,
        }
      : {
          ok: false,
          status: 404,
          statusText: "Local file not found",
          json: async () => null,
        };
  };
  const fetchContent: ContentQueueFetch = async (url) => {
    const file = fileFor(url);
    return file
      ? {
          ok: true,
          status: 200,
          statusText: "OK",
          arrayBuffer: () => file.arrayBuffer(),
        }
      : {
          ok: false,
          status: 404,
          statusText: "Local file not found",
          arrayBuffer: async () => new ArrayBuffer(0),
        };
  };
  return {
    endpoint,
    label: selected[0]!.webkitRelativePath.split("/")[0] || "Local 3D Tiles",
    fetchTileset,
    fetchContent,
  };
};
