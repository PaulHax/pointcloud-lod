/**
 * Strict, renderer-neutral reader for the explicit/implicit 3D Tiles 1.1 profile used
 * by the streamed-scene mesh member.
 */

import { multiplyMat4Values } from "./rtc";

export type TilesetFetchResponse = {
  readonly ok: boolean;
  readonly status: number;
  readonly statusText: string;
  json(): Promise<unknown>;
};

export type TilesetFetch = (
  url: string,
  init: { readonly signal?: AbortSignal },
) => Promise<TilesetFetchResponse>;

export type TilesetBox = {
  readonly center: readonly [number, number, number];
  /** Three column vectors (x, y, z half axes), flattened in that order. */
  readonly halfAxes: readonly [
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
  ];
};

export type TilesetTile = {
  /** Stable explicit-hierarchy path (`root`, `root/0`, ...). */
  readonly id: string;
  readonly geometricError: number;
  readonly boundingVolume: TilesetBox;
  /** Tile-local column-major affine transform. */
  readonly transform: readonly number[];
  /** Parent-to-child accumulated column-major transform, retained as f64 numbers. */
  readonly worldTransform: readonly number[];
  /** Omitted for a standards-defined contentless hierarchy tile. */
  readonly contentUri?: string;
  /** Omitted for a standards-defined contentless hierarchy tile. */
  readonly contentUrl?: string;
  readonly children: readonly TilesetTile[];
  /** Present only on the document root of an implicit quadtree. */
  readonly implicitTiling?: TilesetImplicitTiling;
  /** Present on lazily materialized implicit tiles. */
  readonly implicitAddress?: ImplicitTileAddress;
};

export type ImplicitTileAddress = {
  readonly level: number;
  readonly x: number;
  readonly y: number;
};

export type TilesetImplicitTiling = {
  readonly subdivisionScheme: "QUADTREE";
  readonly subtreeLevels: number;
  readonly availableLevels: number;
  readonly subtreeUriTemplate: string;
  readonly subtreeUrlTemplate: string;
  readonly contentUriTemplate: string;
  readonly contentUrlTemplate: string;
  /** Tileset-level schema used by subtree property tables. */
  readonly metadataSchema: Readonly<Record<string, unknown>>;
};

export type TilesetSource = {
  readonly endpoint: string;
  readonly tilesetUrl: string;
  readonly assetVersion: "1.1";
  readonly geometricError: number;
  readonly root: TilesetTile;
  /** Stable pre-order traversal in source-document child order. */
  readonly tiles: readonly TilesetTile[];
  readonly tileById: ReadonlyMap<string, TilesetTile>;
};

export type LoadTilesetOptions = {
  readonly endpoint: string;
  readonly fetch?: TilesetFetch;
  readonly signal?: AbortSignal;
};

/** Base class for errors which identify a constrained-profile path. */
export class TilesetProfileError extends Error {
  readonly path: string;

  constructor(message: string, path: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "TilesetProfileError";
    this.path = path;
  }
}

/** The document is shaped correctly enough to identify an unsupported feature. */
export class TilesetUnsupportedError extends TilesetProfileError {
  readonly feature: string;

  constructor(feature: string, path: string) {
    super(`unsupported 3D Tiles profile feature ${feature} at ${path}`, path);
    this.name = "TilesetUnsupportedError";
    this.feature = feature;
  }
}

/** The document or URI violates the accepted profile. */
export class TilesetValidationError extends TilesetProfileError {
  constructor(path: string, reason: string, options?: ErrorOptions) {
    super(`invalid 3D Tiles value at ${path}: ${reason}`, path, options);
    this.name = "TilesetValidationError";
  }
}

/** Network, HTTP-status, or JSON transport failure. */
export class TilesetFetchError extends Error {
  readonly url: string;
  readonly status?: number;

  constructor(
    url: string,
    message: string,
    status?: number,
    options?: ErrorOptions,
  ) {
    super(`failed to fetch 3D Tiles source ${url}: ${message}`, options);
    this.name = "TilesetFetchError";
    this.url = url;
    this.status = status;
  }
}

const IDENTITY = Object.freeze([
  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
]);

const objectAt = (value: unknown, path: string): Record<string, unknown> => {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new TilesetValidationError(path, "expected an object");
  }
  return value as Record<string, unknown>;
};

const finiteAtLeastZero = (value: unknown, path: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new TilesetValidationError(path, "expected a finite number >= 0");
  }
  return value;
};

const integerAtLeast = (
  value: unknown,
  minimum: number,
  path: string,
): number => {
  if (!Number.isSafeInteger(value) || (value as number) < minimum) {
    throw new TilesetValidationError(
      path,
      `expected a safe integer >= ${minimum}`,
    );
  }
  return value as number;
};

const finiteArray = (
  value: unknown,
  length: number,
  path: string,
): number[] => {
  if (
    !Array.isArray(value) ||
    value.length !== length ||
    !value.every((entry) => typeof entry === "number" && Number.isFinite(entry))
  ) {
    throw new TilesetValidationError(
      path,
      `expected exactly ${length} finite numbers`,
    );
  }
  return [...value] as number[];
};

const determinant3 = (m: readonly number[]): number =>
  m[0]! * (m[4]! * m[8]! - m[7]! * m[5]!) -
  m[3]! * (m[1]! * m[8]! - m[7]! * m[2]!) +
  m[6]! * (m[1]! * m[5]! - m[4]! * m[2]!);

const affineMatrix = (value: unknown, path: string): readonly number[] => {
  if (value === undefined) return IDENTITY;
  const matrix = finiteArray(value, 16, path);
  if (
    Math.abs(matrix[3]!) > 1e-12 ||
    Math.abs(matrix[7]!) > 1e-12 ||
    Math.abs(matrix[11]!) > 1e-12 ||
    Math.abs(matrix[15]! - 1) > 1e-12
  ) {
    throw new TilesetValidationError(
      path,
      "expected a column-major affine matrix",
    );
  }
  const linear = [
    matrix[0]!,
    matrix[1]!,
    matrix[2]!,
    matrix[4]!,
    matrix[5]!,
    matrix[6]!,
    matrix[8]!,
    matrix[9]!,
    matrix[10]!,
  ];
  if (Math.abs(determinant3(linear)) <= Number.EPSILON) {
    throw new TilesetValidationError(path, "transform must be invertible");
  }
  return Object.freeze(matrix);
};

/** Column-major f64 matrix multiplication. */
/** Unvalidated product: tile transforms are checked when the tileset is read. */
export const multiplyTilesetMatrices = (
  left: readonly number[],
  right: readonly number[],
): readonly number[] => Object.freeze(multiplyMat4Values(left, right));

const normalizeEndpoint = (endpoint: string): string => {
  if (typeof endpoint !== "string" || endpoint.length === 0) {
    throw new TilesetValidationError("endpoint", "expected a non-empty string");
  }
  if (endpoint.includes("?") || endpoint.includes("#")) {
    throw new TilesetValidationError(
      "endpoint",
      "query and fragment are not allowed",
    );
  }
  const normalized = endpoint.replace(/\/+$/, "");
  if (normalized.length === 0) {
    throw new TilesetValidationError(
      "endpoint",
      "root endpoint is not allowed",
    );
  }
  return normalized;
};

const decodeRepeatedly = (uri: string): string[] => {
  const values = [uri];
  for (let count = 0; count < 4; count += 1) {
    let decoded: string;
    try {
      decoded = decodeURIComponent(values.at(-1)!);
    } catch (error) {
      throw new TilesetValidationError("uri", "malformed percent escape", {
        cause: error,
      });
    }
    if (decoded === values.at(-1)) break;
    values.push(decoded);
  }
  return values;
};

/**
 * Resolve a content URI without granting URL semantics to the reference.
 * Only relative, same-root POSIX paths survive; endpoint identity is retained.
 */
export const resolveTilesetContentUri = (
  endpoint: string,
  uri: string,
): string => {
  const base = normalizeEndpoint(endpoint);
  if (typeof uri !== "string" || uri.length === 0) {
    throw new TilesetValidationError("uri", "expected a non-empty string");
  }
  if (
    /^[A-Za-z][A-Za-z\d+.-]*:/.test(uri) ||
    uri.startsWith("//") ||
    uri.startsWith("/") ||
    uri.includes("\\") ||
    uri.includes("?") ||
    uri.includes("#")
  ) {
    throw new TilesetValidationError(
      "uri",
      "expected a relative same-root path",
    );
  }

  const decodedForms = decodeRepeatedly(uri);
  for (const decoded of decodedForms) {
    if (
      /^[A-Za-z][A-Za-z\d+.-]*:/.test(decoded) ||
      decoded.includes("\\") ||
      decoded.startsWith("/") ||
      decoded.startsWith("//") ||
      decoded.includes("?") ||
      decoded.includes("#")
    ) {
      throw new TilesetValidationError(
        "uri",
        "encoded absolute path is not allowed",
      );
    }
    if (decoded.split("/").some((segment) => segment === "..")) {
      throw new TilesetValidationError("uri", "path escapes the tileset root");
    }
  }

  // Percent-encoded separators change path topology and are rejected even if
  // the decoded result happens to normalize inside the root.
  if (decodedForms.some((decoded) => /%(?:2f|5c)/i.test(decoded))) {
    throw new TilesetValidationError(
      "uri",
      "encoded path separators are not allowed",
    );
  }

  const normalizedSegments = uri
    .split("/")
    .filter((segment) => segment !== ".");
  if (
    normalizedSegments.length === 0 ||
    normalizedSegments.some((segment) => segment.length === 0)
  ) {
    throw new TilesetValidationError(
      "uri",
      "empty path segment is not allowed",
    );
  }
  return `${base}/${normalizedSegments.join("/")}`;
};

const boundingBox = (value: unknown, path: string): TilesetBox => {
  const volume = objectAt(value, path);
  if ("sphere" in volume || "region" in volume || !("box" in volume)) {
    throw new TilesetUnsupportedError("boundingVolume", path);
  }
  const raw = finiteArray(volume.box, 12, `${path}.box`);
  const halfAxes = raw.slice(3);
  if (Math.abs(determinant3(halfAxes)) <= Number.EPSILON) {
    throw new TilesetValidationError(
      `${path}.box`,
      "box half axes must span a volume",
    );
  }
  return Object.freeze({
    center: Object.freeze(raw.slice(0, 3)) as unknown as TilesetBox["center"],
    halfAxes: Object.freeze(halfAxes) as unknown as TilesetBox["halfAxes"],
  });
};

const contentUri = (value: unknown, path: string): string => {
  const content = objectAt(value, path);
  if ("url" in content) {
    throw new TilesetUnsupportedError("content.url", `${path}.url`);
  }
  if (typeof content.uri !== "string" || content.uri.length === 0) {
    throw new TilesetValidationError(
      `${path}.uri`,
      "expected a non-empty string",
    );
  }
  const decoded = decodeRepeatedly(content.uri).at(-1)!.toLowerCase();
  const pathOnly = decoded.split(/[?#]/, 1)[0]!;
  const extension = pathOnly.includes(".")
    ? pathOnly.slice(pathOnly.lastIndexOf("."))
    : "";
  if ([".b3dm", ".pnts", ".i3dm", ".cmpt"].includes(extension)) {
    throw new TilesetUnsupportedError("content.uri", `${path}.uri`);
  }
  if (extension !== ".glb" && extension !== ".gltf") {
    throw new TilesetUnsupportedError("content.uri", `${path}.uri`);
  }
  return content.uri;
};

const implicitTemplate = (
  endpoint: string,
  value: unknown,
  path: string,
  extension: ".subtree" | ".glb" | ".gltf",
): { readonly uri: string; readonly url: string } => {
  if (typeof value !== "string" || value.length === 0) {
    throw new TilesetValidationError(path, "expected a non-empty string");
  }
  const placeholders = value.match(/\{[^}]*\}/gu) ?? [];
  for (const name of ["level", "x", "y"] as const) {
    if (
      placeholders.filter((candidate) => candidate === `{${name}}`).length !== 1
    ) {
      throw new TilesetValidationError(
        path,
        `expected exactly one {${name}} placeholder`,
      );
    }
  }
  if (
    placeholders.length !== 3 ||
    value.replace(/\{(?:level|x|y)\}/gu, "").includes("{") ||
    value.replace(/\{(?:level|x|y)\}/gu, "").includes("}")
  ) {
    throw new TilesetValidationError(path, "unsupported template placeholder");
  }
  const substituted = substituteImplicitTemplate(value, {
    level: 0,
    x: 0,
    y: 0,
  });
  if (!substituted.toLowerCase().endsWith(extension)) {
    throw new TilesetValidationError(path, `expected a ${extension} path`);
  }
  try {
    resolveTilesetContentUri(endpoint, substituted);
  } catch (error) {
    throw new TilesetValidationError(
      path,
      "template must be a relative same-root path",
      {
        cause: error,
      },
    );
  }
  return Object.freeze({
    uri: value,
    url: `${normalizeEndpoint(endpoint)}/${value}`,
  });
};

export const substituteImplicitTemplate = (
  template: string,
  address: ImplicitTileAddress,
): string =>
  template
    .replace("{level}", String(address.level))
    .replace("{x}", String(address.x))
    .replace("{y}", String(address.y));

type ParseContext = {
  readonly endpoint: string;
  readonly metadataSchema?: Readonly<Record<string, unknown>>;
  readonly tiles: TilesetTile[];
  readonly tileById: Map<string, TilesetTile>;
};

const parseTile = (
  value: unknown,
  path: string,
  id: string,
  parentTransform: readonly number[],
  inheritedRefine: "REPLACE" | undefined,
  context: ParseContext,
): TilesetTile => {
  const raw = objectAt(value, path);
  if ("implicitTiling" in raw) {
    if (id !== "root") {
      throw new TilesetValidationError(
        `${path}.implicitTiling`,
        "implicit tiling is only allowed on the root",
      );
    }
    if ("contents" in raw) {
      throw new TilesetUnsupportedError("contents", `${path}.contents`);
    }
    const refine = raw.refine ?? inheritedRefine;
    if (refine !== "REPLACE") {
      throw new TilesetValidationError(`${path}.refine`, 'expected "REPLACE"');
    }
    if (raw.children !== undefined) {
      throw new TilesetValidationError(
        `${path}.children`,
        "explicit children are not allowed on an implicit root",
      );
    }
    if (raw.content === undefined) {
      throw new TilesetValidationError(
        `${path}.content`,
        "implicit roots require one content URI template",
      );
    }
    const content = objectAt(raw.content, `${path}.content`);
    if ("url" in content) {
      throw new TilesetUnsupportedError("content.url", `${path}.content.url`);
    }
    const contentPath = `${path}.content.uri`;
    const contentValue = content.uri;
    const contentExtension =
      typeof contentValue === "string" &&
      contentValue.toLowerCase().endsWith(".gltf")
        ? ".gltf"
        : ".glb";
    const contentTemplate = implicitTemplate(
      context.endpoint,
      contentValue,
      contentPath,
      contentExtension,
    );
    const implicit = objectAt(raw.implicitTiling, `${path}.implicitTiling`);
    if (implicit.subdivisionScheme !== "QUADTREE") {
      throw new TilesetUnsupportedError(
        `implicitTiling.${String(implicit.subdivisionScheme)}`,
        `${path}.implicitTiling.subdivisionScheme`,
      );
    }
    const subtreeLevels = integerAtLeast(
      implicit.subtreeLevels,
      1,
      `${path}.implicitTiling.subtreeLevels`,
    );
    if (subtreeLevels !== 4) {
      throw new TilesetValidationError(
        `${path}.implicitTiling.subtreeLevels`,
        "expected 4 for the supported producer profile",
      );
    }
    const availableLevels = integerAtLeast(
      implicit.availableLevels,
      1,
      `${path}.implicitTiling.availableLevels`,
    );
    // Tile IDs use 32-bit bitwise quadrant extraction. Levels 0..30 keep
    // coordinates nonnegative and exactly representable throughout traversal.
    if (availableLevels > 31) {
      throw new TilesetValidationError(
        `${path}.implicitTiling.availableLevels`,
        "expected at most 31 for the supported producer profile",
      );
    }
    const subtrees = objectAt(
      implicit.subtrees,
      `${path}.implicitTiling.subtrees`,
    );
    if ("url" in subtrees) {
      throw new TilesetUnsupportedError(
        "implicitTiling.subtrees.url",
        `${path}.implicitTiling.subtrees.url`,
      );
    }
    const subtreeTemplate = implicitTemplate(
      context.endpoint,
      subtrees.uri,
      `${path}.implicitTiling.subtrees.uri`,
      ".subtree",
    );
    const transform = affineMatrix(raw.transform, `${path}.transform`);
    const worldTransform = multiplyTilesetMatrices(parentTransform, transform);
    const descriptor: TilesetImplicitTiling = Object.freeze({
      subdivisionScheme: "QUADTREE",
      subtreeLevels,
      availableLevels,
      subtreeUriTemplate: subtreeTemplate.uri,
      subtreeUrlTemplate: subtreeTemplate.url,
      contentUriTemplate: contentTemplate.uri,
      contentUrlTemplate: contentTemplate.url,
      metadataSchema:
        context.metadataSchema ??
        (() => {
          throw new TilesetValidationError(
            "schema",
            "implicit tilesets require an inline metadata schema",
          );
        })(),
    });
    const parsed: TilesetTile = Object.freeze({
      id,
      geometricError: finiteAtLeastZero(
        raw.geometricError,
        `${path}.geometricError`,
      ),
      boundingVolume: boundingBox(raw.boundingVolume, `${path}.boundingVolume`),
      transform,
      worldTransform,
      children: Object.freeze([]),
      implicitTiling: descriptor,
      implicitAddress: Object.freeze({ level: 0, x: 0, y: 0 }),
    });
    context.tiles.push(parsed);
    context.tileById.set(id, parsed);
    return parsed;
  }
  if ("contents" in raw) {
    throw new TilesetUnsupportedError("contents", `${path}.contents`);
  }
  const refine = raw.refine ?? inheritedRefine;
  if (refine !== "REPLACE") {
    throw new TilesetValidationError(`${path}.refine`, 'expected "REPLACE"');
  }

  const transform = affineMatrix(raw.transform, `${path}.transform`);
  const worldTransform = multiplyTilesetMatrices(parentTransform, transform);
  let uri: string | undefined;
  let resolved: string | undefined;
  if (raw.content !== undefined) {
    uri = contentUri(raw.content, `${path}.content`);
    try {
      resolved = resolveTilesetContentUri(context.endpoint, uri);
    } catch (error) {
      if (error instanceof TilesetValidationError) {
        throw new TilesetValidationError(`${path}.content.uri`, error.message, {
          cause: error,
        });
      }
      throw error;
    }
  }

  if (raw.children !== undefined && !Array.isArray(raw.children)) {
    throw new TilesetValidationError(`${path}.children`, "expected an array");
  }
  const childrenRaw = (raw.children ?? []) as unknown[];
  const children: TilesetTile[] = [];
  const parsed: TilesetTile = {
    id,
    geometricError: finiteAtLeastZero(
      raw.geometricError,
      `${path}.geometricError`,
    ),
    boundingVolume: boundingBox(raw.boundingVolume, `${path}.boundingVolume`),
    transform,
    worldTransform,
    ...(uri === undefined || resolved === undefined
      ? {}
      : { contentUri: uri, contentUrl: resolved }),
    children,
  };
  context.tiles.push(parsed);
  context.tileById.set(id, parsed);
  for (let index = 0; index < childrenRaw.length; index += 1) {
    children.push(
      parseTile(
        childrenRaw[index],
        `${path}.children[${index}]`,
        `${id}/${index}`,
        worldTransform,
        refine,
        context,
      ),
    );
  }
  Object.freeze(children);
  return Object.freeze(parsed);
};

export const parseTileset = (
  value: unknown,
  endpointValue: string,
): TilesetSource => {
  const endpoint = normalizeEndpoint(endpointValue);
  const raw = objectAt(value, "tileset");
  const asset = objectAt(raw.asset, "asset");
  if (asset.version !== "1.1") {
    throw new TilesetValidationError("asset.version", 'expected "1.1"');
  }
  const geometricError = finiteAtLeastZero(
    raw.geometricError,
    "geometricError",
  );
  if (raw.root === undefined) {
    throw new TilesetValidationError("root", "expected an object");
  }
  const context: ParseContext = {
    endpoint,
    ...(raw.schema === undefined
      ? {}
      : { metadataSchema: objectAt(raw.schema, "schema") }),
    tiles: [],
    tileById: new Map(),
  };
  const root = parseTile(
    raw.root,
    "root",
    "root",
    IDENTITY,
    undefined,
    context,
  );
  return Object.freeze({
    endpoint,
    tilesetUrl: `${endpoint}/tileset.json`,
    assetVersion: "1.1" as const,
    geometricError,
    root,
    tiles: Object.freeze(context.tiles),
    tileById: context.tileById,
  });
};

export const loadTileset = async (
  options: LoadTilesetOptions,
): Promise<TilesetSource> => {
  const endpoint = normalizeEndpoint(options.endpoint);
  const url = `${endpoint}/tileset.json`;
  const fetcher =
    options.fetch ?? (globalThis.fetch as unknown as TilesetFetch);
  if (typeof fetcher !== "function") {
    throw new TilesetFetchError(url, "fetch is unavailable");
  }

  let response: TilesetFetchResponse;
  try {
    response = await fetcher(url, { signal: options.signal });
  } catch (error) {
    throw new TilesetFetchError(url, "network request failed", undefined, {
      cause: error,
    });
  }
  if (!response.ok) {
    throw new TilesetFetchError(
      url,
      `HTTP ${response.status} ${response.statusText}`.trim(),
      response.status,
    );
  }

  let document: unknown;
  try {
    document = await response.json();
  } catch (error) {
    throw new TilesetFetchError(url, "response is not valid JSON", undefined, {
      cause: error,
    });
  }
  return parseTileset(document, endpoint);
};
