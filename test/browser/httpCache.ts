/**
 * A recording HTTP cache for the benchmark's remote datasets.
 *
 * A measurement over the open internet measures the internet: the same gesture
 * replayed twice reads the same tiles at whatever rate the WAN felt like that
 * minute, and a tuning change worth 8 % disappears into that. A measurement
 * that never touches the network measures something no user has, because
 * streaming decisions exist to hide latency.
 *
 * So both, chosen per run. `live` goes to the origin. `record` goes to the
 * origin and keeps every response. `replay` serves what was kept, at a stated
 * latency and rate, and fails loudly on a request nothing recorded rather than
 * silently reaching for the network mid-measurement.
 *
 * Entries are keyed by method, URL and Range, because these datasets are read
 * almost entirely through range requests and two ranges of one COPC file are
 * two different responses.
 */

import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

import type { Page, Route } from "playwright";

export type CacheMode = "live" | "record" | "replay";

export type NetworkShape = {
  /** Delay before the first byte, modelling round-trip latency. */
  readonly latencyMs: number;
  /** Per-request transfer rate; zero or absent serves at full speed. */
  readonly bytesPerSecond?: number;
};

export type HttpCacheOptions = {
  readonly mode: CacheMode;
  readonly directory: string;
  /** Only these origins are cached or shaped; everything else passes through. */
  readonly origins: readonly string[];
  /** Applied in `replay` only: `live` and `record` already pay a real network. */
  readonly shape?: NetworkShape;
};

export type HttpCacheStats = {
  readonly requests: number;
  readonly hits: number;
  readonly misses: number;
  readonly recorded: number;
  readonly bytes: number;
  /** Requests `replay` could not answer, by URL. Empty is the only good value. */
  readonly unrecorded: readonly string[];
};

type CacheEntryMeta = {
  readonly url: string;
  readonly method: string;
  readonly range: string | null;
  readonly status: number;
  readonly headers: Record<string, string>;
  readonly bodyBytes: number;
};

/**
 * A stable path per request. The digest covers the range too, so one file's
 * ranges never collide, and the URL is not reconstructable from the path — a
 * cache directory can be kept beside a repository without publishing which
 * datasets a deployment reads.
 */
const entryKey = (method: string, url: string, range: string | null): string =>
  createHash("sha256")
    .update(`${method}\n${url}\n${range ?? ""}`)
    .digest("hex");

const entryPaths = (
  directory: string,
  key: string,
): { readonly meta: string; readonly body: string } => ({
  meta: resolve(directory, key.slice(0, 2), `${key}.json`),
  body: resolve(directory, key.slice(0, 2), `${key}.bin`),
});

/** Headers worth keeping: the ones a range-reading client actually consults. */
const KEPT_HEADERS = new Set([
  "content-type",
  "content-range",
  "content-length",
  "accept-ranges",
  "etag",
  "last-modified",
]);

const keptHeaders = (headers: Record<string, string>): Record<string, string> =>
  Object.fromEntries(
    Object.entries(headers).filter(([name]) =>
      KEPT_HEADERS.has(name.toLowerCase()),
    ),
  );

const sleep = (ms: number): Promise<void> =>
  ms <= 0 ? Promise.resolve() : new Promise((done) => setTimeout(done, ms));

const writeAtomic = async (path: string, body: Uint8Array): Promise<void> => {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.${process.pid}.tmp`;
  await writeFile(temporary, body);
  await rename(temporary, path);
};

export type HttpCache = {
  /** Install request handling on a page. Safe to call once per page. */
  install(page: Page): Promise<void>;
  stats(): HttpCacheStats;
};

export const createHttpCache = (options: HttpCacheOptions): HttpCache => {
  let requests = 0;
  let hits = 0;
  let misses = 0;
  let recorded = 0;
  let bytes = 0;
  const unrecorded: string[] = [];

  const shaped = async (byteLength: number): Promise<void> => {
    const shape = options.shape;
    if (shape === undefined) return;
    const transferMs =
      shape.bytesPerSecond === undefined || shape.bytesPerSecond <= 0
        ? 0
        : (byteLength / shape.bytesPerSecond) * 1000;
    await sleep(shape.latencyMs + transferMs);
  };

  const handled = (url: string): boolean =>
    options.origins.some((origin) => url.startsWith(origin));

  const handle = async (route: Route): Promise<void> => {
    const request = route.request();
    const url = request.url();
    if (!handled(url)) {
      await route.fallback();
      return;
    }
    requests += 1;
    const range = request.headers()["range"] ?? null;
    const key = entryKey(request.method(), url, range);
    const paths = entryPaths(options.directory, key);

    if (options.mode === "replay") {
      let meta: CacheEntryMeta;
      let body: Buffer;
      try {
        meta = JSON.parse(await readFile(paths.meta, "utf8")) as CacheEntryMeta;
        body = await readFile(paths.body);
      } catch {
        misses += 1;
        unrecorded.push(
          `${request.method()} ${url}${range ? ` ${range}` : ""}`,
        );
        // Aborting rather than passing through: a replay run that quietly
        // fetched a missing entry would report a cached measurement it did not
        // take, which is worse than a run that fails.
        await route.abort("failed");
        return;
      }
      hits += 1;
      bytes += body.byteLength;
      await shaped(body.byteLength);
      await route.fulfill({
        status: meta.status,
        headers: meta.headers,
        body,
      });
      return;
    }

    const response = await route.fetch();
    const body = await response.body();
    bytes += body.byteLength;
    if (options.mode === "record") {
      const meta: CacheEntryMeta = {
        url,
        method: request.method(),
        range,
        status: response.status(),
        headers: keptHeaders(response.headers()),
        bodyBytes: body.byteLength,
      };
      await writeAtomic(paths.body, body);
      await writeAtomic(
        paths.meta,
        new TextEncoder().encode(JSON.stringify(meta, null, 2)),
      );
      recorded += 1;
    }
    await route.fulfill({ response, body });
  };

  return {
    install: async (page) => {
      if (options.mode === "live" && options.shape === undefined) return;
      await page.route("**/*", (route) => {
        void handle(route).catch(() => route.abort("failed"));
      });
    },
    stats: () => ({
      requests,
      hits,
      misses,
      recorded,
      bytes,
      unrecorded: [...unrecorded],
    }),
  };
};
