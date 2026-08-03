/**
 * Static file server for the browser checks.
 *
 * COPC is read entirely through HTTP Range requests, so a server that ignored
 * `Range` and returned whole files would still render — it would just quietly
 * stop testing the transport the library actually uses, and would pull whole
 * multi-gigabyte clouds into memory. Range handling here is the point, not a
 * convenience.
 */

import { createReadStream, promises as fs } from "node:fs";
import { createServer, type Server } from "node:http";
import { join, normalize, resolve, sep } from "node:path";

const CONTENT_TYPES = new Map([
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".css", "text/css; charset=utf-8"],
  [".wasm", "application/wasm"],
  [".laz", "application/octet-stream"],
  [".copc", "application/octet-stream"],
]);

const contentType = (path: string): string => {
  const dot = path.lastIndexOf(".");
  return CONTENT_TYPES.get(path.slice(dot)) ?? "application/octet-stream";
};

/** `bytes=start-end`, either end optional. Null when absent or unusable. */
const parseRange = (
  header: string | undefined,
  size: number,
): { start: number; end: number } | null => {
  if (header === undefined) return null;
  const match = /^bytes=(\d*)-(\d*)$/.exec(header.trim());
  if (match === null) return null;
  const [, rawStart, rawEnd] = match;
  if (rawStart === "" && rawEnd === "") return null;
  // A suffix range (`bytes=-N`) asks for the last N bytes.
  const start =
    rawStart === "" ? Math.max(0, size - Number(rawEnd)) : Number(rawStart);
  const end = rawStart === "" || rawEnd === "" ? size - 1 : Number(rawEnd);
  if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
  if (start > end || start < 0 || start >= size) return null;
  return { start, end: Math.min(end, size - 1) };
};

export type StaticServer = {
  readonly origin: string;
  close(): Promise<void>;
};

export type NetworkProfile = {
  /** URL paths whose file responses should be delayed. */
  readonly paths: readonly string[];
  /** Fixed delay before response headers, modelling request latency. */
  readonly latencyMs: number;
  /** Per-request transfer rate. The browser still receives real range bodies. */
  readonly bytesPerSecond: number;
};

const waitForResponse = (
  milliseconds: number,
  response: import("node:http").ServerResponse,
): Promise<boolean> =>
  new Promise((done) => {
    if (milliseconds <= 0) {
      done(!response.destroyed);
      return;
    }
    const timer = setTimeout(() => {
      response.removeListener("close", closed);
      done(!response.destroyed);
    }, milliseconds);
    const closed = (): void => {
      clearTimeout(timer);
      done(false);
    };
    response.once("close", closed);
  });

/** Deterministic body delay for a shaped response of `bytes` bytes. */
export const transferDelayMs = (
  bytes: number,
  bytesPerSecond: number,
): number => Math.ceil((bytes / bytesPerSecond) * 1000);

/**
 * Serve `roots` under their given URL prefixes. Every root is resolved and
 * every request is checked against it, so a `..` in a URL cannot walk out of
 * the directory being served.
 */
export const startStaticServer = async (
  roots: Readonly<Record<string, string>>,
  /**
   * Exact URL path to a single file, for serving something whose directory
   * should not be exposed — or whose name should not appear in a URL.
   */
  files: Readonly<Record<string, string>> = {},
  network?: NetworkProfile,
): Promise<StaticServer> => {
  if (
    network !== undefined &&
    (!Number.isFinite(network.latencyMs) ||
      network.latencyMs < 0 ||
      !Number.isFinite(network.bytesPerSecond) ||
      network.bytesPerSecond <= 0)
  ) {
    throw new Error(
      "network latency must be non-negative and transfer rate must be positive",
    );
  }
  const resolved = Object.entries(roots).map(
    ([prefix, dir]) => [prefix, resolve(dir)] as const,
  );
  const exact = new Map(
    Object.entries(files).map(([urlPath, file]) => [urlPath, resolve(file)]),
  );

  /** A prefix owns a path only at a segment boundary, so `/fixtures` does not
   * swallow `/fixtures-of-something-else`. */
  const under = (urlPath: string, prefix: string): boolean =>
    prefix === "/" ||
    urlPath === prefix ||
    urlPath.startsWith(prefix.endsWith("/") ? prefix : `${prefix}/`);

  const locate = (urlPath: string): string | null => {
    const named = exact.get(urlPath);
    if (named !== undefined) return named;
    for (const [prefix, dir] of resolved) {
      if (!under(urlPath, prefix)) continue;
      const rest = (prefix === "/" ? urlPath : urlPath.slice(prefix.length))
        // join() would treat a leading slash as absolute and escape the root.
        .replace(/^\/+/, "");
      const candidate = join(dir, normalize(rest === "" ? "index.html" : rest));
      if (candidate === dir || candidate.startsWith(dir + sep))
        return candidate;
    }
    return null;
  };

  const server: Server = createServer((request, response) => {
    void (async () => {
      const urlPath = decodeURIComponent(
        new URL(request.url ?? "/", "http://localhost").pathname,
      );
      const path = locate(urlPath);
      if (path === null) {
        response.writeHead(404).end("not found");
        return;
      }
      let size: number;
      try {
        const stat = await fs.stat(path);
        if (!stat.isFile()) throw new Error("not a file");
        size = stat.size;
      } catch {
        response.writeHead(404).end("not found");
        return;
      }

      const range = parseRange(request.headers.range, size);
      const shaped = network?.paths.includes(urlPath) ? network : undefined;
      const headers: Record<string, string> = {
        "content-type": contentType(path),
        // Without this the reader cannot discover it may range-read at all.
        "accept-ranges": "bytes",
        "cache-control": "no-store",
      };

      if (range === null) {
        if (
          shaped !== undefined &&
          !(await waitForResponse(shaped.latencyMs, response))
        )
          return;
        response.writeHead(200, { ...headers, "content-length": String(size) });
        response.flushHeaders();
        if (
          shaped !== undefined &&
          !(await waitForResponse(
            transferDelayMs(size, shaped.bytesPerSecond),
            response,
          ))
        )
          return;
        createReadStream(path).pipe(response);
        return;
      }
      if (
        shaped !== undefined &&
        !(await waitForResponse(shaped.latencyMs, response))
      )
        return;
      const rangeLength = range.end - range.start + 1;
      response.writeHead(206, {
        ...headers,
        "content-length": String(rangeLength),
        "content-range": `bytes ${range.start}-${range.end}/${size}`,
      });
      response.flushHeaders();
      if (
        shaped !== undefined &&
        !(await waitForResponse(
          transferDelayMs(rangeLength, shaped.bytesPerSecond),
          response,
        ))
      )
        return;
      createReadStream(path, { start: range.start, end: range.end }).pipe(
        response,
      );
    })();
  });

  await new Promise<void>((done) => server.listen(0, "127.0.0.1", done));
  const address = server.address();
  if (address === null || typeof address === "string") {
    throw new Error("static server did not bind a port");
  }
  return {
    origin: `http://127.0.0.1:${address.port}`,
    close: () =>
      new Promise<void>((done, fail) =>
        server.close((error) => (error ? fail(error) : done())),
      ),
  };
};
