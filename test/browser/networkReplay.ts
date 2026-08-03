import { createHash } from "node:crypto";
import { createWriteStream } from "node:fs";
import { mkdir, rename, stat, unlink } from "node:fs/promises";
import { resolve } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

export const REPLAY_URL_PATH = "/telemetry/replay.copc.laz";

/** A stable, URL-specific cache path that does not disclose the source URL. */
export const replayCachePath = (url: string, directory: string): string => {
  const key = createHash("sha256").update(url).digest("hex").slice(0, 16);
  return resolve(directory, `${key}.copc.laz`);
};

const usableFile = async (path: string): Promise<boolean> => {
  try {
    const file = await stat(path);
    return file.isFile() && file.size > 0;
  } catch {
    return false;
  }
};

/**
 * Materialise a remote COPC once so later captures do not depend on WAN
 * conditions. A complete file is cached rather than the ranges from one run,
 * because the camera path under investigation determines which ranges it asks
 * for. The rename is atomic, so an interrupted first download is never reused.
 */
export const ensureReplayCloud = async (
  url: string,
  directory: string,
  refresh = false,
): Promise<string> => {
  const output = replayCachePath(url, directory);
  if (!refresh && (await usableFile(output))) return output;

  await mkdir(directory, { recursive: true });
  const temporary = `${output}.${process.pid}.${Date.now()}.tmp`;
  process.stdout.write(`Caching telemetry cloud at ${output}\n`);
  try {
    const response = await fetch(url);
    if (!response.ok || response.body === null) {
      throw new Error(
        `could not cache telemetry cloud: HTTP ${response.status} ${response.statusText}`,
      );
    }
    await pipeline(
      Readable.from(response.body as unknown as AsyncIterable<Uint8Array>),
      createWriteStream(temporary, { flags: "wx" }),
    );
    const downloaded = await stat(temporary);
    const expected = Number(response.headers.get("content-length"));
    if (
      Number.isFinite(expected) &&
      expected > 0 &&
      downloaded.size !== expected
    ) {
      throw new Error(
        `telemetry cloud download was truncated: expected ${expected} bytes, received ${downloaded.size}`,
      );
    }
    await rename(temporary, output);
    return output;
  } catch (error) {
    await unlink(temporary).catch(() => undefined);
    throw error;
  }
};
