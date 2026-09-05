/** Revision-scoped bounded fetch/parse store for implicit hierarchy subtrees. */

import { finiteAtLeast } from "../numeric";
import { safeCall } from "../observers";
import { parseSubtree, type ParsedSubtree } from "./subtree";

export type SubtreeRequest = {
  readonly id: string;
  readonly url: string;
};

export type SubtreeFetchResponse = {
  readonly ok: boolean;
  readonly status: number;
  readonly statusText: string;
  arrayBuffer(): Promise<ArrayBuffer>;
};

export type SubtreeFetch = (
  url: string,
  init: { readonly signal: AbortSignal },
) => Promise<SubtreeFetchResponse>;

export type SubtreeStoreEntryStatus =
  | "queued"
  | "fetching"
  | "retrying"
  | "ready"
  | "failed";

export type SubtreeStoreEntrySnapshot = SubtreeRequest & {
  readonly status: SubtreeStoreEntryStatus;
  readonly attempt: number;
};

export type SubtreeStoreSnapshot = {
  readonly revision: string;
  readonly configGeneration: number;
  readonly selected: number;
  readonly active: number;
  readonly queued: number;
  readonly retrying: number;
  readonly ready: number;
  readonly failed: number;
  readonly cached: number;
  readonly cachedBytes: number;
  readonly cacheHits: number;
  readonly cacheMisses: number;
  readonly cacheEvictions: number;
  readonly workPending: boolean;
  readonly disposed: boolean;
  readonly entries: readonly SubtreeStoreEntrySnapshot[];
  readonly subtreeById: ReadonlyMap<string, ParsedSubtree>;
};

export class SubtreeStoreError extends Error {
  readonly id: string;
  readonly url: string;

  constructor(
    name: string,
    request: SubtreeRequest,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${request.id}: ${message}`, options);
    this.name = name;
    this.id = request.id;
    this.url = request.url;
  }
}

export class SubtreeFetchError extends SubtreeStoreError {
  readonly status?: number;

  constructor(
    request: SubtreeRequest,
    message: string,
    status?: number,
    options?: ErrorOptions,
  ) {
    super("SubtreeFetchError", request, message, options);
    this.status = status;
  }
}

export class SubtreeParseError extends SubtreeStoreError {
  constructor(
    request: SubtreeRequest,
    message: string,
    options?: ErrorOptions,
  ) {
    super("SubtreeParseError", request, message, options);
  }
}

export class SubtreeRetryError extends SubtreeStoreError {
  constructor(request: SubtreeRequest, options?: ErrorOptions) {
    super("SubtreeRetryError", request, "retry scheduling failed", options);
  }
}

export type SubtreeStoreOptions = {
  readonly revision: string;
  readonly configGeneration: number;
  readonly subtreeLevels: number;
  readonly metadataSchema: Readonly<Record<string, unknown>>;
  readonly maxConcurrency: number;
  readonly maxBytes: number;
  readonly maxAttempts?: number;
  readonly retryBackoffMs?: (failedAttempt: number) => number;
  readonly fetch?: SubtreeFetch;
  readonly parse?: (bytes: ArrayBuffer, subtreeLevels: number) => ParsedSubtree;
  readonly onSubtree?: (
    request: SubtreeRequest,
    subtree: ParsedSubtree,
  ) => void;
  readonly onError?: (request: SubtreeRequest, error: Error) => void;
  readonly onStateChange?: (snapshot: SubtreeStoreSnapshot) => void;
};

export type SubtreeStore = {
  setSelection(requests: readonly SubtreeRequest[]): void;
  configure(configuration: { readonly maxConcurrency: number }): void;
  snapshot(): SubtreeStoreSnapshot;
  dispose(): void;
};

type Selected = {
  readonly request: SubtreeRequest;
  status: SubtreeStoreEntryStatus;
  attempt: number;
  retry?: ReturnType<typeof setTimeout>;
};

type Cached = {
  readonly request: SubtreeRequest;
  readonly subtree: ParsedSubtree;
  readonly bytes: number;
};

const positiveInteger = (value: number, path: string): number => {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`${path} must be an integer >= 1`);
  }
  return value;
};

const normalize = (request: SubtreeRequest): SubtreeRequest => {
  if (typeof request.id !== "string" || request.id.length === 0) {
    throw new TypeError("subtree request id must be a non-empty string");
  }
  if (typeof request.url !== "string" || request.url.length === 0) {
    throw new TypeError(`subtree request ${request.id} URL must be non-empty`);
  }
  return Object.freeze({ id: request.id, url: request.url });
};

export const createSubtreeStore = (
  options: SubtreeStoreOptions,
): SubtreeStore => {
  if (typeof options.revision !== "string" || options.revision.length === 0) {
    throw new TypeError("revision must be a non-empty string");
  }
  positiveInteger(options.configGeneration + 1, "configGeneration + 1");
  let maxConcurrency = positiveInteger(
    options.maxConcurrency,
    "maxConcurrency",
  );
  const maxBytes = finiteAtLeast("maxBytes", options.maxBytes, 0);
  const maxAttempts = positiveInteger(options.maxAttempts ?? 3, "maxAttempts");
  const retryBackoffMs =
    options.retryBackoffMs ?? ((attempt: number) => 250 * 2 ** (attempt - 1));
  const fetcher =
    options.fetch ?? (globalThis.fetch as unknown as SubtreeFetch);
  if (typeof fetcher !== "function")
    throw new TypeError("fetch is unavailable");
  const parser = options.parse ?? parseSubtree;

  let disposed = false;
  let epoch = 0;
  let selected = new Map<string, Selected>();
  const active = new Map<string, AbortController>();
  const cache = new Map<string, Cached>();
  let cachedBytes = 0;
  let cacheHits = 0;
  let cacheMisses = 0;
  let cacheEvictions = 0;

  const evict = (id: string): void => {
    const cached = cache.get(id);
    if (!cached) return;
    cache.delete(id);
    cachedBytes -= cached.bytes;
    cacheEvictions += 1;
  };

  const makeRoom = (bytes: number): boolean => {
    if (bytes > maxBytes) return false;
    for (const id of cache.keys()) {
      if (cachedBytes + bytes <= maxBytes) break;
      if (!selected.has(id)) evict(id);
    }
    return cachedBytes + bytes <= maxBytes;
  };

  const put = (request: SubtreeRequest, subtree: ParsedSubtree): boolean => {
    const prior = cache.get(request.id);
    if (prior) {
      cache.delete(request.id);
      cachedBytes -= prior.bytes;
    }
    if (!makeRoom(subtree.byteLength)) {
      if (prior) {
        cache.set(request.id, prior);
        cachedBytes += prior.bytes;
      }
      return false;
    }
    cache.set(request.id, { request, subtree, bytes: subtree.byteLength });
    cachedBytes += subtree.byteLength;
    return true;
  };

  const snapshot = (): SubtreeStoreSnapshot => {
    const entries = [...selected.values()].map((entry) =>
      Object.freeze({
        ...entry.request,
        status: entry.status,
        attempt: entry.attempt,
      }),
    );
    const count = (status: SubtreeStoreEntryStatus): number =>
      entries.filter((entry) => entry.status === status).length;
    const queued = count("queued");
    const retrying = count("retrying");
    const fetching = count("fetching");
    // Traversal needs every parsed cached ancestor, not only today's fetch
    // selection: once a subtree arrives, selection advances to its children.
    const subtreeById = new Map(
      [...cache].map(([id, cached]) => [id, cached.subtree] as const),
    );
    return Object.freeze({
      revision: options.revision,
      configGeneration: options.configGeneration,
      selected: selected.size,
      active: active.size,
      queued,
      retrying,
      ready: count("ready"),
      failed: count("failed"),
      cached: cache.size,
      cachedBytes,
      cacheHits,
      cacheMisses,
      cacheEvictions,
      workPending: queued + retrying + fetching > 0,
      disposed,
      entries: Object.freeze(entries),
      subtreeById,
    });
  };

  const emit = (): void => {
    if (options.onStateChange)
      safeCall(() => options.onStateChange!(snapshot()));
  };

  const current = (
    id: string,
    entry: Selected,
    operationEpoch: number,
  ): boolean =>
    !disposed && epoch === operationEpoch && selected.get(id) === entry;

  let pump: () => void;

  const fail = (
    entry: Selected,
    error: Error,
    operationEpoch: number,
  ): void => {
    if (!current(entry.request.id, entry, operationEpoch)) return;
    if (entry.attempt >= maxAttempts || error instanceof SubtreeParseError) {
      entry.status = "failed";
      safeCall(() => options.onError?.(entry.request, error));
      return;
    }
    try {
      const delay = finiteAtLeast(
        "retry backoff",
        retryBackoffMs(entry.attempt),
        0,
      );
      entry.status = "retrying";
      entry.retry = setTimeout(() => {
        if (!current(entry.request.id, entry, operationEpoch)) return;
        entry.retry = undefined;
        entry.status = "queued";
        pump();
        emit();
      }, delay);
    } catch (cause) {
      entry.status = "failed";
      safeCall(() =>
        options.onError?.(
          entry.request,
          new SubtreeRetryError(entry.request, { cause }),
        ),
      );
    }
  };

  const start = (entry: Selected): void => {
    const operationEpoch = epoch;
    const controller = new AbortController();
    entry.status = "fetching";
    entry.attempt += 1;
    active.set(entry.request.id, controller);
    emit();
    void (async () => {
      let failure: Error | undefined;
      try {
        let response: SubtreeFetchResponse;
        try {
          response = await fetcher(entry.request.url, {
            signal: controller.signal,
          });
        } catch (cause) {
          throw new SubtreeFetchError(
            entry.request,
            "network request failed",
            undefined,
            {
              cause,
            },
          );
        }
        if (!response.ok) {
          throw new SubtreeFetchError(
            entry.request,
            `HTTP ${response.status} ${response.statusText}`.trim(),
            response.status,
          );
        }
        let bytes: ArrayBuffer;
        try {
          bytes = await response.arrayBuffer();
        } catch (cause) {
          throw new SubtreeFetchError(
            entry.request,
            "response body read failed",
            response.status,
            { cause },
          );
        }
        let subtree: ParsedSubtree;
        try {
          subtree = parser(
            bytes,
            options.subtreeLevels,
            options.metadataSchema,
          );
        } catch (cause) {
          throw new SubtreeParseError(entry.request, "subtree parse failed", {
            cause,
          });
        }
        if (current(entry.request.id, entry, operationEpoch)) {
          if (!put(entry.request, subtree)) {
            throw new SubtreeParseError(
              entry.request,
              `subtree exceeds the ${maxBytes}-byte hierarchy cache`,
            );
          }
          entry.status = "ready";
          safeCall(() => options.onSubtree?.(entry.request, subtree));
        }
      } catch (error) {
        failure =
          error instanceof Error
            ? error
            : new SubtreeStoreError(
                "SubtreeStoreError",
                entry.request,
                "operation failed with a non-Error value",
              );
      } finally {
        if (active.get(entry.request.id) === controller)
          active.delete(entry.request.id);
      }
      if (failure) fail(entry, failure, operationEpoch);
      pump();
      emit();
    })();
  };

  pump = (): void => {
    if (disposed) return;
    for (const entry of selected.values()) {
      if (active.size >= maxConcurrency) break;
      if (entry.status === "queued") start(entry);
    }
  };

  return {
    setSelection(requests) {
      if (disposed) throw new Error("subtree store is disposed");
      const normalized = new Map<string, SubtreeRequest>();
      for (const value of requests) {
        const request = normalize(value);
        const duplicate = normalized.get(request.id);
        if (duplicate && duplicate.url !== request.url) {
          throw new Error(`conflicting subtree requests for ${request.id}`);
        }
        if (!duplicate) normalized.set(request.id, request);
      }
      for (const [id, entry] of selected) {
        const next = normalized.get(id);
        if (next && next.url !== entry.request.url) {
          throw new Error(`conflicting subtree request URL for ${id}`);
        }
      }
      for (const [id, entry] of selected) {
        if (normalized.has(id)) continue;
        active.get(id)?.abort();
        if (entry.retry !== undefined) clearTimeout(entry.retry);
      }
      const nextSelection = new Map<string, Selected>();
      for (const request of normalized.values()) {
        const retained = selected.get(request.id);
        if (retained) {
          nextSelection.set(request.id, retained);
          continue;
        }
        const cached = cache.get(request.id);
        if (cached && cached.request.url !== request.url) evict(request.id);
        const usable = cache.get(request.id);
        const entry: Selected = {
          request,
          status: usable ? "ready" : "queued",
          attempt: 0,
        };
        nextSelection.set(request.id, entry);
        if (usable) {
          cacheHits += 1;
          cache.delete(request.id);
          cache.set(request.id, usable);
          safeCall(() => options.onSubtree?.(request, usable.subtree));
        } else {
          cacheMisses += 1;
        }
      }
      selected = nextSelection;
      pump();
      emit();
    },
    configure(configuration) {
      if (disposed) throw new Error("subtree store is disposed");
      maxConcurrency = positiveInteger(
        configuration.maxConcurrency,
        "maxConcurrency",
      );
      pump();
      emit();
    },
    snapshot,
    dispose() {
      if (disposed) return;
      disposed = true;
      epoch += 1;
      for (const controller of active.values()) controller.abort();
      for (const entry of selected.values()) {
        if (entry.retry !== undefined) clearTimeout(entry.retry);
      }
      active.clear();
      selected.clear();
      cache.clear();
      cachedBytes = 0;
      emit();
    },
  };
};
