/** Bounded, revision-safe fetch/decode queue for renderer-neutral tile content. */

export type TileContentRequest = {
  readonly id: string;
  readonly url: string;
};

export type ContentQueueFetchResponse = {
  readonly ok: boolean;
  readonly status: number;
  readonly statusText: string;
  arrayBuffer(): Promise<ArrayBuffer>;
};

export type ContentQueueFetch = (
  url: string,
  init: { readonly signal: AbortSignal },
) => Promise<ContentQueueFetchResponse>;

export type ContentDecodeContext = {
  readonly signal: AbortSignal;
  readonly revision: string;
  readonly configGeneration: number;
};

export type ContentQueueClock = {
  now(): number;
  setTimeout(callback: () => void, delayMs: number): unknown;
  clearTimeout(handle: unknown): void;
};

export type ContentQueueEntryStatus =
  | "queued"
  | "fetching"
  | "retrying"
  | "ready"
  | "failed";

export type ContentQueueEntrySnapshot = TileContentRequest & {
  readonly status: ContentQueueEntryStatus;
  readonly attempt: number;
};

export type ContentQueueSnapshot = {
  readonly revision: string;
  readonly configGeneration: number;
  readonly maxConcurrency: number;
  readonly maxDecodedBytes: number;
  readonly selected: number;
  readonly active: number;
  readonly queued: number;
  readonly retrying: number;
  readonly ready: number;
  readonly failed: number;
  readonly cached: number;
  readonly decodedBytes: number;
  readonly cacheHits: number;
  readonly cacheMisses: number;
  readonly cacheEvictions: number;
  readonly cacheRevisits: number;
  readonly workPending: boolean;
  readonly disposed: boolean;
  readonly entries: readonly ContentQueueEntrySnapshot[];
};

export type ContentQueueOptions<T> = {
  readonly revision: string;
  readonly configGeneration: number;
  readonly maxConcurrency: number;
  readonly maxDecodedBytes: number;
  readonly fetch?: ContentQueueFetch;
  readonly decode: (
    bytes: ArrayBuffer,
    request: TileContentRequest,
    context: ContentDecodeContext,
  ) => Promise<T> | T;
  readonly decodedByteLength: (
    content: T,
    request: TileContentRequest,
  ) => number;
  readonly maxAttempts?: number;
  readonly retryBackoffMs?: (failedAttempt: number) => number;
  readonly clock?: ContentQueueClock;
  readonly onContent?: (request: TileContentRequest, content: T) => void;
  readonly onError?: (request: TileContentRequest, error: Error) => void;
  readonly onEvict?: (request: TileContentRequest, content: T) => void;
  readonly onStateChange?: (snapshot: ContentQueueSnapshot) => void;
};

export type ContentQueueConfiguration = {
  readonly revision?: string;
  readonly configGeneration?: number;
  readonly maxConcurrency?: number;
  readonly maxDecodedBytes?: number;
};

export type ContentQueue<T> = {
  setSelection(requests: readonly TileContentRequest[]): void;
  configure(configuration: ContentQueueConfiguration): void;
  /** Look up decoded content and mark it most recently used. */
  get(id: string): T | undefined;
  snapshot(): ContentQueueSnapshot;
  dispose(): void;
};

export class ContentQueueError extends Error {
  readonly id: string;
  readonly url: string;

  constructor(
    name: string,
    request: TileContentRequest,
    message: string,
    options?: ErrorOptions,
  ) {
    super(`${request.id}: ${message}`, options);
    this.name = name;
    this.id = request.id;
    this.url = request.url;
  }
}

export class ContentQueueFetchError extends ContentQueueError {
  readonly status?: number;

  constructor(
    request: TileContentRequest,
    message: string,
    status?: number,
    options?: ErrorOptions,
  ) {
    super("ContentQueueFetchError", request, message, options);
    this.status = status;
  }
}

export class ContentQueueDecodeError extends ContentQueueError {
  constructor(request: TileContentRequest, options?: ErrorOptions) {
    super("ContentQueueDecodeError", request, "content decode failed", options);
  }
}

export class ContentQueueRetryError extends ContentQueueError {
  constructor(
    request: TileContentRequest,
    message: string,
    options?: ErrorOptions,
  ) {
    super("ContentQueueRetryError", request, message, options);
  }
}

type SelectedEntry = {
  readonly request: TileContentRequest;
  status: ContentQueueEntryStatus;
  attempt: number;
};

type ActiveEntry = {
  readonly controller: AbortController;
};

type RetryEntry = {
  handle?: unknown;
  handleAssigned: boolean;
};

type CacheEntry<T> = {
  readonly request: TileContentRequest;
  readonly content: T;
  readonly bytes: number;
};

const systemClock: ContentQueueClock = {
  now: () => Date.now(),
  setTimeout: (callback, delayMs) => globalThis.setTimeout(callback, delayMs),
  clearTimeout: (handle) =>
    globalThis.clearTimeout(handle as ReturnType<typeof setTimeout>),
};

const integerAtLeast = (
  name: string,
  value: number,
  minimum: number,
): number => {
  if (!Number.isInteger(value) || value < minimum) {
    throw new RangeError(`${name} must be an integer >= ${minimum}`);
  }
  return value;
};

const finiteAtLeast = (
  name: string,
  value: number,
  minimum: number,
): number => {
  if (!Number.isFinite(value) || value < minimum) {
    throw new RangeError(`${name} must be finite and >= ${minimum}`);
  }
  return value;
};

const safeCall = (callback: (() => void) | undefined): void => {
  if (!callback) return;
  try {
    callback();
  } catch {
    // Consumer notification cannot break queue accounting or create a rejected
    // internal task. Applications surface their own callback errors.
  }
};

const normalizeRequest = (request: TileContentRequest): TileContentRequest => {
  if (typeof request.id !== "string" || request.id.length === 0) {
    throw new TypeError("content request id must be a non-empty string");
  }
  if (typeof request.url !== "string" || request.url.length === 0) {
    throw new TypeError(
      `content request ${request.id} URL must be a non-empty string`,
    );
  }
  return Object.freeze({ id: request.id, url: request.url });
};

export const createContentQueue = <T>(
  options: ContentQueueOptions<T>,
): ContentQueue<T> => {
  let revision = options.revision;
  let configGeneration = integerAtLeast(
    "configGeneration",
    options.configGeneration,
    0,
  );
  let maxConcurrency = integerAtLeast(
    "maxConcurrency",
    options.maxConcurrency,
    1,
  );
  let maxDecodedBytes = finiteAtLeast(
    "maxDecodedBytes",
    options.maxDecodedBytes,
    0,
  );
  const maxAttempts = integerAtLeast(
    "maxAttempts",
    options.maxAttempts ?? 3,
    1,
  );
  const retryBackoffMs =
    options.retryBackoffMs ?? ((attempt) => 250 * 2 ** (attempt - 1));
  const clock = options.clock ?? systemClock;
  const fetcher =
    options.fetch ?? (globalThis.fetch as unknown as ContentQueueFetch);
  if (typeof revision !== "string" || revision.length === 0) {
    throw new TypeError("revision must be a non-empty string");
  }
  if (typeof fetcher !== "function") {
    throw new TypeError("fetch is unavailable");
  }

  let epoch = 0;
  let disposed = false;
  let selected = new Map<string, SelectedEntry>();
  const active = new Map<string, ActiveEntry>();
  const retries = new Map<string, RetryEntry>();
  const cache = new Map<string, CacheEntry<T>>();
  let decodedBytes = 0;
  let cacheHits = 0;
  let cacheMisses = 0;
  let cacheEvictions = 0;
  let cacheRevisits = 0;
  const previouslyEvicted = new Set<string>();

  const snapshot = (): ContentQueueSnapshot => {
    const entries = [...selected.values()].map(({ request, status, attempt }) =>
      Object.freeze({ ...request, status, attempt }),
    );
    const count = (status: ContentQueueEntryStatus): number =>
      entries.filter((entry) => entry.status === status).length;
    const queued = count("queued");
    const retrying = count("retrying");
    const fetching = count("fetching");
    return Object.freeze({
      revision,
      configGeneration,
      maxConcurrency,
      maxDecodedBytes,
      selected: selected.size,
      active: active.size,
      queued,
      retrying,
      ready: count("ready"),
      failed: count("failed"),
      cached: cache.size,
      decodedBytes,
      cacheHits,
      cacheMisses,
      cacheEvictions,
      cacheRevisits,
      workPending: queued + retrying + fetching > 0,
      disposed,
      entries: Object.freeze(entries),
    });
  };

  const emit = (): void =>
    safeCall(
      options.onStateChange
        ? () => options.onStateChange!(snapshot())
        : undefined,
    );

  const evict = (id: string): boolean => {
    const entry = cache.get(id);
    if (!entry) return false;
    cache.delete(id);
    decodedBytes -= entry.bytes;
    cacheEvictions += 1;
    previouslyEvicted.add(id);
    safeCall(
      options.onEvict
        ? () => options.onEvict!(entry.request, entry.content)
        : undefined,
    );
    return true;
  };

  const enforceCacheBudget = (): void => {
    while (decodedBytes > maxDecodedBytes) {
      const oldest = cache.keys().next();
      if (oldest.done === true) break;
      evict(oldest.value);
    }
  };

  const putCache = (
    request: TileContentRequest,
    content: T,
    bytes: number,
  ): void => {
    evict(request.id);
    // Oversize content is still delivered to the submission consumer, but is
    // never transiently charged to the CPU cache. Existing LRU entries remain
    // intact and the uncached value never produces an eviction notification.
    if (bytes > maxDecodedBytes) return;
    cache.set(request.id, { request, content, bytes });
    decodedBytes += bytes;
    enforceCacheBudget();
  };

  const clearCache = (): void => {
    for (const id of cache.keys()) evict(id);
  };

  const cancelActive = (id: string): void => {
    active.get(id)?.controller.abort();
  };

  const cancelRetry = (id: string): void => {
    const retry = retries.get(id);
    if (!retry) return;
    retries.delete(id);
    if (retry.handleAssigned) {
      try {
        clock.clearTimeout(retry.handle);
      } catch {
        // Removing the identity makes even an unsuccessfully canceled timer a
        // no-op. A hostile clock must not leave queue state half-mutated.
      }
    }
  };

  const isCurrent = (
    id: string,
    entry: SelectedEntry,
    operationEpoch: number,
  ): boolean =>
    !disposed && operationEpoch === epoch && selected.get(id) === entry;

  let pump: () => void;

  const start = (entry: SelectedEntry): void => {
    const { request } = entry;
    const operationEpoch = epoch;
    const operationRevision = revision;
    const operationGeneration = configGeneration;
    const controller = new AbortController();
    const operation: ActiveEntry = { controller };
    entry.status = "fetching";
    entry.attempt += 1;
    active.set(request.id, operation);
    emit();

    void (async () => {
      let failure: Error | undefined;
      try {
        let response: ContentQueueFetchResponse;
        try {
          response = await fetcher(request.url, { signal: controller.signal });
        } catch (error) {
          throw new ContentQueueFetchError(
            request,
            "network request failed",
            undefined,
            { cause: error },
          );
        }
        if (!response.ok) {
          throw new ContentQueueFetchError(
            request,
            `HTTP ${response.status} ${response.statusText}`.trim(),
            response.status,
          );
        }
        let bytes: ArrayBuffer;
        try {
          bytes = await response.arrayBuffer();
        } catch (error) {
          throw new ContentQueueFetchError(
            request,
            "response body read failed",
            response.status,
            { cause: error },
          );
        }
        let content: T;
        try {
          content = await options.decode(bytes, request, {
            signal: controller.signal,
            revision: operationRevision,
            configGeneration: operationGeneration,
          });
        } catch (error) {
          throw new ContentQueueDecodeError(request, { cause: error });
        }
        if (isCurrent(request.id, entry, operationEpoch)) {
          let byteLength: number;
          try {
            byteLength = options.decodedByteLength(content, request);
            finiteAtLeast(`decoded bytes for ${request.id}`, byteLength, 0);
          } catch (error) {
            throw new ContentQueueDecodeError(request, { cause: error });
          }
          putCache(request, content, byteLength);
          entry.status = "ready";
          safeCall(
            options.onContent
              ? () => options.onContent!(request, content)
              : undefined,
          );
        }
      } catch (error) {
        failure =
          error instanceof Error
            ? error
            : new ContentQueueError(
                "ContentQueueError",
                request,
                "operation failed with a non-Error value",
              );
      } finally {
        if (active.get(request.id) === operation) active.delete(request.id);
      }

      if (!isCurrent(request.id, entry, operationEpoch)) {
        pump();
        emit();
        return;
      }
      if (failure) {
        if (entry.attempt < maxAttempts) {
          const retry: RetryEntry = {
            handleAssigned: false,
          };
          try {
            const delay = retryBackoffMs(entry.attempt);
            finiteAtLeast("retry backoff", delay, 0);
            entry.status = "retrying";
            retries.set(request.id, retry);
            const handle = clock.setTimeout(() => {
              if (
                retries.get(request.id) !== retry ||
                !isCurrent(request.id, entry, operationEpoch)
              ) {
                return;
              }
              retries.delete(request.id);
              entry.status = "queued";
              pump();
              emit();
            }, delay);
            if (retries.get(request.id) === retry) {
              retry.handle = handle;
              retry.handleAssigned = true;
            }
          } catch (error) {
            if (retries.get(request.id) === retry) {
              retries.delete(request.id);
            }
            entry.status = "failed";
            const retryError = new ContentQueueRetryError(
              request,
              "retry policy or timer scheduling failed",
              { cause: error },
            );
            safeCall(
              options.onError
                ? () => options.onError!(request, retryError)
                : undefined,
            );
          }
        } else {
          entry.status = "failed";
          safeCall(
            options.onError
              ? () => options.onError!(request, failure!)
              : undefined,
          );
        }
      }
      pump();
      emit();
    })();
  };

  pump = (): void => {
    if (disposed) return;
    for (const entry of selected.values()) {
      if (active.size >= maxConcurrency) break;
      if (entry.status === "queued" && !active.has(entry.request.id)) {
        start(entry);
      }
    }
  };

  return {
    setSelection(requests) {
      if (disposed) throw new Error("content queue is disposed");
      const normalized = new Map<string, TileContentRequest>();
      for (const raw of requests) {
        const item = normalizeRequest(raw);
        const duplicate = normalized.get(item.id);
        if (duplicate && duplicate.url !== item.url) {
          throw new Error(`conflicting content requests for ${item.id}`);
        }
        if (!duplicate) normalized.set(item.id, item);
      }

      // Validate retained IDs as a complete phase. No abort, timer cancel, or
      // cache mutation occurs unless the whole requested selection is valid.
      for (const [id, entry] of selected) {
        const next = normalized.get(id);
        if (next && next.url !== entry.request.url) {
          throw new Error(`conflicting content request URL for ${id}`);
        }
      }
      for (const [id] of selected) {
        const next = normalized.get(id);
        if (!next) {
          cancelActive(id);
          cancelRetry(id);
        }
      }

      const nextSelection = new Map<string, SelectedEntry>();
      for (const item of normalized.values()) {
        const retained = selected.get(item.id);
        if (retained) {
          nextSelection.set(item.id, retained);
          continue;
        }
        let cached = cache.get(item.id);
        if (cached && cached.request.url !== item.url) {
          evict(item.id);
          cached = undefined;
        }
        const entry: SelectedEntry = {
          request: item,
          status: cached ? "ready" : "queued",
          attempt: 0,
        };
        nextSelection.set(item.id, entry);
        if (cached) {
          cacheHits += 1;
          cache.delete(item.id);
          cache.set(item.id, cached);
          safeCall(
            options.onContent
              ? () => options.onContent!(item, cached.content)
              : undefined,
          );
        } else {
          cacheMisses += 1;
          if (previouslyEvicted.delete(item.id)) cacheRevisits += 1;
        }
      }
      selected = nextSelection;
      pump();
      emit();
    },

    configure(configuration) {
      if (disposed) throw new Error("content queue is disposed");
      const nextRevision = configuration.revision ?? revision;
      const nextGeneration =
        configuration.configGeneration === undefined
          ? configGeneration
          : integerAtLeast(
              "configGeneration",
              configuration.configGeneration,
              0,
            );
      if (typeof nextRevision !== "string" || nextRevision.length === 0) {
        throw new TypeError("revision must be a non-empty string");
      }
      if (configuration.maxConcurrency !== undefined) {
        maxConcurrency = integerAtLeast(
          "maxConcurrency",
          configuration.maxConcurrency,
          1,
        );
      }
      if (configuration.maxDecodedBytes !== undefined) {
        maxDecodedBytes = finiteAtLeast(
          "maxDecodedBytes",
          configuration.maxDecodedBytes,
          0,
        );
      }

      if (nextRevision !== revision || nextGeneration !== configGeneration) {
        epoch += 1;
        for (const id of active.keys()) cancelActive(id);
        for (const id of retries.keys()) cancelRetry(id);
        clearCache();
        revision = nextRevision;
        configGeneration = nextGeneration;
        for (const entry of selected.values()) {
          entry.status = "queued";
          entry.attempt = 0;
        }
      } else {
        enforceCacheBudget();
      }
      pump();
      emit();
    },

    get(id) {
      const entry = cache.get(id);
      if (!entry) return undefined;
      cache.delete(id);
      cache.set(id, entry);
      return entry.content;
    },

    snapshot,

    dispose() {
      if (disposed) return;
      disposed = true;
      epoch += 1;
      for (const id of active.keys()) cancelActive(id);
      for (const id of retries.keys()) cancelRetry(id);
      selected.clear();
      clearCache();
      emit();
    },
  };
};
