/**
 * Byte-budgeted LRU cache. Intended for tile payloads / GPU-backed resources:
 * the eviction callback is the hook to dispose whatever the value owns.
 */

export interface LruCacheOptions<K, V> {
  /** Maximum total bytes retained. Must be > 0. */
  maxBytes: number;
  /** Called for every entry removed by eviction or `clear()` (not `delete()`). */
  onEvict?: (key: K, value: V, bytes: number) => void;
}

export interface LruCache<K, V> {
  /** Look up and mark as most recently used. */
  get(key: K): V | undefined;
  /** Look up without touching recency. */
  peek(key: K): V | undefined;
  /**
   * Insert or replace an entry, then evict least-recently-used entries until
   * the byte budget holds. An entry larger than the whole budget is evicted
   * immediately (via `onEvict`) and never retained.
   */
  set(key: K, value: V, bytes: number): void;
  /** Mark as most recently used. Returns false if absent. */
  touch(key: K): boolean;
  /** Remove without invoking `onEvict`. Returns false if absent. */
  delete(key: K): boolean;
  has(key: K): boolean;
  /** Evict every entry (each reported to `onEvict`). */
  clear(): void;
  /** Current total bytes retained. */
  totalBytes(): number;
  /** Current entry count. */
  count(): number;
}

interface Entry<V> {
  value: V;
  bytes: number;
}

export const createLruCache = <K, V>(
  options: LruCacheOptions<K, V>,
): LruCache<K, V> => {
  const { maxBytes, onEvict } = options;
  if (!(maxBytes > 0)) {
    throw new Error(`maxBytes must be > 0, got ${maxBytes}`);
  }

  // Map iteration order is insertion order; re-inserting on access makes the
  // first key the least recently used.
  const entries = new Map<K, Entry<V>>();
  let totalBytes = 0;

  const refresh = (key: K, entry: Entry<V>): void => {
    entries.delete(key);
    entries.set(key, entry);
  };

  const evictOldest = (): void => {
    const oldest = entries.entries().next();
    if (oldest.done === true) return;
    const [key, entry] = oldest.value;
    entries.delete(key);
    totalBytes -= entry.bytes;
    onEvict?.(key, entry.value, entry.bytes);
  };

  const enforceBudget = (): void => {
    while (totalBytes > maxBytes && entries.size > 0) {
      evictOldest();
    }
  };

  return {
    get(key) {
      const entry = entries.get(key);
      if (entry === undefined) return undefined;
      refresh(key, entry);
      return entry.value;
    },

    peek(key) {
      return entries.get(key)?.value;
    },

    set(key, value, bytes) {
      if (!(bytes >= 0)) {
        throw new Error(`bytes must be >= 0, got ${bytes}`);
      }
      const existing = entries.get(key);
      if (existing !== undefined) {
        totalBytes -= existing.bytes;
        entries.delete(key);
      }
      entries.set(key, { value, bytes });
      totalBytes += bytes;
      enforceBudget();
    },

    touch(key) {
      const entry = entries.get(key);
      if (entry === undefined) return false;
      refresh(key, entry);
      return true;
    },

    delete(key) {
      const entry = entries.get(key);
      if (entry === undefined) return false;
      entries.delete(key);
      totalBytes -= entry.bytes;
      return true;
    },

    has(key) {
      return entries.has(key);
    },

    clear() {
      while (entries.size > 0) {
        evictOldest();
      }
    },

    totalBytes() {
      return totalBytes;
    },

    count() {
      return entries.size;
    },
  };
};
