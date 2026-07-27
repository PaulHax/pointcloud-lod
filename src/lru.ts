/**
 * Byte-budgeted LRU cache for decoded tile payloads: inserting past the
 * budget evicts least-recently-used entries until the budget holds again.
 */

import { finiteAbove, finiteAtLeast } from "./numeric";

export type LruCacheOptions = {
  /** Maximum total bytes retained. Must be > 0. */
  maxBytes: number;
};

export type LruCache<K, V> = {
  /** Look up and mark as most recently used. */
  get(key: K): V | undefined;
  /**
   * Insert or replace an entry, then evict least-recently-used entries until
   * the byte budget holds. An entry larger than the whole budget is evicted
   * immediately and never retained.
   */
  set(key: K, value: V, bytes: number): void;
  /** Remove. Returns false if absent. */
  delete(key: K): boolean;
  has(key: K): boolean;
  /** Remove every entry. */
  clear(): void;
  /** Current total bytes retained. */
  totalBytes(): number;
  /** Current entry count. */
  count(): number;
};

type Entry<V> = {
  value: V;
  bytes: number;
};

export const createLruCache = <K, V>(
  options: LruCacheOptions,
): LruCache<K, V> => {
  const { maxBytes } = options;
  finiteAbove("maxBytes", maxBytes, 0);

  // Map iteration order is insertion order; re-inserting on access makes the
  // first key the least recently used.
  const entries = new Map<K, Entry<V>>();
  let totalBytes = 0;

  /** Drop an entry and un-charge its bytes. Returns false if absent. */
  const remove = (key: K): boolean => {
    const entry = entries.get(key);
    if (entry === undefined) return false;
    entries.delete(key);
    totalBytes -= entry.bytes;
    return true;
  };

  const enforceBudget = (): void => {
    while (totalBytes > maxBytes) {
      const oldest = entries.keys().next();
      if (oldest.done === true) return;
      remove(oldest.value);
    }
  };

  return {
    get(key) {
      const entry = entries.get(key);
      if (entry === undefined) return undefined;
      entries.delete(key);
      entries.set(key, entry);
      return entry.value;
    },

    set(key, value, bytes) {
      finiteAtLeast("bytes", bytes, 0);
      remove(key);
      entries.set(key, { value, bytes });
      totalBytes += bytes;
      enforceBudget();
    },

    delete: remove,

    has(key) {
      return entries.has(key);
    },

    clear() {
      entries.clear();
      totalBytes = 0;
    },

    totalBytes() {
      return totalBytes;
    },

    count() {
      return entries.size;
    },
  };
};
