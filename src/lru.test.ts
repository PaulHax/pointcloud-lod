import { describe, expect, it, vi } from 'vitest';

import { createLruCache } from './lru';

describe('createLruCache', () => {
  it('rejects a non-positive byte budget', () => {
    expect(() => createLruCache({ maxBytes: 0 })).toThrow();
    expect(() => createLruCache({ maxBytes: -5 })).toThrow();
  });

  it('stores and retrieves values with byte accounting', () => {
    const cache = createLruCache<string, string>({ maxBytes: 100 });
    cache.set('a', 'alpha', 10);
    cache.set('b', 'beta', 20);

    expect(cache.get('a')).toBe('alpha');
    expect(cache.peek('b')).toBe('beta');
    expect(cache.totalBytes()).toBe(30);
    expect(cache.count()).toBe(2);
    expect(cache.has('c')).toBe(false);
  });

  it('evicts least-recently-used first when over budget', () => {
    const evicted: string[] = [];
    const cache = createLruCache<string, number>({
      maxBytes: 30,
      onEvict: (key) => evicted.push(key),
    });
    cache.set('a', 1, 10);
    cache.set('b', 2, 10);
    cache.set('c', 3, 10);
    cache.set('d', 4, 10); // 40 bytes -> evict 'a'

    expect(evicted).toEqual(['a']);
    expect(cache.has('a')).toBe(false);
    expect(cache.totalBytes()).toBe(30);
  });

  it('get refreshes recency', () => {
    const evicted: string[] = [];
    const cache = createLruCache<string, number>({
      maxBytes: 30,
      onEvict: (key) => evicted.push(key),
    });
    cache.set('a', 1, 10);
    cache.set('b', 2, 10);
    cache.set('c', 3, 10);
    cache.get('a'); // now 'b' is oldest
    cache.set('d', 4, 10);

    expect(evicted).toEqual(['b']);
    expect(cache.has('a')).toBe(true);
  });

  it('touch refreshes recency without returning the value', () => {
    const evicted: string[] = [];
    const cache = createLruCache<string, number>({
      maxBytes: 20,
      onEvict: (key) => evicted.push(key),
    });
    cache.set('a', 1, 10);
    cache.set('b', 2, 10);

    expect(cache.touch('a')).toBe(true);
    expect(cache.touch('missing')).toBe(false);

    cache.set('c', 3, 10); // 'b' is now oldest
    expect(evicted).toEqual(['b']);
    expect(cache.has('a')).toBe(true);
  });

  it('peek does not refresh recency', () => {
    const evicted: string[] = [];
    const cache = createLruCache<string, number>({
      maxBytes: 20,
      onEvict: (key) => evicted.push(key),
    });
    cache.set('a', 1, 10);
    cache.set('b', 2, 10);
    cache.peek('a');
    cache.set('c', 3, 10);

    expect(evicted).toEqual(['a']);
  });

  it('replacing a key updates byte accounting', () => {
    const cache = createLruCache<string, string>({ maxBytes: 100 });
    cache.set('a', 'small', 10);
    cache.set('a', 'large', 60);

    expect(cache.totalBytes()).toBe(60);
    expect(cache.count()).toBe(1);
    expect(cache.get('a')).toBe('large');
  });

  it('evicts multiple entries to fit one large insert', () => {
    const evicted: string[] = [];
    const cache = createLruCache<string, number>({
      maxBytes: 30,
      onEvict: (key) => evicted.push(key),
    });
    cache.set('a', 1, 10);
    cache.set('b', 2, 10);
    cache.set('c', 3, 10);
    cache.set('big', 4, 25);

    expect(evicted).toEqual(['a', 'b', 'c']);
    expect(cache.has('big')).toBe(true);
    expect(cache.totalBytes()).toBe(25);
  });

  it('an entry larger than the whole budget is evicted immediately', () => {
    const onEvict = vi.fn();
    const cache = createLruCache<string, number>({ maxBytes: 30, onEvict });
    cache.set('huge', 1, 50);

    expect(cache.has('huge')).toBe(false);
    expect(cache.totalBytes()).toBe(0);
    expect(onEvict).toHaveBeenCalledWith('huge', 1, 50);
  });

  it('onEvict receives key, value, and bytes', () => {
    const onEvict = vi.fn();
    const cache = createLruCache<string, string>({ maxBytes: 10, onEvict });
    cache.set('a', 'alpha', 8);
    cache.set('b', 'beta', 8);

    expect(onEvict).toHaveBeenCalledTimes(1);
    expect(onEvict).toHaveBeenCalledWith('a', 'alpha', 8);
  });

  it('delete removes without calling onEvict', () => {
    const onEvict = vi.fn();
    const cache = createLruCache<string, number>({ maxBytes: 100, onEvict });
    cache.set('a', 1, 10);

    expect(cache.delete('a')).toBe(true);
    expect(cache.delete('a')).toBe(false);
    expect(cache.totalBytes()).toBe(0);
    expect(onEvict).not.toHaveBeenCalled();
  });

  it('clear evicts everything through onEvict', () => {
    const evicted: string[] = [];
    const cache = createLruCache<string, number>({
      maxBytes: 100,
      onEvict: (key) => evicted.push(key),
    });
    cache.set('a', 1, 10);
    cache.set('b', 2, 10);
    cache.clear();

    expect(evicted).toEqual(['a', 'b']);
    expect(cache.count()).toBe(0);
    expect(cache.totalBytes()).toBe(0);
  });
});
