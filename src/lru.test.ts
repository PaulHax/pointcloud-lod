import { describe, expect, it } from "vitest";

import { createLruCache } from "./lru";

describe("createLruCache", () => {
  it("rejects a non-positive byte budget", () => {
    expect(() => createLruCache({ maxBytes: 0 })).toThrow();
    expect(() => createLruCache({ maxBytes: -5 })).toThrow();
  });

  it("stores and retrieves values with byte accounting", () => {
    const cache = createLruCache<string, string>({ maxBytes: 100 });
    cache.set("a", "alpha", 10);
    cache.set("b", "beta", 20);

    expect(cache.get("a")).toBe("alpha");
    expect(cache.get("b")).toBe("beta");
    expect(cache.totalBytes()).toBe(30);
    expect(cache.count()).toBe(2);
    expect(cache.has("c")).toBe(false);
  });

  // Which key survived an over-budget insert is exactly what eviction order
  // means, so `has` is the observation channel throughout.
  it("evicts least-recently-used first when over budget", () => {
    const cache = createLruCache<string, number>({ maxBytes: 30 });
    cache.set("a", 1, 10);
    cache.set("b", 2, 10);
    cache.set("c", 3, 10);
    cache.set("d", 4, 10); // 40 bytes -> evict 'a'

    expect(cache.has("a")).toBe(false);
    expect(cache.has("b")).toBe(true);
    expect(cache.has("c")).toBe(true);
    expect(cache.has("d")).toBe(true);
    expect(cache.totalBytes()).toBe(30);
  });

  it("get refreshes recency", () => {
    const cache = createLruCache<string, number>({ maxBytes: 30 });
    cache.set("a", 1, 10);
    cache.set("b", 2, 10);
    cache.set("c", 3, 10);
    cache.get("a"); // now 'b' is oldest
    cache.set("d", 4, 10);

    expect(cache.has("b")).toBe(false);
    expect(cache.has("a")).toBe(true);
    expect(cache.has("c")).toBe(true);
    expect(cache.has("d")).toBe(true);
  });

  it("replacing a key updates byte accounting", () => {
    const cache = createLruCache<string, string>({ maxBytes: 100 });
    cache.set("a", "small", 10);
    cache.set("a", "large", 60);

    expect(cache.totalBytes()).toBe(60);
    expect(cache.count()).toBe(1);
    expect(cache.get("a")).toBe("large");
  });

  it("evicts multiple entries to fit one large insert", () => {
    const cache = createLruCache<string, number>({ maxBytes: 30 });
    cache.set("a", 1, 10);
    cache.set("b", 2, 10);
    cache.set("c", 3, 10);
    cache.set("big", 4, 25);

    expect(cache.has("a")).toBe(false);
    expect(cache.has("b")).toBe(false);
    expect(cache.has("c")).toBe(false);
    expect(cache.has("big")).toBe(true);
    expect(cache.totalBytes()).toBe(25);
  });

  it("an entry larger than the whole budget is evicted immediately", () => {
    const cache = createLruCache<string, number>({ maxBytes: 30 });
    cache.set("huge", 1, 50);

    expect(cache.has("huge")).toBe(false);
    expect(cache.totalBytes()).toBe(0);
    expect(cache.count()).toBe(0);
  });

  it("delete removes an entry", () => {
    const cache = createLruCache<string, number>({ maxBytes: 100 });
    cache.set("a", 1, 10);

    expect(cache.delete("a")).toBe(true);
    expect(cache.delete("a")).toBe(false);
    expect(cache.has("a")).toBe(false);
    expect(cache.totalBytes()).toBe(0);
  });

  it("clear removes everything", () => {
    const cache = createLruCache<string, number>({ maxBytes: 100 });
    cache.set("a", 1, 10);
    cache.set("b", 2, 10);
    cache.clear();

    expect(cache.has("a")).toBe(false);
    expect(cache.has("b")).toBe(false);
    expect(cache.count()).toBe(0);
    expect(cache.totalBytes()).toBe(0);
  });
});
