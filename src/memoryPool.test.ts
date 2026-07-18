import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  DEFAULT_MEMORY_BUDGET_BYTES,
  createMemoryPool,
  defaultMemoryBudgetBytes,
} from './memoryPool';

const MiB = 1024 * 1024;

describe('createMemoryPool', () => {
  it('gives a lone member the whole budget and splits evenly as members join', () => {
    const pool = createMemoryPool({ totalBytes: 900 });
    const a = pool.register();
    expect(a.budgetBytes()).toBe(900);
    const b = pool.register();
    expect(a.budgetBytes()).toBe(450);
    expect(b.budgetBytes()).toBe(450);
    const c = pool.register();
    expect(a.budgetBytes()).toBe(300);
    c.release();
    expect(a.budgetBytes()).toBe(450);
  });

  it('notifies existing members on join and leave, but never the joiner during register', () => {
    const pool = createMemoryPool({ totalBytes: 100 });
    const aChanges = vi.fn();
    const bChanges = vi.fn();
    pool.register(aChanges);
    expect(aChanges).not.toHaveBeenCalled();
    const b = pool.register(bChanges);
    expect(aChanges).toHaveBeenCalledTimes(1);
    expect(bChanges).not.toHaveBeenCalled();
    b.release();
    expect(aChanges).toHaveBeenCalledTimes(2);
  });

  it('release is idempotent and zeroes the released member', () => {
    const pool = createMemoryPool({ totalBytes: 100 });
    const aChanges = vi.fn();
    pool.register(aChanges);
    const b = pool.register();
    b.release();
    b.release();
    expect(b.budgetBytes()).toBe(0);
    expect(aChanges).toHaveBeenCalledTimes(2); // join + one release
    expect(pool.memberCount()).toBe(1);
  });

  it('setTotalBytes resizes shares and notifies every member', () => {
    const pool = createMemoryPool({ totalBytes: 100 });
    const changes = vi.fn();
    const member = pool.register(changes);
    pool.setTotalBytes(600);
    expect(member.budgetBytes()).toBe(600);
    expect(changes).toHaveBeenCalledTimes(1);
    pool.setTotalBytes(600); // unchanged → no notification
    expect(changes).toHaveBeenCalledTimes(1);
  });
});

describe('defaultMemoryBudgetBytes', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('falls back to the fixed default without a deviceMemory signal', () => {
    vi.stubGlobal('navigator', {});
    expect(defaultMemoryBudgetBytes()).toBe(DEFAULT_MEMORY_BUDGET_BYTES);
  });

  it('takes an eighth of reported device memory, clamped to [256 MiB, 1 GiB]', () => {
    vi.stubGlobal('navigator', { deviceMemory: 8 });
    expect(defaultMemoryBudgetBytes()).toBe(1024 * MiB);
    vi.stubGlobal('navigator', { deviceMemory: 4 });
    expect(defaultMemoryBudgetBytes()).toBe(512 * MiB);
    vi.stubGlobal('navigator', { deviceMemory: 0.5 });
    expect(defaultMemoryBudgetBytes()).toBe(256 * MiB);
  });
});
