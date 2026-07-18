/**
 * Shared GPU-memory budget for resident tiles.
 *
 * The adaptive budget loop only measures render duration, and frame time
 * stays healthy right up until GPU memory runs out — then the failure is an
 * allocation error or a lost context, not a slow frame. The pool is the
 * memory half of the governor: a byte budget for resident tile data that the
 * controller converts into a point ceiling the frame-time loop can never
 * exceed.
 *
 * One pool per GPU (in practice: per page). Every controller registers as a
 * member and receives an even share of the total, so N clouds never multiply
 * the memory footprint by N. Membership changes and total changes notify the
 * remaining members so they can re-derive their ceilings and reselect.
 */

const MiB = 1024 * 1024;

/** Fallback byte budget when the environment reports nothing about memory. */
export const DEFAULT_MEMORY_BUDGET_BYTES = 512 * MiB;

/**
 * Estimate a byte budget for resident tile data on this device.
 *
 * WebGL exposes no VRAM size, so this leans on `navigator.deviceMemory`
 * (coarse, capped at 8 GiB, Chromium-only): an eighth of reported RAM,
 * clamped to [256 MiB, 1 GiB]. Integrated GPUs share system RAM, and on
 * discrete GPUs driver paging makes system RAM a serviceable proxy. Absent
 * the signal, a fixed default — still strictly better than a point count,
 * which ignores bytes per point entirely.
 */
export const defaultMemoryBudgetBytes = (): number => {
  const nav = (globalThis as { navigator?: { deviceMemory?: unknown } })
    .navigator;
  const deviceGb = nav?.deviceMemory;
  if (typeof deviceGb === 'number' && Number.isFinite(deviceGb) && deviceGb > 0) {
    return Math.min(Math.max((deviceGb * 1024 * MiB) / 8, 256 * MiB), 1024 * MiB);
  }
  return DEFAULT_MEMORY_BUDGET_BYTES;
};

export interface MemoryPoolMember {
  /** This member's current byte allowance (an even share of the total). */
  budgetBytes(): number;
  /** Leave the pool; the remaining members' shares grow. Idempotent. */
  release(): void;
}

export interface MemoryPoolOptions {
  /** Total byte budget to divide. Default `defaultMemoryBudgetBytes()`. */
  totalBytes?: number;
}

export interface MemoryPool {
  /**
   * Join the pool. `onChange` fires whenever this member's share moves —
   * another member joined or left, or the total changed — but never during
   * this `register` call itself.
   */
  register(onChange?: () => void): MemoryPoolMember;
  totalBytes(): number;
  /** Change the total; every member is notified. */
  setTotalBytes(bytes: number): void;
  memberCount(): number;
}

export const createMemoryPool = (
  options: MemoryPoolOptions = {},
): MemoryPool => {
  let totalBytes = Math.max(
    1,
    Math.floor(options.totalBytes ?? defaultMemoryBudgetBytes()),
  );
  const members = new Set<{ onChange?: () => void }>();

  const notify = (except?: object): void => {
    for (const member of [...members]) {
      if (member !== except) member.onChange?.();
    }
  };

  return {
    register(onChange) {
      const entry = { onChange };
      members.add(entry);
      // The new member reads its share lazily via budgetBytes(); only the
      // existing members need to hear that their shares shrank.
      notify(entry);
      let released = false;
      return {
        budgetBytes: () =>
          released ? 0 : Math.floor(totalBytes / Math.max(1, members.size)),
        release: () => {
          if (released) return;
          released = true;
          members.delete(entry);
          notify();
        },
      };
    },

    totalBytes: () => totalBytes,

    setTotalBytes(bytes) {
      const next = Math.max(1, Math.floor(bytes));
      if (next === totalBytes) return;
      totalBytes = next;
      notify();
    },

    memberCount: () => members.size,
  };
};
