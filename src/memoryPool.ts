/**
 * Shared GPU-memory budget for resident tiles.
 *
 * The adaptive quality loop only measures render duration, and frame time
 * stays healthy right up until GPU memory runs out — then the failure is an
 * allocation error or a lost context, not a slow frame. Memory is therefore
 * a separate byte axis: each streamed member interprets its allowance in its
 * own format-specific way.
 *
 * One pool per GPU (in practice: per page). Every active member registers as a
 * member and receives an even share of the total, so N clouds never multiply
 * the memory footprint by N. Membership changes notify the remaining members
 * so they can re-derive their ceilings and reselect.
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
  if (
    typeof deviceGb === "number" &&
    Number.isFinite(deviceGb) &&
    deviceGb > 0
  ) {
    return Math.min(
      Math.max((deviceGb * 1024 * MiB) / 8, 256 * MiB),
      1024 * MiB,
    );
  }
  return DEFAULT_MEMORY_BUDGET_BYTES;
};

export type MemoryPoolMember = {
  /** This member's current byte allowance (an even share of the total). */
  budgetBytes(): number;
  /** Leave the pool; the remaining members' shares grow. Idempotent. */
  release(): void;
};

export type MemoryPoolOptions = {
  /** Total byte budget to divide. Default `defaultMemoryBudgetBytes()`. */
  totalBytes?: number;
};

export type MemoryPool = {
  /**
   * Join the pool. `onChange` fires whenever this member's share moves —
   * another member joined or left, or the total changed — but never during
   * this `register` call itself.
   */
  register(onChange?: () => void): MemoryPoolMember;
  memberCount(): number;
  /** Aggregate public diagnostics for cross-format/page-wide enforcement. */
  stats?(): {
    readonly totalBytes: number;
    readonly memberCount: number;
    readonly shareBytes: number;
  };
};

export const createMemoryPool = (
  options: MemoryPoolOptions = {},
): MemoryPool => {
  const totalBytes = Math.max(
    1,
    Math.floor(options.totalBytes ?? defaultMemoryBudgetBytes()),
  );
  const members = new Set<{ onChange?: () => void }>();

  const notify = (except?: object): void => {
    // Snapshot deliberately: an `onChange` may release ANOTHER member, and a
    // Set drops a not-yet-visited entry deleted mid-iteration. Every member
    // registered when the change happened is entitled to hear about it.
    // oxlint-disable-next-line unicorn/no-useless-spread
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
      return {
        budgetBytes: () =>
          members.has(entry) ? Math.floor(totalBytes / members.size) : 0,
        release: () => {
          // Set.delete answers false on the second call — idempotence for free.
          if (members.delete(entry)) notify();
        },
      };
    },

    memberCount: () => members.size,
    stats: () => ({
      totalBytes,
      memberCount: members.size,
      shareBytes: members.size
        ? Math.floor(totalBytes / members.size)
        : totalBytes,
    }),
  };
};
