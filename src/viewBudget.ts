import type { AllocationRegime, GovernorInputs } from "./streamedMember";

/** The adaptive view governor never asks a member to render below this. */
export const MIN_VIEW_QUALITY_FRACTION = 0.05;
export const MAX_VIEW_QUALITY_FRACTION = 1;

export type ViewQualityContender<Key> = {
  readonly key: Key;
  readonly inputs: GovernorInputs;
};

/**
 * Weighted, demand-capped water filling for one normalized view-quality
 * fraction. The view owns `viewFraction * activeMemberCount` units of quality.
 * Equal-importance members therefore receive the view fraction; a member that
 * cannot spend its share returns it to the remaining contenders.
 */
export const allocateViewQuality = <Key>(
  contenders: readonly ViewQualityContender<Key>[],
  viewFraction: number,
): ReadonlyMap<Key, number> => {
  const allocations = new Map<Key, number>();
  const usableFraction = Number.isFinite(viewFraction)
    ? Math.min(MAX_VIEW_QUALITY_FRACTION, Math.max(0, viewFraction))
    : 0;
  let remaining = usableFraction * contenders.length;
  let open = contenders.filter(
    ({ inputs }) =>
      Number.isFinite(inputs.projectedImportance) &&
      inputs.projectedImportance > 0 &&
      Number.isFinite(inputs.qualityDemand) &&
      inputs.qualityDemand > 0,
  );
  for (const contender of contenders) allocations.set(contender.key, 0);

  // Every pass either settles a capped contender or assigns the full
  // remainder, so the loop is bounded by the contender count.
  while (open.length > 0 && remaining > 0) {
    const importance = open.reduce(
      (sum, contender) => sum + contender.inputs.projectedImportance,
      0,
    );
    if (!(importance > 0)) break;
    const proposed = new Map(
      open.map((contender) => [
        contender,
        (remaining * contender.inputs.projectedImportance) / importance,
      ]),
    );
    const capped = open.filter((contender) => {
      const cap = Math.min(1, contender.inputs.qualityDemand);
      return cap <= proposed.get(contender)!;
    });
    if (capped.length === 0) {
      for (const contender of open) {
        allocations.set(
          contender.key,
          Math.min(1, contender.inputs.qualityDemand, proposed.get(contender)!),
        );
      }
      break;
    }
    for (const contender of capped) {
      const allocation = Math.min(1, contender.inputs.qualityDemand);
      allocations.set(contender.key, allocation);
      remaining = Math.max(0, remaining - allocation);
    }
    open = open.filter((contender) => !capped.includes(contender));
  }
  return allocations;
};

export type ViewQualityAllocation<Key> = {
  readonly viewFraction: number;
  readonly regime: AllocationRegime;
  readonly allocations: ReadonlyMap<Key, number>;
};
