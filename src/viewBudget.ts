import type { GovernorInputs } from "./streamedMember";

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
 *
 * Importance and demand must already be normalized to [0, 1]: the
 * coordinator does that once, where it reads each member's inputs.
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
    ({ inputs }) => inputs.projectedImportance > 0 && inputs.qualityDemand > 0,
  );
  for (const contender of contenders) allocations.set(contender.key, 0);

  // Every pass either settles a capped contender or assigns the full
  // remainder, so the loop is bounded by the contender count.
  while (open.length > 0 && remaining > 0) {
    const importance = open.reduce(
      (sum, { inputs }) => sum + inputs.projectedImportance,
      0,
    );
    const proposed = new Map(
      open.map((contender) => [
        contender,
        (remaining * contender.inputs.projectedImportance) / importance,
      ]),
    );
    const capped = open.filter(
      (contender) => contender.inputs.qualityDemand <= proposed.get(contender)!,
    );
    if (capped.length === 0) {
      for (const contender of open) {
        allocations.set(
          contender.key,
          Math.min(contender.inputs.qualityDemand, proposed.get(contender)!),
        );
      }
      break;
    }
    for (const { key, inputs } of capped) {
      allocations.set(key, inputs.qualityDemand);
      remaining = Math.max(0, remaining - inputs.qualityDemand);
    }
    open = open.filter((contender) => !capped.includes(contender));
  }
  return allocations;
};
