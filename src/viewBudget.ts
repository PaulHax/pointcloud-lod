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
  // Importance is contractually [0, 1]. Clamping here rather than trusting the
  // producers means one member reporting on the wrong scale can at worst take
  // a full share, never crowd every other member down to the quality floor.
  const importanceOf = ({ inputs }: ViewQualityContender<Key>): number =>
    Number.isFinite(inputs.projectedImportance)
      ? Math.min(1, Math.max(0, inputs.projectedImportance))
      : 0;
  let open = contenders.filter(
    (contender) =>
      importanceOf(contender) > 0 &&
      Number.isFinite(contender.inputs.qualityDemand) &&
      contender.inputs.qualityDemand > 0,
  );
  for (const contender of contenders) allocations.set(contender.key, 0);

  // Every pass either settles a capped contender or assigns the full
  // remainder, so the loop is bounded by the contender count.
  while (open.length > 0 && remaining > 0) {
    const importance = open.reduce(
      (sum, contender) => sum + importanceOf(contender),
      0,
    );
    if (!(importance > 0)) break;
    const proposed = new Map(
      open.map((contender) => [
        contender,
        (remaining * importanceOf(contender)) / importance,
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
