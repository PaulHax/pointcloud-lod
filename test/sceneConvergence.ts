import type { ViewGovernorStats } from "../src/viewGovernor";

/** Dataset idleness can precede the governor's next quality adjustment. */
export const sceneConverged = (
  activity: {
    readonly loading: number;
    readonly datasets: readonly { readonly state: string }[];
  },
  governor: Pick<ViewGovernorStats, "regime" | "needsFrame" | "activity">,
  members: readonly {
    readonly active: boolean;
    readonly qualityManaged: boolean;
  }[],
  queuedSubmissions: number,
): boolean =>
  activity.loading === 0 &&
  activity.datasets.length > 0 &&
  activity.datasets.every(
    (dataset) => dataset.state === "settled" || dataset.state === "error",
  ) &&
  governor.regime === "stationary" &&
  !governor.activity.workPending &&
  queuedSubmissions === 0 &&
  (!members.some((member) => member.active && member.qualityManaged) ||
    !governor.needsFrame);
