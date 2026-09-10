// @ts-check

/** @typedef {Pick<import('../src/streamedSceneCoordinator').StreamedSceneCoordinatorStats, 'governor' | 'members' | 'submissions'>} Coordinator */

/**
 * Shared by the live replay wait and the offline settle-marker check.
 * Fixed-only scenes do not train the governor or consume its frame requests.
 * @param {Coordinator | null | undefined} coordinator
 */
export const coordinatorConverged = (coordinator) => {
  if (!coordinator) return false;
  const { governor } = coordinator;
  return (
    governor?.regime === "stationary" &&
    governor.activity.workPending === false &&
    coordinator.submissions.queuedJobs === 0 &&
    (!coordinator.members.some(
      (member) => member.active && member.qualityManaged,
    ) ||
      governor.needsFrame === false)
  );
};

/**
 * Dataset idleness can precede the next quality adjustment.
 * @param {{ loading: number, datasets: readonly { state: string }[] }} activity
 * @param {Coordinator} coordinator
 */
export const sceneConverged = (activity, coordinator) =>
  activity.loading === 0 &&
  activity.datasets.length > 0 &&
  activity.datasets.every(
    ({ state }) => state === "settled" || state === "error",
  ) &&
  coordinatorConverged(coordinator);
