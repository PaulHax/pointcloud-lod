import { copyCameraView, sameCameraView, type CameraView } from "./camera";
import { sameMatrix, type Mat16 } from "./mat4";
import type { MemoryPool, MemoryPoolMember } from "./memoryPool";
import { finiteNonNegative } from "./numeric";
import { allocateViewQuality } from "./viewBudget";
import {
  createSubmissionScheduler,
  type SubmissionScheduler,
} from "./submissionScheduler";
import {
  CULLED,
  type Allocation,
  type AllocationRegime,
  type DecodeWorkerPool,
  type GovernorInputs,
  type Importance,
  type StreamedMember,
  type StreamedMemberContext,
  type TextureCapabilities,
} from "./streamedMember";
import {
  createViewGovernor,
  type FrameVerdict,
  type HostFrameMetrics,
  type MotionReference,
  type ViewGovernorOptions,
  type ViewGovernorStats,
} from "./viewGovernor";

const EMPTY_WORKERS: DecodeWorkerPool = {
  decode: () => {
    throw new Error("decode workers are unavailable");
  },
};
const EMPTY_TEXTURE_CAPABILITIES: TextureCapabilities = {
  capabilityKey: "none",
  compressedFormats: [],
};

export type AdaptiveQualityTargets = {
  readonly interactionTargetMs?: number;
  readonly stationaryTargetMs?: number;
};

export type StreamedMemberRegistrationOptions = {
  readonly id?: string;
  readonly active?: boolean;
  /** Fixed members receive bytes/work tracking but bypass quality allocation. */
  readonly qualityManaged?: boolean;
  /** Only managed point members state these; stable registry order breaks ties. */
  readonly qualityTargets?: AdaptiveQualityTargets;
};

export type StreamedMemberRegistration = {
  setCamera(view: CameraView): void;
  setModelMatrix(matrix: Mat16 | null): void;
  setDevicePixelRatio(devicePixelRatio: number): void;
  setActive(active: boolean): void;
  setConfig(kindConfig: object): void;
  setQualityPolicy(managed: boolean, targets?: AdaptiveQualityTargets): void;
  release(): void;
};

export type StreamedSceneCoordinatorOptions = {
  readonly scheduleRender: () => void;
  /** Required page-wide pool shared by every view coordinator on the GPU. */
  readonly memory: MemoryPool;
  readonly workers?: DecodeWorkerPool;
  readonly textureCapabilities?: TextureCapabilities;
  readonly devicePixelRatio?: number;
  /** Injectable monotonic-ish clock for deterministic stall tests. */
  readonly now?: () => number;
  /** Motion/sample policy. Quality range and member target override are owned here. */
  readonly governor?: Omit<
    ViewGovernorOptions,
    "initialFraction" | "interactionTargetMs" | "stationaryTargetMs"
  >;
};

export type StreamedCoordinatorMemberStats = {
  readonly id: string | null;
  readonly active: boolean;
  readonly qualityManaged: boolean;
  readonly governorInputs: GovernorInputs;
  readonly allocation: Allocation;
};

export type StreamedSceneCoordinatorStats = {
  /** The governor's fraction now; members hold it from the next prepared frame. */
  readonly viewQualityFraction: number;
  readonly targetOverrideMemberId: string | null;
  readonly governor: ViewGovernorStats;
  readonly submissions: ReturnType<SubmissionScheduler["stats"]>;
  readonly stalledMembers: readonly string[];
  readonly members: readonly StreamedCoordinatorMemberStats[];
};

export type StreamedSceneCoordinator = {
  /**
   * Build one member context. Renderer and initial DPR belong to the anchor,
   * while memory/workers/submissions remain shared by the view/page.
   */
  context(renderer: unknown, devicePixelRatio?: number): StreamedMemberContext;
  register(
    member: StreamedMember,
    options?: StreamedMemberRegistrationOptions,
  ): StreamedMemberRegistration;
  /** Feed the complete rendered-camera map to the view motion classifier. */
  noteRenderedCameras(
    views: ReadonlyMap<unknown, CameraView | null | undefined>,
  ): void;
  /** Set the view-wide frame targets independently of any scene member. */
  setQualityTargets(targets: AdaptiveQualityTargets): void;
  beginInteraction(): void;
  endInteraction(): void;
  /**
   * Run immediately before each paint: member preparation, the shared
   * admission drain, one read of every member's inputs, and the frame's
   * allocations. This is the only place members receive allocations.
   */
  prepareFrame(frameSerial: number): void;
  /**
   * Report one presented frame. The verdict says whether the governor took it
   * as a capacity sample, whether that sample may train quality, and the
   * regime it counted toward, so a host never has to read `stats()` per frame.
   * It applies nothing: a quality change lands at the next `prepareFrame`.
   */
  recordHostFrame(metrics: HostFrameMetrics): FrameVerdict;
  /**
   * Whether the view owes another frame: queued submissions, a governor that
   * is still measuring, or an allocation that has changed since it was applied.
   */
  needsFrame(): boolean;
  /** Diagnostics as last observed. Reading them calls no member. */
  stats(): StreamedSceneCoordinatorStats;
  dispose(): void;
};

type MemberState = {
  readonly member: StreamedMember;
  readonly id: string | null;
  active: boolean;
  qualityManaged: boolean;
  qualityTargets: AdaptiveQualityTargets | undefined;
  memoryMember: MemoryPoolMember | null;
  inputs: GovernorInputs;
  allocation: Allocation;
  lastProgressSerial: number;
  lastProgressAt: number;
  stalled: boolean;
  stallReported: boolean;
};

const EMPTY_INPUTS: GovernorInputs = {
  projectedImportance: CULLED,
  qualityDemand: 0,
  work: { operations: 0, progressSerial: 0 },
  physicalTileOperations: 0,
  physicalHierarchyOperations: 0,
  residentBytes: 0,
};

const usableNonNegative = (value: number): number =>
  finiteNonNegative(value) ? value : 0;

const usableFraction = (value: number): number =>
  Math.min(1, usableNonNegative(value));

/**
 * The one boundary for member inputs. Custom members are an extension point,
 * so a member reporting on the wrong scale takes at most a full share here,
 * and nothing non-finite reaches the allocator or the governor.
 */
const normalizeInputs = (inputs: GovernorInputs): GovernorInputs => ({
  projectedImportance: usableFraction(inputs.projectedImportance) as Importance,
  qualityDemand: usableFraction(inputs.qualityDemand),
  work: {
    operations: Math.floor(usableNonNegative(inputs.work.operations)),
    progressSerial: Math.floor(usableNonNegative(inputs.work.progressSerial)),
  },
  physicalTileOperations: Math.floor(
    usableNonNegative(inputs.physicalTileOperations),
  ),
  physicalHierarchyOperations: Math.floor(
    usableNonNegative(inputs.physicalHierarchyOperations),
  ),
  residentBytes: Math.floor(usableNonNegative(inputs.residentBytes)),
});

/** Lack-of-progress window before one member is isolated from view gating. */
const STALL_WINDOW_MS = 4_000;

const sameAllocation = (left: Allocation, right: Allocation): boolean =>
  left.qualityFraction === right.qualityFraction &&
  left.memoryBudgetBytes === right.memoryBudgetBytes &&
  left.regime === right.regime;

const sameTargets = (
  left: AdaptiveQualityTargets | undefined,
  right: AdaptiveQualityTargets | undefined,
): boolean =>
  left?.interactionTargetMs === right?.interactionTargetMs &&
  left?.stationaryTargetMs === right?.stationaryTargetMs;

const isAdaptive = (state: MemberState): boolean =>
  state.active && state.qualityManaged;

/**
 * What every member should hold: the view fraction water-filled across the
 * adaptive members by their last-read inputs, full quality for fixed members,
 * each active member's page memory share, and nothing for inactive members.
 *
 * `fractionApplied` says the current allocations were made at this same view
 * fraction. While the camera moves, adaptive members keep their shares if
 * the visible contenders have not changed: demand follows the camera, so
 * passing every change on would swap their tiles all through a gesture.
 */
const allocate = (
  states: readonly MemberState[],
  viewFraction: number,
  regime: AllocationRegime,
  fractionApplied: boolean,
): ReadonlyMap<MemberState, Allocation> => {
  const adaptive = states.filter(isAdaptive);
  const quality = allocateViewQuality(
    adaptive.map((state) => ({ key: state, inputs: state.inputs })),
    viewFraction,
  );
  const holding =
    fractionApplied &&
    regime === "moving" &&
    adaptive.every((state) => {
      const hadShare = state.allocation.qualityFraction > 0;
      const hasShare = (quality.get(state) ?? 0) > 0;
      return state.allocation.regime === "moving" && hadShare === hasShare;
    });
  return new Map(
    states.map((state) => {
      const share = !state.active
        ? 0
        : state.qualityManaged
          ? (quality.get(state) ?? 0)
          : 1;
      const held =
        holding &&
        state.qualityManaged &&
        state.allocation.regime === "moving" &&
        state.allocation.qualityFraction > 0 &&
        share > 0;
      return [
        state,
        {
          qualityFraction: held ? state.allocation.qualityFraction : share,
          memoryBudgetBytes: state.active
            ? (state.memoryMember?.budgetBytes() ?? 0)
            : 0,
          regime,
        },
      ];
    }),
  );
};

export const createStreamedSceneCoordinator = (
  options: StreamedSceneCoordinatorOptions,
): StreamedSceneCoordinator => {
  if (!options.memory) {
    throw new Error(
      "createStreamedSceneCoordinator requires the page-wide MemoryPool",
    );
  }
  const memory = options.memory;
  const submissions = createSubmissionScheduler({
    scheduleRender: options.scheduleRender,
  });
  const members = new Set<MemberState>();
  let disposed = false;
  const now = options.now ?? Date.now;
  let stallTimer: ReturnType<typeof setTimeout> | null = null;
  let lastPreparedFrameSerial = -1;
  let capacityWorkSinceLastReport = false;
  // Some member may now report different inputs than the ones last read.
  let inputsStale = false;
  let checkQueued = false;
  let globalQualityTargets: AdaptiveQualityTargets | undefined;
  let appliedTargetState: MemberState | null = null;
  let appliedTargets: AdaptiveQualityTargets | undefined;
  const motionReferences: MotionReference[] = [];

  const governorOptions = (
    targets?: AdaptiveQualityTargets,
  ): ViewGovernorOptions => ({
    ...options.governor,
    initialFraction: 1,
    ...(targets?.interactionTargetMs === undefined
      ? {}
      : { interactionTargetMs: targets.interactionTargetMs }),
    ...(targets?.stationaryTargetMs === undefined
      ? {}
      : { stationaryTargetMs: targets.stationaryTargetMs }),
  });
  const governor = createViewGovernor(governorOptions());

  const hasAdaptiveMember = (): boolean => {
    for (const state of members) if (isAdaptive(state)) return true;
    return false;
  };

  // The view fraction the current allocations were applied at.
  let appliedViewFraction: number | null = null;

  const plannedAllocations = (): ReadonlyMap<MemberState, Allocation> => {
    const viewFraction = governor.qualityFraction();
    return allocate(
      [...members],
      viewFraction,
      governor.regime() === "interaction" ? "moving" : "stationary",
      viewFraction === appliedViewFraction,
    );
  };

  const allocationOutdated = (): boolean => {
    for (const [state, next] of plannedAllocations()) {
      if (!sameAllocation(next, state.allocation)) return true;
    }
    return false;
  };

  const applyAllocations = (): void => {
    for (const [state, next] of plannedAllocations()) {
      if (sameAllocation(next, state.allocation)) continue;
      state.allocation = next;
      state.member.applyAllocation(next);
    }
    appliedViewFraction = governor.qualityFraction();
  };

  const invalidateCapacity = (): void => {
    if (disposed || governor.frameCount() === 0) return;
    governor.invalidateCapacity();
    // Reject the presentation spanning the mutation as well as the previous
    // workload's samples. A style or visibility change may enqueue no work.
    capacityWorkSinceLastReport = true;
    if (hasAdaptiveMember()) options.scheduleRender();
  };

  const updateTargets = (): void => {
    // Insertion order is the conflict rule. A hidden managed member yields
    // to the first active one and regains precedence if it is shown again.
    // Only adaptive point members supply a target bag. A tiles member is
    // quality-managed too, but must not mask the first adaptive point's
    // stable registry-order override; a tiles-only view uses defaults.
    const targetState = globalQualityTargets
      ? null
      : ([...members].find(
          (state) => isAdaptive(state) && state.qualityTargets !== undefined,
        ) ?? null);
    const nextTargets = globalQualityTargets ?? targetState?.qualityTargets;
    if (
      targetState === appliedTargetState &&
      sameTargets(nextTargets, appliedTargets)
    ) {
      return;
    }
    governor.setOptions(governorOptions(nextTargets));
    appliedTargetState = targetState;
    appliedTargets = nextTargets ? { ...nextTargets } : undefined;
  };

  const updateStalls = (): void => {
    if (stallTimer !== null) clearTimeout(stallTimer);
    stallTimer = null;
    const checkedAt = now();
    let nextStallDelay = Number.POSITIVE_INFINITY;
    for (const state of members) {
      const operations =
        state.inputs.work.operations +
        state.inputs.physicalTileOperations +
        state.inputs.physicalHierarchyOperations;
      if (!state.active || operations === 0) {
        state.lastProgressSerial = state.inputs.work.progressSerial;
        state.lastProgressAt = checkedAt;
        state.stalled = false;
        state.stallReported = false;
        continue;
      }
      if (state.inputs.work.progressSerial !== state.lastProgressSerial) {
        state.lastProgressSerial = state.inputs.work.progressSerial;
        state.lastProgressAt = checkedAt;
        state.stalled = false;
        state.stallReported = false;
      }
      const remaining = STALL_WINDOW_MS - (checkedAt - state.lastProgressAt);
      if (remaining <= 0) {
        state.stalled = true;
        if (!state.stallReported) {
          state.stallReported = true;
          state.member.onStall?.(
            new Error(
              `streamed member ${state.id ?? "<unnamed>"} made no progress for ${STALL_WINDOW_MS} ms`,
            ),
          );
        }
      } else {
        nextStallDelay = Math.min(nextStallDelay, remaining);
      }
    }
    if (Number.isFinite(nextStallDelay)) {
      stallTimer = setTimeout(
        markStale,
        Math.max(1, Math.ceil(nextStallDelay)),
      );
    }
  };

  /**
   * Read every member's inputs once and derive the view's work state from
   * them. Members are only read here, apart from a first stall report.
   * Returns whether the view's work just drained.
   */
  const observe = (): boolean => {
    inputsStale = false;
    for (const state of members) {
      state.inputs = state.active
        ? normalizeInputs(state.member.governorInputs())
        : EMPTY_INPUTS;
    }
    updateStalls();
    // Fixed members do not consume normalized quality, but their frame cost
    // and incomplete work still contaminate the same view-wide sample.
    let tileOperations = 0;
    let hierarchyOperations = 0;
    let pending = submissions.hasPending();
    for (const state of members) {
      if (!state.active || state.stalled) continue;
      tileOperations += state.inputs.physicalTileOperations;
      hierarchyOperations += state.inputs.physicalHierarchyOperations;
      pending ||= state.inputs.work.operations > 0;
    }
    const wasPending = governor.workPending();
    governor.setWorkState({
      physicalTileOperations: tileOperations,
      physicalHierarchyOperations: hierarchyOperations,
      workPending: pending,
    });
    const nowPending = governor.workPending();
    capacityWorkSinceLastReport ||= nowPending;
    return wasPending && !nowPending;
  };

  /**
   * Cancellation and budget backoff can finish without submitting an actor,
   * and the governor still needs a frame to measure the result.
   */
  const owesMeasurement = (drained: boolean): boolean =>
    drained && hasAdaptiveMember();

  /**
   * The coalesced answer to stale inputs: read them, keep the work state
   * current, and ask for a frame when one is owed. It applies nothing.
   *
   * The check counts as queued until it returns, so nothing it calls can
   * queue another: a member that reports work whenever it is read cannot hold
   * the page inside microtasks. Such a report still marks the inputs stale,
   * and the next frame reads them.
   */
  const checkInputs = (): void => {
    try {
      if (disposed || !inputsStale) return;
      if (owesMeasurement(observe()) || allocationOutdated()) {
        options.scheduleRender();
      }
    } finally {
      checkQueued = false;
    }
  };

  /**
   * Record that some member's inputs, or an allocation, may have changed.
   * At most one check is queued at a time, and this never calls a member.
   */
  const markStale = (): void => {
    if (disposed) return;
    inputsStale = true;
    if (checkQueued) return;
    checkQueued = true;
    queueMicrotask(checkInputs);
  };

  return {
    context: (renderer, devicePixelRatio = options.devicePixelRatio ?? 1) => ({
      renderer,
      scheduleRender: options.scheduleRender,
      memory,
      workers: options.workers ?? EMPTY_WORKERS,
      submissions,
      textureCapabilities:
        options.textureCapabilities ?? EMPTY_TEXTURE_CAPABILITIES,
      devicePixelRatio,
      onWorkChange: markStale,
    }),

    register(member, registration = {}) {
      const state: MemberState = {
        member,
        id: registration.id ?? null,
        active: registration.active ?? true,
        qualityManaged: registration.qualityManaged ?? false,
        qualityTargets: registration.qualityTargets,
        memoryMember: null,
        inputs: EMPTY_INPUTS,
        allocation: {
          qualityFraction: -1,
          memoryBudgetBytes: -1,
          regime: "stationary",
        },
        lastProgressSerial: 0,
        lastProgressAt: now(),
        stalled: false,
        stallReported: false,
      };
      if (disposed) {
        member.dispose();
        return {
          setCamera() {},
          setModelMatrix() {},
          setDevicePixelRatio() {},
          setActive() {},
          setConfig() {},
          setQualityPolicy() {},
          release() {},
        };
      }
      members.add(state);
      appliedViewFraction = null;
      if (state.active) {
        invalidateCapacity();
        state.memoryMember = memory.register(markStale);
      }
      member.setActive(state.active);
      // A member can be realized or rebuilt while the view is already inside
      // one or more nested interactions. Replay every held begin so its
      // eventual end calls remain balanced and format-local interaction state
      // never starts stationary in the middle of a gesture.
      for (let index = 0; index < motionReferences.length; index += 1) {
        member.beginInteraction();
      }
      updateTargets();
      markStale();
      let released = false;
      let lastCamera: CameraView | null = null;
      let lastModelMatrix: Mat16 | null = null;
      let modelMatrixSet = false;
      let lastDevicePixelRatio: number | null = null;
      return {
        setCamera(view) {
          if (released || disposed) return;
          if (lastCamera !== null && sameCameraView(lastCamera, view)) return;
          lastCamera = copyCameraView(view);
          state.member.setCamera(view);
          markStale();
        },
        setModelMatrix(matrix) {
          if (released || disposed) return;
          if (modelMatrixSet && sameMatrix(lastModelMatrix, matrix)) return;
          modelMatrixSet = true;
          lastModelMatrix =
            matrix === null ? null : (Array.from(matrix) as Mat16);
          state.member.setModelMatrix(matrix);
          markStale();
        },
        setDevicePixelRatio(devicePixelRatio) {
          if (released || disposed) return;
          if (devicePixelRatio === lastDevicePixelRatio) return;
          lastDevicePixelRatio = devicePixelRatio;
          state.member.setDevicePixelRatio(devicePixelRatio);
        },
        setActive(active) {
          if (released || disposed || active === state.active) return;
          state.active = active;
          appliedViewFraction = null;
          invalidateCapacity();
          if (active) state.memoryMember = memory.register(markStale);
          else {
            state.memoryMember?.release();
            state.memoryMember = null;
          }
          state.member.setActive(active);
          updateTargets();
          markStale();
        },
        setConfig(kindConfig) {
          if (released || disposed) return;
          if (state.active) invalidateCapacity();
          state.member.setConfig(kindConfig);
          markStale();
        },
        setQualityPolicy(managed, targets) {
          if (released || disposed) return;
          if (
            managed === state.qualityManaged &&
            sameTargets(state.qualityTargets, targets)
          )
            return;
          state.qualityManaged = managed;
          appliedViewFraction = null;
          state.qualityTargets = targets;
          if (state.active) invalidateCapacity();
          updateTargets();
          markStale();
        },
        release() {
          if (released) return;
          released = true;
          if (!members.delete(state)) return;
          appliedViewFraction = null;
          if (state.active) invalidateCapacity();
          state.memoryMember?.release();
          state.memoryMember = null;
          state.member.dispose();
          updateTargets();
          markStale();
        },
      };
    },

    noteRenderedCameras(views) {
      if (disposed) return;
      governor.noteRenderedCameras(views, options.scheduleRender);
    },

    setQualityTargets(targets) {
      if (disposed || sameTargets(globalQualityTargets, targets)) return;
      globalQualityTargets = { ...targets };
      updateTargets();
      markStale();
    },

    beginInteraction() {
      if (disposed) return;
      motionReferences.push(governor.beginMotion("explicit"));
      for (const state of members) state.member.beginInteraction();
      markStale();
    },

    endInteraction() {
      if (disposed) return;
      motionReferences.pop()?.release();
      for (const state of members) state.member.endInteraction();
      markStale();
    },

    prepareFrame(frameSerial) {
      if (disposed) return;
      if (!Number.isSafeInteger(frameSerial) || frameSerial < 0) {
        throw new TypeError("frameSerial must be a non-negative safe integer");
      }
      if (frameSerial <= lastPreparedFrameSerial) return;
      lastPreparedFrameSerial = frameSerial;
      for (const state of members) {
        if (state.active) state.member.prepareFrame();
      }
      submissions.prepareFrame();
      capacityWorkSinceLastReport ||=
        submissions.stats().lastFrameAdmittedJobs > 0;
      const owed = owesMeasurement(observe());
      // One application per frame and no loop: whatever a member reports in
      // answer is read by the next check and applied at the next frame.
      applyAllocations();
      if (owed) options.scheduleRender();
    },

    recordHostFrame(metrics) {
      const unsampled = (): FrameVerdict => ({
        sampled: false,
        eligible: false,
        regime: governor.regime(),
      });
      if (disposed || !finiteNonNegative(metrics?.hostFrameMs)) {
        return unsampled();
      }
      // Fixed-only views have no adaptive fraction to train. Fixed members do
      // still contaminate samples whenever at least one adaptive member is
      // present, because their frame cost belongs to that same host frame.
      const verdict = hasAdaptiveMember()
        ? governor.recordHostFrame({
            ...metrics,
            capacitySampleEligible:
              (metrics.capacitySampleEligible ?? true) &&
              !capacityWorkSinceLastReport,
          })
        : unsampled();
      // Work that finished during this presentation still contaminated its
      // cost; the next interval starts clean.
      capacityWorkSinceLastReport = false;
      return verdict;
    },

    needsFrame: () =>
      !disposed &&
      (submissions.hasPending() ||
        (hasAdaptiveMember() && governor.needsFrame()) ||
        allocationOutdated()),

    stats() {
      return {
        viewQualityFraction: governor.qualityFraction(),
        targetOverrideMemberId: appliedTargetState?.id ?? null,
        governor: governor.stats(),
        submissions: submissions.stats(),
        stalledMembers: [...members]
          .filter((state) => state.stalled)
          .map((state) => state.id ?? "<unnamed>"),
        members: [...members].map((state) => ({
          id: state.id,
          active: state.active,
          qualityManaged: state.qualityManaged,
          governorInputs: state.inputs,
          allocation: state.allocation,
        })),
      };
    },

    dispose() {
      if (disposed) return;
      disposed = true;
      if (stallTimer !== null) clearTimeout(stallTimer);
      for (const reference of motionReferences) reference.release();
      motionReferences.length = 0;
      for (const state of members) {
        state.memoryMember?.release();
        state.member.dispose();
      }
      members.clear();
      governor.dispose();
      submissions.dispose();
    },
  };
};
