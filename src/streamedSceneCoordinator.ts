import {
  sameCameraView,
  sameMatrix,
  type CameraView,
  type Mat16,
} from "./camera";
import type { MemoryPool, MemoryPoolMember } from "./memoryPool";
import { allocateViewQuality } from "./viewBudget";
import {
  createSubmissionScheduler,
  type SubmissionScheduler,
  type SubmissionSchedulerOptions,
} from "./submissionScheduler";
import {
  CULLED,
  type Allocation,
  type DecodeWorkerPool,
  type GovernorInputs,
  type Importance,
  type StreamedMember,
  type StreamedMemberContext,
  type TextureCapabilities,
} from "./streamedMember";
import {
  createViewGovernor,
  type HostFrameMetrics,
  type MotionReference,
  type ViewGovernorOptions,
  type ViewGovernorStats,
} from "./viewGovernor";

const EMPTY_WORKERS: DecodeWorkerPool = {
  size: 0,
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
  readonly submissions?: SubmissionScheduler;
  readonly submissionLimits?: Omit<
    SubmissionSchedulerOptions,
    "scheduleRender"
  >;
  readonly textureCapabilities?: TextureCapabilities;
  readonly devicePixelRatio?: number;
  /** Lack-of-progress window before one member is isolated from view gating. */
  readonly stallWindowMs?: number;
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
  /** Member preparation followed by the shared admission drain. */
  prepareFrame(frameSerial: number): void;
  recordHostFrame(metrics: HostFrameMetrics): void;
  needsFrame(): boolean;
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
  Number.isFinite(value) && value >= 0 ? value : 0;

const usableFraction = (value: number): number =>
  Math.min(1, usableNonNegative(value));

const normalizeInputs = (inputs: GovernorInputs): GovernorInputs => ({
  // Importance is already a branded [0, 1] weight; this only defends against a
  // non-finite value reaching the allocator.
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

const copyCameraView = (view: CameraView): CameraView => ({
  ...view,
  position: [...view.position],
  viewProj: Array.from(view.viewProj) as Mat16,
});

export const createStreamedSceneCoordinator = (
  options: StreamedSceneCoordinatorOptions,
): StreamedSceneCoordinator => {
  if (!options.memory) {
    throw new Error(
      "createStreamedSceneCoordinator requires the page-wide MemoryPool",
    );
  }
  const memory = options.memory;
  const submissions =
    options.submissions ??
    createSubmissionScheduler({
      scheduleRender: options.scheduleRender,
      ...options.submissionLimits,
    });
  const members = new Set<MemberState>();
  let disposed = false;
  const now = options.now ?? Date.now;
  const stallWindowMs = Math.max(1, options.stallWindowMs ?? 4_000);
  let stallTimer: ReturnType<typeof setTimeout> | null = null;
  let lastPreparedFrameSerial = -1;
  let refreshing = false;
  let refreshPending = false;
  let viewQualityFraction = 1;
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

  const applyAllocations = (): void => {
    if (disposed) return;
    const regime =
      governor.stats().regime === "interaction" ? "moving" : "stationary";
    const contenders = [];
    for (const state of members) {
      if (isAdaptive(state))
        contenders.push({ key: state, inputs: state.inputs });
    }
    viewQualityFraction = governor.qualityFraction();
    const quality = allocateViewQuality(contenders, viewQualityFraction);
    for (const state of members) {
      const next: Allocation = {
        qualityFraction:
          state.active && state.qualityManaged
            ? (quality.get(state) ?? 0)
            : state.active
              ? 1
              : 0,
        memoryBudgetBytes: state.active
          ? (state.memoryMember?.budgetBytes() ?? 0)
          : 0,
        regime,
      };
      if (sameAllocation(next, state.allocation)) continue;
      state.allocation = next;
      state.member.applyAllocation(next);
    }
  };

  const isAdaptive = (state: MemberState): boolean =>
    state.active && state.qualityManaged;

  const adaptiveStates = (): MemberState[] => [...members].filter(isAdaptive);

  const hasAdaptiveMember = (): boolean => {
    for (const state of members) if (isAdaptive(state)) return true;
    return false;
  };

  /**
   * Passes one `refresh` will make before leaving the rest to the next one.
   *
   * Applying an allocation calls back into the member, which reports the work
   * that allocation created, which is another refresh — so a refresh has to
   * re-run until the allocations stop changing, and two members sharing one
   * quality budget can trade it back and forth without ever settling. The cap
   * bounds that. Nothing is lost by stopping: every frame refreshes again, and
   * the work state a dropped pass would have written is recomputed from the
   * members rather than accumulated.
   */
  const MAX_REFRESH_PASSES = 8;

  const refreshOnce = (): void => {
    for (const state of members) {
      state.inputs = state.active
        ? normalizeInputs(state.member.governorInputs())
        : EMPTY_INPUTS;
    }
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
      const remaining = stallWindowMs - (checkedAt - state.lastProgressAt);
      if (remaining <= 0) {
        state.stalled = true;
        if (!state.stallReported) {
          state.stallReported = true;
          state.member.onStall?.(
            new Error(
              `streamed member ${state.id ?? "<unnamed>"} made no progress for ${stallWindowMs} ms`,
            ),
          );
        }
      } else {
        nextStallDelay = Math.min(nextStallDelay, remaining);
      }
    }
    if (Number.isFinite(nextStallDelay)) {
      stallTimer = setTimeout(refresh, Math.max(1, Math.ceil(nextStallDelay)));
    }
    // Insertion order is the conflict rule. A hidden managed member yields
    // to the first active one and regains precedence if it is shown again.
    const adaptive = adaptiveStates();
    // Only adaptive point members supply a target bag. A tiles member is
    // quality-managed too, but must not mask the first adaptive point's
    // stable registry-order override; a tiles-only view uses defaults.
    const targetState = globalQualityTargets
      ? null
      : (adaptive.find((state) => state.qualityTargets !== undefined) ?? null);
    const nextTargets = globalQualityTargets ?? targetState?.qualityTargets;
    if (
      targetState !== appliedTargetState ||
      !sameTargets(nextTargets, appliedTargets)
    ) {
      governor.setOptions(governorOptions(nextTargets));
      appliedTargetState = targetState;
      appliedTargets = nextTargets ? { ...nextTargets } : undefined;
    }
    // Fixed members do not consume normalized quality, but their frame cost
    // and incomplete work still contaminate the same view-wide sample.
    let tileOperations = 0;
    let hierarchyOperations = 0;
    let pending = submissions.hasPending();
    for (const state of members) {
      if (!state.active) continue;
      if (!state.stalled) {
        tileOperations += state.inputs.physicalTileOperations;
        hierarchyOperations += state.inputs.physicalHierarchyOperations;
        pending = pending || state.inputs.work.operations > 0;
      }
    }
    governor.setWorkState({
      physicalTileOperations: tileOperations,
      physicalHierarchyOperations: hierarchyOperations,
      workPending: pending,
    });
    applyAllocations();
  };

  const refresh = (): void => {
    if (disposed) return;
    // A refresh raised from inside a refresh — an allocation's own work change
    // — is remembered rather than dropped, and rather than recursing into a
    // second pass on top of the first. Recursing is what this guard exists to
    // stop: the member callbacks are several frames deep already, and a scene
    // whose members trade quality would exhaust the stack rather than settle.
    if (refreshing) {
      refreshPending = true;
      return;
    }
    refreshing = true;
    try {
      let pass = 0;
      do {
        refreshPending = false;
        refreshOnce();
        pass += 1;
      } while (refreshPending && !disposed && pass < MAX_REFRESH_PASSES);
    } finally {
      refreshing = false;
    }
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
      onWorkChange: refresh,
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
      if (state.active) {
        state.memoryMember = memory.register(refresh);
      }
      member.setActive(state.active);
      // A member can be realized or rebuilt while the view is already inside
      // one or more nested interactions. Replay every held begin so its
      // eventual end calls remain balanced and format-local interaction state
      // never starts stationary in the middle of a gesture.
      for (let index = 0; index < motionReferences.length; index += 1) {
        member.beginInteraction();
      }
      refresh();
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
          refresh();
        },
        setModelMatrix(matrix) {
          if (released || disposed) return;
          if (modelMatrixSet && sameMatrix(lastModelMatrix, matrix)) return;
          modelMatrixSet = true;
          lastModelMatrix =
            matrix === null ? null : (Array.from(matrix) as Mat16);
          state.member.setModelMatrix(matrix);
          refresh();
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
          if (active) state.memoryMember = memory.register(refresh);
          else {
            state.memoryMember?.release();
            state.memoryMember = null;
          }
          state.member.setActive(active);
          refresh();
        },
        setConfig(kindConfig) {
          if (released || disposed) return;
          state.member.setConfig(kindConfig);
          refresh();
        },
        setQualityPolicy(managed, targets) {
          if (released || disposed) return;
          state.qualityManaged = managed;
          state.qualityTargets = targets;
          refresh();
        },
        release() {
          if (released) return;
          released = true;
          if (members.delete(state)) {
            state.memoryMember?.release();
            state.memoryMember = null;
            state.member.dispose();
            refresh();
          }
        },
      };
    },

    noteRenderedCameras(views) {
      if (disposed) return;
      governor.noteRenderedCameras(views, options.scheduleRender);
      refresh();
    },

    setQualityTargets(targets) {
      if (disposed || sameTargets(globalQualityTargets, targets)) return;
      globalQualityTargets = { ...targets };
      refresh();
    },

    beginInteraction() {
      if (disposed) return;
      motionReferences.push(governor.beginMotion("explicit"));
      for (const state of members) state.member.beginInteraction();
      refresh();
    },

    endInteraction() {
      if (disposed) return;
      motionReferences.pop()?.release();
      for (const state of members) state.member.endInteraction();
      refresh();
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
      refresh();
    },

    recordHostFrame(metrics) {
      if (disposed) return;
      refresh();
      // Fixed-only views have no adaptive fraction to train. Fixed members do
      // still contaminate samples whenever at least one adaptive member is
      // present, because their frame cost belongs to that same host frame.
      if (hasAdaptiveMember()) governor.recordHostFrame(metrics);
      applyAllocations();
    },

    needsFrame: () =>
      !disposed &&
      (submissions.hasPending() ||
        (hasAdaptiveMember() && governor.needsFrame())),

    stats() {
      refresh();
      return {
        viewQualityFraction,
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
