import type { CameraView, Mat16 } from "./camera";
import type { MemoryPool, MemoryPoolMember } from "./memoryPool";
import { allocateViewQuality } from "./viewBudget";
import {
  createSubmissionScheduler,
  type SubmissionScheduler,
  type SubmissionSchedulerOptions,
} from "./submissionScheduler";
import type {
  Allocation,
  DecodeWorkerPool,
  GovernorInputs,
  StreamedMember,
  StreamedMemberContext,
  TextureCapabilities,
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
  beginInteraction(): void;
  endInteraction(): void;
  /** Member preparation followed by the shared admission drain. */
  prepareFrame(): void;
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
};

const EMPTY_INPUTS: GovernorInputs = {
  projectedImportance: 0,
  qualityDemand: 0,
  workPending: false,
  physicalTileOperations: 0,
  physicalHierarchyOperations: 0,
  residentBytes: 0,
};

const usableNonNegative = (value: number): number =>
  Number.isFinite(value) && value >= 0 ? value : 0;

const usableFraction = (value: number): number =>
  Math.min(1, usableNonNegative(value));

const normalizeInputs = (inputs: GovernorInputs): GovernorInputs => ({
  projectedImportance: usableNonNegative(inputs.projectedImportance),
  qualityDemand: usableFraction(inputs.qualityDemand),
  workPending: !!inputs.workPending,
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
  let refreshing = false;
  let viewQualityFraction = 1;
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

  const refresh = (): void => {
    if (disposed || refreshing) return;
    refreshing = true;
    try {
      for (const state of members) {
        state.inputs = state.active
          ? normalizeInputs(state.member.governorInputs())
          : EMPTY_INPUTS;
      }
      // Insertion order is the conflict rule. A hidden managed member yields
      // to the first active one and regains precedence if it is shown again.
      const adaptive = adaptiveStates();
      // Only adaptive point members supply a target bag. A tiles member is
      // quality-managed too, but must not mask the first adaptive point's
      // stable registry-order override; a tiles-only view uses defaults.
      const targetState =
        adaptive.find((state) => state.qualityTargets !== undefined) ?? null;
      if (
        targetState !== appliedTargetState ||
        !sameTargets(targetState?.qualityTargets, appliedTargets)
      ) {
        governor.setOptions(governorOptions(targetState?.qualityTargets));
        appliedTargetState = targetState;
        appliedTargets = targetState?.qualityTargets
          ? { ...targetState.qualityTargets }
          : undefined;
      }
      // Fixed members do not consume normalized quality, but their frame cost
      // and incomplete work still contaminate the same view-wide sample.
      let tileOperations = 0;
      let hierarchyOperations = 0;
      let pending = submissions.hasPending();
      for (const state of members) {
        if (!state.active) continue;
        tileOperations += state.inputs.physicalTileOperations;
        hierarchyOperations += state.inputs.physicalHierarchyOperations;
        pending = pending || state.inputs.workPending;
      }
      governor.setWorkState({
        physicalTileOperations: tileOperations,
        physicalHierarchyOperations: hierarchyOperations,
        workPending: pending,
      });
    } finally {
      refreshing = false;
    }
    applyAllocations();
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
      return {
        setCamera(view) {
          if (released || disposed) return;
          state.member.setCamera(view);
          refresh();
        },
        setModelMatrix(matrix) {
          if (released || disposed) return;
          state.member.setModelMatrix(matrix);
          refresh();
        },
        setDevicePixelRatio(devicePixelRatio) {
          if (released || disposed) return;
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

    prepareFrame() {
      if (disposed) return;
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
