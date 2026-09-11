import { ADAPTIVE_QUALITY_DEFAULTS } from "./adaptiveBudget";
import {
  createLodController,
  type LodController,
  type PointPresentation,
} from "./controller";
import { createRendererAdapter, type RendererAdapter } from "./rendererAdapter";
import {
  occlusionFromPick,
  importanceFromRootSseCssPx,
  type Allocation,
  type GovernorInputs,
  type MemberPickResult,
  type OcclusionResult,
  type StreamedMember,
  type StreamedMemberContext,
} from "./streamedMember";
import type { TileSource } from "./tileSource";

export const DEFAULT_FIXED_POINT_BUDGET = 2_000_000;
export const DEFAULT_MIN_POINT_BUDGET = ADAPTIVE_QUALITY_DEFAULTS.minBudget;

export type PointCloudAdaptiveOptions = {
  readonly minBudget?: number;
  readonly maxBudget?: number;
  readonly interactionTargetMs?: number;
  readonly stationaryTargetMs?: number;
};

export type PointCloudMemberConfig = {
  readonly source: TileSource;
  readonly pointCount?: number;
  readonly presentation: PointPresentation;
  readonly adaptive: boolean;
  readonly adaptiveOptions?: PointCloudAdaptiveOptions;
  readonly pointBudget?: number;
  readonly refinementCutoffPx?: number;
  readonly fetchConcurrency?: number;
  readonly hierarchyConcurrency?: number;
  readonly cacheBytes?: number;
  readonly selectionDelayMs?: number;
  readonly onError?: (error: unknown) => void;
};

export type PointCloudMemberStats = {
  readonly kind: "pointCloud";
  readonly adaptive: boolean;
  readonly minimumPointBudget: number;
  /** Explicit/default point ceiling, before camera demand and page memory. */
  readonly configuredPointCeiling: number | null;
  readonly fullPointCeiling: number;
  readonly allocation: Allocation;
  readonly controller: ReturnType<LodController["stats"]>;
  readonly renderer: ReturnType<RendererAdapter["stats"]>;
};

const pointCountOf = (config: PointCloudMemberConfig): number => {
  const stated = config.pointCount ?? config.source.metadata().pointCount;
  return Number.isFinite(stated) && stated >= 0 ? Math.floor(stated) : 0;
};

const adaptiveMinimum = (config: PointCloudMemberConfig): number =>
  Math.max(
    1,
    Math.floor(config.adaptiveOptions?.minBudget ?? DEFAULT_MIN_POINT_BUDGET),
  );

const configuredCeiling = (config: PointCloudMemberConfig): number => {
  const ceiling = config.adaptive
    ? config.adaptiveOptions?.maxBudget
    : (config.pointBudget ?? DEFAULT_FIXED_POINT_BUDGET);
  return ceiling === undefined || !Number.isFinite(ceiling)
    ? Number.POSITIVE_INFINITY
    : Math.max(0, Math.floor(ceiling));
};

export const createPointCloudMember = (
  context: StreamedMemberContext,
  initialConfig: PointCloudMemberConfig,
): StreamedMember => {
  let config = initialConfig;
  let active = true;
  let disposed = false;
  let stationaryQualityFraction = 1;
  let allocation: Allocation = {
    qualityFraction: 1,
    memoryBudgetBytes: 0,
    regime: "stationary",
  };

  const adapter = createRendererAdapter({
    renderer: context.renderer as {
      addActor(actor: unknown): void;
      removeActor(actor: unknown): void;
    },
    scheduleRender: context.scheduleRender,
    devicePixelRatio: context.devicePixelRatio,
    visible: active,
  });
  const controller = createLodController({
    source: config.source,
    onTiles: adapter.applyBatch,
    onDrawPlan: adapter.applyDrawPlan,
    scheduleRender: context.scheduleRender,
    onWorkChange: () => context.onWorkChange?.(),
    onPointDiameterCssPx: adapter.setPointDiameterCssPx,
    active,
    pointBudget: config.adaptive
      ? adaptiveMinimum(config)
      : (config.pointBudget ?? DEFAULT_FIXED_POINT_BUDGET),
    memoryBudgetBytes: 0,
    presentation: config.presentation,
    ...(config.refinementCutoffPx === undefined
      ? {}
      : { refinementCutoffPx: config.refinementCutoffPx }),
    ...(config.fetchConcurrency === undefined
      ? {}
      : { fetchConcurrency: config.fetchConcurrency }),
    ...(config.hierarchyConcurrency === undefined
      ? {}
      : { hierarchyConcurrency: config.hierarchyConcurrency }),
    ...(config.cacheBytes === undefined
      ? {}
      : { cacheBytes: config.cacheBytes }),
    ...(config.selectionDelayMs === undefined
      ? {}
      : { selectionDelayMs: config.selectionDelayMs }),
    ...(config.onError === undefined ? {} : { onError: config.onError }),
  });

  let cachedWorkRevision = -1;
  let cachedWorkPending = false;

  const fullCeiling = (): number => {
    const memoryCeiling = controller.governorInputs().memoryCeilingPoints;
    return Math.max(
      0,
      Math.min(pointCountOf(config), configuredCeiling(config), memoryCeiling),
    );
  };

  const applyPointAllocation = (next: Allocation): void => {
    allocation = next;
    controller.setMemoryBudgetBytes(next.memoryBudgetBytes);
    adapter.setResourceCeilingBytes(next.memoryBudgetBytes);
    if (!config.adaptive) {
      controller.setPointBudget(
        config.pointBudget ?? DEFAULT_FIXED_POINT_BUDGET,
      );
      controller.setDensityFraction(1);
      return;
    }

    const controllerInputs = controller.governorInputs();
    const full = fullCeiling();
    const cameraDemand = controllerInputs.demandPoints;
    const budgetAt = (qualityFraction: number): number => {
      const requested = Math.floor(full * qualityFraction);
      const demandCapped =
        cameraDemand > 0 ? Math.min(requested, cameraDemand) : requested;
      return Math.min(full, Math.max(adaptiveMinimum(config), demandCapped));
    };
    if (next.regime === "stationary") {
      stationaryQualityFraction = next.qualityFraction;
      const points = budgetAt(next.qualityFraction);
      controller.setPointBudget(points);
      controller.setDensityFraction(1);
      return;
    }
    const drawPoints = budgetAt(next.qualityFraction);
    const selectionPoints = budgetAt(
      Math.max(stationaryQualityFraction, next.qualityFraction),
    );
    controller.setPointBudget(selectionPoints);
    controller.setDensityFraction(
      selectionPoints > 0 ? drawPoints / selectionPoints : 0,
    );
  };

  return {
    setCamera: controller.setCamera,

    setModelMatrix(matrix) {
      if (disposed) return;
      controller.setModelMatrix(matrix);
      adapter.setBaseMatrix(matrix);
    },

    setDevicePixelRatio(devicePixelRatio) {
      if (!disposed) adapter.setDevicePixelRatio(devicePixelRatio);
    },

    setActive(nextActive) {
      if (disposed || active === nextActive) return;
      active = nextActive;
      controller.setActive(nextActive);
      adapter.setVisible(nextActive);
    },

    setConfig(kindConfig) {
      if (disposed) return;
      const next = kindConfig as PointCloudMemberConfig;
      if (next.source !== config.source) controller.setSource(next.source);
      config = next;
      controller.setPresentation(config.presentation);
      controller.setRefinementCutoffPx(config.refinementCutoffPx ?? 1);
      applyPointAllocation(allocation);
    },

    beginInteraction: controller.beginInteraction,
    endInteraction: controller.endInteraction,
    prepareFrame() {},

    governorInputs(): GovernorInputs {
      const narrow = controller.governorInputs();
      if (narrow.workRevision !== cachedWorkRevision) {
        cachedWorkPending = controller.stats().workPending;
        cachedWorkRevision = narrow.workRevision;
      }
      const full = fullCeiling();
      return {
        // The controller reports root SSE in CSS pixels; the governor compares
        // members, so it is normalized against this cloud's own cutoff here.
        projectedImportance: importanceFromRootSseCssPx(
          narrow.projectedImportance,
          config.refinementCutoffPx ?? 1,
        ),
        qualityDemand:
          narrow.projectedImportance > 0 && full > 0
            ? Math.min(1, narrow.demandPoints / full)
            : 0,
        work: {
          // Backoff remains one logical obligation even when physical counts
          // are temporarily zero.
          operations: cachedWorkPending
            ? Math.max(
                1,
                narrow.physicalTileOperations +
                  narrow.physicalHierarchyOperations,
              )
            : 0,
          progressSerial: narrow.workRevision,
        },
        physicalTileOperations: narrow.physicalTileOperations,
        physicalHierarchyOperations: narrow.physicalHierarchyOperations,
        residentBytes: adapter.workState().gpuResidentBytes,
      };
    },

    applyAllocation(next) {
      if (!disposed) applyPointAllocation(next);
    },

    pick(view, cssX, cssY): MemberPickResult | null {
      return controller.pickPoint(view, cssX, cssY);
    },

    occlusionDepth(view, cssX, cssY): OcclusionResult | null {
      return occlusionFromPick(controller.pickPoint(view, cssX, cssY));
    },

    stats(): PointCloudMemberStats {
      return {
        kind: "pointCloud",
        adaptive: config.adaptive,
        minimumPointBudget: adaptiveMinimum(config),
        configuredPointCeiling: Number.isFinite(configuredCeiling(config))
          ? configuredCeiling(config)
          : null,
        fullPointCeiling: fullCeiling(),
        allocation,
        controller: controller.stats(),
        renderer: adapter.stats(),
      };
    },

    dispose() {
      if (disposed) return;
      disposed = true;
      controller.dispose();
      adapter.dispose();
    },
  };
};
