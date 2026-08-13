/** StreamedMember implementation for the constrained explicit 3D Tiles profile. */

import {
  sameCameraView,
  sameMatrix,
  type CameraView,
  type Mat16,
} from "../camera";
import { integerAtLeast } from "../numeric";
import {
  CULLED,
  importanceFromRootSseCssPx,
  type Allocation,
  type GovernorInputs,
  type StreamedMember,
  type StreamedMemberContext,
} from "../streamedMember";
import {
  createContentQueue,
  type ContentQueue,
  type ContentQueueEntrySnapshot,
  type ContentQueueSnapshot,
} from "./contentQueue";
import type { DecodedTileContent, DecodeWasmUrls } from "./decode";
import { createMeshAdapter, type MeshAdapter } from "./meshAdapter";
import { createMeshPickSet } from "./meshPicking";
import {
  DEFAULT_MAXIMUM_SCREEN_SPACE_ERROR_PX,
  DEFAULT_TILES3D_CACHE_BYTES,
  DEFAULT_TILES3D_MAX_CONCURRENCY,
  DEFAULT_TILES3D_MIN_CONCURRENCY,
  DEFAULT_VERTICAL_EXAGGERATION,
  DEFAULT_VERTICAL_PIVOT_Z,
  type Tiles3dMemberConfig,
  type Tiles3dMemberStats,
} from "./memberTypes";
import { createVerticalExaggerationTransform } from "./rtc";
import {
  loadTileset,
  multiplyTilesetMatrices,
  type TilesetSource,
} from "./tilesetSource";
import { traverseTileset, type TilesetTraversalResult } from "./traversal";

const errorMessage = (error: unknown): string => {
  if (!(error instanceof Error)) return String(error);
  const messages = [error.message];
  let cause = error.cause;
  while (cause instanceof Error) {
    if (!messages.includes(cause.message)) messages.push(cause.message);
    cause = cause.cause;
  }
  return messages.join(": ");
};

/** Whether two wasm URL sets decode to the same bytes.
 *
 * A host resolves these URLs against the page on every config push, so the
 * object is fresh each time. Only the URLs it names are decode inputs.
 */
const sameDecodeWasm = (
  left: DecodeWasmUrls | undefined,
  right: DecodeWasmUrls | undefined,
): boolean =>
  left === right ||
  (left?.draco?.wrapperUrl === right?.draco?.wrapperUrl &&
    left?.draco?.wasmUrl === right?.draco?.wasmUrl &&
    left?.basis?.encoderUrl === right?.basis?.encoderUrl &&
    left?.basis?.wasmUrl === right?.basis?.wasmUrl);

const finiteMatrix = (
  matrix: readonly number[],
  label: string,
): readonly number[] => {
  if (matrix.length !== 16 || matrix.some((value) => !Number.isFinite(value))) {
    throw new TypeError(`${label} must contain 16 finite numbers`);
  }
  if (
    Math.abs(matrix[3]!) > 1e-12 ||
    Math.abs(matrix[7]!) > 1e-12 ||
    Math.abs(matrix[11]!) > 1e-12 ||
    Math.abs(matrix[15]! - 1) > 1e-12
  ) {
    throw new TypeError(`${label} must be an affine column-major matrix`);
  }
  const determinant =
    matrix[0]! * (matrix[5]! * matrix[10]! - matrix[9]! * matrix[6]!) -
    matrix[4]! * (matrix[1]! * matrix[10]! - matrix[9]! * matrix[2]!) +
    matrix[8]! * (matrix[1]! * matrix[6]! - matrix[5]! * matrix[2]!);
  if (!Number.isFinite(determinant) || Math.abs(determinant) <= 1e-12) {
    throw new TypeError(`${label} must be invertible`);
  }
  return [...matrix];
};

const validateConfig = (config: Tiles3dMemberConfig): Tiles3dMemberConfig => {
  if (typeof config.endpoint !== "string" || config.endpoint.length === 0)
    throw new TypeError("tiles endpoint must be non-empty");
  if (typeof config.revision !== "string" || config.revision.length === 0)
    throw new TypeError("tiles revision must be non-empty");
  finiteMatrix(config.tilesetToScene, "tilesetToScene");
  const maximum =
    config.maximumScreenSpaceErrorPx ?? DEFAULT_MAXIMUM_SCREEN_SPACE_ERROR_PX;
  if (!Number.isFinite(maximum) || maximum <= 0)
    throw new RangeError("maximumScreenSpaceErrorPx must be finite and > 0");
  const minimumConcurrency = integerAtLeast(
    "minConcurrency",
    config.minConcurrency ?? DEFAULT_TILES3D_MIN_CONCURRENCY,
    1,
  );
  const maximumConcurrency = integerAtLeast(
    "maxConcurrency",
    config.maxConcurrency ?? DEFAULT_TILES3D_MAX_CONCURRENCY,
    minimumConcurrency,
  );
  const cacheBytes = config.cacheBytes ?? DEFAULT_TILES3D_CACHE_BYTES;
  if (!Number.isFinite(cacheBytes) || cacheBytes < 0)
    throw new RangeError("cacheBytes must be finite and >= 0");
  const verticalExaggeration =
    config.verticalExaggeration === undefined
      ? DEFAULT_VERTICAL_EXAGGERATION
      : config.verticalExaggeration;
  if (!Number.isFinite(verticalExaggeration) || verticalExaggeration <= 0) {
    throw new RangeError("verticalExaggeration must be finite and > 0");
  }
  const verticalPivotZ =
    config.verticalPivotZ === undefined
      ? DEFAULT_VERTICAL_PIVOT_Z
      : config.verticalPivotZ;
  if (!Number.isFinite(verticalPivotZ)) {
    throw new RangeError("verticalPivotZ must be finite");
  }
  return {
    ...config,
    minConcurrency: minimumConcurrency,
    maxConcurrency: maximumConcurrency,
    cacheBytes,
    verticalExaggeration,
    verticalPivotZ,
  };
};

const decodedBytes = (content: DecodedTileContent): number =>
  content.byteEstimate.geometry + content.byteEstimate.textures;

const workerUrl = (value: string): string =>
  new URL(value, globalThis.location?.href ?? "http://localhost/").toString();

const safeCallback = (
  callback: ((error: unknown) => void) | undefined,
  error: unknown,
): void => {
  try {
    callback?.(error);
  } catch {
    // Member state and retry accounting must survive observers.
  }
};

const contentRequests = (
  tileset: TilesetSource,
  ids: readonly string[],
): { readonly id: string; readonly url: string }[] =>
  ids.flatMap((id) => {
    const url = tileset.tileById.get(id)?.contentUrl;
    return url === undefined ? [] : [{ id, url }];
  });

export const createTiles3dMember = (
  context: StreamedMemberContext,
  initialConfig: Tiles3dMemberConfig,
): StreamedMember => {
  let config = validateConfig(initialConfig);
  let active = true;
  let disposed = false;
  let sourceState: Tiles3dMemberStats["sourceState"] = "idle";
  let source: TilesetSource | null = null;
  let camera: CameraView | null = null;
  let modelMatrix: readonly number[] | null = null;
  let devicePixelRatio = context.devicePixelRatio;
  let allocation: Allocation = {
    qualityFraction: 1,
    memoryBudgetBytes: 0,
    regime: "stationary",
  };
  let interactionDepth = 0;
  let memoryConstrained = false;
  let irreducibleBudget = false;
  let constrainedDesiredSignature: string | null = null;
  let configGeneration = 1;
  let loadGeneration = 0;
  let loadController: AbortController | null = null;
  let queue: ContentQueue<DecodedTileContent> | null = null;
  let queueSnapshot: ContentQueueSnapshot | null = null;
  // Traversal asks readiness for every tile it visits, so the snapshot is
  // indexed on arrival instead of scanned per tile.
  let queueEntryById = new Map<string, ContentQueueEntrySnapshot>();
  let traversal: TilesetTraversalResult | null = null;
  let errorCount = 0;
  let lastError: string | null = null;
  let workProgressSerial = 0;
  let selectionPasses = 0;
  const requested = new Set<string>();
  const decoded = new Set<string>();
  const admissionBlocked = new Set<string>();
  const admissionFailed = new Set<string>();
  let refreshing = false;
  let refreshAgain = false;
  let drawSetDirty = true;

  const report = (error: unknown): void => {
    errorCount += 1;
    lastError = errorMessage(error);
    safeCallback(config.onError, error);
    context.onWorkChange?.();
  };

  const adapter: MeshAdapter = createMeshAdapter({
    renderer: context.renderer as {
      addActor(actor: unknown): void;
      removeActor(actor: unknown): void;
    },
    scheduleRender: context.scheduleRender,
    submissions: context.submissions,
    visible: active,
    onError: report,
  });
  const pickSet = createMeshPickSet();

  const maximumSse = (): number =>
    config.maximumScreenSpaceErrorPx ?? DEFAULT_MAXIMUM_SCREEN_SPACE_ERROR_PX;

  const verticalExaggeration = (): readonly number[] =>
    createVerticalExaggerationTransform(
      config.verticalExaggeration ?? DEFAULT_VERTICAL_EXAGGERATION,
      config.verticalPivotZ ?? DEFAULT_VERTICAL_PIVOT_Z,
    );

  const exaggeratedEcefToScene = (): readonly number[] =>
    multiplyTilesetMatrices(verticalExaggeration(), config.tilesetToScene);

  const traversalModelMatrix = (): readonly number[] =>
    modelMatrix
      ? multiplyTilesetMatrices(modelMatrix, exaggeratedEcefToScene())
      : exaggeratedEcefToScene();

  /**
   * Anchor placement composed after each tile's scene-local origin. Decoded
   * vertices stay in unexaggerated scene coordinates, so exaggeration is a matrix the
   * renderer applies rather than a property of the downloaded bytes.
   */
  const placementMatrix = (): readonly number[] =>
    modelMatrix
      ? multiplyTilesetMatrices(modelMatrix, verticalExaggeration())
      : verticalExaggeration();

  /**
   * The same placement without exaggeration — the frame a picked point must be
   * reported in, since anything the app stores (control points, registration
   * pairs) is canonical scene coordinates and must not move when the user changes a
   * display-only vertical scale.
   */
  const scenePlacementMatrix = (): readonly number[] | null => modelMatrix;

  const applyPlacement = (): void => {
    const placement = Array.from(placementMatrix()) as Mat16;
    adapter.setBaseMatrix(placement);
    const scene = scenePlacementMatrix();
    pickSet.setPlacement({
      drawn: placement,
      scene: scene === null ? null : (Array.from(scene) as Mat16),
    });
  };

  const traversalMaximumSse = (): number =>
    memoryConstrained ? Number.MAX_VALUE : maximumSse();

  const traversalQualityFraction = (): number =>
    memoryConstrained ? 1 : allocation.qualityFraction;

  const concurrency = (): number => {
    const minimum = config.minConcurrency ?? DEFAULT_TILES3D_MIN_CONCURRENCY;
    const maximum = config.maxConcurrency ?? DEFAULT_TILES3D_MAX_CONCURRENCY;
    return Math.round(
      minimum + (maximum - minimum) * allocation.qualityFraction,
    );
  };

  const setQueueSnapshot = (snapshot: ContentQueueSnapshot | null): void => {
    queueSnapshot = snapshot;
    queueEntryById = new Map(
      snapshot ? snapshot.entries.map((entry) => [entry.id, entry]) : [],
    );
  };

  /** Submitted descendants a newly admitted tile replaces under REPLACE. */
  const replacedDescendants = (id: string): string[] =>
    adapter
      .submittedTiles()
      .map((tile) => tile.id)
      .filter((submittedId) => submittedId.startsWith(`${id}/`));

  const submittedAncestorReplacement = (id: string): string | null => {
    if (!traversal) return null;
    const ancestors = adapter
      .submittedTiles()
      .map((tile) => tile.id)
      .filter((candidate) => id.startsWith(`${candidate}/`))
      .sort((left, right) => right.length - left.length);
    for (const ancestor of ancestors) {
      const desired = traversal.desiredTileIds.filter((candidate) =>
        candidate.startsWith(`${ancestor}/`),
      );
      if (
        desired.length > 0 &&
        desired.every(
          (candidate) =>
            candidate === id || adapter.tileState(candidate) === "submitted",
        )
      ) {
        return ancestor;
      }
    }
    return null;
  };

  const replacementsForAdmission = (id: string): string[] => {
    const replacements = replacedDescendants(id);
    const ancestor = submittedAncestorReplacement(id);
    return ancestor === null ? replacements : [...replacements, ancestor];
  };

  const waitingForSiblingBeforeReplacement = (id: string): boolean => {
    if (!traversal) return false;
    return adapter.submittedTiles().some((tile) => {
      if (!id.startsWith(`${tile.id}/`)) return false;
      const siblings = traversal!.desiredTileIds.filter(
        (candidate) => candidate !== id && candidate.startsWith(`${tile.id}/`),
      );
      return siblings.some((candidate) => {
        const state = adapter.tileState(candidate);
        return (
          state === "queued" ||
          decoded.has(candidate) ||
          (requested.has(candidate) && readiness(candidate) !== "failed")
        );
      });
    });
  };

  const coveredSoonByDesiredDescendants = (id: string): boolean => {
    if (!traversal) return false;
    const descendants = traversal.desiredTileIds.filter((candidate) =>
      candidate.startsWith(`${id}/`),
    );
    return (
      descendants.length > 0 &&
      descendants.every((candidate) => {
        const state = adapter.tileState(candidate);
        return (
          state === "queued" || state === "submitted" || decoded.has(candidate)
        );
      })
    );
  };

  const trySubmitDesiredGroup = (id: string) => {
    if (!traversal || !queue) return null;
    const ancestor = adapter
      .submittedTiles()
      .map((tile) => tile.id)
      .filter((candidate) => id.startsWith(`${candidate}/`))
      .sort((left, right) => right.length - left.length)[0];
    if (!ancestor) return null;
    const desired = traversal.desiredTileIds.filter((candidate) =>
      candidate.startsWith(`${ancestor}/`),
    );
    if (
      desired.length < 2 ||
      desired.some(
        (candidate) =>
          adapter.tileState(candidate) !== "absent" || !decoded.has(candidate),
      )
    ) {
      return null;
    }
    const entries = desired.flatMap((candidate) => {
      const content = queue!.get(candidate);
      return content
        ? [
            {
              id: candidate,
              content,
              onSubmitted: () => onAdmitted(candidate),
            },
          ]
        : [];
    });
    if (entries.length !== desired.length) return null;
    const outcome = adapter.submitTileGroup(entries, [ancestor]);
    if (outcome === "queued") {
      for (const candidate of desired) admissionBlocked.delete(candidate);
    }
    return outcome;
  };

  const unconstrainedDesiredSignature = (): string | null => {
    if (!source || !camera) return null;
    selectionPasses += 1;
    return traverseTileset({
      root: source.root,
      camera,
      maximumScreenSpaceErrorPx: maximumSse(),
      qualityFraction: allocation.qualityFraction,
      modelMatrix: traversalModelMatrix(),
      readiness,
    }).desiredTileIds.join("\n");
  };

  const retryIfDesiredSelectionChanged = (): void => {
    if (
      memoryConstrained &&
      constrainedDesiredSignature !== unconstrainedDesiredSignature()
    ) {
      memoryConstrained = false;
      irreducibleBudget = false;
      constrainedDesiredSignature = null;
    }
  };

  const readiness = (id: string) => {
    const state = adapter.tileState(id);
    if (state === "submitted") return "submitted" as const;
    if (state === "failed" || admissionFailed.has(id)) return "failed" as const;
    // A root that will not fit the member's whole byte allowance is terminal
    // for this budget, not still arriving. Reporting it as outstanding kept
    // `workPending` true forever, and the view governor stops sampling
    // capacity for EVERY member while any of them claims pending work — so one
    // over-budget tileset froze adaptive quality view-wide with nothing drawn.
    // A larger allowance clears the latch and reopens the tile.
    if (irreducibleBudget && source && id === source.root.id)
      return "failed" as const;
    if (decoded.has(id)) return "decoded" as const;
    const entry = queueEntryById.get(id);
    if (entry?.status === "failed") return "failed" as const;
    if (entry) return "loading" as const;
    return "unloaded" as const;
  };

  /**
   * Give up on fitting this tileset in the current byte allowance.
   *
   * Surfaced through `stats().irreducibleBudget` rather than `onError`: the
   * member recovers by itself as soon as it is given more memory, so this is a
   * state worth seeing in diagnostics, not a failure worth counting. Without
   * it the only evidence is an absence of geometry.
   */
  const latchIrreducibleBudget = (): void => {
    if (irreducibleBudget) return;
    irreducibleBudget = true;
    adapter.clearTiles();
    pickSet.replaceDrawn([]);
  };

  const updateDrawSet = (): void => {
    if (!drawSetDirty) return;
    drawSetDirty = false;
    if (!source || !camera || !active || disposed) {
      adapter.setDrawnTiles([]);
      pickSet.replaceDrawn([]);
      return;
    }
    traversal = traverseTileset({
      root: source.root,
      camera,
      maximumScreenSpaceErrorPx: traversalMaximumSse(),
      qualityFraction: traversalQualityFraction(),
      modelMatrix: traversalModelMatrix(),
      readiness,
    });
    selectionPasses += 1;
    adapter.setDrawnTiles(traversal.drawnTileIds);
    pickSet.replaceDrawn(adapter.submittedTiles());
  };

  const refreshSelection = (): void => {
    if (disposed) return;
    drawSetDirty = true;
    if (refreshing) {
      refreshAgain = true;
      return;
    }
    refreshing = true;
    try {
      if (!active || !source || !camera || !queue) {
        queue?.setSelection([]);
        for (const id of requested) adapter.cancelTile(id);
        requested.clear();
        updateDrawSet();
        return;
      }
      selectionPasses += 1;
      const next = traverseTileset({
        root: source.root,
        camera,
        maximumScreenSpaceErrorPx: traversalMaximumSse(),
        qualityFraction: traversalQualityFraction(),
        modelMatrix: traversalModelMatrix(),
        readiness,
      });
      traversal = next;
      const nextContentRequests = contentRequests(
        source,
        next.requestedTileIds,
      );
      const nextRequested = new Set(
        nextContentRequests.map((request) => request.id),
      );
      for (const id of requested) {
        if (nextRequested.has(id)) continue;
        adapter.cancelTile(id);
        if (!next.drawnTileIds.includes(id)) adapter.retireTile(id);
        decoded.delete(id);
        admissionBlocked.delete(id);
        admissionFailed.delete(id);
      }
      requested.clear();
      for (const id of nextRequested) requested.add(id);
      queue.setSelection(nextContentRequests);
      for (const id of nextRequested) {
        if (!admissionBlocked.has(id) || adapter.tileState(id) !== "absent")
          continue;
        const content = queue.get(id);
        if (content && !irreducibleBudget) {
          const replacements = replacementsForAdmission(id);
          const outcome = adapter.submitTile(
            id,
            content,
            () => onAdmitted(id),
            replacements,
          );
          if (outcome === "queued") admissionBlocked.delete(id);
          else if (outcome === "failed") {
            admissionBlocked.delete(id);
            admissionFailed.add(id);
          } else if (id === source.root.id && memoryConstrained) {
            latchIrreducibleBudget();
            refreshAgain = true;
          }
        }
      }
      updateDrawSet();
    } finally {
      refreshing = false;
      context.onWorkChange?.();
      if (refreshAgain) {
        refreshAgain = false;
        refreshSelection();
      }
    }
  };

  const onAdmitted = (id: string): void => {
    if (disposed || !active || !requested.has(id)) {
      adapter.retireTile(id);
      return;
    }
    admissionBlocked.delete(id);
    // Admission changes REPLACE coverage. Re-run selection so an ancestor is
    // removed from both the fetch obligation and renderer only after every
    // required descendant has crossed the shared admission boundary.
    refreshSelection();
    context.scheduleRender();
    context.onWorkChange?.();
  };

  const createQueue = (tileset: TilesetSource): void => {
    queue?.dispose();
    queue = createContentQueue({
      revision: config.revision,
      configGeneration,
      maxConcurrency: concurrency(),
      maxDecodedBytes: config.cacheBytes ?? DEFAULT_TILES3D_CACHE_BYTES,
      ...(config.fetchContent ? { fetch: config.fetchContent } : {}),
      ...(config.maxAttempts === undefined
        ? {}
        : { maxAttempts: config.maxAttempts }),
      ...(config.retryBackoffMs === undefined
        ? {}
        : { retryBackoffMs: config.retryBackoffMs }),
      decode: async (bytes, request, decodeContext) => {
        const tile = tileset.tileById.get(request.id);
        if (!tile) throw new Error(`unknown 3D tile ${request.id}`);
        const job = context.workers.decode({
          content: bytes,
          contentUrl: workerUrl(request.url),
          dependencyRootUrl: workerUrl(`${config.endpoint}/`),
          revision: decodeContext.revision,
          accumulatedTransform: [...tile.worldTransform] as any,
          tilesetToScene: [...config.tilesetToScene] as any,
          textureCapabilities: context.textureCapabilities,
          ...(config.wasm ? { wasm: config.wasm } : {}),
        });
        const abort = () => job.cancel();
        decodeContext.signal.addEventListener("abort", abort, { once: true });
        try {
          return await job.promise;
        } finally {
          decodeContext.signal.removeEventListener("abort", abort);
        }
      },
      decodedByteLength: decodedBytes,
      onContent: (request, content) => {
        workProgressSerial += 1;
        if (disposed || !active || !requested.has(request.id)) return;
        decoded.add(request.id);
        const groupOutcome = trySubmitDesiredGroup(request.id);
        if (groupOutcome === "queued") {
          updateDrawSet();
          return;
        }
        if (groupOutcome === "budget-blocked") {
          memoryConstrained = true;
          constrainedDesiredSignature = unconstrainedDesiredSignature();
          irreducibleBudget = false;
          refreshSelection();
          return;
        }
        const replacements = replacementsForAdmission(request.id);
        const outcome = adapter.submitTile(
          request.id,
          content,
          () => onAdmitted(request.id),
          replacements,
        );
        if (outcome === "budget-blocked") {
          admissionBlocked.add(request.id);
          if (coveredSoonByDesiredDescendants(request.id)) {
            updateDrawSet();
            return;
          }
          if (waitingForSiblingBeforeReplacement(request.id)) {
            updateDrawSet();
            return;
          }
          if (!memoryConstrained) {
            memoryConstrained = true;
            constrainedDesiredSignature = unconstrainedDesiredSignature();
            irreducibleBudget = false;
            refreshSelection();
          } else if (request.id === tileset.root.id) {
            latchIrreducibleBudget();
            refreshSelection();
          }
        } else if (outcome === "failed") {
          admissionFailed.add(request.id);
        }
        updateDrawSet();
      },
      onEvict: (request) => {
        workProgressSerial += 1;
        decoded.delete(request.id);
      },
      onError: (_request, error) => {
        workProgressSerial += 1;
        report(error);
      },
      onStateChange: (snapshot) => {
        workProgressSerial += 1;
        setQueueSnapshot(snapshot);
        context.onWorkChange?.();
      },
    });
  };

  const beginLoad = (): void => {
    if (disposed || !active) return;
    const generation = ++loadGeneration;
    loadController?.abort();
    const controller = new AbortController();
    loadController = controller;
    sourceState = "loading";
    source = null;
    traversal = null;
    setQueueSnapshot(null);
    context.onWorkChange?.();
    void loadTileset({
      endpoint: config.endpoint,
      ...(config.fetchTileset ? { fetch: config.fetchTileset } : {}),
      signal: controller.signal,
    }).then(
      (loaded) => {
        if (
          disposed ||
          generation !== loadGeneration ||
          controller.signal.aborted
        )
          return;
        source = loaded;
        workProgressSerial += 1;
        sourceState = "ready";
        createQueue(loaded);
        refreshSelection();
        context.scheduleRender();
      },
      (error) => {
        if (
          disposed ||
          generation !== loadGeneration ||
          controller.signal.aborted
        )
          return;
        sourceState = "failed";
        workProgressSerial += 1;
        report(error);
      },
    );
  };

  applyPlacement();
  beginLoad();

  return {
    setCamera(view) {
      if (disposed) return;
      // Hosts restate the camera unconditionally — every pre-paint pass and
      // twice per pick — so acting on an unchanged view costs a full tileset
      // traversal for nothing, and does it inside the span the host times.
      if (camera !== null && sameCameraView(camera, view)) return;
      camera = view;
      retryIfDesiredSelectionChanged();
      refreshSelection();
    },

    setModelMatrix(matrix: Mat16 | null) {
      if (disposed) return;
      if (sameMatrix(modelMatrix, matrix ?? null)) return;
      try {
        modelMatrix =
          matrix === null
            ? null
            : finiteMatrix(Array.from(matrix), "model matrix");
      } catch (error) {
        report(error);
        return;
      }
      applyPlacement();
      drawSetDirty = true;
      retryIfDesiredSelectionChanged();
      refreshSelection();
    },

    setDevicePixelRatio(next) {
      if (!disposed && Number.isFinite(next) && next > 0)
        devicePixelRatio = next;
    },

    setActive(next) {
      if (disposed || active === next) return;
      active = next;
      adapter.setVisible(next);
      drawSetDirty = true;
      if (!next) {
        loadController?.abort();
        loadGeneration += 1;
        queue?.setSelection([]);
        for (const id of requested) adapter.cancelTile(id);
        adapter.clearTiles();
        pickSet.replaceDrawn([]);
        requested.clear();
        decoded.clear();
        admissionBlocked.clear();
        admissionFailed.clear();
        memoryConstrained = false;
        irreducibleBudget = false;
        constrainedDesiredSignature = null;
        updateDrawSet();
      } else if (!source) beginLoad();
      else {
        retryIfDesiredSelectionChanged();
        refreshSelection();
      }
      context.onWorkChange?.();
    },

    setConfig(kindConfig) {
      if (disposed) return;
      let next: Tiles3dMemberConfig;
      try {
        next = validateConfig(kindConfig as Tiles3dMemberConfig);
      } catch (error) {
        report(error);
        return;
      }
      const sourceChanged =
        next.endpoint !== config.endpoint || next.revision !== config.revision;
      const decodeChanged =
        next.tilesetToScene.some(
          (value, index) => value !== config.tilesetToScene[index],
        ) || !sameDecodeWasm(next.wasm, config.wasm);
      const placementChanged =
        next.verticalExaggeration !== config.verticalExaggeration ||
        next.verticalPivotZ !== config.verticalPivotZ;
      config = next;
      if (placementChanged) applyPlacement();
      if (sourceChanged || decodeChanged) {
        configGeneration += 1;
        queue?.dispose();
        queue = null;
        setQueueSnapshot(null);
        adapter.clearTiles();
        pickSet.replaceDrawn([]);
        source = null;
        sourceState = "idle";
        traversal = null;
        requested.clear();
        decoded.clear();
        admissionBlocked.clear();
        admissionFailed.clear();
        memoryConstrained = false;
        irreducibleBudget = false;
        constrainedDesiredSignature = null;
        if (active) beginLoad();
      } else {
        queue?.configure({
          maxConcurrency: concurrency(),
          maxDecodedBytes: config.cacheBytes ?? DEFAULT_TILES3D_CACHE_BYTES,
        });
        retryIfDesiredSelectionChanged();
        refreshSelection();
      }
    },

    beginInteraction() {
      if (!disposed) interactionDepth += 1;
    },
    endInteraction() {
      if (!disposed && interactionDepth > 0) interactionDepth -= 1;
    },
    prepareFrame() {
      if (!disposed && drawSetDirty) updateDrawSet();
    },

    governorInputs(): GovernorInputs {
      const renderer = adapter.stats();
      const hasContent = !!traversal && traversal.requestedTileIds.length > 0;
      return {
        projectedImportance:
          active && hasContent
            ? importanceFromRootSseCssPx(
                traversal!.rootScreenSpaceErrorPx,
                maximumSse(),
              )
            : CULLED,
        qualityDemand: active && hasContent ? 1 : 0,
        work: {
          operations:
            active &&
            (sourceState === "loading" ||
              !!queueSnapshot?.workPending ||
              renderer.pendingJobs > 0 ||
              (traversal?.requestedTileIds.some((id) => {
                const state = readiness(id);
                return state !== "submitted" && state !== "failed";
              }) ??
                false))
              ? Math.max(1, (queueSnapshot?.active ?? 0) + renderer.pendingJobs)
              : 0,
          progressSerial: workProgressSerial + renderer.workRevision,
        },
        physicalTileOperations: active ? (queueSnapshot?.active ?? 0) : 0,
        physicalHierarchyOperations:
          active && sourceState === "loading" ? 1 : 0,
        residentBytes: renderer.residentBytes,
      };
    },

    onStall(error) {
      if (!disposed) report(error);
    },

    applyAllocation(next) {
      if (disposed) return;
      const previousMemoryBudgetBytes = allocation.memoryBudgetBytes;
      allocation = {
        qualityFraction: Math.min(1, Math.max(0, next.qualityFraction)),
        memoryBudgetBytes: Math.max(0, Math.floor(next.memoryBudgetBytes)),
        regime: next.regime,
      };
      if (allocation.memoryBudgetBytes > previousMemoryBudgetBytes) {
        memoryConstrained = false;
        irreducibleBudget = false;
        constrainedDesiredSignature = null;
      } else if (adapter.stats().residentBytes > allocation.memoryBudgetBytes) {
        memoryConstrained = true;
        irreducibleBudget = false;
        constrainedDesiredSignature = unconstrainedDesiredSignature();
      } else {
        retryIfDesiredSelectionChanged();
      }
      for (const id of adapter.setResourceCeilingBytes(
        allocation.memoryBudgetBytes,
      )) {
        if (requested.has(id)) admissionBlocked.add(id);
      }
      queue?.configure({
        maxConcurrency: concurrency(),
        maxDecodedBytes: config.cacheBytes ?? DEFAULT_TILES3D_CACHE_BYTES,
      });
      refreshSelection();
      // An oversize decoded value is deliberately not cached by ContentQueue.
      // If it was blocked by the previous GPU allowance, deselect/reselect it
      // so the larger allocation can fetch/decode again instead of leaving a
      // permanently-ready entry with no retained payload.
      const uncachedBlocked = new Set(
        [...admissionBlocked].filter(
          (id) => requested.has(id) && queue?.get(id) === undefined,
        ),
      );
      if (queue && uncachedBlocked.size > 0) {
        queue.setSelection(
          contentRequests(
            source!,
            [...requested].filter((id) => !uncachedBlocked.has(id)),
          ),
        );
        for (const id of uncachedBlocked) {
          admissionBlocked.delete(id);
          admissionFailed.delete(id);
          decoded.delete(id);
        }
        refreshSelection();
      }
    },

    pick: (view, cssX, cssY) =>
      active ? pickSet.pick(view, cssX, cssY) : null,
    occlusionDepth: (view, cssX, cssY) =>
      active ? pickSet.occlusionDepth(view, cssX, cssY) : null,

    stats(): Tiles3dMemberStats {
      const submissions = context.submissions.stats();
      const effectiveSse =
        traversal?.effectiveScreenSpaceErrorPx ??
        maximumSse() / Math.max(allocation.qualityFraction, 0.05);
      return {
        kind: "tiles3d",
        active,
        disposed,
        sourceState,
        irreducibleBudget,
        revision: config.revision,
        capabilityKey: context.textureCapabilities.capabilityKey,
        devicePixelRatio,
        interactionDepth,
        configGeneration,
        verticalExaggeration:
          config.verticalExaggeration ?? DEFAULT_VERTICAL_EXAGGERATION,
        verticalPivotZ: config.verticalPivotZ ?? DEFAULT_VERTICAL_PIVOT_Z,
        allocation,
        maximumScreenSpaceErrorPx: maximumSse(),
        effectiveScreenSpaceErrorPx: effectiveSse,
        sseMultiplier: effectiveSse / maximumSse(),
        memoryConstrained,
        selectedTiles: traversal?.desiredTileIds.length ?? 0,
        requestedTiles: traversal?.requestedTileIds.length ?? 0,
        selectionPasses,
        errorCount,
        lastError,
        queue: queueSnapshot,
        decode: context.workers.stats?.() ?? null,
        renderer: adapter.stats(),
        submissions: {
          queuedJobs: submissions.queuedJobs,
          queuedBytes: submissions.queuedBytes,
          lastFrameAdmittedJobs: submissions.lastFrameAdmittedJobs,
          lastFrameAdmittedBytes: submissions.lastFrameAdmittedBytes,
          admittedJobs: submissions.admittedJobs,
          admittedBytes: submissions.admittedBytes,
          peakQueuedJobs: submissions.peakQueuedJobs,
          peakQueuedBytes: submissions.peakQueuedBytes,
          peakFrameAdmittedBytes: submissions.peakFrameAdmittedBytes,
          peakFrameElapsedMs: submissions.peakFrameElapsedMs,
          admissionFrames: submissions.admissionFrames,
        },
      };
    },

    dispose() {
      if (disposed) return;
      disposed = true;
      active = false;
      sourceState = "disposed";
      loadGeneration += 1;
      loadController?.abort();
      queue?.dispose();
      queue = null;
      requested.clear();
      decoded.clear();
      admissionBlocked.clear();
      admissionFailed.clear();
      adapter.dispose();
      pickSet.dispose();
      source = null;
      camera = null;
    },
  };
};
