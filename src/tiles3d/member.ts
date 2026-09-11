/** StreamedMember implementation for the constrained 3D Tiles profile. */

import {
  sameCameraView,
  sameMatrix,
  type CameraView,
  type Mat16,
} from "../camera";
import { safeCall } from "../observers";
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
  DEFAULT_TILES3D_CONCURRENCY,
  DEFAULT_TILES3D_SUBTREE_CACHE_BYTES,
  DEFAULT_VERTICAL_EXAGGERATION,
  DEFAULT_VERTICAL_PIVOT_Z,
  type Tiles3dMemberConfig,
  type Tiles3dMemberStats,
} from "./memberTypes";
import {
  finiteAffineMatrix,
  validateTiles3dMemberConfig,
} from "./memberConfig";
import { createVerticalExaggerationTransform } from "./rtc";
import {
  loadTileset,
  multiplyTilesetMatrices,
  type TilesetSource,
  type TilesetTile,
} from "./tilesetSource";
import { parseSubtree, type ParsedSubtree } from "./subtree";
import {
  createTilesetTraversal,
  type SubtreeHierarchyState,
  type TilesetTraversalResult,
} from "./traversal";

/**
 * Proper ancestor ids of a tile, deepest first.
 *
 * Tile ids are `/`-joined child indices under the tileset root, so an ancestor
 * is exactly a `/`-prefix. Walking them costs the tile's depth, where asking
 * which of a set of ids is an ancestor costs the size of that set.
 */
const ancestorIds = (id: string): string[] => {
  const ancestors: string[] = [];
  for (let cut = id.indexOf("/"); cut !== -1; cut = id.indexOf("/", cut + 1)) {
    ancestors.push(id.slice(0, cut));
  }
  return ancestors.reverse();
};

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

const decodedBytes = (content: DecodedTileContent): number =>
  content.byteEstimate.geometry + content.byteEstimate.textures;

const workerUrl = (value: string): string =>
  new URL(value, globalThis.location?.href ?? "http://localhost/").toString();

const contentRequests = (
  tileById: ReadonlyMap<string, TilesetTile>,
  ids: readonly string[],
): { readonly id: string; readonly url: string }[] =>
  ids.flatMap((id) => {
    const url = tileById.get(id)?.contentUrl;
    return url === undefined ? [] : [{ id, url }];
  });

export const createTiles3dMember = (
  context: StreamedMemberContext,
  initialConfig: Tiles3dMemberConfig,
): StreamedMember => {
  let traverseTileset = createTilesetTraversal();
  let config = validateTiles3dMemberConfig(initialConfig);
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
  /**
   * Submitted tiles whose complete replacement group did not fit.
   *
   * Distinct from `memoryConstrained`, which says the member's whole desired
   * frontier is unaffordable and answers by falling back to the root. A
   * blocked group is a fact about one branch: the rest of the tileset is still
   * affordable at full quality, so refinement stops at this tile and nowhere
   * else.
   */
  const blockedGroupAncestors = new Set<string>();
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
  let subtreeQueue: ContentQueue<ParsedSubtree> | null = null;
  let subtreeSnapshot: ContentQueueSnapshot | null = null;
  let subtreeHierarchy: SubtreeHierarchyState | null = null;
  let materializedTileById = new Map<string, TilesetTile>();
  let traversal: TilesetTraversalResult | null = null;
  let errorCount = 0;
  let lastError: string | null = null;
  let workProgressSerial = 0;
  let selectionPasses = 0;
  const requested = new Set<string>();
  const desiredRequests = new Set<string>();
  const decoded = new Set<string>();
  const admissionBlocked = new Set<string>();
  const admissionFailed = new Set<string>();
  let refreshing = false;
  let refreshAgain = false;
  let drawSetDirty = true;

  const report = (error: unknown): void => {
    errorCount += 1;
    lastError = errorMessage(error);
    safeCall(() => config.onError?.(error));
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

  const concurrency = (): number =>
    config.concurrency ?? DEFAULT_TILES3D_CONCURRENCY;

  const setQueueSnapshot = (snapshot: ContentQueueSnapshot | null): void => {
    queueSnapshot = snapshot;
    queueEntryById = new Map(
      snapshot ? snapshot.entries.map((entry) => [entry.id, entry]) : [],
    );
  };

  const setSubtreeSnapshot = (snapshot: ContentQueueSnapshot | null): void => {
    subtreeSnapshot = snapshot;
    subtreeHierarchy =
      snapshot && subtreeQueue
        ? {
            revision: snapshot.revision,
            configGeneration: snapshot.configGeneration,
            entries: snapshot.entries,
            subtreeById: subtreeQueue.contents(),
          }
        : null;
  };

  const select = (
    maximumScreenSpaceErrorPx: number,
    qualityFraction: number,
    honourBlockedGroups = true,
  ): TilesetTraversalResult => {
    const retainedAncestors = new Set<string>();
    if (
      !memoryConstrained &&
      allocation.regime === "moving" &&
      qualityFraction < 1
    ) {
      const drawn = adapter.stats();
      if (
        config.interactionRetentionMaxTriangles! > 0 &&
        config.interactionRetentionMaxActors! > 0 &&
        drawn.drawnActors > 0 &&
        drawn.drawnActors <= config.interactionRetentionMaxActors! &&
        drawn.drawnTriangles <= config.interactionRetentionMaxTriangles!
      ) {
        for (const id of drawn.drawnTileIds)
          for (const ancestor of ancestorIds(id))
            retainedAncestors.add(ancestor);
      }
    }
    const result = traverseTileset({
      root: source!.root,
      camera: camera!,
      maximumScreenSpaceErrorPx,
      qualityFraction,
      retainRefinement: (id) => retainedAncestors.has(id),
      modelMatrix: traversalModelMatrix(),
      geometricErrorScale: config.geometricErrorScale,
      readiness,
      ...(honourBlockedGroups && blockedGroupAncestors.size > 0
        ? { refinable: (id: string) => !blockedGroupAncestors.has(id) }
        : {}),
      subtrees: subtreeHierarchy,
    });
    // This hierarchy is derived from the bounded subtree snapshot. Replacing
    // the map keeps camera exploration from retaining every historical tile.
    materializedTileById = new Map(result.tileById);
    return result;
  };

  /**
   * Ids grouped under every ancestor prefix, so "the subtree under x" is one
   * lookup instead of a prefix scan of the whole list.
   */
  const indexByAncestor = (ids: readonly string[]): Map<string, string[]> => {
    const index = new Map<string, string[]>();
    for (const id of ids) {
      for (const ancestor of ancestorIds(id)) {
        const bucket = index.get(ancestor);
        if (bucket) bucket.push(id);
        else index.set(ancestor, [id]);
      }
    }
    return index;
  };

  // The desired set changes only with the selection that produced it, and the
  // admitted set only when the adapter says so, so each index is rebuilt once
  // per change rather than once per admission. Both are read many times per
  // admission and once per blocked tile across a whole selection refresh,
  // which is where scanning instead used to cost O(tiles^2).
  let desiredIndex = new Map<string, string[]>();
  let desiredIndexOf: TilesetTraversalResult | null = null;
  let submittedIndex = new Map<string, string[]>();
  let submittedIndexRevision: number | null = null;

  const desiredDescendants = (id: string): readonly string[] => {
    if (!traversal) return [];
    if (desiredIndexOf !== traversal) {
      desiredIndex = indexByAncestor(traversal.desiredTileIds);
      desiredIndexOf = traversal;
    }
    return desiredIndex.get(id) ?? [];
  };

  /** Submitted descendants a newly admitted tile replaces under REPLACE. */
  const replacedDescendants = (id: string): string[] => {
    const revision = adapter.submissionRevision();
    if (submittedIndexRevision !== revision) {
      submittedIndex = indexByAncestor(adapter.submittedTileIds());
      submittedIndexRevision = revision;
    }
    return [...(submittedIndex.get(id) ?? [])];
  };

  /** Submitted ancestors of a tile, deepest first. */
  const submittedAncestors = (id: string): string[] =>
    ancestorIds(id).filter(
      (candidate) => adapter.tileState(candidate) === "submitted",
    );

  const submittedAncestorReplacement = (id: string): string | null => {
    for (const ancestor of submittedAncestors(id)) {
      const desired = desiredDescendants(ancestor);
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

  const waitingForSiblingBeforeReplacement = (id: string): boolean =>
    submittedAncestors(id).some((ancestor) =>
      desiredDescendants(ancestor).some((candidate) => {
        if (candidate === id) return false;
        const state = adapter.tileState(candidate);
        return (
          state === "queued" ||
          decoded.has(candidate) ||
          (requested.has(candidate) && readiness(candidate) !== "failed")
        );
      }),
    );

  const coveredSoonByDesiredDescendants = (id: string): boolean => {
    const descendants = desiredDescendants(id);
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
    const ancestor = submittedAncestors(id)[0];
    if (!ancestor) return null;
    // Some siblings can fit alongside the ancestor before the rest arrive.
    // Keep those admitted resources and replace the ancestor with only the
    // remaining siblings; requiring the entire group to be absent can leave
    // a partially admitted frontier waiting forever at its memory ceiling.
    const desired = desiredDescendants(ancestor).filter(
      (candidate) => adapter.tileState(candidate) !== "submitted",
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

  const blockGroupUnder = (id: string): void => {
    // Coarsen the nearest content-bearing region even when its fallback is
    // no longer resident. Traversal can request it again without discarding
    // affordable detail elsewhere in the view.
    const ancestor = ancestorIds(id).find(
      (candidate) =>
        materializedTileById.get(candidate)?.contentUrl !== undefined,
    );
    if (ancestor === undefined || ancestor === source?.root.id) {
      // No smaller content region can cover this request: the member as a
      // whole must fall back to its root.
      memoryConstrained = true;
      irreducibleBudget = false;
    } else {
      blockedGroupAncestors.add(ancestor);
      for (const candidate of desiredDescendants(ancestor)) {
        admissionBlocked.add(candidate);
      }
    }
    constrainedDesiredSignature = unconstrainedDesiredSignature();
    refreshSelection();
  };

  const unconstrainedDesiredSignature = (): string | null => {
    if (!source || !camera) return null;
    selectionPasses += 1;
    // Deliberately blind to both backoffs: this is the frontier the camera
    // would ask for if nothing were blocked, and it is the only thing that can
    // tell a genuinely new view from the one that was already refused.
    return select(
      maximumSse(),
      allocation.qualityFraction,
      false,
    ).desiredTileIds.join("\n");
  };

  const clearBudgetConstraint = (): void => {
    memoryConstrained = false;
    blockedGroupAncestors.clear();
    irreducibleBudget = false;
    constrainedDesiredSignature = null;
  };

  const clearAdmissionState = (): void => {
    requested.clear();
    desiredRequests.clear();
    decoded.clear();
    admissionBlocked.clear();
    admissionFailed.clear();
    lostBlockedContent.clear();
  };

  const lostBlockedContent = new Set<string>();
  let cacheBackoffQueued = false;
  const backoffAfterCacheLoss = (id: string): void => {
    if (!admissionBlocked.has(id)) return;
    lostBlockedContent.add(id);
    if (cacheBackoffQueued) return;
    cacheBackoffQueued = true;
    // Eviction runs inside the queue's cache mutation. Wait until that
    // transaction and its content delivery finish before changing selection.
    queueMicrotask(() => {
      cacheBackoffQueued = false;
      const lost = [...lostBlockedContent];
      lostBlockedContent.clear();
      if (disposed || !active) return;
      for (const candidate of lost) {
        if (
          requested.has(candidate) &&
          admissionBlocked.has(candidate) &&
          adapter.tileState(candidate) === "absent" &&
          !queue?.get(candidate)
        ) {
          if (candidate === source?.root.id && memoryConstrained) {
            latchIrreducibleBudget();
            refreshSelection();
          } else blockGroupUnder(candidate);
        }
      }
    });
  };

  const retryIfDesiredSelectionChanged = (): void => {
    if (!memoryConstrained && blockedGroupAncestors.size === 0) return;
    if (constrainedDesiredSignature === unconstrainedDesiredSignature()) return;
    clearBudgetConstraint();
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

  const updateDrawSet = (selectionComplete = false): void => {
    // Cached-content callbacks can run synchronously inside setSelection.
    // Publish coverage once that batch has finished, not once per callback.
    if (!drawSetDirty || (refreshing && !selectionComplete)) return;
    drawSetDirty = false;
    if (!source || !camera || !active || disposed) {
      adapter.setDrawnTiles([]);
      pickSet.replaceDrawn([]);
      return;
    }
    traversal = select(traversalMaximumSse(), traversalQualityFraction());
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
        subtreeQueue?.setSelection([]);
        for (const id of requested) adapter.cancelTile(id);
        requested.clear();
        desiredRequests.clear();
        updateDrawSet(true);
        return;
      }
      selectionPasses += 1;
      const next = select(traversalMaximumSse(), traversalQualityFraction());
      const newlyDesired = new Set(
        next.desiredTileIds.filter(
          (id) =>
            !desiredRequests.has(id) && adapter.tileState(id) === "absent",
        ),
      );
      desiredRequests.clear();
      for (const id of next.desiredTileIds) desiredRequests.add(id);
      traversal = next;
      subtreeQueue?.setSelection(next.neededSubtreeRequests);
      const nextContentRequests = contentRequests(
        next.tileById,
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
      // Renderer residency outlives the decoded CPU cache. Restore those
      // immutable resources before asking the content queue to fetch them.
      for (const id of nextRequested) {
        if (adapter.restoreTile(id) === "failed") admissionFailed.add(id);
      }
      queue.setSelection(
        nextContentRequests.filter(
          (request) => adapter.tileState(request.id) !== "submitted",
        ),
        { requeueUncached: newlyDesired },
      );
      for (const id of nextRequested) {
        if (!admissionBlocked.has(id) || adapter.tileState(id) !== "absent")
          continue;
        const content = queue.get(id);
        if (content && !irreducibleBudget) {
          // A regional backoff can complete a replacement set without a new
          // decode callback. Reconsider the group against the new frontier.
          const groupOutcome = trySubmitDesiredGroup(id);
          if (groupOutcome === "queued") continue;
          if (groupOutcome === "budget-blocked") {
            blockGroupUnder(id);
            break;
          }
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
      updateDrawSet(true);
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
        const tile = materializedTileById.get(request.id);
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
          // The whole replacement set is decoded and it did not fit, so
          // nothing further is coming for this branch and neither patience
          // guard below applies. Stop refining past the ancestor it would have
          // replaced and leave the rest of the tileset at full quality.
          blockGroupUnder(request.id);
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
          if (!queue?.get(request.id)) backoffAfterCacheLoss(request.id);
          if (coveredSoonByDesiredDescendants(request.id)) {
            updateDrawSet();
            return;
          }
          if (waitingForSiblingBeforeReplacement(request.id)) {
            updateDrawSet();
            return;
          }
          if (!memoryConstrained) {
            blockGroupUnder(request.id);
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
        backoffAfterCacheLoss(request.id);
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

  const createHierarchyQueue = (tileset: TilesetSource): void => {
    subtreeQueue?.dispose();
    subtreeQueue = null;
    setSubtreeSnapshot(null);
    const implicit = tileset.root.implicitTiling;
    if (!implicit) return;
    subtreeQueue = createContentQueue<ParsedSubtree>({
      revision: config.revision,
      configGeneration,
      maxConcurrency: concurrency(),
      maxDecodedBytes: DEFAULT_TILES3D_SUBTREE_CACHE_BYTES,
      ...(config.fetchSubtree ? { fetch: config.fetchSubtree } : {}),
      ...(config.maxAttempts === undefined
        ? {}
        : { maxAttempts: config.maxAttempts }),
      ...(config.retryBackoffMs === undefined
        ? {}
        : { retryBackoffMs: config.retryBackoffMs }),
      decode: (bytes) =>
        parseSubtree(bytes, implicit.subtreeLevels, implicit.metadataSchema),
      decodedByteLength: (subtree) => subtree.byteLength,
      onContent: () => {
        workProgressSerial += 1;
        if (disposed || !active) return;
        setSubtreeSnapshot(subtreeQueue?.snapshot() ?? null);
        refreshSelection();
        context.scheduleRender();
      },
      onError: (_request, error) => {
        workProgressSerial += 1;
        report(error);
        refreshSelection();
      },
      onStateChange: (snapshot) => {
        workProgressSerial += 1;
        setSubtreeSnapshot(snapshot);
        context.onWorkChange?.();
      },
    });
    setSubtreeSnapshot(subtreeQueue.snapshot());
  };

  const beginLoad = (): void => {
    if (disposed || !active) return;
    const generation = ++loadGeneration;
    loadController?.abort();
    const controller = new AbortController();
    loadController = controller;
    sourceState = "loading";
    source = null;
    materializedTileById.clear();
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
        materializedTileById = new Map(loaded.tileById);
        workProgressSerial += 1;
        sourceState = "ready";
        createQueue(loaded);
        createHierarchyQueue(loaded);
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
            : finiteAffineMatrix(Array.from(matrix), "model matrix");
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
        subtreeQueue?.setSelection([]);
        for (const id of requested) adapter.cancelTile(id);
        adapter.clearTiles();
        pickSet.replaceDrawn([]);
        clearAdmissionState();
        clearBudgetConstraint();
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
        next = validateTiles3dMemberConfig(kindConfig as Tiles3dMemberConfig);
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
        next.verticalPivotZ !== config.verticalPivotZ ||
        next.geometricErrorScale !== config.geometricErrorScale;
      const cacheIncreased =
        (next.cacheBytes ?? DEFAULT_TILES3D_CACHE_BYTES) >
        (config.cacheBytes ?? DEFAULT_TILES3D_CACHE_BYTES);
      config = next;
      if (placementChanged) applyPlacement();
      if (sourceChanged || decodeChanged) {
        traverseTileset = createTilesetTraversal();
        configGeneration += 1;
        queue?.dispose();
        queue = null;
        setQueueSnapshot(null);
        subtreeQueue?.dispose();
        subtreeQueue = null;
        setSubtreeSnapshot(null);
        adapter.clearTiles();
        pickSet.replaceDrawn([]);
        source = null;
        materializedTileById.clear();
        sourceState = "idle";
        traversal = null;
        clearAdmissionState();
        clearBudgetConstraint();
        if (active) beginLoad();
      } else {
        if (cacheIncreased) clearBudgetConstraint();
        queue?.configure({
          maxConcurrency: concurrency(),
          maxDecodedBytes: config.cacheBytes ?? DEFAULT_TILES3D_CACHE_BYTES,
        });
        subtreeQueue?.configure({ maxConcurrency: concurrency() });
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
      const renderer = adapter.workState();
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
              !!subtreeSnapshot?.workPending ||
              renderer.pendingJobs > 0 ||
              (traversal?.requestedTileIds.some((id) => {
                const state = readiness(id);
                return state !== "submitted" && state !== "failed";
              }) ??
                false))
              ? Math.max(
                  1,
                  (queueSnapshot?.active ?? 0) +
                    (subtreeSnapshot?.active ?? 0) +
                    renderer.pendingJobs,
                )
              : 0,
          progressSerial: workProgressSerial + renderer.workRevision,
        },
        physicalTileOperations: active ? (queueSnapshot?.active ?? 0) : 0,
        physicalHierarchyOperations: active
          ? (sourceState === "loading" ? 1 : 0) + (subtreeSnapshot?.active ?? 0)
          : 0,
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
        clearBudgetConstraint();
      } else if (
        adapter.workState().residentBytes > allocation.memoryBudgetBytes
      ) {
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
      refreshSelection();
      // An oversize decoded value is deliberately not cached by ContentQueue.
      // If it was blocked by the previous GPU allowance, explicitly reopen it
      // so the larger allocation can fetch/decode again instead of leaving a
      // permanently-ready entry with no retained payload.
      const uncachedBlocked = new Set(
        [...admissionBlocked].filter(
          (id) => requested.has(id) && queue?.get(id) === undefined,
        ),
      );
      if (queue && uncachedBlocked.size > 0) {
        for (const id of uncachedBlocked) {
          admissionBlocked.delete(id);
          admissionFailed.delete(id);
          decoded.delete(id);
        }
        queue.setSelection(
          contentRequests(
            materializedTileById,
            [...requested].filter(
              (id) => adapter.tileState(id) !== "submitted",
            ),
          ),
          { requeueUncached: uncachedBlocked },
        );
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
        geometricErrorScale: config.geometricErrorScale ?? "maximum",
        allocation,
        maximumScreenSpaceErrorPx: maximumSse(),
        effectiveScreenSpaceErrorPx: effectiveSse,
        sseMultiplier: effectiveSse / maximumSse(),
        memoryConstrained,
        blockedGroups: blockedGroupAncestors.size,
        selectedTiles: traversal?.desiredTileIds.length ?? 0,
        requestedTiles: traversal?.requestedTileIds.length ?? 0,
        selectionPasses,
        errorCount,
        lastError,
        queue: queueSnapshot,
        subtrees: subtreeSnapshot,
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
      traverseTileset = createTilesetTraversal();
      loadGeneration += 1;
      loadController?.abort();
      queue?.dispose();
      queue = null;
      subtreeQueue?.dispose();
      subtreeQueue = null;
      setSubtreeSnapshot(null);
      clearAdmissionState();
      blockedGroupAncestors.clear();
      adapter.dispose();
      pickSet.dispose();
      source = null;
      materializedTileById.clear();
      camera = null;
    },
  };
};
