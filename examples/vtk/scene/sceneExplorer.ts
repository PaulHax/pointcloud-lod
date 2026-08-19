import {
  createCopcWorkerTileSource,
  wgs84ToEcef,
  type PointPresentation,
  type StreamedMemberRegistration,
  type TileSource,
} from "../../../src";
import { createPointCloudMember, createTiles3dMember } from "../../../src/vtk";
import type {
  PointCloudMemberConfig,
  PointCloudMemberStats,
} from "../../../src/pointCloudMember";
import type {
  Tiles3dMemberConfig,
  Tiles3dMemberStats,
} from "../../../src/tiles3d/memberTypes";
import { createBag3dTilesetFetch } from "./bag3d";
import { decodeWasmUrls } from "./decodeAssets";
import { renderDiagnostics, type MemberRow } from "./diagnostics";
import { renderExplorerShell, type ExplorerPreset } from "./explorerShell";
import { createFrameRateMonitor } from "./frameRate";
import { createSceneHost } from "./host";
import { createLocalTilesSource } from "./localTiles";
import {
  BAG3D_ENDPOINT,
  PLACES,
  contentToScene,
  rdToScene,
  type Place,
} from "./places";
import { HOSTED_POINT_CLOUDS } from "./sceneCatalog";

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

const element = <T extends Element>(selector: string): T => {
  const found = document.querySelector<T>(selector);
  if (!found) throw new Error(`The explorer is missing ${selector}`);
  return found;
};

type Dataset = {
  readonly id: string;
  readonly kind: "points" | "tiles";
  readonly label: string;
  readonly card: HTMLDetailsElement;
  readonly statsElement: HTMLElement;
  readonly registration: StreamedMemberRegistration;
  readonly source?: TileSource;
  readonly location: DatasetLocation | null;
  view: ViewTarget | null;
  stats(): PointCloudMemberStats | Tiles3dMemberStats;
  release(): void;
};

type ViewTarget = {
  readonly center: [number, number, number];
  readonly distance: number;
};

type DatasetLocation = {
  readonly kind: "points-url" | "points-place" | "tiles-url" | "tiles-place";
  readonly value: string;
};

type PointSource = {
  readonly source: string | Blob;
  readonly label: string;
  readonly place?: Place;
  readonly location: DatasetLocation | null;
};

type TilesSource = {
  readonly label: string;
  readonly endpoint: string;
  readonly place?: Place;
  readonly fetchTileset?: Tiles3dMemberConfig["fetchTileset"];
  readonly fetchContent?: Tiles3dMemberConfig["fetchContent"];
  readonly location: DatasetLocation | null;
};

const pointPresets: readonly PointSource[] = [
  ...HOSTED_POINT_CLOUDS.map(({ label, url }) => ({
    label,
    source: url,
    location: { kind: "points-url" as const, value: url },
  })),
  ...PLACES.map((place) => ({
    label: place.label,
    source: place.copcUrl,
    place,
    location: { kind: "points-place" as const, value: place.label },
  })),
];

const tilesPreset = (place: Place, radiusMeters = 3_000): TilesSource => ({
  label: `3DBAG buildings · ${place.label}`,
  endpoint: BAG3D_ENDPOINT,
  place,
  fetchTileset: createBag3dTilesetFetch({
    endpoint: BAG3D_ENDPOINT,
    center: wgs84ToEcef(place.longitude, place.latitude, place.napZeroHeight),
    radiusMeters,
  }),
  location: { kind: "tiles-place", value: place.label },
});

const basename = (value: string): string => {
  try {
    const path = new URL(value).pathname;
    return decodeURIComponent(path.slice(path.lastIndexOf("/") + 1)) || value;
  } catch {
    return value;
  }
};

const tilesEndpoint = (value: string): string => {
  const url = new URL(value);
  if (url.pathname.toLowerCase().endsWith("/tileset.json")) {
    url.pathname = url.pathname.slice(0, -"/tileset.json".length);
  }
  return url.href.replace(/\/$/, "");
};

const boundsView = (
  source: TileSource,
): { center: [number, number, number]; distance: number } | null => {
  const bounds = source.metadata().bounds;
  if (!bounds) return null;
  const center: [number, number, number] = [
    (bounds.min[0] + bounds.max[0]) / 2,
    (bounds.min[1] + bounds.max[1]) / 2,
    (bounds.min[2] + bounds.max[2]) / 2,
  ];
  const radius =
    Math.hypot(
      bounds.max[0] - bounds.min[0],
      bounds.max[1] - bounds.min[1],
      bounds.max[2] - bounds.min[2],
    ) / 2;
  return radius > 0 ? { center, distance: radius * 2.8 } : null;
};

const makeCard = (
  id: string,
  label: string,
  subtitle: string,
): {
  card: HTMLDetailsElement;
  body: HTMLElement;
  stats: HTMLElement;
  title: HTMLElement;
} => {
  const card = document.createElement("details");
  card.className = "dataset-card";
  card.innerHTML = `
    <summary>
      <span class="dataset-heading">
        <strong></strong>
        <small data-role="subtitle"></small>
      </span>
    </summary>
    <div class="dataset-body" data-role="body"></div>
  `;
  const title = card.querySelector<HTMLElement>("strong")!;
  title.textContent = label;
  card.querySelector<HTMLElement>('[data-role="subtitle"]')!.textContent =
    subtitle;
  const body = card.querySelector<HTMLElement>('[data-role="body"]')!;
  const stats = document.createElement("div");
  stats.id = `${id}-stats`;
  stats.className = "dataset-stats";
  return { card, body, stats, title };
};

type Activity = {
  readonly state: "loading" | "processing" | "rendering" | "settled" | "error";
  readonly label: string;
  readonly detail: string;
};

const setActivity = (element: HTMLElement, activity: Activity): void => {
  element.dataset.state = activity.state;
  element.querySelector<HTMLElement>(
    '[data-role="activity-label"]',
  )!.textContent = activity.label;
  element.title = activity.detail;
  element.setAttribute("aria-label", activity.detail);
};

const datasetActivity = (
  stats: PointCloudMemberStats | Tiles3dMemberStats,
): Activity => {
  if (stats.kind === "pointCloud") {
    const controller = stats.controller;
    const activeOperations =
      controller.physicalTileOperations +
      controller.physicalHierarchyOperations;
    if (activeOperations > 0) {
      return {
        state: "loading",
        label: "Fetching data",
        detail: `${activeOperations} point-cloud network or decode operation${activeOperations === 1 ? "" : "s"} active`,
      };
    }
    const queued = controller.queuedTiles + controller.queuedPages;
    if (queued > 0) {
      return {
        state: "loading",
        label: "Waiting to fetch",
        detail: `${queued} point-cloud request${queued === 1 ? "" : "s"} waiting for a load slot`,
      };
    }
    if (controller.selectionPending || controller.workPending) {
      return {
        state: "processing",
        label: "Processing data",
        detail:
          "Selecting and preparing point-cloud detail for the current view",
      };
    }
    return {
      state: "settled",
      label: "Up to date",
      detail: "Point cloud is up to date for the current view",
    };
  }
  if (stats.sourceState === "failed") {
    return {
      state: "error",
      label: "Error",
      detail: stats.lastError ?? "The 3D Tiles source failed to load",
    };
  }
  if (stats.sourceState === "loading") {
    return {
      state: "loading",
      label: "Fetching data",
      detail: "Fetching the 3D Tiles tileset",
    };
  }
  const decodeJobs =
    (stats.decode?.activeJobs ?? 0) + (stats.decode?.queuedJobs ?? 0);
  if (decodeJobs > 0) {
    return {
      state: "processing",
      label: "Processing data",
      detail: `${decodeJobs} 3D Tiles decode job${decodeJobs === 1 ? "" : "s"} active or queued`,
    };
  }
  if ((stats.queue?.active ?? 0) > 0) {
    return {
      state: "loading",
      label: "Fetching data",
      detail: `${stats.queue!.active} 3D Tiles content request${stats.queue!.active === 1 ? "" : "s"} active`,
    };
  }
  if ((stats.queue?.queued ?? 0) + (stats.queue?.retrying ?? 0) > 0) {
    return {
      state: "loading",
      label: "Waiting to fetch",
      detail: "3D Tiles content is waiting for a network slot or retry",
    };
  }
  if (stats.submissions.queuedJobs > 0 || stats.renderer.pendingJobs > 0) {
    return {
      state: "rendering",
      label: "Updating renderer",
      detail:
        "Decoded 3D Tiles content is waiting to be submitted to the renderer",
    };
  }
  if (stats.queue?.workPending) {
    return {
      state: "processing",
      label: "Processing data",
      detail: "Updating 3D Tiles detail for the current view",
    };
  }
  return {
    state: "settled",
    label: "Up to date",
    detail: "3D Tiles are up to date for the current view",
  };
};

const infoTip = (description: string): string =>
  `<span class="info-tip" tabindex="0" title="${description}" aria-label="${description}" data-tooltip="${description}">i</span>`;

const locationKey = (location: DatasetLocation): string =>
  `${location.kind}:${location.value}`;

const parseLocation = (value: string): DatasetLocation | null => {
  const separator = value.indexOf(":");
  if (separator < 0) return null;
  const kind = value.slice(0, separator);
  if (
    kind !== "points-url" &&
    kind !== "points-place" &&
    kind !== "tiles-url" &&
    kind !== "tiles-place"
  ) {
    return null;
  }
  return { kind, value: value.slice(separator + 1) };
};

const sourceForLocation = (
  location: DatasetLocation,
): PointSource | TilesSource | null => {
  if (location.kind === "points-url") {
    return {
      source: location.value,
      label: basename(location.value),
      location,
    };
  }
  if (location.kind === "tiles-url") {
    return {
      endpoint: tilesEndpoint(location.value),
      label: basename(location.value) || "3D Tiles",
      location,
    };
  }
  const place = PLACES.find((candidate) => candidate.label === location.value);
  if (!place) return null;
  return location.kind === "points-place"
    ? {
        source: place.copcUrl,
        label: place.label,
        place,
        location,
      }
    : tilesPreset(place);
};

const pointPresentation = (
  mode: "auto" | "fixed",
  value: number,
): PointPresentation =>
  mode === "auto" ? { mode, userScale: value } : { mode, diameterCssPx: value };

export const startSceneExplorer = (preset: ExplorerPreset): void => {
  renderExplorerShell(preset);
  const host = createSceneHost(element<HTMLElement>("#viewer"));
  const frameRate = createFrameRateMonitor(
    host,
    element<HTMLElement>("#frame-rate"),
  );
  const governorActivity = element<HTMLElement>("#governor-activity");
  const datasetList = element<HTMLElement>("#dataset-list");
  const message = element<HTMLOutputElement>("#message");
  const stats = element<HTMLElement>("#stats");
  const resetView = element<HTMLButtonElement>("#reset-view");
  const movingTarget = element<HTMLInputElement>("#moving-target");
  const stationaryTarget = element<HTMLInputElement>("#stationary-target");
  const dialog = element<HTMLDialogElement>("#add-dataset-dialog");
  const addForm = element<HTMLFormElement>("#add-dataset-form");
  const datasetType = element<HTMLSelectElement>("#dataset-type");
  const sourceType = element<HTMLSelectElement>("#source-type");
  const presetSelect = element<HTMLSelectElement>("#dataset-preset");
  const urlInput = element<HTMLInputElement>("#dataset-url");
  const filesInput = element<HTMLInputElement>("#dataset-files");
  const nameInput = element<HTMLInputElement>("#dataset-name");
  const addError = element<HTMLOutputElement>("#add-dataset-error");
  const localHint = element<HTMLElement>("#local-source-hint");
  const dialogHeading = element<HTMLElement>("#add-dataset-heading");
  const dialogDescription = element<HTMLElement>("#add-dataset-description");
  const submitDataset = element<HTMLButtonElement>("#submit-dataset");
  const datasets: Dataset[] = [];
  let nextDatasetId = 1;
  let loading = 0;
  let sceneError: string | null = null;
  const pendingTilesFrames = new Set<Dataset>();
  let replacing: Dataset | null = null;

  const syncTargets = (): void => {
    host.coordinator.setQualityTargets({
      interactionTargetMs: Math.max(1, Number(movingTarget.value)),
      stationaryTargetMs: Math.max(1, Number(stationaryTarget.value)),
    });
  };
  movingTarget.addEventListener("change", syncTargets);
  stationaryTarget.addEventListener("change", syncTargets);
  syncTargets();

  const syncUrl = (): void => {
    const url = new URL(window.location.href);
    for (const name of ["url", "place", "pointPlace", "tiles", "data"]) {
      url.searchParams.delete(name);
    }
    const locations = datasets.flatMap((dataset) =>
      dataset.location ? [dataset.location] : [],
    );
    if (locations.length === datasets.length && locations.length === 1) {
      const location = locations[0]!;
      if (location.kind === "points-url")
        url.searchParams.set("url", location.value);
      else if (location.kind === "points-place")
        url.searchParams.set("pointPlace", location.value);
      else if (location.kind === "tiles-url")
        url.searchParams.set("tiles", location.value);
      else url.searchParams.set("place", location.value);
    } else {
      for (const location of locations) {
        url.searchParams.append("data", locationKey(location));
      }
    }
    window.history.replaceState(null, "", url);
  };

  const frameDataset = (dataset: Dataset): boolean =>
    dataset.view
      ? (host.lookAt(dataset.view.center, dataset.view.distance), true)
      : host.frameVisible();

  const removeDataset = (dataset: Dataset, frameRemaining = true): void => {
    const index = datasets.indexOf(dataset);
    if (index >= 0) datasets.splice(index, 1);
    pendingTilesFrames.delete(dataset);
    dataset.release();
    dataset.card.remove();
    syncUrl();
    if (frameRemaining && datasets.length === 1) frameDataset(datasets[0]!);
    host.scheduleRender();
  };

  const addPointDataset = async (
    input: PointSource,
    displayName = input.label,
    frame = datasets.length === 0,
    before: Element | null = null,
  ): Promise<Dataset> => {
    loading += 1;
    try {
      const source = await createCopcWorkerTileSource({
        source: input.source,
        createWorker: () =>
          new Worker(new URL("../copc.worker.ts", import.meta.url), {
            type: "module",
          }),
        lazPerfWasmUrl: new URL("/laz-perf.wasm", window.location.href).href,
      });
      const id = `dataset-${nextDatasetId++}`;
      const built = makeCard(id, displayName, "Point cloud · COPC");
      built.card.open = preset !== "combined" && datasets.length === 0;
      built.body.innerHTML = `
        <div class="toggle-row">
          <label><input data-role="visible" type="checkbox" checked /> Visible</label>
          <button data-role="remove" class="text-button danger" type="button">Remove</button>
        </div>
        <div class="field-grid">
          <div>
            <label>Point budget ${infoTip("Adaptive lets the view governor vary this dataset's point count to meet the frame-time target. Fixed always requests the entered budget.")}</label>
            <select data-role="budget-mode"><option value="adaptive">Adaptive</option><option value="fixed">Fixed</option></select>
          </div>
          <div>
            <label>Point size ${infoTip("Auto derives a screen-space point diameter from projected density. Fixed uses one CSS-pixel diameter at every zoom level.")}</label>
            <select data-role="size-mode"><option value="auto">Auto</option><option value="fixed">Fixed</option></select>
          </div>
        </div>
        <div>
          <label data-role="size-label">Point size scale ${infoTip("Multiplier for the automatically calculated point diameter. 1× uses the computed size; lower values make finer points and reveal more gaps.")}</label>
          <div class="range-output">
            <input data-role="size" type="range" min="0.25" max="2" step="0.05" value="0.5" />
            <output data-role="size-value">0.50×</output>
          </div>
        </div>
        <div data-role="fixed-controls" hidden>
          <label>Fixed budget (points) ${infoTip("Maximum number of points selected when Point budget is Fixed.")}</label>
          <input data-role="fixed-budget" type="number" min="1000" step="100000" value="2000000" />
        </div>
        <div data-role="adaptive-controls">
          <label>Maximum budget ${infoTip("Optional ceiling for adaptive point selection. Leave blank to let the shared memory limit be the ceiling.")} <span class="optional">blank = memory only</span></label>
          <input data-role="max-points" type="number" min="200000" step="100000" placeholder="none" />
        </div>
      `;
      built.body.append(built.stats);
      datasetList.insertBefore(built.card, before);

      const query = <T extends Element>(role: string): T =>
        built.body.querySelector<T>(`[data-role="${role}"]`)!;
      const visible = query<HTMLInputElement>("visible");
      const budgetMode = query<HTMLSelectElement>("budget-mode");
      const sizeMode = query<HTMLSelectElement>("size-mode");
      const size = query<HTMLInputElement>("size");
      const sizeLabel = query<HTMLElement>("size-label");
      const sizeValue = query<HTMLOutputElement>("size-value");
      const fixedControls = query<HTMLElement>("fixed-controls");
      const adaptiveControls = query<HTMLElement>("adaptive-controls");
      const fixedBudget = query<HTMLInputElement>("fixed-budget");
      const maxPoints = query<HTMLInputElement>("max-points");
      let autoSize = 0.5;
      let fixedSize = 2;
      let config: PointCloudMemberConfig = {
        source,
        presentation: pointPresentation("auto", 0.5),
        adaptive: true,
      };
      const member = createPointCloudMember(
        host.coordinator.context(host.renderer),
        config,
      );
      const registration = host.coordinator.register(member, {
        id,
        qualityManaged: true,
      });
      if (input.place) registration.setModelMatrix(rdToScene(input.place));
      registration.setCamera(host.cameraView());
      const dataset: Dataset = {
        id,
        kind: "points",
        label: displayName,
        card: built.card,
        statsElement: built.stats,
        registration,
        source,
        location: input.location,
        view: input.place
          ? { center: [0, 0, 30], distance: input.place.viewDistance }
          : boundsView(source),
        stats: () => member.stats() as PointCloudMemberStats,
        release: () => {
          registration.release();
          source.dispose?.();
        },
      };
      datasets.push(dataset);
      installPresetSwitcher(dataset, built.title);
      syncUrl();

      const updateConfig = (): void => {
        const adaptive = budgetMode.value === "adaptive";
        const pointMode = sizeMode.value as "auto" | "fixed";
        const pointValue = Number(size.value);
        config = {
          ...config,
          adaptive,
          presentation: pointPresentation(pointMode, pointValue),
          pointBudget: Math.max(1_000, Number(fixedBudget.value)),
          adaptiveOptions: maxPoints.value
            ? { maxBudget: Math.max(1, Number(maxPoints.value)) }
            : {},
        };
        registration.setQualityPolicy(adaptive);
        registration.setConfig(config);
      };
      const syncSize = (): void => {
        const auto = sizeMode.value === "auto";
        size.min = "0.25";
        size.max = auto ? "2" : "8";
        size.step = auto ? "0.05" : "0.25";
        sizeLabel.childNodes[0]!.textContent = auto
          ? "Point size scale "
          : "Point size (CSS px) ";
        sizeValue.textContent = auto
          ? `${Number(size.value).toFixed(2)}×`
          : `${Number(size.value).toFixed(2)} px`;
        updateConfig();
      };
      visible.addEventListener("change", () =>
        registration.setActive(visible.checked),
      );
      budgetMode.addEventListener("change", () => {
        const adaptive = budgetMode.value === "adaptive";
        fixedControls.hidden = adaptive;
        adaptiveControls.hidden = !adaptive;
        updateConfig();
      });
      sizeMode.addEventListener("change", () => {
        size.value = String(sizeMode.value === "auto" ? autoSize : fixedSize);
        syncSize();
      });
      size.addEventListener("input", () => {
        if (sizeMode.value === "auto") autoSize = Number(size.value);
        else fixedSize = Number(size.value);
        syncSize();
      });
      fixedBudget.addEventListener("change", updateConfig);
      maxPoints.addEventListener("change", updateConfig);
      query<HTMLButtonElement>("remove").addEventListener("click", () =>
        removeDataset(dataset),
      );

      if (frame) frameDataset(dataset);
      return dataset;
    } finally {
      loading -= 1;
    }
  };

  const addTilesDataset = (
    input: TilesSource,
    displayName = input.label,
    frame = datasets.length === 0,
    before: Element | null = null,
  ): Dataset => {
    const id = `dataset-${nextDatasetId++}`;
    const built = makeCard(id, displayName, "3D Tiles 1.1");
    built.card.open = preset !== "combined" && datasets.length === 0;
    built.body.innerHTML = `
      <div class="toggle-row">
        <label><input data-role="visible" type="checkbox" checked /> Visible</label>
        <button data-role="remove" class="text-button danger" type="button">Remove</button>
      </div>
      <div>
        <label>Maximum screen-space error ${infoTip("Largest projected tile error the renderer accepts. Lower values load finer tiles and cost more CPU, network, and GPU work.")}</label>
        <div class="range-output">
          <input data-role="sse" type="range" min="4" max="64" step="2" value="16" />
          <output data-role="sse-value">16 px</output>
        </div>
      </div>
    `;
    built.body.append(built.stats);
    datasetList.insertBefore(built.card, before);
    const query = <T extends Element>(role: string): T =>
      built.body.querySelector<T>(`[data-role="${role}"]`)!;
    const visible = query<HTMLInputElement>("visible");
    const sse = query<HTMLInputElement>("sse");
    const sseValue = query<HTMLOutputElement>("sse-value");
    let config: Tiles3dMemberConfig = {
      endpoint: input.endpoint,
      revision: `${id}:${Date.now()}`,
      tilesetToScene: input.place ? contentToScene(input.place) : IDENTITY,
      maximumScreenSpaceErrorPx: Number(sse.value),
      wasm: decodeWasmUrls(),
      ...(input.fetchTileset ? { fetchTileset: input.fetchTileset } : {}),
      ...(input.fetchContent ? { fetchContent: input.fetchContent } : {}),
    };
    const member = createTiles3dMember(
      host.coordinator.context(host.renderer),
      config,
    );
    const registration = host.coordinator.register(member, {
      id,
      qualityManaged: true,
    });
    registration.setCamera(host.cameraView());
    const dataset: Dataset = {
      id,
      kind: "tiles",
      label: displayName,
      card: built.card,
      statsElement: built.stats,
      registration,
      location: input.location,
      view: input.place
        ? { center: [0, 0, 30], distance: input.place.viewDistance }
        : null,
      stats: () => member.stats() as Tiles3dMemberStats,
      release: () => registration.release(),
    };
    datasets.push(dataset);
    installPresetSwitcher(dataset, built.title);
    syncUrl();
    visible.addEventListener("change", () =>
      registration.setActive(visible.checked),
    );
    sse.addEventListener("input", () => {
      sseValue.textContent = `${Number(sse.value)} px`;
      config = { ...config, maximumScreenSpaceErrorPx: Number(sse.value) };
      registration.setConfig(config);
    });
    query<HTMLButtonElement>("remove").addEventListener("click", () =>
      removeDataset(dataset),
    );
    if (frame) {
      if (dataset.view) {
        frameDataset(dataset);
      } else {
        pendingTilesFrames.add(dataset);
        host.lookAt([0, 0, 0], 1_000);
      }
    }
    return dataset;
  };

  host.onBeforeFrame((view, devicePixelRatio) => {
    for (const dataset of datasets) {
      dataset.registration.setCamera(view);
      dataset.registration.setDevicePixelRatio(devicePixelRatio);
    }
  });

  resetView.addEventListener("click", () => {
    if (datasets.length === 1) frameDataset(datasets[0]!);
    else host.frameVisible();
  });

  const updateDiagnostics = (): void => {
    const rows: MemberRow[] = datasets.map((dataset) => ({
      id: dataset.id,
      label: dataset.label,
      stats: dataset.stats(),
    }));
    const coordinator = host.coordinator.stats();
    const activities = rows.map((row) => datasetActivity(row.stats));
    const overall: Activity =
      loading > 0
        ? {
            state: "loading",
            label: "Fetching data",
            detail: `Opening ${loading} new dataset${loading === 1 ? "" : "s"}`,
          }
        : datasets.length === 0
          ? {
              state: "settled",
              label: "No data",
              detail: "Add a dataset to begin streaming",
            }
          : activities.some((activity) => activity.state === "error")
            ? {
                state: "error",
                label: "Needs attention",
                detail: "One or more datasets reported an error",
              }
            : activities.some((activity) => activity.state === "loading")
              ? {
                  state: "loading",
                  label: "Fetching data",
                  detail:
                    "One or more datasets are fetching or waiting for data",
                }
              : activities.some((activity) => activity.state === "rendering")
                ? {
                    state: "rendering",
                    label: "Updating renderer",
                    detail:
                      "One or more datasets are submitting decoded content to the renderer",
                  }
                : activities.some(
                      (activity) => activity.state === "processing",
                    ) || coordinator.governor.activity.workPending
                  ? {
                      state: "processing",
                      label: "Processing data",
                      detail:
                        "One or more datasets are decoding or selecting detail",
                    }
                  : {
                      state: "settled",
                      label: "Up to date",
                      detail:
                        "All loaded datasets are up to date for the current view",
                    };
    setActivity(governorActivity, overall);
    renderDiagnostics(
      stats,
      {
        coordinator,
        rows,
        frameMs: host.lastFrameMs(),
        rendererName: host.rendererName,
        textureFormat: host.textureCapabilities.compressedFormats[0] ?? "rgba",
      },
      Object.fromEntries(
        datasets.map((dataset) => [dataset.id, dataset.statsElement]),
      ),
    );
    frameRate.setTargetFrameMs(coordinator.governor.targetFrameTimeMs);
    message.hidden =
      sceneError === null && (loading > 0 || datasets.length > 0);
    message.textContent =
      sceneError ?? "Add a point cloud or 3D Tiles dataset.";
    for (const dataset of pendingTilesFrames) {
      if (
        (dataset.stats() as Tiles3dMemberStats).renderer.submittedActors > 0
      ) {
        pendingTilesFrames.delete(dataset);
        if (host.frameVisible()) {
          const camera = host.camera;
          dataset.view = {
            center: [...camera.getFocalPoint()] as [number, number, number],
            distance: camera.getDistance(),
          };
        }
      }
    }
  };
  updateDiagnostics();
  setInterval(updateDiagnostics, 250);

  const syncPresetChoices = (): void => {
    presetSelect.replaceChildren();
    const choices =
      datasetType.value === "points"
        ? pointPresets.map((item, index) => ({
            value: String(index),
            label: item.label,
          }))
        : PLACES.map((place, index) => ({
            value: String(index),
            label: `3DBAG buildings · ${place.label}`,
          }));
    for (const choice of choices) {
      const option = document.createElement("option");
      option.value = choice.value;
      option.textContent = choice.label;
      presetSelect.append(option);
    }
    const tiles = datasetType.value === "tiles";
    urlInput.placeholder = tiles
      ? "https://example.com/tileset.json"
      : "https://example.com/cloud.copc.laz";
    filesInput.value = "";
    filesInput.accept = tiles ? ".json,.glb,.b3dm" : ".laz,.copc.laz";
    filesInput.multiple = tiles;
    if (tiles) filesInput.setAttribute("webkitdirectory", "");
    else filesInput.removeAttribute("webkitdirectory");
    localHint.textContent = tiles
      ? "Choose the tileset directory so tileset.json and every referenced payload remain available."
      : "Choose one COPC .laz file.";
  };
  const syncSourceFields = (): void => {
    element<HTMLElement>("#preset-source-fields").hidden =
      sourceType.value !== "preset";
    element<HTMLElement>("#url-source-fields").hidden =
      sourceType.value !== "url";
    element<HTMLElement>("#local-source-fields").hidden =
      sourceType.value !== "local";
  };
  datasetType.addEventListener("change", syncPresetChoices);
  sourceType.addEventListener("change", syncSourceFields);
  syncPresetChoices();
  syncSourceFields();

  async function replaceDataset(
    current: Dataset,
    source: PointSource | TilesSource,
    displayName = source.label,
  ): Promise<void> {
    const currentIndex = datasets.indexOf(current);
    const replacement =
      "source" in source
        ? await addPointDataset(source, displayName, true, current.card)
        : addTilesDataset(source, displayName, true, current.card);
    const appendedIndex = datasets.indexOf(replacement);
    if (currentIndex >= 0 && appendedIndex >= 0) {
      datasets.splice(appendedIndex, 1);
      datasets.splice(currentIndex, 0, replacement);
    }
    removeDataset(current, false);
  }

  function openDatasetDialog(target: Dataset | null = null): void {
    replacing = target;
    addError.textContent = "";
    dialogHeading.textContent = target ? "Change dataset" : "Add dataset";
    dialogDescription.textContent = target
      ? `Replace ${target.label} while keeping the rest of the scene unchanged.`
      : "Add another independently controlled member to this view.";
    submitDataset.textContent = target ? "Change dataset" : "Add dataset";
    datasetType.disabled = false;
    if (target) datasetType.value = target.kind;
    sourceType.value = "preset";
    nameInput.value = "";
    urlInput.value = "";
    syncPresetChoices();
    syncSourceFields();
    dialog.showModal();
  }

  function installPresetSwitcher(dataset: Dataset, title: HTMLElement): void {
    const select = document.createElement("select");
    select.className = "dataset-source-select";
    select.setAttribute("aria-label", `Change ${dataset.label}`);
    const groups: readonly {
      readonly label: string;
      readonly sources: readonly (PointSource | TilesSource)[];
    }[] = [
      { label: "Point clouds", sources: pointPresets },
      {
        label: "3D Tiles",
        sources: PLACES.map((place) => tilesPreset(place)),
      },
    ];
    for (const group of groups) {
      const options = document.createElement("optgroup");
      options.label = group.label;
      for (const source of group.sources) {
        const option = document.createElement("option");
        option.value = locationKey(source.location!);
        option.textContent = source.label;
        options.append(option);
      }
      select.append(options);
    }
    const selectedValue = dataset.location
      ? locationKey(dataset.location)
      : "current";
    if (![...select.options].some((option) => option.value === selectedValue)) {
      const current = document.createElement("option");
      current.value = "current";
      current.textContent = dataset.label;
      select.prepend(current);
    }
    const custom = document.createElement("option");
    custom.value = "custom";
    custom.textContent = "Custom source…";
    select.append(custom);
    select.value = selectedValue;
    for (const eventName of ["click", "pointerdown"]) {
      select.addEventListener(eventName, (event) => event.stopPropagation());
    }
    select.addEventListener("change", () => {
      const wanted = select.value;
      select.value = selectedValue;
      if (wanted === "custom") {
        openDatasetDialog(dataset);
        return;
      }
      const location = parseLocation(wanted);
      const source = location ? sourceForLocation(location) : null;
      if (!source) return;
      sceneError = null;
      void replaceDataset(dataset, source).catch((error: unknown) => {
        sceneError = error instanceof Error ? error.message : String(error);
      });
    });
    title.replaceWith(select);
  }

  const closeDialog = (): void => {
    dialog.close();
    replacing = null;
  };
  element<HTMLButtonElement>("#open-add-dataset").addEventListener(
    "click",
    () => openDatasetDialog(),
  );
  element<HTMLButtonElement>("#close-add-dataset").addEventListener(
    "click",
    closeDialog,
  );
  element<HTMLButtonElement>("#cancel-add-dataset").addEventListener(
    "click",
    closeDialog,
  );
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) closeDialog();
  });

  addForm.addEventListener("submit", (event) => {
    event.preventDefault();
    addError.textContent = "";
    const add = async (): Promise<void> => {
      const customName = nameInput.value.trim();
      const target = replacing;
      if (datasetType.value === "points") {
        let source: PointSource;
        if (sourceType.value === "preset") {
          source = pointPresets[Number(presetSelect.value)]!;
        } else if (sourceType.value === "url") {
          if (!urlInput.value.trim()) throw new Error("Enter a COPC URL.");
          source = {
            source: urlInput.value.trim(),
            label: basename(urlInput.value.trim()),
            location: {
              kind: "points-url",
              value: urlInput.value.trim(),
            },
          };
        } else {
          const file = filesInput.files?.[0];
          if (!file) throw new Error("Choose a COPC file.");
          source = { source: file, label: file.name, location: null };
        }
        if (target) {
          await replaceDataset(target, source, customName || source.label);
        } else {
          await addPointDataset(source, customName || source.label);
        }
      } else {
        let source: TilesSource;
        if (sourceType.value === "preset") {
          source = tilesPreset(PLACES[Number(presetSelect.value)]!);
        } else if (sourceType.value === "url") {
          if (!urlInput.value.trim()) throw new Error("Enter a 3D Tiles URL.");
          source = {
            endpoint: tilesEndpoint(urlInput.value.trim()),
            label: basename(urlInput.value.trim()) || "3D Tiles",
            location: {
              kind: "tiles-url",
              value: urlInput.value.trim(),
            },
          };
        } else {
          const local = createLocalTilesSource([...(filesInput.files ?? [])]);
          source = { ...local, location: null };
        }
        if (target) {
          await replaceDataset(target, source, customName || source.label);
        } else {
          addTilesDataset(source, customName || source.label);
        }
      }
      nameInput.value = "";
      closeDialog();
    };
    void add().catch((error: unknown) => {
      addError.textContent =
        error instanceof Error ? error.message : String(error);
    });
  });

  const initial = async (): Promise<void> => {
    const parameters = new URLSearchParams(window.location.search);
    const encoded = parameters
      .getAll("data")
      .map(parseLocation)
      .filter((location): location is DatasetLocation => location !== null)
      .map(sourceForLocation)
      .filter((source): source is PointSource | TilesSource => source !== null);
    if (encoded.length > 0) {
      for (const [index, source] of encoded.entries()) {
        if ("source" in source) {
          await addPointDataset(source, undefined, index === 0);
        } else {
          addTilesDataset(source, undefined, index === 0);
        }
      }
      syncUrl();
      return;
    }
    const place =
      PLACES.find((candidate) =>
        candidate.label.startsWith(parameters.get("place") ?? ""),
      ) ?? PLACES[0]!;
    if (preset === "points") {
      const url = parameters.get("url");
      const pointPlace = parameters.get("pointPlace");
      const pointPlaceSource = pointPlace
        ? sourceForLocation({ kind: "points-place", value: pointPlace })
        : null;
      const source = url
        ? {
            source: url,
            label: basename(url),
            location: { kind: "points-url" as const, value: url },
          }
        : pointPlaceSource && "source" in pointPlaceSource
          ? pointPlaceSource
          : pointPresets[0]!;
      await addPointDataset(source);
    } else if (preset === "tiles") {
      const tiles = parameters.get("tiles");
      addTilesDataset(
        tiles
          ? {
              endpoint: tilesEndpoint(tiles),
              label: basename(tiles) || "3D Tiles",
              location: { kind: "tiles-url", value: tiles },
            }
          : tilesPreset(place),
      );
    } else {
      addTilesDataset(tilesPreset(place), undefined, true);
      await addPointDataset(
        {
          source: place.copcUrl,
          label: place.label,
          place,
          location: { kind: "points-place", value: place.label },
        },
        undefined,
        false,
      );
    }
    syncUrl();
  };
  void initial().catch((error: unknown) => {
    sceneError = error instanceof Error ? error.message : String(error);
  });
};
