/**
 * pointcloud-lod combined streamed-scene example.
 *
 * One coordinator, one memory pool, one submission scheduler and one view
 * governor, with two members drawing the same place: AHN4 lidar as points and
 * 3DBAG buildings as mesh. That is the configuration the streamed-member
 * architecture exists for, and the only one where the members have to argue
 * about a budget — the panel shows each one's claimed share of the view beside
 * the quality it was actually given.
 *
 * Both datasets are Dutch national open data, streamed live with no key. They
 * meet in a local ENU frame whose Z is NAP: the mesh arrives in ECEF and is
 * placed by `ecefToScene`, the cloud arrives in RD New and is placed by a
 * model matrix. `scene/places.ts` derives both from one origin, which is what
 * makes a shared camera meaningful.
 */

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
import {
  BAG3D_ENDPOINT,
  PLACES,
  contentToScene,
  rdToScene,
  type Place,
} from "../scene/places";
import { bag3dContentFetch, createBag3dTilesetFetch } from "../scene/bag3d";
import { decodeWasmUrls } from "../scene/decodeAssets";
import { createSceneHost } from "../scene/host";
import { renderDiagnostics, type MemberRow } from "../scene/diagnostics";

const element = <T extends Element>(selector: string): T => {
  const found = document.querySelector<T>(selector);
  if (!found) throw new Error(`The example page is missing ${selector}`);
  return found;
};

const placeSelect = element<HTMLSelectElement>("#place");
const radiusInput = element<HTMLInputElement>("#radius");
const radiusValue = element<HTMLOutputElement>("#radius-value");
const sseInput = element<HTMLInputElement>("#sse");
const sseValue = element<HTMLOutputElement>("#sse-value");
const pointSizeInput = element<HTMLInputElement>("#point-size");
const pointSizeValue = element<HTMLOutputElement>("#point-size-value");
const meshToggle = element<HTMLInputElement>("#show-mesh");
const pointsToggle = element<HTMLInputElement>("#show-points");
const resetViewButton = element<HTMLButtonElement>("#reset-view");
const messageOutput = element<HTMLOutputElement>("#message");
const statsElement = element<HTMLDivElement>("#stats");

const host = createSceneHost(element<HTMLElement>("#viewer"));

type Member = {
  readonly registration: StreamedMemberRegistration;
  readonly source?: TileSource;
  stats(): Tiles3dMemberStats | PointCloudMemberStats;
};

const releaseMember = (member: Member | null): void => {
  member?.registration.release();
  member?.source?.dispose?.();
};

let mesh: Member | null = null;
let points: Member | null = null;
let place: Place = PLACES[0]!;
let loadRevision = 0;

const radiusMeters = (): number => Number(radiusInput.value);
const maximumSse = (): number => Number(sseInput.value);
const presentation = (): PointPresentation => ({
  mode: "auto",
  userScale: Number(pointSizeInput.value),
});

const copcSource = (url: string): Promise<TileSource> =>
  createCopcWorkerTileSource({
    source: url,
    createWorker: () =>
      new Worker(new URL("../copc.worker.ts", import.meta.url), {
        type: "module",
      }),
    lazPerfWasmUrl: new URL("/laz-perf.wasm", window.location.href).href,
  });

let externalTilesets = 0;
let failure: string | null = null;
let meshConfig: Tiles3dMemberConfig | null = null;
let pointsConfig: PointCloudMemberConfig | null = null;

/** One line of state, rather than whichever callback fired last. */
const message = (): string => {
  if (failure) return failure;
  const meshStats = mesh?.stats() as Tiles3dMemberStats | undefined;
  if (!meshStats || meshStats.sourceState !== "ready" || !points) {
    return `Loading ${place.label}…`;
  }
  return `${place.label} — ${externalTilesets} building tilesets, ${meshStats.selectedTiles} mesh tiles and lidar streaming`;
};

const buildingsConfig = (revision: string): Tiles3dMemberConfig => ({
  endpoint: BAG3D_ENDPOINT,
  revision,
  ecefToScene: contentToScene(place),
  maximumScreenSpaceErrorPx: maximumSse(),
  wasm: decodeWasmUrls(),
  fetchContent: bag3dContentFetch,
  fetchTileset: createBag3dTilesetFetch({
    endpoint: BAG3D_ENDPOINT,
    center: wgs84ToEcef(place.longitude, place.latitude, place.napZeroHeight),
    radiusMeters: radiusMeters(),
    onExternalLoaded: (loaded) => {
      externalTilesets = loaded;
    },
  }),
  onError: (error) => {
    failure = `buildings: ${error instanceof Error ? error.message : String(error)}`;
  },
});

const loadMesh = (revision: string): Member => {
  // Members take a whole configuration, never a patch, so the object a control
  // later edits is the one the member was built from.
  meshConfig = buildingsConfig(revision);
  const member = createTiles3dMember(
    host.coordinator.context(host.renderer),
    meshConfig,
  );
  const registration = host.coordinator.register(member, {
    id: "buildings",
    qualityManaged: true,
  });
  registration.setActive(meshToggle.checked);
  registration.setCamera(host.cameraView());
  return { registration, stats: () => member.stats() as Tiles3dMemberStats };
};

const loadPoints = async (
  target: Place,
  generation: number,
): Promise<Member | null> => {
  const source = await copcSource(target.copcUrl);
  // Place identity can cycle A -> B -> A while the first A is still loading.
  if (generation !== loadRevision) {
    source.dispose?.();
    return null;
  }
  pointsConfig = {
    source,
    presentation: presentation(),
    adaptive: true,
    onError: (error: unknown) => {
      failure = `lidar: ${error instanceof Error ? error.message : String(error)}`;
    },
  };
  const member = createPointCloudMember(
    host.coordinator.context(host.renderer),
    pointsConfig,
  );
  const registration = host.coordinator.register(member, {
    id: "lidar",
    qualityManaged: true,
    qualityTargets: { interactionTargetMs: 16, stationaryTargetMs: 33 },
  });
  // RD New metres into the scene's ENU frame: a rotation by the meridian
  // convergence and a uniform scale, which is the similarity the controller's
  // screen-space error math accepts.
  registration.setModelMatrix(rdToScene(target));
  registration.setActive(pointsToggle.checked);
  registration.setCamera(host.cameraView());
  return {
    registration,
    source,
    stats: () => member.stats() as PointCloudMemberStats,
  };
};

/**
 * Reloads both members, because the streamed radius is baked into the resolved
 * document and the revision that keys it. Only a change of place reframes:
 * widening the radius from where you are standing should bring more buildings
 * to the same view, not move you.
 */
const loadPlace = async (next: Place, reframe: boolean): Promise<void> => {
  place = next;
  loadRevision += 1;
  const generation = loadRevision;
  const revision = `${place.label}@${radiusMeters()}m#${loadRevision}`;
  releaseMember(mesh);
  releaseMember(points);
  mesh = null;
  points = null;
  externalTilesets = 0;
  failure = null;
  if (reframe) host.lookAt([0, 0, 30], place.viewDistance);
  mesh = loadMesh(revision);
  const loadedPoints = await loadPoints(place, generation);
  if (generation === loadRevision) points = loadedPoints;
  else releaseMember(loadedPoints);
};

host.onBeforeFrame((view, devicePixelRatio) => {
  mesh?.registration.setCamera(view);
  mesh?.registration.setDevicePixelRatio(devicePixelRatio);
  points?.registration.setCamera(view);
  points?.registration.setDevicePixelRatio(devicePixelRatio);
});

for (const candidate of PLACES) {
  const option = document.createElement("option");
  option.value = candidate.label;
  option.textContent = candidate.label;
  placeSelect.append(option);
}

placeSelect.addEventListener("change", () => {
  const next = PLACES.find(
    (candidate) => candidate.label === placeSelect.value,
  );
  if (next) void loadPlace(next, true);
});

radiusInput.addEventListener("input", () => {
  radiusValue.textContent = `${(radiusMeters() / 1000).toFixed(1)} km`;
});
radiusInput.addEventListener("change", () => void loadPlace(place, false));

sseInput.addEventListener("input", () => {
  sseValue.textContent = `${maximumSse()} px`;
  if (!meshConfig) return;
  meshConfig = { ...meshConfig, maximumScreenSpaceErrorPx: maximumSse() };
  mesh?.registration.setConfig(meshConfig);
});

pointSizeInput.addEventListener("input", () => {
  pointSizeValue.textContent = `${Number(pointSizeInput.value).toFixed(2)}×`;
  if (!pointsConfig) return;
  pointsConfig = { ...pointsConfig, presentation: presentation() };
  points?.registration.setConfig(pointsConfig);
});

meshToggle.addEventListener("change", () =>
  mesh?.registration.setActive(meshToggle.checked),
);
pointsToggle.addEventListener("change", () =>
  points?.registration.setActive(pointsToggle.checked),
);

resetViewButton.addEventListener("click", () =>
  host.lookAt([0, 0, 30], place.viewDistance),
);

const rows = (): MemberRow[] => {
  const built: MemberRow[] = [];
  if (mesh)
    built.push({
      id: "buildings",
      label: "3DBAG buildings",
      stats: mesh.stats(),
    });
  if (points)
    built.push({ id: "lidar", label: "AHN4 lidar", stats: points.stats() });
  return built;
};

const updateDiagnostics = (): void => {
  messageOutput.textContent = message();
  renderDiagnostics(statsElement, {
    coordinator: host.coordinator.stats(),
    rows: rows(),
    frameMs: host.lastFrameMs(),
    rendererName: host.rendererName,
    textureFormat: host.textureCapabilities.compressedFormats[0] ?? "rgba",
  });
};

updateDiagnostics();
setInterval(updateDiagnostics, 250);

const initial = new URLSearchParams(window.location.search).get("place");
const requested = PLACES.find((candidate) =>
  candidate.label.startsWith(initial ?? ""),
);
placeSelect.value = (requested ?? PLACES[0]!).label;
radiusValue.textContent = `${(radiusMeters() / 1000).toFixed(1)} km`;
sseValue.textContent = `${maximumSse()} px`;
pointSizeValue.textContent = `${Number(pointSizeInput.value).toFixed(2)}×`;
void loadPlace(requested ?? PLACES[0]!, true);

// Driving handles for browser checks. Everything here moves the page the way a
// user would, so a passing check says the assembled scene works.
Object.assign(window, {
  combinedExample: {
    stats: () => ({
      coordinator: host.coordinator.stats(),
      mesh: mesh?.stats() ?? null,
      points: points?.stats() ?? null,
      frameMs: host.lastFrameMs(),
      place: place.label,
    }),
    view: () => host.cameraView(),
    lookAt: (distanceMeters: number) => host.lookAt([0, 0, 30], distanceMeters),
    setMemberActive: (id: "buildings" | "lidar", active: boolean) => {
      const toggle = id === "buildings" ? meshToggle : pointsToggle;
      toggle.checked = active;
      toggle.dispatchEvent(new Event("change"));
    },
  },
});
