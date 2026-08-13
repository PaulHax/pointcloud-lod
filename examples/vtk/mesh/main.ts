/**
 * pointcloud-lod streamed mesh example.
 *
 * Streams 3D Tiles through the member API — one coordinator owning the memory
 * pool, the submission scheduler and the view governor, and one tiles3d member
 * registered with it. That is the same wiring the trame bridge uses, and it is
 * what the combined example extends by registering a second member against the
 * same budget.
 *
 * The data is 3DBAG: every building in the Netherlands at LoD2.2, streamed
 * live from data.3dbag.nl with no key. See `scene/bag3d.ts` for why the
 * published index needs a host-side pass before it fits the library's profile.
 */

import {
  TilesetProfileError,
  wgs84ToEcef,
  type StreamedMemberRegistration,
} from "../../../src";
import { createTiles3dMember } from "../../../src/vtk";
import type {
  Tiles3dMemberConfig,
  Tiles3dMemberStats,
} from "../../../src/tiles3d/memberTypes";
import {
  BAG3D_ENDPOINT,
  PLACES,
  contentToScene,
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
const resetViewButton = element<HTMLButtonElement>("#reset-view");
const messageOutput = element<HTMLOutputElement>("#message");
const statsElement = element<HTMLDivElement>("#stats");

const host = createSceneHost(element<HTMLElement>("#viewer"));

let registration: StreamedMemberRegistration | null = null;
let member: ReturnType<typeof createTiles3dMember> | null = null;
let place: Place = PLACES[0]!;
let loadRevision = 0;
let externalTilesets = 0;
let failure: string | null = null;
let config: Tiles3dMemberConfig | null = null;

const radiusMeters = (): number => Number(radiusInput.value);
const maximumSse = (): number => Number(sseInput.value);

const memberStats = (): Tiles3dMemberStats | null =>
  (member?.stats() as Tiles3dMemberStats | undefined) ?? null;

const buildingsConfig = (
  target: Place,
  revision: string,
): Tiles3dMemberConfig => ({
  endpoint: BAG3D_ENDPOINT,
  revision,
  ecefToScene: contentToScene(target),
  maximumScreenSpaceErrorPx: maximumSse(),
  wasm: decodeWasmUrls(),
  fetchContent: bag3dContentFetch,
  fetchTileset: createBag3dTilesetFetch({
    endpoint: BAG3D_ENDPOINT,
    center: wgs84ToEcef(
      target.longitude,
      target.latitude,
      target.napZeroHeight,
    ),
    radiusMeters: radiusMeters(),
    onExternalLoaded: (loaded) => {
      externalTilesets = loaded;
    },
  }),
  onError: (error) => {
    failure =
      error instanceof TilesetProfileError
        ? `outside the supported 3D Tiles profile: ${error.message}`
        : error instanceof Error
          ? error.message
          : String(error);
  },
});

/**
 * Reloads the member, because the streamed radius is baked into the resolved
 * document and the revision that keys it. Only a change of place reframes:
 * widening the radius from where you are standing should bring more buildings
 * to the same view, not move you.
 */
const loadPlace = (next: Place, reframe: boolean): void => {
  place = next;
  loadRevision += 1;
  registration?.release();
  externalTilesets = 0;
  failure = null;
  // The member takes a whole configuration, never a patch: a later change
  // resends this object with one field replaced.
  config = buildingsConfig(
    place,
    `${place.label}@${radiusMeters()}m#${loadRevision}`,
  );
  member = createTiles3dMember(host.coordinator.context(host.renderer), config);
  registration = host.coordinator.register(member, {
    id: "buildings",
    qualityManaged: true,
  });
  registration.setCamera(host.cameraView());
  if (reframe) host.lookAt([0, 0, 0], place.viewDistance);
};

/** One line of state, rather than whichever callback fired last. */
const message = (stats: Tiles3dMemberStats | null): string => {
  if (failure) return `${place.label}: ${failure}`;
  if (!stats || stats.sourceState !== "ready")
    return `Resolving ${place.label}…`;
  return `${place.label} — ${externalTilesets} external tilesets resolved, ${stats.selectedTiles} tiles selected`;
};

host.onFrame((view) => registration?.setCamera(view));

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
  if (next) loadPlace(next, true);
});

radiusInput.addEventListener("input", () => {
  radiusValue.textContent = `${(radiusMeters() / 1000).toFixed(1)} km`;
});
radiusInput.addEventListener("change", () => loadPlace(place, false));

sseInput.addEventListener("input", () => {
  sseValue.textContent = `${maximumSse()} px`;
  if (!config) return;
  config = { ...config, maximumScreenSpaceErrorPx: maximumSse() };
  registration?.setConfig(config);
});

resetViewButton.addEventListener("click", () =>
  host.lookAt([0, 0, 0], place.viewDistance),
);

const meshRow = (stats: Tiles3dMemberStats): MemberRow => ({
  id: "buildings",
  label: place.label,
  stats,
});

const updateDiagnostics = (): void => {
  const stats = memberStats();
  messageOutput.textContent = message(stats);
  renderDiagnostics(statsElement, {
    coordinator: host.coordinator.stats(),
    rows: stats ? [meshRow(stats)] : [],
    frameMs: host.lastFrameMs(),
    rendererName: host.rendererName,
    textureFormat: host.textureCapabilities.compressedFormats[0] ?? "rgba",
  });
};

// The panel runs on its own clock: driving it from the render loop would leave
// it a frame stale exactly when the view stops painting, and would charge its
// cost to the budget it is reporting on.
updateDiagnostics();
setInterval(updateDiagnostics, 250);

const initial = new URLSearchParams(window.location.search).get("place");
const requested = PLACES.find((candidate) =>
  candidate.label.startsWith(initial ?? ""),
);
placeSelect.value = (requested ?? PLACES[0]!).label;
radiusValue.textContent = `${(radiusMeters() / 1000).toFixed(1)} km`;
sseValue.textContent = `${maximumSse()} px`;
loadPlace(requested ?? PLACES[0]!, true);

// Driving handles for browser checks: they move the camera and read the panel
// the way a user would, rather than reaching past the page into the library.
Object.assign(window, {
  meshExample: {
    stats: () => ({
      coordinator: host.coordinator.stats(),
      member: memberStats(),
      frameMs: host.lastFrameMs(),
      place: place.label,
      paints: host.paintCount(),
    }),
    view: () => host.cameraView(),
    lookAt: (distanceMeters: number) => host.lookAt([0, 0, 0], distanceMeters),
    load: (label: string) => {
      const next = PLACES.find((candidate) =>
        candidate.label.startsWith(label),
      );
      if (next) loadPlace(next, true);
    },
  },
});
