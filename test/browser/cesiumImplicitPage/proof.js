import {
  Cartesian3,
  Cesium3DTileset,
  Color,
  HeadingPitchRange,
  SceneTransforms,
  Viewer,
} from "/cesium/index.js";

const failures = [];
globalThis.addEventListener("error", (event) => failures.push(event.message));
globalThis.addEventListener("unhandledrejection", (event) =>
  failures.push(String(event.reason)),
);

const viewer = new Viewer("view", {
  animation: false,
  baseLayer: false,
  baseLayerPicker: false,
  fullscreenButton: false,
  geocoder: false,
  globe: false,
  homeButton: false,
  infoBox: false,
  navigationHelpButton: false,
  sceneModePicker: false,
  selectionIndicator: false,
  timeline: false,
});
viewer.scene.backgroundColor = Color.BLACK;
viewer.scene.highDynamicRange = false;
viewer.resolutionScale = 1;

try {
  const quadrantResponse = await fetch("/fixture/quadrants.json");
  const { quadrants } = await quadrantResponse.json();
  const tileset = await Cesium3DTileset.fromUrl("/fixture/tileset.json", {
    maximumScreenSpaceError: 1,
    skipLevelOfDetail: false,
  });
  tileset.tileFailed.addEventListener((error) =>
    failures.push(error?.message ?? String(error)),
  );
  viewer.scene.primitives.add(tileset);
  await viewer.zoomTo(tileset, new HeadingPitchRange(0, -Math.PI / 2, 0));

  await new Promise((resolve, reject) => {
    const timeout = setTimeout(
      () => reject(new Error("Cesium implicit fixture did not finish loading")),
      30_000,
    );
    const remove = tileset.allTilesLoaded.addEventListener(() => {
      if (
        tileset.statistics.numberOfPendingRequests !== 0 ||
        tileset.statistics.numberOfTilesWithContentReady < 5
      )
        return;
      remove();
      clearTimeout(timeout);
      requestAnimationFrame(() => requestAnimationFrame(resolve));
    });
  });

  const samples = quadrants.map((quadrant) => {
    const world = Cartesian3.fromDegrees(
      quadrant.sampleWgs84.longitude,
      quadrant.sampleWgs84.latitude,
      quadrant.sampleHeight,
    );
    const windowPosition = SceneTransforms.worldToWindowCoordinates(
      viewer.scene,
      world,
    );
    return {
      ...quadrant,
      x: windowPosition?.x,
      y: windowPosition?.y,
    };
  });
  globalThis.cesiumImplicitProof = {
    failures,
    samples,
    commands: tileset.statistics.numberOfCommands,
    readyTiles: tileset.statistics.numberOfTilesWithContentReady,
    totalTiles: tileset.statistics.numberOfTilesTotal,
  };
  document.documentElement.dataset.ready = "true";
} catch (error) {
  failures.push(error instanceof Error ? error.stack : String(error));
  globalThis.cesiumImplicitProof = { failures };
  document.documentElement.dataset.ready = "failed";
}
