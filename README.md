# pointcloud-lod

Octree LOD point-cloud streaming for [vtk.js](https://kitware.github.io/vtk-js/).

**Status: early development.** APIs are provisional and will change without
notice.

## What it is

A standalone library that streams massive point clouds into vtk.js scenes by
walking an octree level-of-detail hierarchy: only the tiles that matter for
the current camera are fetched, decoded, and submitted to the renderer. Decoded
CPU payloads, renderer/GPU residency, and active draw are separate lifecycle
states; hiding a controller releases renderer resources while its byte-bounded
decoded cache remains reusable.

## Install

```bash
npm install pointcloud-lod
```

To track unreleased work, install from GitHub instead — the `prepare` script
builds `dist/` on install, so a git reference works with no extra steps:

```bash
npm install github:PaulHax/pointcloud-lod#<commit-sha>
```

### Requirements

The package has two entry points, and only one of them needs vtk.js:

- **`pointcloud-lod`** — tile sources, LOD controller, and camera math. No
  vtk.js dependency; runs standalone (e.g. in a worker or a test).
- **`pointcloud-lod/vtk`** — the renderer adapter. This requires
  `vtkPointGaussianMapper`, which is **not present in any released
  `@kitware/vtk.js`**. Until it lands upstream you must build vtk.js from
  [the fork](https://github.com/PaulHax/vtk-js) at commit `804faaf46b`.

vtk.js is deliberately not declared as a peer dependency: no published version
satisfies the adapter, so any semver range would be false.

## Usage

```js
import { createCopcTileSource, createLodController } from "pointcloud-lod";
import { createRendererAdapter } from "pointcloud-lod/vtk";

// A coalescing render request the host owns (must not render synchronously
// more than once per event-loop turn).
const scheduleRender = () => renderWindow.render();

// 1. A tile source. COPC reads a static .copc.laz over HTTP Range requests,
//    so any static file host works with no tile server:
const source = await createCopcTileSource({
  source: "https://host/cloud.copc.laz",
});

// 2. A renderer adapter turns tile batches into vtk.js actors in your renderer.
const adapter = createRendererAdapter({ renderer, scheduleRender });

// 3. The controller decides which octree nodes are resident for the camera
//    and streams batches to the adapter.
const controller = createLodController({
  source,
  onTiles: (batch) => adapter.applyBatch(batch),
  onPointDiameterCssPx: (diameter) => adapter.setPointDiameterCssPx(diameter),
  presentation: { mode: "auto", userScale: 1 },
  scheduleRender,
});

// Feed the controller a plain camera description on every camera change:
controller.setCamera({ viewProj, position, fovY, viewportHeightCssPx });
adapter.setDevicePixelRatio(window.devicePixelRatio);

// Tear down when done (both are idempotent):
controller.dispose();
adapter.dispose();
```

For servers that reproject or transform points per tile, use
`createHttpTileSource({ endpoint, metadata })` instead of the COPC source; it
speaks a compact binary tile protocol (`PCT1`).

### Adaptive quality

A controller on its own draws to whatever fixed budget you set. To adapt
quality to what the machine can actually paint, add a view governor: one per
view, shared by every controller drawing into that view. It splits one budget
across its members from measured host-frame timings, so several clouds in a
view compete for a single frame-time target instead of each chasing its own.

```js
import { createViewGovernor } from "pointcloud-lod";

const governor = createViewGovernor();

// Register each controller in the view. The governor pushes budgets in.
const member = governor.register({
  setPointBudget: (points) => controller.setPointBudget(points),
  active: true,
});

// Report every completed host frame, including non-VTK work:
governor.recordHostFrame({ hostFrameMs, vtkFrameMs });

// Bracket camera interaction so quality relaxes while moving and settles
// after release. Controllers take the same pair:
governor.beginInteraction();
controller.beginInteraction();
// ...camera moves...
governor.endInteraction();
controller.endInteraction();

// Drop a controller out of the split without disposing it:
member.update({ active: false });
member.release();
governor.dispose();
```

## Architecture

```
TileSource  ──▶  LOD controller  ──▶  renderer adapter
(hierarchy +     (frustum cull,       (vtkPolyData +
 tile payloads)   screen-space         vtkPointGaussianMapper
                  error, point         per tile)
                  budget, LRU)
```

- **TileSource** — abstract interface over an octree tile store: dataset
  metadata, hierarchy pages with conservative render-space bounds and effective
  spacing for every node, and per-node point payloads. Two
  implementations ship:
  - `createCopcTileSource` reads [COPC](https://copc.io/) files directly
    over HTTP Range requests (via the `copc` package), so any static file
    host works with no tile server at all;
  - `createHttpTileSource` speaks a small revision-scoped hierarchy/tile
    HTTP protocol with a compact binary tile format (`PCT1`: Float64 tile
    origin + tile-local Float32 positions + Uint8 RGB) for servers that
    reproject or transform points per tile.
- **LOD controller** (`createLodController`) — decides which nodes are
  resident: frustum culling, screen-space error priority, a visible-point
  budget with a parent-closed selection invariant (COPC hierarchies are
  additive, so that invariant alone guarantees hole-free refinement),
  coarse-first fetching with bounded concurrency and cancellation,
  byte-budgeted LRU caching of deselected tiles, and batched delivery.
  Fixed presentation keeps one CSS-pixel diameter; Auto presentation derives
  one damped settled diameter from the p75 projected spacing of the ready
  terminal coverage frontier and uses two CSS pixels during interaction.
- **Renderer adapter** (`createRendererAdapter`) — turns tile batches into
  vtk.js actors, one `vtkPolyData` + `vtkPointGaussianMapper` per tile
  (one gl.POINTS vertex per point, no cell topology), with an anchor base
  matrix composed onto each tile's origin translation. This is the only
  module importing `@kitware/vtk.js`, which is why it ships under a separate
  `pointcloud-lod/vtk` entry point. The `vtkPointGaussianMapper` it uses is
  not yet in a released vtk.js — see [Requirements](#requirements).
  CSS diameter stays separate from framebuffer density: the adapter applies
  device pixel ratio through the mapper at the final rendering boundary.
- **View governor** (`createViewGovernor`) — optional, one per view. Owns the
  point budget for every controller registered to that view and adapts it from
  measured host-frame timings, holding VTK to a fraction of the frame and
  guaranteeing each active member a floor so one busy cloud cannot starve the
  rest. Controllers have no adaptive loop of their own; this is the only route
  to adaptive quality.

Camera math (`frustumPlanes`, `screenSpaceError`) is pure and
renderer-agnostic: the controller takes a view-projection matrix and camera
parameters as plain arrays, and requests renders only through an injected
coalescing `scheduleRender` callback — the host owns render pacing.

## License

[MIT](./LICENSE)
