# vtk-pointcloud-lod

Octree LOD point-cloud streaming for [vtk.js](https://kitware.github.io/vtk-js/).

**Status: early development.** APIs are provisional and will change without
notice. Not published to npm yet — install straight from GitHub (see below).

## What it is

A standalone library that streams massive point clouds into vtk.js scenes by
walking an octree level-of-detail hierarchy: only the tiles that matter for
the current camera are fetched, decoded, and kept resident.

## Install

Install from GitHub. The `prepare` script builds `dist/` on install, so a
git reference works with no extra steps:

```bash
npm install github:PaulHax/vtk-pointcloud-lod
# or pin a commit for reproducible builds:
npm install github:PaulHax/vtk-pointcloud-lod#<commit-sha>
```

`@kitware/vtk.js` (`>=36`) is an **optional** peer dependency: it is only
needed for the renderer adapter. The tile sources, LOD controller, and camera
math have no vtk.js dependency and run standalone (e.g. in a worker or a test).

## Usage

```js
import {
  createCopcTileSource,
  createLodController,
  createRendererAdapter,
} from "vtk-pointcloud-lod";

// A coalescing render request the host owns (must not render synchronously
// more than once per event-loop turn).
const scheduleRender = () => renderWindow.render();

// 1. A tile source. COPC reads a static .copc.laz over HTTP Range requests,
//    so any static file host works with no tile server:
const source = await createCopcTileSource({ source: "https://host/cloud.copc.laz" });

// 2. A renderer adapter turns tile batches into vtk.js actors in your renderer.
const adapter = createRendererAdapter({ renderer, scheduleRender });

// 3. The controller decides which octree nodes are resident for the camera
//    and streams batches to the adapter.
const controller = createLodController({
  source,
  onTiles: (batch) => adapter.applyBatch(batch),
  scheduleRender,
});

// Feed the controller a plain camera description on every camera change:
controller.setCamera({ viewProj, position, fovY, viewportHeight });

// Tear down when done (both are idempotent):
controller.dispose();
adapter.dispose();
```

For servers that reproject or transform points per tile, use
`createHttpTileSource({ endpoint, metadata })` instead of the COPC source; it
speaks a compact binary tile protocol (`PCT1`).

## Architecture

```
TileSource  ──▶  LOD controller  ──▶  renderer adapter
(hierarchy +     (frustum cull,       (vtkPolyData +
 tile payloads)   screen-space         vtkPointGaussianMapper
                  error, point         per tile)
                  budget, LRU)
```

- **TileSource** — abstract interface over an octree tile store: dataset
  metadata, hierarchy pages, and per-node point payloads. Two
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
- **Renderer adapter** (`createRendererAdapter`) — turns tile batches into
  vtk.js actors, one `vtkPolyData` + `vtkPointGaussianMapper` per tile
  (one gl.POINTS vertex per point, no cell topology), with an anchor base
  matrix composed onto each tile's origin translation. This is the only
  module importing `@kitware/vtk.js`; the `vtkPointGaussianMapper` it uses
  is being upstreamed to vtk.js.

Camera math (`frustumPlanes`, `screenSpaceError`) is pure and
renderer-agnostic: the controller takes a view-projection matrix and camera
parameters as plain arrays, and requests renders only through an injected
coalescing `scheduleRender` callback — the host owns render pacing.

## License

[MIT](./LICENSE)
