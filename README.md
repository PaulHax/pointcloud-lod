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
  [the fork](https://github.com/PaulHax/vtk-js) at the exact commit in
  [`vtkjs-fork.env`](./vtkjs-fork.env) — the one commit this package's CI
  builds and tests the example against:

  ```bash
  source ./vtkjs-fork.env
  git clone "$VTKJS_FORK_REPO" vtk-js && cd vtk-js
  git checkout "$VTKJS_FORK_COMMIT"
  npm ci && npm run build:esm
  ```

vtk.js is deliberately not declared as a peer dependency: no published version
satisfies the adapter, so any semver range would be false.

## Usage

```js
import "@kitware/vtk.js/Rendering/Profiles/Geometry";
import { createCopcTileSource, createLodController } from "pointcloud-lod";
import { createRendererAdapter } from "pointcloud-lod/vtk";

// A coalescing render request the host owns (must not render synchronously
// more than once per event-loop turn).
let frameQueued = false;
const scheduleRender = () => {
  if (frameQueued) return;
  frameQueued = true;
  requestAnimationFrame(() => {
    frameQueued = false;
    renderWindow.render();
  });
};

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

// Feed the controller a plain camera description on every camera change. The
// projection mode is explicit — an orthographic view carries `parallelScale`
// (the world-space half-height of the viewport) instead of `fovY`:
controller.setCamera({
  projection: "perspective",
  viewProj,
  position,
  fovY,
  viewportHeightCssPx,
});
adapter.setDevicePixelRatio(window.devicePixelRatio);

// Tear down when done. Both are idempotent, and BOTH are required: the
// controller's teardown hands its tiles back as removals, which the adapter
// answers by pooling those actors for reuse, so disposing only the controller
// leaves GPU resources alive with nothing left to reclaim them.
controller.dispose();
adapter.dispose();
```

For servers that reproject or transform points per tile, use
`createHttpTileSource({ endpoint, metadata })` instead of the COPC source; it
speaks a compact binary tile protocol (`PCT1`).

### vtk.js example

The runnable example in [`examples/vtk`](./examples/vtk/) loads either a local
`.copc.laz` file (read through `File.slice()` range reads) or a COPC URL (HTTP
Range), frames it automatically, and streams it through the same three pieces
an application wires: one controller, one renderer adapter, and one view
governor for the view.

Its controls are all runtime, so one build loads any dataset:

- **Point budget** — `Adaptive` gives the number to a `ViewGovernor`; `Fixed`
  gives it straight to `setPointBudget`.
- **Moving / settled target** — the two regimes' frame-time targets. Changing
  either replaces the governor, since the library fixes its options at
  construction.
- **Maximum budget** — the optional configured maximum. Blank leaves the
  memory-derived ceiling as the only upper bound.
- **Projection** — perspective or orthographic, preserving the world height the
  viewport covers, so the only thing a toggle changes is the law selection
  refines by (an orthographic zoom moves `parallelScale`, not the eye).

The panel answers "why is it drawing N points" without reading internal state:
regime and what is holding it, explicit versus inferred motion, target frame
time, percentile estimate, sample count, adaptive track budget, configured
maximum, memory ceiling, aggregate view budget, this cloud's share, the last
adjustment and its reason, the binding constraint, and the physical tile and
hierarchy work counts — next to the controller's selection, decoded-memory,
GPU-residency and frame-time statistics. A badge shows the current regime.

The page holds an explicit motion reference for the interactor's gestures and
infers the rest by comparing the camera it hands to LOD on every painted frame,
releasing that reference after 250 ms of stillness. It times every paint,
reports it through `recordHostFrame`, and repaints while `needsFrame()` is
true. `window.pointCloudExample` exposes `stats()`, `setProjection()`,
`setBudgetMode()` and `dispose()` for browser tests.

Install a vtk.js build containing `vtkPointGaussianMapper` as described in
[Requirements](#requirements), then run:

```bash
npm run example
```

If that vtk.js build is outside this package's `node_modules`, point the example
at its ESM package directory:

```bash
VTK_JS_DIR=/path/to/vtk-js/dist/esm npm run example
```

A `?url=` query parameter loads a cloud on startup.

Remote URLs must allow cross-origin `Range` requests. Raw LAS/LAZ is not
streamable by the COPC source; convert it with a proven native COPC writer —
PDAL's `writers.copc` or `untwine` — which is also what server-side preparation
pipelines should run. No JavaScript/browser COPC writer is used or accepted
here, not even as a development dependency: the one evaluated preserved the
point count but silently rewrote RGB point format 7 as non-RGB format 6.

### Adaptive quality

A controller on its own draws to whatever fixed budget you set. To adapt
quality to what the machine can actually paint, add a view governor: one per
view, shared by every controller drawing into that view. It splits one budget
across its members from measured host-frame timings, so several clouds in a
view compete for a single frame-time target instead of each chasing its own.

Two regimes, two targets. A moving camera is being steered, so it gets the
tighter **16 ms** target and trades points for responsiveness; a settled camera
is being read, so it gets the looser **33 ms** target and spends the extra
frame time on detail. With the default 20% hysteresis the no-change bands are
12.8-19.2 ms while moving and 26.4-39.6 ms while settled. Both tracks start at
**1,000,000** points (`initialBudget`) and never drop below **200,000**
(`minBudget`). Releasing the camera never changes the picture on its own: the
stationary track starts from exactly the density the moving regime just
sustained and refines from there.

```js
import { createViewGovernor } from "pointcloud-lod";

const governor = createViewGovernor();

// Register each controller in the view. The governor pushes budgets in.
const member = governor.register({
  id: "cloud-1",
  setPointBudget: (points) => controller.setPointBudget(points),
  active: true,
});

// Report every completed host frame, including non-VTK work, then ask whether
// the view still needs painting. The governor never schedules anything itself.
governor.recordHostFrame({ hostFrameMs, vtkFrameMs });
if (governor.needsFrame()) scheduleRender();

// Feed the split and the diagnostics from the controller's own statistics:
const stats = controller.stats();
member.update({
  projectedImportance: stats.selection.projectedImportance,
  memoryCeilingPoints: stats.memoryCeilingPoints,
  physicalTileOperations: stats.physicalTileOperations,
  physicalHierarchyOperations: stats.physicalHierarchyOperations,
});

// Hold the moving regime while the camera moves. References compose across
// sources and the regime ends only when the last one is released, so an
// inferred playback motion overlapping a pointer gesture behaves correctly.
const gesture = governor.beginMotion("explicit");
controller.beginInteraction();
// ...camera moves...
gesture.release();
controller.endInteraction();

// Motion the host cannot announce — playback, scrubbing, programmatic
// animation — is inferred by comparing the camera actually handed to LOD from
// frame to frame, holding one "inferred" reference for the whole burst and
// releasing it after a quiet debounce (250 ms in the shipped integrations).
// A scene change with an unchanged camera is not motion.

// Drop a controller out of the split without disposing it:
member.update({ active: false });
member.release();
governor.dispose();
```

`governor.stats()` explains any drawn point count without reading internal
state: the regime and what is holding it, the target frame time, the recent
percentile estimate and sample count, the adaptive track budget, the optional
configured maximum (`maxBudget`), the memory-derived ceiling, the aggregate
view budget and each member's share of it, the last adjustment's time,
direction and reason, and which of `adaptive | configured-maximum | memory |
inactive` is the binding constraint.

The effective budget is `min(adaptive track budget, configured maximum,
memory-derived ceiling)`, applied to the aggregate before the split so no
member's share is sized against memory another member owns. The memory ceiling
stays authoritative; `maxBudget` is an optional policy, diagnostics, and
hardware-safety bound, and omitting it leaves memory as the only ceiling.

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
  Tiles and hierarchy pages get separate ceilings (`fetchConcurrency`,
  default 6, and `hierarchyConcurrency`, default 4) because a page unblocks
  selection for a whole subtree and must not queue behind tile fetches that
  cannot be chosen correctly until it lands. Both ceilings count _physical_
  operations: cancellation is advisory wherever the underlying reader takes
  no signal, so an abandoned read keeps its slot until its promise settles
  and a look-away/look-back storm cannot multiply real I/O.
  Fixed presentation keeps one CSS-pixel diameter; Auto presentation derives
  the diameter from the p75 projected spacing of the ready terminal coverage
  frontier, scaled by `userScale` and clamped to the presentation's min/max,
  and emits it as soon as the frontier is remeasured. Two CSS pixels is the
  seed it starts from before any frontier exists.
- **Renderer adapter** (`createRendererAdapter`) — turns tile batches into
  vtk.js actors, one `vtkPolyData` + `vtkPointGaussianMapper` per tile
  (one gl.POINTS vertex per point, no cell topology), with an anchor base
  matrix composed onto each tile's origin translation. This is the only
  module importing `@kitware/vtk.js`, which is why it ships under a separate
  `pointcloud-lod/vtk` entry point. The `vtkPointGaussianMapper` it uses is
  not yet in a released vtk.js — see [Requirements](#requirements).
  CSS diameter stays separate from framebuffer density: the adapter applies
  device pixel ratio through the mapper at the final rendering boundary.
  The controller owns residency and the adapter owns actors: `setVisible` is
  a draw switch that keeps every actor alive (batches still apply while
  hidden, so showing again restores exactly the submitted set), while
  releasing a hidden cloud's tiles is the controller's `setActive(false)`.
- **View governor** (`createViewGovernor`) — optional, one per view. Owns the
  point budget for every controller registered to that view and adapts it from
  measured host-frame timings, holding VTK to a fraction of the frame and
  guaranteeing each active member a floor so one busy cloud cannot starve the
  rest. Controllers have no adaptive loop of their own; this is the only route
  to adaptive quality.

Camera math (`frustumPlanes`, `nodeScreenSpaceError`) is pure and
renderer-agnostic: the controller takes a view-projection matrix and camera
parameters as plain arrays, and requests renders only through an injected
coalescing `scheduleRender` callback — the host owns render pacing. Both
projections are first class: a perspective view shrinks a node's projected
spacing with distance, a parallel one is set purely by `parallelScale`, so an
orthographic camera refines on zoom rather than on approach.

## License

[MIT](./LICENSE)
