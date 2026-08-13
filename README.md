# pointcloud-lod

Governed point-cloud and 3D Tiles streaming for
[vtk.js](https://kitware.github.io/vtk-js/).

**Status: early development.** APIs are provisional and will change without
notice.

## What it is

A standalone library that streams large point clouds and explicit 3D Tiles 1.1
trees into vtk.js scenes. A shared scene coordinator governs frame time, memory,
submission work, interaction, and cross-member quality while each format owns
its traversal and renderer adapter. Only content relevant to the current camera
is fetched, decoded, and submitted. Decoded CPU payloads, renderer/GPU
residency, and active draw are separate lifecycle states.

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

The package has three entry points, and only one of them needs vtk.js:

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

- **`pointcloud-lod/copc-worker`** — the optional COPC worker endpoint. Pair it
  with `createCopcWorkerTileSource` to keep range reads, LAZ decoding, and
  point extraction off the interaction/render thread.

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
  onDrawPlan: (plan) => adapter.applyDrawPlan(plan),
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
  viewportWidthCssPx,
  viewportHeightCssPx,
});
adapter.setDevicePixelRatio(window.devicePixelRatio);

// Tear down when done. Both are idempotent, and BOTH are required: the
// controller's teardown hands its tiles back as removals, which the adapter
// answers by pooling those actors for reuse, so disposing only the controller
// leaves GPU resources alive with nothing left to reclaim them.
controller.dispose();
adapter.dispose();
source.dispose?.();
```

For servers that reproject or transform points per tile, use
`createHttpTileSource({ endpoint, metadata })` instead of the COPC source; it
speaks a compact binary tile protocol (`PCT1`).

In an interactive browser, put the COPC source behind one module worker. The
worker owns its laz-perf instance, accepts concurrent range reads, and transfers
decoded position/color buffers back without copying. For example, make the
worker entry:

```js
// copc.worker.js
import { serveCopcTileSourceWorker } from "pointcloud-lod/copc-worker";

serveCopcTileSourceWorker();
```

Then create the source from the main thread (serve `laz-perf.wasm` at the URL
you provide):

```js
import { createCopcWorkerTileSource } from "pointcloud-lod";

const source = await createCopcWorkerTileSource({
  source: "https://host/cloud.copc.laz", // Blob/File also works
  createWorker: () =>
    new Worker(new URL("./copc.worker.js", import.meta.url), {
      type: "module",
    }),
  lazPerfWasmUrl: new URL("./laz-perf.wasm", document.baseURI).href,
});
```

The source owns that worker; call `source.dispose()` after disposing its
controller. Cancellation still settles only when the worker's physical read or
decode ends, so controller concurrency remains bounded.

### Application flows and primary controls

There are two loops in a typical integration:

- the required camera → selection → streaming → render loop;
- an optional frame-time loop that automatically changes the point budget.

Point presentation is a separate choice: it can follow the streamed density
automatically or stay at a fixed size.

#### Camera, selection, streaming, and render

```text
camera or budget changes
  → select a parent-closed set of visible octree nodes
  → update Auto point diameter for the selected terminal density
  → fetch missing hierarchy pages and tile payloads
  → make each decoded tile resident
  → recompute the ready terminal coverage frontier
  → submit one batched renderer delta
  → paint one coalesced frame
```

`setCamera()` is the normal entry point. It should receive the actual camera
used for rendering, including the CSS viewport height. The first change
selects immediately; repeated changes are rate-limited by `selectionDelayMs`
and the last one still gets a trailing selection.

Selection is parent-closed: a child adds points without replacing its parent.
While a selected child is still loading, missing from the hierarchy, or
excluded by the point budget, the closest ready ancestor continues to cover
that region. At a same-level point-budget boundary, an already selected node
keeps an eight-CSS-pixel-equivalent centre-cone advantage. This hysteresis
prevents nearly tied tiles from trading places under tiny camera changes while
allowing culling,
refinement-cutoff transitions, and materially better candidates to take effect.
Candidates are visited breadth-first and in concentric 3D cones around the
camera's centre ray, with projected detail breaking ties. When the next
candidate does not fit, that pass stops adding optional detail instead of
spending the remainder on either a cheaper horizon tile or a deeper foreground
child. The next budget increase therefore extends the current band before
descending another level.

The **ready terminal coverage frontier** is the set of ready nodes currently
responsible for the ends of those visible branches. Some may be fine children
while another is still a coarse parent, so its statistics describe exactly
what is on screen while payloads stream.

Auto presentation follows the corresponding **selected** terminal density. It
projects each selected terminal's world-space point spacing into CSS pixels and
uses the largest spacing as the common point diameter, after applying the
configured bounds and `userScale`. Hierarchy- and budget-blocked branches retain
their closest sampled parent. Tile readiness does not change the diameter: the
selection adopts its completed density once, then newly decoded tiles add
detail without shrinking every point already on screen.

Tile arrivals and diameter changes can both request a render. `scheduleRender`
must therefore coalesce requests, normally with `requestAnimationFrame`, and
must not paint synchronously from either callback. That lets a newly resident
tile, the frontier-derived diameter, and the renderer batch land before the
same frame.

#### View-quality flow: adaptive or fixed

With a `ViewGovernor`, the host closes a frame-time feedback loop:

```text
paint frame
  → recordTransientFrame({ hostFrameMs, vtkFrameMs })
  → recordCapacitySample(clean asynchronous GPU result)
  → governor adjusts one normalized view-quality fraction
  → coordinator demand-caps and water-fills that fraction across members
  → each point member maps its allocation to selection + draw density
  → moving changes thin existing VBO prefixes; settled changes may reselect
  → needsFrame() says whether another measurement is useful
```

The governor is optional. It controls a format-neutral fraction in `[0.05, 1]`
and never reads the renderer or schedules a frame on its own. The streamed
scene coordinator reports completed frames, camera motion, and aggregate work,
then water-fills quality by projected importance and member demand. Memory is
a separate byte allocation from the page-wide pool. See
[Adaptive quality](#adaptive-quality) for the complete wiring.

For one cloud, a fixed budget can still be set directly:

```js
controller.setPointBudget(1_500_000);
```

The effective budget is still capped by the controller's memory-derived point
ceiling. A higher budget permits more selected detail; it does not require
that every dataset contain that many useful visible points.

#### Point-presentation flow: Auto or Fixed

Auto presentation follows the density of the selected coverage:

```js
controller.setPresentation({
  mode: "auto",
  userScale: 1,
  minDiameterCssPx: 1.25,
  maxDiameterCssPx: 4,
});
```

- `userScale` changes overlap without changing selection: 1 matches the full
  estimated point spacing, above 1 overlaps footprints, and below 1 leaves
  separation between them.
- `minDiameterCssPx` prevents very dense detail from becoming sub-pixel.
- `maxDiameterCssPx` limits how large coarse points may become.

For a constant point footprint, use Fixed presentation. No frontier
measurement changes it:

```js
controller.setPresentation({ mode: "fixed", diameterCssPx: 2 });
```

Keep CSS size separate from framebuffer density. Report display-density
changes through `adapter.setDevicePixelRatio(window.devicePixelRatio)`.

#### Direct controls and lifecycle

These methods are useful when application policy lives outside the library:

| Control                                               | Effect                                                                                                                                                                     |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `controller.setPointBudget(points)`                   | Set the visible-point target directly. Use this instead of a governor for a fixed or externally managed budget.                                                            |
| `controller.setDensityFraction(fraction)`             | Redistribute that fraction of selected points into parent-closed, importance-ranked tile prefixes without changing selection, I/O, actors, or VBO contents.                |
| `controller.setPresentation(...)`                     | Switch live between Auto and Fixed point presentation.                                                                                                                     |
| `controller.setRefinementCutoffPx(pixels)`            | Stop descending when a node's projected spacing is below this threshold. Lower values allow finer traversal; the point and memory budgets still apply.                     |
| `controller.refresh()`                                | Force immediate reselection against the current camera, useful after external state changes that do not produce a new camera value.                                        |
| `controller.beginInteraction()` / `endInteraction()`  | Mark explicit camera interaction for controller diagnostics and immediate selection. Calls may be nested.                                                                  |
| `governor.beginMotion(kind)`                          | Hold the adaptive budget in its moving-camera regime. Use `"explicit"` for announced gestures and `"inferred"` for playback or programmatic motion detected by the host.   |
| `governor.recordCameraChange()`                       | Report a rendered-camera change directly. The governor owns the trailing camera-stability timer. Prefer `noteRenderedCameras()`, which infers this.                        |
| `governor.noteRenderedCameras(views, scheduleRender)` | Hand the governor the camera each view actually rendered, keyed by anything stable, and let it classify the motion. Covers playback, scrubbing and programmatic motion.    |
| `governor.resetMotionBaselines()`                     | Forget those baselines when a view stops feeding cameras, so its next one is a baseline again rather than every metre travelled since.                                     |
| `governor.setOptions(options)`                        | Re-target a running governor. Memberships, motion references and camera stability survive; only the adaptive tracks restart. An unusable value throws and changes nothing. |
| `adapter.setPointDiameterCssPx(pixels)`               | Set point size outside the controller. Omit `onPointDiameterCssPx` when the application owns this value so two policies do not compete.                                    |
| `adapter.setDevicePixelRatio(ratio)`                  | Update CSS-to-framebuffer scaling without changing selection or the CSS point diameter.                                                                                    |
| `adapter.setDensityFraction(fraction)`                | Apply progressive prefix drawing directly when policy lives outside the controller.                                                                                        |
| `adapter.setResourceCeilingBytes(bytes)`              | Bound GPU resources retained in the adapter's actor-reuse pool. A host can feed it the controller's current `memoryBudgetBytes`.                                           |
| `adapter.setVisible(false)`                           | Hide drawing only. Actors, GPU resources, selection, and streaming remain live for an immediate show.                                                                      |
| `controller.setActive(false)`                         | Stop selection and tile fetches, emit removals that move actors into the bounded adapter pool, and retain decoded payloads in the bounded CPU cache.                       |
| `controller.setSource(source)`                        | Replace the dataset or revision, dropping old hierarchy, residency, cache, and pending results before bootstrapping the new source.                                        |
| `controller.governorInputs()`                         | Point-specific demand, importance, memory ceiling, and physical-operation inputs used by the point member. Prefer it to `stats()` on a per-frame path.                     |
| `controller.setModelMatrix(matrix)`                   | The transform the tiles draw under, so cameras and picks are given in world coordinates. Must be a similarity; an unchanged matrix is a no-op, so forward it every pass.   |
| `adapter.setBaseMatrix(matrix)`                       | Apply or update the registration transform without rebuilding tile payloads. Pair it with `controller.setModelMatrix()` so selection and drawing agree.                    |

The main dials and their tradeoffs are:

| Dial                              | Primary effect                                           | Tune when                                                      |
| --------------------------------- | -------------------------------------------------------- | -------------------------------------------------------------- |
| `pointBudget` or governor targets | Visible detail and render cost                           | First performance/quality control                              |
| `presentation`                    | Point coverage and apparent density, not selected detail | Points look porous or overly solid                             |
| `refinementCutoffPx`              | How far hierarchy traversal is allowed to descend        | Storage has detail finer than the view needs                   |
| `memory`                          | GPU-resident byte ceiling, converted to a point ceiling  | Multiple clouds compete for GPU memory                         |
| `cacheBytes`                      | Decoded CPU payloads retained for fast reselection       | Revisiting views causes too much decoding or uses too much RAM |
| `fetchConcurrency`                | Parallel tile fetch/decode work                          | The source is under-filled or decoding saturates the client    |
| `hierarchyConcurrency`            | Parallel hierarchy-page work                             | Deep traversal stalls waiting for hierarchy                    |
| `selectionDelayMs`                | Reselection rate and recent-read cancellation grace      | Camera motion causes selection or cancel/refetch churn         |
| `interactionSettleMs`             | Governor delay for declaring the rendered camera stable  | Quality rises too early or too late after motion               |

Visibility, activity, and disposal answer different questions:

```text
setVisible(false)  → draw nothing; keep actors and streaming
setActive(false)   → stop tile fetches; pool submitted actors and cache payloads
dispose both       → release controller state and every adapter-owned actor
```

Always dispose both the controller and adapter when the cloud is permanently
removed.

### vtk.js examples

The runnable app in [`examples/vtk`](./examples/vtk/) contains four examples.
Two use `createLodController` directly, which is all a single point cloud
needs:

- **Simple** (`/simple/`) is the integration starting point: one worker-backed
  COPC source, one fixed-budget controller, one renderer adapter, and the
  default vtk.js camera interaction. It intentionally has no governor,
  performance chart, telemetry recorder, diagnostic table, or browser-test
  API.
- **Complete** (`/`) demonstrates adaptive quality, fixed/adaptive presentation,
  custom point-cloud camera interaction, local telemetry, and the diagnostic
  hooks exercised by the browser suite.

Two more use the member API — `createStreamedSceneCoordinator` with
`createTiles3dMember` and `createPointCloudMember` — which is what a scene of
more than one streamed dataset needs, and are the reference for it:

- **Mesh** (`/mesh/`) streams 3D Tiles: every building in the Netherlands at
  LoD2.2, live from [3DBAG](https://docs.3dbag.nl/) with no key.
- **Combined** (`/combined/`) puts AHN4 lidar and 3DBAG buildings in one scene
  sharing one memory pool, one submission scheduler and one view governor. Its
  panel shows each member's claimed share of the view beside the quality the
  governor gave it back, which is the only place that negotiation is visible.

Both meet in a local ENU frame whose Z is NAP: the mesh arrives in ECEF and is
placed by `tilesetToScene`, the lidar arrives in RD New (EPSG:28992) and is placed
by a model matrix, and `examples/vtk/scene/places.ts` derives both from one
origin. Public data is rarely shaped the way a strict reader wants, and
`examples/vtk/scene/` carries what that costs at the host boundary: resolving
3DBAG's external tilesets on the `fetchTileset` seam, rotating bounding volumes
into the frame its Y-up glTF content uses, and repairing colliding meshopt
fallback-buffer offsets on the way past `fetchContent`. Each is commented where
it lives.

Both load either a local `.copc.laz` file (read through `Blob.slice()` in a
source worker) or a COPC URL (HTTP Range from that worker), frame it
automatically, and stream it through a controller and renderer adapter. The
complete example also wires one view governor for the view.

The complete example defaults to Auto presentation at `0.5×`, using half the
estimated point spacing as its diameter. Its sidebar can switch to a constant
Fixed CSS-pixel diameter when comparing presentation policies.

Its controls are all runtime, so one build loads any dataset:

- **Point budget** — `Adaptive` gives the number to a `ViewGovernor`; `Fixed`
  gives it straight to `setPointBudget`.
- **Point size** — `Fixed` directly controls the CSS-pixel diameter; `Auto`
  controls the density-derived diameter's scale and defaults to `0.5×`.
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
GPU-residency and frame-time statistics. The current moving, settling, or
settled status is the first row of that budget chain.

The page holds an explicit motion reference for the interactor's gestures and
reports rendered-camera changes to the governor, which owns the single trailing
stability timer. It uses displayed cadence for immediate pressure and
asynchronous WebGL timer queries for lasting capacity. Controller work and
renderer resource revisions reject frames crossed by streaming, decode, batch,
or first-upload work. Clean displayed cadence is the fallback when hardware GPU
timing is unavailable. The page repaints while `needsFrame()` is true.
`window.pointCloudExample` exposes `stats()`, `setProjection()`,
`setBudgetMode()` and `dispose()` for browser tests.

Install a vtk.js build containing `vtkPointGaussianMapper` as described in
[Requirements](#requirements), then run:

```bash
npm run example
```

Vite serves the complete example at the printed root URL, and the others at
`/simple/`, `/mesh/` and `/combined/`. The streamed-scene pages decode in a
classic worker built by `npm run build`, so build the package before serving
them.

If that vtk.js build is outside this package's `node_modules`, point the example
at its ESM package directory:

```bash
VTK_JS_DIR=/path/to/vtk-js/dist/esm npm run example
```

A `?place=` query parameter opens the mesh and combined examples on a named
place. A `?url=` query parameter loads a cloud on startup in either point-cloud
example. On the
complete example, add `&telemetry=1` to begin a local performance trace before
that source opens.

The **Local telemetry** controls record only in the current browser tab. Start,
stop, clear, and download produce a bounded JSON trace; nothing is uploaded.
The trace identifies the WebGL vendor and renderer (including an explicit
software-renderer flag), viewport and device-pixel ratio, camera and control
state, controller/adapter/governor statistics, rAF cadence, synchronous vtk
time, long tasks, source/hierarchy/tile work, and renderer-batch handoff.

Every source or renderer work transition increments a revision. A frame is
`clean` only when that revision stayed unchanged across the complete interval
since the previous presentation, no work remained pending, and the controller
reported no queued, physical, or undecoded selected-tile work. This preserves
the evidence needed to distinguish steady rendering capacity from a frame that
overlapped streaming. The same controls are scriptable through
`window.pointCloudExample.telemetry` (`start`, `stop`, `clear`, `environment`,
`summary`, `trace`, `download`, and `mark`).

Browser checks use SwiftShader by default and must not be treated as GPU
benchmarks. On a WSLg machine with GPU forwarding, run the telemetry check in a
headed hardware browser with:

```bash
POINTCLOUD_LOD_BROWSER_GPU=1 npm run test:browser -- test/browser/telemetry.spec.ts
```

That mode asserts the reported renderer is not software, so a silent fallback
cannot pass as a hardware measurement.

To capture a repeatable initial load, picked ROI, rapid wheel zoom, long close
orbit sweep, near-horizontal local pan, and return to overview, supply a cloud URL
and optional artifact path:

```bash
POINTCLOUD_LOD_TELEMETRY_URL='https://example.test/cloud.copc.laz' \
POINTCLOUD_LOD_TELEMETRY_OUT='artifacts/telemetry/cloud.json' \
VTK_JS_DIR=/path/to/vtk-js/dist/esm \
npm run telemetry:capture
```

The capture starts before the source opens and adds phase markers to the JSON,
including the tight and restored broad views after each has settled.
By default, the first run caches the complete COPC under
`artifacts/telemetry/cache/`; later runs replay real range requests from that
local file with deterministic 40 ms request latency and an 80 Mbps per-request
transfer rate. This removes changing WAN conditions while preserving range
traffic, concurrency, cancellation, decoding, and rendering. Adjust the replay
with `POINTCLOUD_LOD_TELEMETRY_LATENCY_MS` and
`POINTCLOUD_LOD_TELEMETRY_MBPS`, force a new download with
`POINTCLOUD_LOD_TELEMETRY_REFRESH=1`, or select the original remote behavior
with `POINTCLOUD_LOD_TELEMETRY_NETWORK=live`.

The same scenario can compare the optional governor with the conventional
fixed-budget policy:

```bash
POINTCLOUD_LOD_TELEMETRY_BUDGET_MODE=fixed \
POINTCLOUD_LOD_TELEMETRY_POINT_BUDGET=5000000 \
POINTCLOUD_LOD_TELEMETRY_INTERACTION_DENSITY=0.6 \
npm run telemetry:capture
```

The defaults are `adaptive`, 2,000,000 points, and full interaction density.
The point and interaction-density values affect only fixed mode.

It rejects software rendering and checks that the trace is structurally usable;
it deliberately does not turn machine-specific frame times into pass/fail
thresholds. If no output path is supplied, the timestamped trace is written
under `artifacts/telemetry/`. It is tagged `*.perf.spec.ts`, runs only through
the opt-in `test:perf`/`telemetry:capture` commands, and is excluded from normal
unit and browser test runs. If no URL is supplied, the capture skips.

Remote URLs must allow cross-origin `Range` requests. Raw LAS/LAZ is not
streamable by the COPC source; convert it with a proven native COPC writer —
PDAL's `writers.copc` or `untwine` — which is also what server-side preparation
pipelines should run. No JavaScript/browser COPC writer is used or accepted
here, not even as a development dependency: the one evaluated preserved the
point count but silently rewrote RGB point format 7 as non-RGB format 6.

### Fixed quality

The controller is a complete fixed-budget viewer without a governor. Fixed
quality does not disable demand rendering, screen-space prioritization,
progressive point order, a worker-backed COPC source, decoded and GPU reuse,
memory ceilings, or per-tile draw plans; only frame-time learning is absent.

```js
controller.setPointBudget(5_000_000);
```

An application that wants a simple fixed interaction policy can lower only the
draw allowance while input is active. This keeps the fixed selection resident
and uses the same importance-ranked tile prefixes as adaptive mode:

```js
controller.beginInteraction();
controller.setDensityFraction(0.6);
// ...camera input...
controller.setDensityFraction(1);
controller.endInteraction();
```

This two-state policy is optional. A constant fixed budget is the smallest and
most predictable configuration.

### Adaptive quality

A controller on its own draws to whatever fixed point budget you set. Adaptive
views use one streamed-scene coordinator, which owns one normalized governor
and allocates its fraction across every format member in the view.

Two regimes, two targets. A moving camera is being steered, so it gets the
tighter **16 ms** target and trades drawn points for responsiveness; a settled
camera is being read, so it gets the looser **33 ms** target and spends the
extra frame time on detail. With the default 20% hysteresis the no-change bands
are 12.8-19.2 ms while moving and 26.4-39.6 ms while settled. Both tracks start
at quality **1** and never drop below **0.05**. Point `minBudget`/`maxBudget`
remain point-member clamps; they are not governor options. The point member
keeps the denser proven stationary selection while moving and expresses the
interaction allocation as parent-closed per-tile prefixes.

```js
import {
  createMemoryPool,
  createStreamedSceneCoordinator,
} from "pointcloud-lod";
import { createPointCloudMember } from "pointcloud-lod/vtk";

const coordinator = createStreamedSceneCoordinator({
  scheduleRender,
  memory: pageMemoryPool ?? createMemoryPool(),
});
const cloud = createPointCloudMember(
  coordinator.context(renderer, dpr),
  config,
);
coordinator.register(cloud, {
  id: "cloud-1",
  qualityManaged: config.adaptive,
  qualityTargets: config.adaptive ? config.adaptiveOptions : undefined,
});

// Hold the moving regime while the camera moves. References compose across
// sources and the regime ends only when the last one is released, so an
// inferred playback motion overlapping a pointer gesture behaves correctly.
coordinator.beginInteraction();
// ...camera moves...
coordinator.endInteraction();

// Motion the host cannot announce — playback, scrubbing, programmatic
// animation — is classified by the governor. Hand it the camera each view
// actually rendered, once per rendered frame, keyed by anything stable:
coordinator.noteRenderedCameras(new Map([[renderer, view]]));
// It owns the jitter epsilon, ignores the clip-z row a host rewrites when
// tiles arrive, holds one inferred motion reference for the whole burst, and
// releases it after the debounce — asking for one frame back, because the
// settled regime cannot refine quality it never measures. A scene change with
// an unchanged camera is not motion.

// When a view stops feeding cameras, drop its baseline. Otherwise the first
// camera after it returns is compared against one from before it left, and all
// the travel between reads as a gesture nobody made.
coordinator.recordHostFrame({ hostFrameMs, vtkFrameMs, gpuMs });
if (coordinator.needsFrame()) scheduleRender();
coordinator.dispose();
```

The point member reports retry-aware `workPending` through the coordinator
context's work callback. Fixed point members still register for byte and work
accounting but bypass normalized quality allocation.

`coordinator.stats()` reports the normalized view fraction, target-override
member, governor motion/samples, byte allocations, and each member's neutral
inputs. Point-specific diagnostics—including configured point ceiling and
effective full point ceiling—live on `PointCloudMember.stats()`.

## Picking

`controller.pickPoint(view, cursorXCssPx, cursorYCssPx)` resolves a cursor to
a point **on the cursor ray** at the support depth of the frontmost rendered
sample near the cursor — it never snaps to the vertex itself. The query runs
only over the tile set last submitted to the renderer (WYSIWYG: what is not
drawn cannot be picked), and answers

- `{ status: "hit", pointOnRay, distancePx }` for a supported depth,
- `{ status: "miss" }` for a valid sweep that found no sample within any
  pick bucket, or
- `null` when the query is unavailable (inactive/disposed controller, invalid
  camera or viewport, singular view-projection, non-finite cursor) — never to
  be conflated with a miss.

Candidates gather in escalating css-pixel buckets around the cursor; the
smallest non-empty bucket wins, then its minimum-depth point. The bucket radii
(`PICK_RADII_CSS_PX` = `DEFAULT_PICK_PIXEL_RADIUS` ×
`DEFAULT_PICK_PIXEL_RADIUS_MULTIPLIERS` = 10 × (1, 2, 10) css px) mirror
`DEFAULT_PICK_PIXEL_RADIUS` and `DEFAULT_PICK_PIXEL_RADIUS_MULTIPLIERS` in
telesculptor-web's `scene/ray_depth.py`; keep the two definitions in lockstep
so a pick one side accepts is never the other side's miss.

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
  - `createCopcWorkerTileSource` exposes the same contract while a dedicated
    worker performs COPC I/O, LAZ decoding, and extraction; final typed arrays
    transfer to the controller without a copy;
  - `createHttpTileSource` speaks a small revision-scoped hierarchy/tile
    HTTP protocol with a compact binary tile format (`PCT1`: Float64 tile
    origin + tile-local Float32 positions + Uint8 RGB) for servers that
    reproject or transform points per tile.
- **LOD controller** (`createLodController`) — decides which nodes are
  resident: frustum culling, 3D centre-ray cone priority with projected-detail
  tie-breaking, and a visible-point budget with a parent-closed selection
  invariant (COPC hierarchies are
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
  the diameter from the largest projected spacing on the selected terminal
  coverage frontier, clamped to the presentation's min/max and scaled by
  `userScale`. Progressive drawing reserves coarse parent coverage, then gives
  complete or partial nested prefixes to the highest projected-importance
  branches. Auto sizing follows each terminal's effective prefix density, and
  CPU picking examines the same per-tile plan the renderer draws. Two CSS
  pixels is the seed before any selected frontier exists. Tiles arrive already
  in progressive order — every source shuffles a tile once, deterministically,
  as it decodes it, which for the worker-backed source is off the main thread —
  keeping arbitrary COPC record order from biasing early prefixes.
- **Renderer adapter** (`createRendererAdapter`) — turns tile batches into
  vtk.js actors, one `vtkPolyData` + `vtkPointGaussianMapper` per tile
  (one gl.POINTS vertex per point, no cell topology), with an anchor base
  matrix composed onto each tile's origin translation. This is the only
  module importing `@kitware/vtk.js`, which is why it ships under a separate
  `pointcloud-lod/vtk` entry point. The `vtkPointGaussianMapper` it uses is
  not yet in a released vtk.js — see [Requirements](#requirements).
  CSS diameter stays separate from framebuffer density: the adapter applies
  device pixel ratio through the mapper at the final rendering boundary.
  `applyDrawPlan` changes each mapper's draw count while its complete
  point/color VBO remains resident. `setDensityFraction` remains available
  when an external policy deliberately wants uniform thinning.
  The controller owns residency and the adapter owns actors: `setVisible` is
  a draw switch that keeps every actor alive (batches still apply while
  hidden, so showing again restores exactly the submitted set), while
  releasing a hidden cloud's tiles is the controller's `setActive(false)`.
- **View governor** (`createViewGovernor`) — one format-neutral normalized
  fraction per adaptive view. It owns frame-time learning, motion, capacity
  eligibility, and nothing point- or memory-specific. The streamed-scene
  coordinator demand-caps and water-fills that fraction across members while
  the page memory pool allocates bytes separately.

Camera math (`frustumPlanes`, `nodeScreenSpaceError`) is pure and
renderer-agnostic: the controller takes a view-projection matrix and camera
parameters as plain arrays, and requests renders only through an injected
coalescing `scheduleRender` callback — the host owns render pacing. Both
projections are first class: a perspective view shrinks a node's projected
spacing with distance, a parallel one is set purely by `parallelScale`, so an
orthographic camera refines on zoom rather than on approach.

## License

[MIT](./LICENSE)
