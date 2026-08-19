# Recorded gestures

Each `.json` here is one captured camera gesture: the pointer and wheel events
a hand actually produced, the viewer size and device pixel ratio they were
aimed at, and the camera path they produced at capture time. The replay
benchmark drives the examples with these instead of a camera path written in
code, because a synthesised path is a straight drag at a constant rate and a
hand is not — it accelerates, overshoots, pauses mid-orbit, and aims at
whatever looked interesting, which is what makes streaming decisions hard.

## Capturing one

Start the examples and open the scene you want with `?record=1`:

```bash
npm run build && npm run example
```

The explorer holds any mix of datasets, and its URL carries the dataset list,
so the page plus its query string is the whole scene description:

- `http://localhost:5173/?record=1&url=<copc-url>` — a point cloud
- `http://localhost:5173/mesh/?record=1&place=Rotterdam` — 3D Tiles
- `http://localhost:5173/combined/?record=1&place=Rotterdam` — both

Everything that measures the page — the capture panel, telemetry, and the
automation surface the benchmark drives — lives in `examples/vtk/harness/` and
is a dynamic import the page only fetches when the URL asks for it, so an
ordinary session runs and downloads none of it. `?record=1` and `?telemetry=1`
both ask; `?harness=1` asks for the surface without starting either, which is
what the benchmark uses.

A capture panel appears at the top right. Let the scene finish loading, press
**Record** (or F9), fly the camera, press **Stop**, then **Download**. Drop the
file in this directory.

**Mark** (F10) drops a labelled boundary the analysis reports against, so a
recording that zooms in, orbits, and pulls back can be read as three phases
rather than one average. Separate recordings of separate intents beat one long
one.

Two things decide whether a recording can be replayed faithfully:

- **Do not resize the window while recording.** Every coordinate after a resize
  aims at a differently sized view. The panel warns when it happens, and the
  recording carries the resize as an event so a replay can refuse it.
- **The device pixel ratio is recorded and reproduced.** A capture on a
  fractional-scaling display replays at that scaling, which is the workload
  that display actually costs.

Capture on real hardware. The recorder itself is passive — it never calls
`preventDefault` and dispatches nothing — so recording does not change how the
gesture behaves, but a gesture performed against a software rasteriser is a
gesture performed against a stalling page.

## Replaying them

```bash
npm run bench:replay
```

Every recording here runs against every configuration, headed on the real GPU,
and writes one artifact per run under `artifacts/replay/`. Then:

```bash
npm run bench:analyze artifacts/replay
```

Nothing in the benchmark asserts a speed — it writes evidence and the analysis
compares it. What it does assert is that a run _is_ evidence: a hardware
renderer, a scene that converged, no page errors, and no request a replayed
network could not answer.

### Choosing what to compare

`POINTCLOUD_LOD_REPLAY_CONFIGS` points at a JSON file of setting sweeps.
Sections left out keep the page's own defaults, and each is applied through the
same control a hand would use:

```json
{
  "configs": [
    { "name": "baseline" },
    { "name": "sse-8", "tiles": { "screenSpaceErrorPx": 8 } },
    { "name": "patient", "quality": { "interactionTargetMs": 33 } },
    {
      "name": "fixed-2m",
      "points": { "budgetMode": "fixed", "fixedPointBudget": 2000000 }
    }
  ]
}
```

### Network

Streaming decisions exist to hide latency, so the network is part of the
measurement rather than something to eliminate:

- `POINTCLOUD_LOD_REPLAY_NETWORK=live` — the real internet, real conditions,
  and a different set of them every run.
- `=record` — the real internet, keeping every response under
  `POINTCLOUD_LOD_REPLAY_CACHE_DIR`.
- `=replay` (default) — serve what was recorded, at
  `POINTCLOUD_LOD_REPLAY_LATENCY_MS` and `POINTCLOUD_LOD_REPLAY_MBPS`. A
  request nothing recorded fails the run rather than quietly reaching for the
  network mid-measurement.

Record once, then sweep settings against the replayed network, and confirm
against a live run when the profile starts to look unlike the real thing.

### What a replayed gesture cannot reproduce exactly

CDP can say that a wheel turned, and where, but not how hard: Chrome pins
`wheelDelta` to a single notch on every injected wheel event whatever `deltaY`
it carries, and `normalizeWheel` — which is what the interactor reads — prefers
`wheelDelta`. So a hand's two-notch flick would replay as one, and because
dolly is multiplicative the shortfall compounds over the rest of the path.

The driver converts magnitude into count instead, dispatching one message per
notch, counted the way the interactor counts: relative to the turn that opened
the burst, with a 200 ms gap ending one. A trackpad's fractional turns still
cannot be dispatched individually, so the remainder is carried to the next
event rather than rounded away on each, which keeps the total right even where
no single event is.

Wheel deltas also cross CDP in device pixels while pointer coordinates cross it
in CSS pixels, so the driver scales them by the device pixel ratio. Both of
these are checked by `replayHarness.spec.ts` on a fractionally scaled page —
the mistakes are invisible at a device pixel ratio of 1, where the two units
coincide.

### Reading the fidelity numbers

Replay is real time: events go out on their recorded schedule and are not
waited on, because a real mouse keeps moving while a page is busy. The camera
therefore does not land in exactly the same place every run, and the analysis
reports two numbers that say whether runs are comparable at all.

- **Path drift** — how far the recorded camera track sits from the replayed
  one, ignoring timing. Near zero means both runs flew the same route and saw
  the same scenery. This is the one that decides comparability.
- **Input lateness** — how far behind schedule the driver fell issuing events.
  Large values mean the gesture was delivered more slowly than it was
  performed, so the view had longer to catch up than a hand would have given
  it.

A run that scores badly on time-aligned drift while scoring perfectly on path
drift is normal and fine: the interactor coalesces a gesture into whatever
frames it got, so a slower configuration arrives at each point slightly later
along exactly the same route. That lateness is the thing being measured.

### Without a GPU

`POINTCLOUD_LOD_REPLAY_ALLOW_SOFTWARE=1` runs the whole pipeline headless on a
software rasteriser. It is for checking that the benchmark still works, never
for measuring: every artifact it writes records `softwareRenderer: true` and
the analysis prints that in place of a headline number.
