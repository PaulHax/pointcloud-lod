# vtk-pointcloud-lod

Octree LOD point-cloud streaming for [vtk.js](https://kitware.github.io/vtk-js/).

**Status: early development.** APIs are provisional and will change without
notice. Nothing here is published yet.

## What it is

A standalone library that streams massive point clouds into vtk.js scenes by
walking an octree level-of-detail hierarchy: only the tiles that matter for
the current camera are fetched, decoded, and kept resident.

## Architecture

```
TileSource  ──▶  LOD controller  ──▶  renderer adapter
(hierarchy +     (frustum cull,       (vtkPolyData +
 tile payloads)   screen-space         vtkPointGaussianMapper
                  error, point         per tile)
                  budget, LRU)
```

- **TileSource** — abstract interface over an octree tile store: dataset
  metadata, hierarchy pages, and per-node point payloads. The reference
  implementation reads [COPC](https://copc.io/) files over HTTP Range
  requests (via the `copc` package), so any static file host works.
- **LOD controller** — pure logic deciding which nodes to show: frustum
  culling, screen-space error, a visible-point budget, progressive
  refinement with request cancellation, byte-budgeted LRU eviction, and
  hole-free parent/child replacement.
- **Renderer adapter** — turns loaded tiles into vtk.js actors, one
  `vtkPolyData` + `vtkPointGaussianMapper` per tile.

Currently only the pure-logic core is implemented (octree keys/bounds,
byte-budgeted LRU, point-budget node selection) plus the provisional
`TileSource` contract. The COPC source, LOD controller loop, and renderer
adapter come next.
