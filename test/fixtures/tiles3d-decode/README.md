# 3D Tiles decode regression fixtures

These minimal glTF 2.0 assets are original test data generated for this
repository and are covered by the repository's MIT license. They isolate the
error path so the tests do not depend on network access.

- `y-up-triangle.gltf` pins the glTF Y-up to scene Z-up transform.
- `meshopt-required.gltf` declares `EXT_meshopt_compression` as required and
  pins the named unsupported-extension failure.
- `invalid-index.gltf` contains an index outside its POSITION accessor and
  pins vertex-index validation before renderer submission.
- `unlit-triangle.gltf` requires `KHR_materials_unlit` and pins the material
  signal used to disable vtk.js lighting.
- `orientation-test.glb` is the unmodified Khronos glTF Sample Assets
  `OrientationTest` model. Its node rotations and translations pin the real-
  world Y-up/profile path; provenance and CC BY 4.0 attribution are recorded in
  `orientation-test-LICENSE.txt`.

The fixtures are self-contained JSON with data-URI buffers so the tests never
depend on a network service.

The required-meshopt error path was also checked against three.js'
`facecap.glb` during implementation. It is not copied here because it does not
carry a standalone asset license record suitable for redistribution.
