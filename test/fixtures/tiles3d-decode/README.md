# 3D Tiles decode regression fixtures

These minimal glTF 2.0 assets are original test data generated for this
repository and are covered by the repository's MIT license. They isolate the
error path so the tests do not depend on network access.

- `y-up-triangle.gltf` pins the glTF Y-up to scene Z-up transform.
- `unsupported-extension.gltf` declares an extension this decoder does not
  implement and pins the named unsupported-extension failure.
- `meshopt-quad.glb` and `meshopt-quad-collision.glb` hold the same
  meshopt-compressed quad, the second with both compressed views claiming
  offset zero of one fallback buffer. Together they pin meshopt decompression
  and the fallback-offset repair that keeps the second from silently decoding a
  wrong mesh. Regenerate with `npm run fixture:meshopt`.
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
