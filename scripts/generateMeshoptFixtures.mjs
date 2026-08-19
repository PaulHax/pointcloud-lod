/**
 * Generates the two meshopt decode fixtures.
 *
 * Both hold the same quad — four vertices, two triangles, every coordinate
 * distinct — encoded with `EXT_meshopt_compression`. They differ only in where
 * the compressed views say their fallback storage begins:
 *
 * - `meshopt-quad.glb` lays the fallback views out end to end, as the
 *   extension intends.
 * - `meshopt-quad-collision.glb` starts both at offset zero of one fallback
 *   buffer sized for the two concatenated, which is how some publishers write
 *   them. A loader that decompresses in place then lands the vertex view on
 *   top of the index view, and the mesh draws as spikes with no error raised.
 *
 * A test decodes both and asserts the same geometry, so the repair is pinned
 * against an asset that genuinely needs it rather than against a description
 * of one.
 *
 * Run with `npm run fixture:meshopt`.
 */

import { writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { MeshoptEncoder } from "meshoptimizer";

const here = dirname(fileURLToPath(import.meta.url));
const output = resolve(here, "../test/fixtures/tiles3d-decode");

/** Distinct coordinates on every axis, so a scrambled index list is visible. */
const POSITIONS = new Float32Array([0, 0, 0, 4, 0, 1, 4, 3, 2, 0, 3, 3]);
const INDICES = new Uint32Array([0, 1, 2, 0, 2, 3]);

const VERTEX_STRIDE = 12;
const INDEX_STRIDE = 4;

const GLB_MAGIC = 0x46546c67;
const JSON_CHUNK = 0x4e4f534a;
const BIN_CHUNK = 0x004e4942;

const padToFour = (value) => (value + 3) & ~3;

const bytesOf = (typed) =>
  new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength);

/** One GLB from a JSON chunk and a binary chunk, both padded as the spec asks. */
const buildGlb = (json, binary) => {
  const jsonBytes = new TextEncoder().encode(JSON.stringify(json));
  const jsonLength = padToFour(jsonBytes.byteLength);
  const binaryLength = padToFour(binary.byteLength);
  const total = 12 + 8 + jsonLength + 8 + binaryLength;
  const output = new Uint8Array(total);
  const view = new DataView(output.buffer);

  view.setUint32(0, GLB_MAGIC, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, total, true);
  view.setUint32(12, jsonLength, true);
  view.setUint32(16, JSON_CHUNK, true);
  output.set(jsonBytes, 20);
  output.fill(0x20, 20 + jsonBytes.byteLength, 20 + jsonLength);
  const binaryHeader = 20 + jsonLength;
  view.setUint32(binaryHeader, binaryLength, true);
  view.setUint32(binaryHeader + 4, BIN_CHUNK, true);
  output.set(binary, binaryHeader + 8);
  return output;
};

const main = async () => {
  await MeshoptEncoder.ready;

  const vertexCount = POSITIONS.length / 3;
  const indexCount = INDICES.length;
  const compressedVertices = MeshoptEncoder.encodeVertexBuffer(
    bytesOf(POSITIONS),
    vertexCount,
    VERTEX_STRIDE,
  );
  const compressedIndices = MeshoptEncoder.encodeIndexBuffer(
    bytesOf(INDICES),
    indexCount,
    INDEX_STRIDE,
  );

  const vertexBytes = vertexCount * VERTEX_STRIDE;
  const indexBytes = indexCount * INDEX_STRIDE;

  // The BIN chunk carries only the compressed payloads; the fallback buffer
  // carries no bytes at all, which is the whole point of declaring it.
  const vertexOffset = 0;
  const indexOffset = padToFour(compressedVertices.byteLength);
  const binary = new Uint8Array(
    padToFour(indexOffset + compressedIndices.byteLength),
  );
  binary.set(compressedVertices, vertexOffset);
  binary.set(compressedIndices, indexOffset);

  const json = (fallbackOffsets) => ({
    asset: { version: "2.0", generator: "pointcloud-lod fixture generator" },
    extensionsUsed: ["EXT_meshopt_compression"],
    extensionsRequired: ["EXT_meshopt_compression"],
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [
      {
        primitives: [{ attributes: { POSITION: 0 }, indices: 1, mode: 4 }],
      },
    ],
    accessors: [
      {
        bufferView: 0,
        componentType: 5126,
        count: vertexCount,
        type: "VEC3",
        min: [0, 0, 0],
        max: [4, 3, 3],
      },
      {
        bufferView: 1,
        componentType: 5125,
        count: indexCount,
        type: "SCALAR",
      },
    ],
    bufferViews: [
      {
        buffer: 1,
        byteOffset: fallbackOffsets.vertices,
        byteLength: vertexBytes,
        byteStride: VERTEX_STRIDE,
        target: 34962,
        extensions: {
          EXT_meshopt_compression: {
            buffer: 0,
            byteOffset: vertexOffset,
            byteLength: compressedVertices.byteLength,
            byteStride: VERTEX_STRIDE,
            count: vertexCount,
            mode: "ATTRIBUTES",
          },
        },
      },
      {
        buffer: 1,
        byteOffset: fallbackOffsets.indices,
        byteLength: indexBytes,
        target: 34963,
        extensions: {
          EXT_meshopt_compression: {
            buffer: 0,
            byteOffset: indexOffset,
            byteLength: compressedIndices.byteLength,
            byteStride: INDEX_STRIDE,
            count: indexCount,
            mode: "TRIANGLES",
          },
        },
      },
    ],
    buffers: [
      { byteLength: binary.byteLength },
      {
        byteLength: vertexBytes + indexBytes,
        extensions: { EXT_meshopt_compression: { fallback: true } },
      },
    ],
  });

  const wellFormed = buildGlb(
    json({ vertices: 0, indices: vertexBytes }),
    binary,
  );
  const colliding = buildGlb(json({ vertices: 0, indices: 0 }), binary);

  await writeFile(resolve(output, "meshopt-quad.glb"), wellFormed);
  await writeFile(resolve(output, "meshopt-quad-collision.glb"), colliding);
  process.stdout.write(
    `meshopt-quad.glb ${wellFormed.byteLength} bytes\n` +
      `meshopt-quad-collision.glb ${colliding.byteLength} bytes\n`,
  );
};

await main();
