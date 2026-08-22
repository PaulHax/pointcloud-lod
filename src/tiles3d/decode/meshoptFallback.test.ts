import { describe, expect, it } from "vitest";

import { repairMeshoptFallbackOffsets } from "./meshoptFallback";

const GLB_MAGIC = 0x46546c67;
const JSON_CHUNK = 0x4e4f534a;
const BIN_CHUNK = 0x004e4942;

/** A GLB carrying `json` and a token BIN chunk, laid out as the spec asks. */
const glb = (json: unknown, binaryBytes = 16): ArrayBuffer => {
  const encoded = new TextEncoder().encode(JSON.stringify(json));
  const jsonLength = (encoded.byteLength + 3) & ~3;
  const binaryLength = (binaryBytes + 3) & ~3;
  const total = 12 + 8 + jsonLength + 8 + binaryLength;
  const output = new Uint8Array(total);
  const view = new DataView(output.buffer);
  view.setUint32(0, GLB_MAGIC, true);
  view.setUint32(4, 2, true);
  view.setUint32(8, total, true);
  view.setUint32(12, jsonLength, true);
  view.setUint32(16, JSON_CHUNK, true);
  output.set(encoded, 20);
  output.fill(0x20, 20 + encoded.byteLength, 20 + jsonLength);
  view.setUint32(20 + jsonLength, binaryLength, true);
  view.setUint32(24 + jsonLength, BIN_CHUNK, true);
  return output.buffer;
};

type RepairedGltf = {
  bufferViews: { byteOffset?: number; byteLength: number }[];
  buffers: { byteLength: number }[];
};

const jsonOf = (buffer: ArrayBuffer): RepairedGltf => {
  const view = new DataView(buffer);
  return JSON.parse(
    new TextDecoder().decode(
      new Uint8Array(buffer, 20, view.getUint32(12, true)),
    ),
  );
};

/**
 * The layout a 3DBAG lod22 tile ships: two compressed views on one fallback
 * buffer, both at offset zero, the first an odd number of bytes long.
 */
const COLLIDING = {
  buffers: [
    { byteLength: 366_332 },
    {
      byteLength: 598_750,
      extensions: { EXT_meshopt_compression: { fallback: true } },
    },
  ],
  bufferViews: [
    {
      buffer: 1,
      byteLength: 90_318,
      extensions: { EXT_meshopt_compression: { byteStride: 2 } },
    },
    {
      buffer: 1,
      byteLength: 508_432,
      byteStride: 16,
      extensions: { EXT_meshopt_compression: { byteStride: 16 } },
    },
  ],
  accessors: [
    { bufferView: 0, componentType: 5123, count: 45_159, type: "SCALAR" },
    { bufferView: 1, componentType: 5126, count: 31_777, type: "SCALAR" },
  ],
};

describe("repairMeshoptFallbackOffsets", () => {
  it("starts every repaired view where an accessor may begin", () => {
    const repaired = jsonOf(repairMeshoptFallbackOffsets(glb(COLLIDING)));
    // 90,318 is where the index view ends and two bytes short of where a
    // float accessor may start; unpadded, loaders.gl refuses the attributes.
    expect(repaired.bufferViews[0]!.byteOffset).toBe(0);
    expect(repaired.bufferViews[1]!.byteOffset).toBe(90_320);
    for (const view of repaired.bufferViews) {
      expect((view.byteOffset ?? 0) % 4).toBe(0);
    }
    // Padding pushed the last view two bytes past a buffer sized for the
    // views concatenated, so the storage a loader allocates has to grow with
    // it. The buffer carries no bytes of its own; only its length matters.
    expect(repaired.buffers[1]!.byteLength).toBe(598_752);
    expect(repaired.buffers[0]!.byteLength).toBe(366_332);
  });

  it("leaves a distinctly laid out asset exactly as authored", () => {
    const authored = {
      ...COLLIDING,
      bufferViews: [
        COLLIDING.bufferViews[0]!,
        { ...COLLIDING.bufferViews[1]!, byteOffset: 90_320 },
      ],
    };
    const source = glb(authored);
    expect(repairMeshoptFallbackOffsets(source)).toBe(source);
  });

  it("repairs a bare glTF JSON tile, which the decoder also accepts", () => {
    // A `.gltf` payload has no GLB magic to key off, and a collision authored
    // in one draws exactly the wrong mesh this module exists to prevent.
    const source = new TextEncoder().encode(JSON.stringify(COLLIDING))
      .buffer as ArrayBuffer;
    const repaired: RepairedGltf = JSON.parse(
      new TextDecoder().decode(repairMeshoptFallbackOffsets(source)),
    );

    expect(repaired.bufferViews[0]!.byteOffset).toBe(0);
    expect(repaired.bufferViews[1]!.byteOffset).toBe(90_320);
    expect(repaired.buffers[1]!.byteLength).toBe(598_752);
  });

  it("leaves content it cannot read, or has nothing to fix in, untouched", () => {
    const cases = [
      new TextEncoder().encode(JSON.stringify({ asset: { version: "2.0" } })),
      new TextEncoder().encode("not json at all"),
      new Uint8Array([1, 2, 3]),
      new Uint8Array(0),
    ];
    for (const bytes of cases) {
      const source = bytes.buffer as ArrayBuffer;
      expect(repairMeshoptFallbackOffsets(source)).toBe(source);
    }
  });
});
