import { subtreeTileCount } from "../../src/tiles3d/subtree";

type JsonObject = Record<string, any>;

export type SubtreeFixtureOptions = {
  readonly subtreeLevels?: number;
  readonly tileAvailability?: readonly boolean[];
  readonly contentAvailability?: readonly boolean[];
  readonly childSubtreeAvailability?: readonly boolean[];
  readonly boxes?: ReadonlyMap<number, readonly number[]>;
  readonly mutateJson?: (json: JsonObject) => void;
};

export const SUBTREE_METADATA_SCHEMA = Object.freeze({
  classes: {
    tile: {
      properties: {
        bounds: {
          type: "SCALAR",
          componentType: "FLOAT64",
          array: true,
          count: 12,
          required: true,
          semantic: "TILE_BOUNDING_BOX",
        },
      },
    },
  },
});

const align8 = (value: number): number => (value + 7) & ~7;

const bits = (values: readonly boolean[]): Uint8Array => {
  const result = new Uint8Array(Math.ceil(values.length / 8));
  values.forEach((available, index) => {
    if (available) {
      const byteIndex = index >> 3;
      result[byteIndex] = result[byteIndex]! | (1 << (index & 7));
    }
  });
  return result;
};

export const makeSubtreeFixture = (
  options: SubtreeFixtureOptions = {},
): ArrayBuffer => {
  const subtreeLevels = options.subtreeLevels ?? 2;
  const tileCount = subtreeTileCount(subtreeLevels);
  const tileAvailability =
    options.tileAvailability ?? Array.from({ length: tileCount }, () => true);
  const contentAvailability =
    options.contentAvailability ??
    Array.from({ length: tileCount }, () => true);
  const childSubtreeAvailability =
    options.childSubtreeAvailability ??
    Array.from({ length: 4 ** subtreeLevels }, () => false);
  if (
    tileAvailability.length !== tileCount ||
    contentAvailability.length !== tileCount ||
    childSubtreeAvailability.length !== 4 ** subtreeLevels
  ) {
    throw new Error("fixture availability length mismatch");
  }

  const chunks: Uint8Array[] = [];
  const bufferViews: { buffer: 0; byteOffset: number; byteLength: number }[] =
    [];
  let binaryLength = 0;
  const add = (bytes: Uint8Array, alignment = 1): number => {
    const offset = Math.ceil(binaryLength / alignment) * alignment;
    if (offset > binaryLength)
      chunks.push(new Uint8Array(offset - binaryLength));
    const index = bufferViews.length;
    bufferViews.push({
      buffer: 0,
      byteOffset: offset,
      byteLength: bytes.byteLength,
    });
    chunks.push(bytes);
    binaryLength = offset + bytes.byteLength;
    return index;
  };
  const tileView = add(bits(tileAvailability));
  const contentView = add(bits(contentAvailability));
  const childView = add(bits(childSubtreeAvailability));
  const availableIndices = tileAvailability.flatMap((available, index) =>
    available ? [index] : [],
  );
  const boundBytes = new Uint8Array(availableIndices.length * 12 * 8);
  const boundView = new DataView(boundBytes.buffer);
  availableIndices.forEach((tileIndex, row) => {
    const values = options.boxes?.get(tileIndex) ?? [
      tileIndex,
      0,
      0,
      1,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      1,
    ];
    values.forEach((value, component) =>
      boundView.setFloat64((row * 12 + component) * 8, value, true),
    );
  });
  const boundsView = add(boundBytes, 8);
  const paddedBinaryLength = align8(binaryLength);
  if (paddedBinaryLength > binaryLength) {
    chunks.push(new Uint8Array(paddedBinaryLength - binaryLength));
  }
  const binary = new Uint8Array(paddedBinaryLength);
  let cursor = 0;
  for (const chunk of chunks) {
    binary.set(chunk, cursor);
    cursor += chunk.byteLength;
  }

  const count = (values: readonly boolean[]): number =>
    values.filter(Boolean).length;
  const json: JsonObject = {
    buffers: [{ byteLength: paddedBinaryLength }],
    bufferViews,
    tileAvailability: {
      bitstream: tileView,
      availableCount: count(tileAvailability),
    },
    contentAvailability: [
      {
        bitstream: contentView,
        availableCount: count(contentAvailability),
      },
    ],
    childSubtreeAvailability: {
      bitstream: childView,
      availableCount: count(childSubtreeAvailability),
    },
    propertyTables: [
      {
        class: "tile",
        count: availableIndices.length,
        properties: { bounds: { values: boundsView } },
      },
    ],
    tileMetadata: 0,
  };
  options.mutateJson?.(json);
  const encoded = new TextEncoder().encode(JSON.stringify(json));
  const jsonLength = align8(encoded.byteLength);
  const result = new ArrayBuffer(24 + jsonLength + binary.byteLength);
  const header = new DataView(result);
  new Uint8Array(result, 0, 4).set(new TextEncoder().encode("subt"));
  header.setUint32(4, 1, true);
  header.setBigUint64(8, BigInt(jsonLength), true);
  header.setBigUint64(16, BigInt(binary.byteLength), true);
  const jsonChunk = new Uint8Array(result, 24, jsonLength);
  jsonChunk.fill(0x20);
  jsonChunk.set(encoded);
  new Uint8Array(result, 24 + jsonLength).set(binary);
  return result;
};
