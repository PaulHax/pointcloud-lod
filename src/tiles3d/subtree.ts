/** Strict reader for the binary 3D Tiles 1.1 subtree producer profile. */

import {
  TilesetUnsupportedError,
  TilesetValidationError,
  type TilesetBox,
} from "./tilesetSource";

export type SubtreeAvailability = {
  readonly length: number;
  readonly availableCount: number;
  isAvailable(index: number): boolean;
};

export type ParsedSubtree = {
  readonly byteLength: number;
  readonly subtreeLevels: number;
  readonly tileAvailability: SubtreeAvailability;
  readonly contentAvailability: SubtreeAvailability;
  readonly childSubtreeAvailability: SubtreeAvailability;
  /** Indexed by availability index; unavailable tiles have no entry. */
  readonly tileBoundingBoxes: readonly (TilesetBox | undefined)[];
};

const objectAt = (value: unknown, path: string): Record<string, unknown> => {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new TilesetValidationError(path, "expected an object");
  }
  return value as Record<string, unknown>;
};

const arrayAt = (value: unknown, path: string): unknown[] => {
  if (!Array.isArray(value)) {
    throw new TilesetValidationError(path, "expected an array");
  }
  return value;
};

const integerAtLeast = (
  value: unknown,
  minimum: number,
  path: string,
): number => {
  if (!Number.isSafeInteger(value) || (value as number) < minimum) {
    throw new TilesetValidationError(
      path,
      `expected a safe integer >= ${minimum}`,
    );
  }
  return value as number;
};

const uint64 = (view: DataView, offset: number, path: string): number => {
  const value = view.getBigUint64(offset, true);
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new TilesetValidationError(
      path,
      "value exceeds the safe integer range",
    );
  }
  return Number(value);
};

const determinant3 = (m: readonly number[]): number =>
  m[0]! * (m[4]! * m[8]! - m[7]! * m[5]!) -
  m[3]! * (m[1]! * m[8]! - m[7]! * m[2]!) +
  m[6]! * (m[1]! * m[5]! - m[4]! * m[2]!);

const boxFrom = (values: readonly number[], path: string): TilesetBox => {
  if (values.length !== 12 || values.some((value) => !Number.isFinite(value))) {
    throw new TilesetValidationError(
      path,
      "expected exactly 12 finite numbers",
    );
  }
  const halfAxes = values.slice(3);
  if (Math.abs(determinant3(halfAxes)) <= Number.EPSILON) {
    throw new TilesetValidationError(path, "box half axes must span a volume");
  }
  return Object.freeze({
    center: Object.freeze(values.slice(0, 3)) as TilesetBox["center"],
    halfAxes: Object.freeze(halfAxes) as TilesetBox["halfAxes"],
  });
};

type BufferView = { readonly offset: number; readonly length: number };

const immutableAvailability = (
  length: number,
  availableCount: number,
  available: (index: number) => boolean,
): SubtreeAvailability =>
  Object.freeze({
    length,
    availableCount,
    isAvailable(index: number): boolean {
      return Number.isSafeInteger(index) && index >= 0 && index < length
        ? available(index)
        : false;
    },
  });

const availabilityAt = (
  rawValue: unknown,
  path: string,
  expectedBits: number,
  views: readonly BufferView[],
  binary: Uint8Array,
): SubtreeAvailability => {
  const raw = objectAt(rawValue, path);
  const hasConstant = raw.constant !== undefined;
  const hasBitstream = raw.bitstream !== undefined;
  if (hasConstant === hasBitstream) {
    throw new TilesetValidationError(
      path,
      "expected exactly one of constant or bitstream",
    );
  }
  if (hasConstant) {
    if (raw.constant !== 0 && raw.constant !== 1) {
      throw new TilesetValidationError(`${path}.constant`, "expected 0 or 1");
    }
    const count = raw.constant === 1 ? expectedBits : 0;
    if (raw.availableCount !== undefined && raw.availableCount !== count) {
      throw new TilesetValidationError(
        `${path}.availableCount`,
        `expected ${count}`,
      );
    }
    return immutableAvailability(expectedBits, count, () => raw.constant === 1);
  }

  const viewIndex = integerAtLeast(raw.bitstream, 0, `${path}.bitstream`);
  const bufferView = views[viewIndex];
  if (!bufferView) {
    throw new TilesetValidationError(
      `${path}.bitstream`,
      `buffer view ${viewIndex} is missing`,
    );
  }
  const requiredBytes = Math.ceil(expectedBits / 8);
  if (bufferView.length < requiredBytes) {
    throw new TilesetValidationError(
      `${path}.bitstream`,
      `buffer view is shorter than ${requiredBytes} bytes`,
    );
  }
  const bytes = binary.subarray(
    bufferView.offset,
    bufferView.offset + requiredBytes,
  );
  const bit = (index: number): boolean =>
    ((bytes[index >> 3]! >> (index & 7)) & 1) === 1;
  let count = 0;
  for (let index = 0; index < expectedBits; index += 1) {
    if (bit(index)) count += 1;
  }
  if (raw.availableCount !== undefined) {
    const declared = integerAtLeast(
      raw.availableCount,
      0,
      `${path}.availableCount`,
    );
    if (declared !== count) {
      throw new TilesetValidationError(
        `${path}.availableCount`,
        `declares ${declared}, but the bitstream contains ${count}`,
      );
    }
  }
  return immutableAvailability(expectedBits, count, bit);
};

const parseMetadataBounds = (
  raw: Record<string, unknown>,
  schemaValue: unknown,
  views: readonly BufferView[],
  binary: Uint8Array,
  availability: SubtreeAvailability,
): readonly (TilesetBox | undefined)[] => {
  const schema = objectAt(schemaValue, "schema");
  const classes = objectAt(schema.classes, "schema.classes");
  const tables = arrayAt(raw.propertyTables, "subtree.propertyTables");
  const tableIndex = integerAtLeast(
    raw.tileMetadata,
    0,
    "subtree.tileMetadata",
  );
  const table = objectAt(
    tables[tableIndex],
    `subtree.propertyTables[${tableIndex}]`,
  );
  if (typeof table.class !== "string" || table.class.length === 0) {
    throw new TilesetValidationError(
      `subtree.propertyTables[${tableIndex}].class`,
      "expected a non-empty string",
    );
  }
  const metadataClass = objectAt(
    classes[table.class],
    `schema.classes.${table.class}`,
  );
  const classProperties = objectAt(
    metadataClass.properties,
    `schema.classes.${table.class}.properties`,
  );
  const semanticProperties = Object.entries(classProperties).filter(
    ([, value]) => {
      if (value === null || typeof value !== "object" || Array.isArray(value))
        return false;
      return (
        (value as Record<string, unknown>).semantic === "TILE_BOUNDING_BOX"
      );
    },
  );
  if (semanticProperties.length !== 1) {
    throw new TilesetValidationError(
      `schema.classes.${table.class}.properties`,
      "expected exactly one TILE_BOUNDING_BOX semantic",
    );
  }
  const [propertyName, propertyValue] = semanticProperties[0]!;
  const classProperty = objectAt(
    propertyValue,
    `schema.classes.${table.class}.properties.${propertyName}`,
  );
  if (
    classProperty.type !== "SCALAR" ||
    (classProperty.componentType !== "FLOAT32" &&
      classProperty.componentType !== "FLOAT64") ||
    classProperty.array !== true ||
    classProperty.count !== 12 ||
    classProperty.required !== true
  ) {
    throw new TilesetValidationError(
      `schema.classes.${table.class}.properties.${propertyName}`,
      "TILE_BOUNDING_BOX must be a required fixed array of 12 FLOAT32 or FLOAT64 scalars",
    );
  }
  const rowCount = integerAtLeast(
    table.count,
    0,
    `subtree.propertyTables[${tableIndex}].count`,
  );
  if (rowCount !== availability.availableCount) {
    throw new TilesetValidationError(
      `subtree.propertyTables[${tableIndex}].count`,
      `expected ${availability.availableCount} available-tile rows`,
    );
  }
  const tableProperties = objectAt(
    table.properties,
    `subtree.propertyTables[${tableIndex}].properties`,
  );
  const property = objectAt(
    tableProperties[propertyName],
    `subtree.propertyTables[${tableIndex}].properties.${propertyName}`,
  );
  const valuesViewIndex = integerAtLeast(
    property.values,
    0,
    `subtree.propertyTables[${tableIndex}].properties.${propertyName}.values`,
  );
  const valuesView = views[valuesViewIndex];
  if (!valuesView) {
    throw new TilesetValidationError(
      `subtree.propertyTables[${tableIndex}].properties.${propertyName}.values`,
      `buffer view ${valuesViewIndex} is missing`,
    );
  }
  const bytesPerComponent = classProperty.componentType === "FLOAT64" ? 8 : 4;
  const requiredBytes = rowCount * 12 * bytesPerComponent;
  if (valuesView.length !== requiredBytes) {
    throw new TilesetValidationError(
      `subtree.bufferViews[${valuesViewIndex}].byteLength`,
      `expected exactly ${requiredBytes} bytes`,
    );
  }
  if (valuesView.offset % bytesPerComponent !== 0) {
    throw new TilesetValidationError(
      `subtree.bufferViews[${valuesViewIndex}].byteOffset`,
      `expected ${bytesPerComponent}-byte alignment`,
    );
  }
  const values = new DataView(
    binary.buffer,
    binary.byteOffset + valuesView.offset,
    valuesView.length,
  );
  const result: (TilesetBox | undefined)[] = Array.from({
    length: availability.length,
  });
  let row = 0;
  for (let tileIndex = 0; tileIndex < availability.length; tileIndex += 1) {
    if (!availability.isAvailable(tileIndex)) continue;
    const boxValues = Array.from({ length: 12 }, () => 0);
    for (let component = 0; component < 12; component += 1) {
      const offset = (row * 12 + component) * bytesPerComponent;
      boxValues[component] =
        bytesPerComponent === 8
          ? values.getFloat64(offset, true)
          : values.getFloat32(offset, true);
    }
    result[tileIndex] = boxFrom(
      boxValues,
      `subtree.tileMetadata[${row}].${propertyName}`,
    );
    row += 1;
  }
  return Object.freeze(result);
};

export const subtreeTileCount = (subtreeLevels: number): number => {
  if (
    !Number.isSafeInteger(subtreeLevels) ||
    subtreeLevels < 1 ||
    subtreeLevels > 16
  ) {
    throw new RangeError("subtreeLevels must be an integer in [1, 16]");
  }
  return (4 ** subtreeLevels - 1) / 3;
};

export const subtreeLevelOffset = (level: number): number =>
  (4 ** level - 1) / 3;

/** Morton index with x in the low bit and y in the high bit of each pair. */
export const quadtreeMortonIndex = (
  level: number,
  x: number,
  y: number,
): number => {
  if (
    !Number.isSafeInteger(level) ||
    level < 0 ||
    !Number.isSafeInteger(x) ||
    !Number.isSafeInteger(y) ||
    x < 0 ||
    y < 0 ||
    x >= 2 ** level ||
    y >= 2 ** level
  ) {
    throw new RangeError("invalid quadtree coordinates");
  }
  let result = 0;
  for (let bit = 0; bit < level; bit += 1) {
    result |= ((x >> bit) & 1) << (bit * 2);
    result |= ((y >> bit) & 1) << (bit * 2 + 1);
  }
  return result;
};

export const subtreeTileIndex = (level: number, x: number, y: number): number =>
  subtreeLevelOffset(level) + quadtreeMortonIndex(level, x, y);

export const parseSubtree = (
  content: ArrayBuffer,
  subtreeLevels: number,
  metadataSchema: unknown,
): ParsedSubtree => {
  const tileCount = subtreeTileCount(subtreeLevels);
  if (content.byteLength < 24) {
    throw new TilesetValidationError("subtree", "binary header is truncated");
  }
  const header = new DataView(content);
  if (
    header.getUint8(0) !== 0x73 ||
    header.getUint8(1) !== 0x75 ||
    header.getUint8(2) !== 0x62 ||
    header.getUint8(3) !== 0x74
  ) {
    throw new TilesetValidationError("subtree.magic", 'expected "subt"');
  }
  if (header.getUint32(4, true) !== 1) {
    throw new TilesetValidationError("subtree.version", "expected 1");
  }
  const jsonLength = uint64(header, 8, "subtree.jsonByteLength");
  const binaryLength = uint64(header, 16, "subtree.binaryByteLength");
  if (jsonLength % 8 !== 0 || binaryLength % 8 !== 0) {
    throw new TilesetValidationError(
      "subtree",
      "JSON and binary chunks must be 8-byte aligned",
    );
  }
  if (24 + jsonLength + binaryLength !== content.byteLength) {
    throw new TilesetValidationError(
      "subtree",
      "header chunk lengths do not match the payload length",
    );
  }
  let document: unknown;
  try {
    const jsonBytes = new Uint8Array(content, 24, jsonLength);
    const paddedText = new TextDecoder("utf-8", { fatal: true }).decode(
      jsonBytes,
    );
    let jsonEnd = paddedText.length;
    while (
      jsonEnd > 0 &&
      (paddedText.charCodeAt(jsonEnd - 1) === 0 ||
        paddedText.charCodeAt(jsonEnd - 1) === 0x20)
    ) {
      jsonEnd -= 1;
    }
    document = JSON.parse(paddedText.slice(0, jsonEnd));
  } catch (error) {
    throw new TilesetValidationError("subtree.json", "invalid UTF-8 JSON", {
      cause: error,
    });
  }
  const raw = objectAt(document, "subtree");
  if (raw.schema !== undefined || raw.schemaUri !== undefined) {
    throw new TilesetValidationError(
      "subtree.schema",
      "schema must be declared by the tileset, not repeated in the subtree",
    );
  }
  const buffers = arrayAt(raw.buffers, "subtree.buffers");
  if (buffers.length !== 1) {
    throw new TilesetValidationError(
      "subtree.buffers",
      "expected exactly one inline binary buffer",
    );
  }
  const buffer = objectAt(buffers[0], "subtree.buffers[0]");
  if (buffer.uri !== undefined) {
    throw new TilesetUnsupportedError(
      "subtree.externalBuffer",
      "subtree.buffers[0].uri",
    );
  }
  if (
    integerAtLeast(buffer.byteLength, 0, "subtree.buffers[0].byteLength") >
    binaryLength
  ) {
    throw new TilesetValidationError(
      "subtree.buffers[0].byteLength",
      "exceeds the binary chunk",
    );
  }
  const views = arrayAt(raw.bufferViews, "subtree.bufferViews").map(
    (value, index): BufferView => {
      const view = objectAt(value, `subtree.bufferViews[${index}]`);
      if (view.buffer !== 0) {
        throw new TilesetValidationError(
          `subtree.bufferViews[${index}].buffer`,
          "expected buffer 0",
        );
      }
      const offset = integerAtLeast(
        view.byteOffset ?? 0,
        0,
        `subtree.bufferViews[${index}].byteOffset`,
      );
      const length = integerAtLeast(
        view.byteLength,
        0,
        `subtree.bufferViews[${index}].byteLength`,
      );
      if (offset + length > binaryLength) {
        throw new TilesetValidationError(
          `subtree.bufferViews[${index}]`,
          "range exceeds the binary chunk",
        );
      }
      return Object.freeze({ offset, length });
    },
  );
  const binary = new Uint8Array(content, 24 + jsonLength, binaryLength);
  const tileAvailability = availabilityAt(
    raw.tileAvailability,
    "subtree.tileAvailability",
    tileCount,
    views,
    binary,
  );
  if (!tileAvailability.isAvailable(0)) {
    throw new TilesetValidationError(
      "subtree.tileAvailability",
      "the subtree root tile must be available",
    );
  }
  const contentAvailabilityValues = arrayAt(
    raw.contentAvailability,
    "subtree.contentAvailability",
  );
  if (contentAvailabilityValues.length !== 1) {
    throw new TilesetUnsupportedError(
      "subtree.multipleContents",
      "subtree.contentAvailability",
    );
  }
  const contentAvailability = availabilityAt(
    contentAvailabilityValues[0],
    "subtree.contentAvailability[0]",
    tileCount,
    views,
    binary,
  );
  for (let index = 0; index < tileCount; index += 1) {
    if (
      contentAvailability.isAvailable(index) &&
      !tileAvailability.isAvailable(index)
    ) {
      throw new TilesetValidationError(
        "subtree.contentAvailability[0]",
        `content ${index} is available for an unavailable tile`,
      );
    }
  }
  const childSubtreeAvailability = availabilityAt(
    raw.childSubtreeAvailability,
    "subtree.childSubtreeAvailability",
    4 ** subtreeLevels,
    views,
    binary,
  );
  const tileBoundingBoxes = parseMetadataBounds(
    raw,
    metadataSchema,
    views,
    binary,
    tileAvailability,
  );
  return Object.freeze({
    byteLength: content.byteLength,
    subtreeLevels,
    tileAvailability,
    contentAvailability,
    childSubtreeAvailability,
    tileBoundingBoxes,
  });
};
