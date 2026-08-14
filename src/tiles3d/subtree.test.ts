import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";

import {
  makeSubtreeFixture,
  SUBTREE_METADATA_SCHEMA,
} from "../../test/fixtures/subtreeFixture";
import { parseSubtree, quadtreeMortonIndex, subtreeTileIndex } from "./subtree";
import { parseTileset } from "./tilesetSource";

describe("binary subtree reader", () => {
  it("consumes the checked-in producer wire fixture and tileset-level schema", () => {
    const directory = new URL(
      "../../test/fixtures/tiles3d-implicit/",
      import.meta.url,
    );
    const source = parseTileset(
      JSON.parse(readFileSync(new URL("tileset.json", directory), "utf8")),
      "/fixture",
    );
    const bytes = readFileSync(new URL("subtrees/0/0/0.subtree", directory));
    const content = bytes.buffer.slice(
      bytes.byteOffset,
      bytes.byteOffset + bytes.byteLength,
    );
    const parsed = parseSubtree(
      content,
      source.root.implicitTiling!.subtreeLevels,
      source.root.implicitTiling!.metadataSchema,
    );

    expect(parsed.subtreeLevels).toBe(4);
    expect(parsed.tileAvailability.availableCount).toBeGreaterThan(4);
    expect(parsed.tileBoundingBoxes[0]?.center.every(Number.isFinite)).toBe(
      true,
    );
  });
  it("reads availability and maps compact metadata rows to breadth-first Morton indices", () => {
    const parsed = parseSubtree(
      makeSubtreeFixture({
        tileAvailability: [true, true, false, true, false],
        contentAvailability: [false, false, false, true, false],
        childSubtreeAvailability: Array.from(
          { length: 16 },
          (_value, index) => index === 9,
        ),
      }),
      2,
      SUBTREE_METADATA_SCHEMA,
    );

    expect(
      [...Array(5).keys()].map(parsed.tileAvailability.isAvailable),
    ).toEqual([true, true, false, true, false]);
    expect(parsed.contentAvailability.isAvailable(3)).toBe(true);
    expect(parsed.childSubtreeAvailability.isAvailable(9)).toBe(true);
    expect(parsed.tileBoundingBoxes.map((box) => box?.center[0])).toEqual([
      0,
      1,
      undefined,
      3,
      undefined,
    ]);
  });

  it("pins quadtree x-low/y-high Morton and level offsets", () => {
    expect(quadtreeMortonIndex(2, 1, 2)).toBe(9);
    expect(subtreeTileIndex(0, 0, 0)).toBe(0);
    expect(subtreeTileIndex(1, 1, 0)).toBe(2);
    expect(subtreeTileIndex(2, 1, 2)).toBe(14);
  });

  it("rejects content for unavailable tiles", () => {
    expect(() =>
      parseSubtree(
        makeSubtreeFixture({
          tileAvailability: [true, false, true, true, true],
          contentAvailability: [true, true, false, false, false],
        }),
        2,
        SUBTREE_METADATA_SCHEMA,
      ),
    ).toThrow(/content 1 is available for an unavailable tile/);
  });

  it.each([
    [
      "missing bound semantic",
      (schema: any) => {
        schema.classes.tile.properties.bounds.semantic = "OTHER";
      },
      /TILE_BOUNDING_BOX/,
    ],
    [
      "variable bound array",
      (schema: any) => {
        delete schema.classes.tile.properties.bounds.count;
      },
      /fixed array of 12/,
    ],
  ])(
    "rejects malformed producer schema: %s",
    (_name, mutateSchema, message) => {
      const schema = structuredClone(SUBTREE_METADATA_SCHEMA);
      mutateSchema(schema);
      expect(() => parseSubtree(makeSubtreeFixture(), 2, schema)).toThrow(
        message,
      );
    },
  );

  it("rejects a property-table row count that differs from availability", () => {
    const bytes = makeSubtreeFixture({
      mutateJson: (json) => {
        json.propertyTables[0].count -= 1;
      },
    });
    expect(() => parseSubtree(bytes, 2, SUBTREE_METADATA_SCHEMA)).toThrow(
      /available-tile rows/,
    );
  });

  it("rejects invalid binary framing with a typed validation path", () => {
    const bytes = makeSubtreeFixture();
    new DataView(bytes).setUint32(4, 2, true);
    expect(() => parseSubtree(bytes, 2, SUBTREE_METADATA_SCHEMA)).toThrow(
      /subtree\.version/,
    );
  });
});
