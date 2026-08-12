import { describe, expect, it } from "vitest";

import { createTiles3dFixtureTileset } from "../../scripts/generateTiles3dFixture.mjs";
import type { PerspectiveCameraView } from "../camera";
import { parseTileset } from "./tilesetSource";
import { traverseTileset } from "./traversal";

const perspective = (
  position: readonly [number, number, number],
  worldToClip: readonly number[],
): PerspectiveCameraView => ({
  projection: "perspective",
  position,
  viewProj: worldToClip,
  fovY: Math.PI / 2,
  viewportWidthCssPx: 100,
  viewportHeightCssPx: 100,
});

const translateScale = (x: number, y: number, z: number, scale: number) => [
  scale,
  0,
  0,
  0,
  0,
  scale,
  0,
  0,
  0,
  0,
  scale,
  0,
  x,
  y,
  z,
  1,
];

const inverseRigid = (matrix: readonly number[]) => {
  const inverse = [
    matrix[0]!,
    matrix[4]!,
    matrix[8]!,
    0,
    matrix[1]!,
    matrix[5]!,
    matrix[9]!,
    0,
    matrix[2]!,
    matrix[6]!,
    matrix[10]!,
    0,
    0,
    0,
    0,
    1,
  ];
  inverse[12] = -(
    inverse[0]! * matrix[12]! +
    inverse[4]! * matrix[13]! +
    inverse[8]! * matrix[14]!
  );
  inverse[13] = -(
    inverse[1]! * matrix[12]! +
    inverse[5]! * matrix[13]! +
    inverse[9]! * matrix[14]!
  );
  inverse[14] = -(
    inverse[2]! * matrix[12]! +
    inverse[6]! * matrix[13]! +
    inverse[10]! * matrix[14]!
  );
  return inverse;
};

const multiply = (left: readonly number[], right: readonly number[]) => {
  const result = Array.from({ length: 16 }, () => 0);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      result[column * 4 + row] =
        left[row]! * right[column * 4]! +
        left[4 + row]! * right[column * 4 + 1]! +
        left[8 + row]! * right[column * 4 + 2]! +
        left[12 + row]! * right[column * 4 + 3]!;
    }
  }
  return result;
};

const source = () => parseTileset(createTiles3dFixtureTileset(), "/fixture");

describe("generated fixture traversal goldens", () => {
  it("pins camera distance to stable selected sets", () => {
    const root = source().root;
    const allReady = () => "submitted" as const;
    // Cancel the representative ECEF placement and map fixture-local ±56 into
    // the identity clip cube. Traversal still accumulates every authored
    // root/per-tile transform itself in f64.
    const fixtureToWorld = multiply(
      translateScale(0, 0, 0, 1 / 64),
      inverseRigid(root.worldTransform),
    );

    const far = traverseTileset({
      root,
      camera: perspective([0, 0, 1000 / 64], translateScale(0, 0, 0, 1)),
      modelMatrix: fixtureToWorld,
      maximumScreenSpaceErrorPx: 8,
      readiness: allReady,
    });
    expect(far.desiredTileIds).toEqual(["root"]);

    const near = traverseTileset({
      root,
      camera: perspective([0, 0, 100 / 64], translateScale(0, 0, 0, 1)),
      modelMatrix: fixtureToWorld,
      maximumScreenSpaceErrorPx: 8,
      readiness: allReady,
    });
    expect(near.desiredTileIds).toEqual([
      "root/0",
      "root/1",
      "root/2",
      "root/3",
    ]);

    const inside = traverseTileset({
      root,
      camera: perspective([0, 0, 0], translateScale(0, 0, 0, 1)),
      modelMatrix: fixtureToWorld,
      maximumScreenSpaceErrorPx: 8,
      readiness: allReady,
    });
    expect(inside.desiredTileIds).toEqual([
      "root/0/0",
      "root/0/1",
      "root/1/0",
      "root/1/1",
      "root/2/0",
      "root/2/1",
      "root/3/0",
      "root/3/1",
    ]);
  });
});
