import { describe, expect, it } from "vitest";

import {
  composeSceneTransform,
  composeVerticalExaggeratedSceneTransform,
  createEcefToEnuTransform,
  createVerticalExaggerationTransform,
  flattenPrimitiveToRtc,
  multiplyMat4,
  transformPoint,
  wgs84ToEcef,
  type Mat4,
} from "./rtc";

const identity: Mat4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

const translation = (x: number, y: number, z: number): Mat4 => [
  1,
  0,
  0,
  0,
  0,
  1,
  0,
  0,
  0,
  0,
  1,
  0,
  x,
  y,
  z,
  1,
];

describe("3D Tiles RTC conversion", () => {
  it("maps known WGS84/ECEF points into local ENU when the tileset establishes the session origin", () => {
    const origin = wgs84ToEcef(-77.0353, 38.8895, 42);
    expect(origin[0]).toBeCloseTo(1_115_263.211, 2);
    expect(origin[1]).toBeCloseTo(-4_844_350.577, 2);
    expect(origin[2]).toBeCloseTo(3_982_802.63, 2);

    const tilesetToScene = createEcefToEnuTransform(-77.0353, 38.8895, 42);
    expect(transformPoint(tilesetToScene, origin)).toEqual(
      expect.arrayContaining([
        expect.closeTo(0, 8),
        expect.closeTo(0, 8),
        expect.closeTo(0, 8),
      ]),
    );

    const longitude = (-77.0353 * Math.PI) / 180;
    const eastOneMeter: [number, number, number] = [
      origin[0] - Math.sin(longitude),
      origin[1] + Math.cos(longitude),
      origin[2],
    ];
    const east = transformPoint(tilesetToScene, eastOneMeter);
    expect(east[0]).toBeCloseTo(1, 5);
    expect(east[1]).toBeCloseTo(0, 5);
    expect(east[2]).toBeCloseTo(0, 5);
  });

  it("uses a different established session anchor and preserves composition order", () => {
    const anchor = translation(7, 11, 13);
    const tilesetToScene: Mat4 = [
      0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 100, 200, 300, 1,
    ];
    const accumulated = translation(2, 3, 5);
    const composed = multiplyMat4(
      anchor,
      composeSceneTransform(tilesetToScene, accumulated),
    );

    expect(transformPoint(composed, [1, 0, 0])).toEqual([104, 214, 318]);
  });

  it("composes anchor, pivoted ENU exaggeration, ECEF placement, and tile-local transforms in that order", () => {
    const anchor: Mat4 = [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 7, 11, 13, 1];
    const tilesetToScene: Mat4 = [
      0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 100, 200, 10, 1,
    ];
    const accumulated = translation(2, 3, 4);
    const scene = composeVerticalExaggeratedSceneTransform(
      3,
      5,
      tilesetToScene,
      accumulated,
    );
    const composed = multiplyMat4(anchor, scene);

    // tile/local -> tilesetToScene gives [97, 203, 16], exaggeration gives
    // [97, 203, 38], then the live anchor gives [-196, 108, 51].
    expect(transformPoint(composed, [1, 0, 2])).toEqual([-196, 108, 51]);
    expect(
      transformPoint(createVerticalExaggerationTransform(3, 5), [0, 0, 5]),
    ).toEqual([0, 0, 5]);
    expect(createVerticalExaggerationTransform()).toEqual(identity);
  });

  it("subtracts a Float64 tile origin before Float32 conversion", () => {
    const result = flattenPrimitiveToRtc(
      {
        positions: new Float32Array([0, 0, 0, 0.25, -0.5, 1]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1]),
      },
      translation(6_378_137.125, -4_841_403.375, 3_985_381.625),
      identity,
    );

    expect(result.origin).toEqual([
      6_378_137.25, -4_841_403.625, 3_985_382.125,
    ]);
    expect([...result.positions]).toEqual([
      -0.125, 0.25, -0.5, 0.125, -0.25, 0.5,
    ]);
    expect(result.normals && [...result.normals]).toEqual([0, 0, 1, 0, 0, 1]);
  });

  it("bakes node transforms and inverse-transpose normals", () => {
    const local: Mat4 = [2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4, 0, 5, 7, 11, 1];
    const invSqrt2 = Math.SQRT1_2;
    const result = flattenPrimitiveToRtc(
      {
        positions: new Float32Array([0, 0, 0, 2, 3, 4]),
        normals: new Float32Array([
          invSqrt2,
          invSqrt2,
          0,
          0,
          invSqrt2,
          invSqrt2,
        ]),
      },
      identity,
      local,
    );

    expect([...result.positions]).toEqual([-2, -4.5, -8, 2, 4.5, 8]);
    expect(result.normals?.[0]).toBeCloseTo(0.83205, 5);
    expect(result.normals?.[1]).toBeCloseTo(0.5547, 5);
    expect(result.normals?.[2]).toBeCloseTo(0, 5);
    expect(result.normals?.[3]).toBeCloseTo(0, 5);
    expect(result.normals?.[4]).toBeCloseTo(0.8, 5);
    expect(result.normals?.[5]).toBeCloseTo(0.6, 5);
  });

  it("uses the inverse transpose for normals under scene-local vertical exaggeration", () => {
    const tilesetToScene: Mat4 = [
      0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 100, 200, 10, 1,
    ];
    const scene = composeVerticalExaggeratedSceneTransform(
      3,
      5,
      tilesetToScene,
      translation(2, 3, 4),
    );
    const invSqrt2 = Math.SQRT1_2;
    const result = flattenPrimitiveToRtc(
      {
        positions: new Float32Array([1, 0, 2]),
        normals: new Float32Array([invSqrt2, 0, invSqrt2]),
      },
      scene,
      identity,
    );

    expect(result.origin).toEqual([97, 203, 38]);
    expect([...result.positions]).toEqual([0, 0, 0]);
    expect(result.normals?.[0]).toBeCloseTo(0, 6);
    expect(result.normals?.[1]).toBeCloseTo(3 / Math.sqrt(10), 6);
    expect(result.normals?.[2]).toBeCloseTo(1 / Math.sqrt(10), 6);
  });

  it.each([
    [0, 0, /verticalExaggeration/],
    [-1, 0, /verticalExaggeration/],
    [Number.NaN, 0, /verticalExaggeration/],
    [Number.POSITIVE_INFINITY, 0, /verticalExaggeration/],
    [1, Number.NaN, /verticalPivotZ/],
    [1, Number.NEGATIVE_INFINITY, /verticalPivotZ/],
  ])(
    "rejects invalid vertical transform values %s, %s",
    (scale, pivot, match) => {
      expect(() =>
        createVerticalExaggerationTransform(scale as number, pivot as number),
      ).toThrow(match as RegExp);
    },
  );

  it("rejects non-finite and degenerate transforms and malformed attributes", () => {
    const positions = new Float32Array([0, 0, 0]);
    expect(() =>
      flattenPrimitiveToRtc(
        { positions },
        [...identity.slice(0, 15), NaN] as Mat4,
        identity,
      ),
    ).toThrow(/finite/i);
    expect(() =>
      flattenPrimitiveToRtc(
        { positions, normals: new Float32Array([0, 0, 1]) },
        identity,
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
      ),
    ).toThrow(/degenerate/i);
    expect(() =>
      flattenPrimitiveToRtc(
        { positions: new Float32Array([0, 1]) },
        identity,
        identity,
      ),
    ).toThrow(/positions/i);
    expect(() =>
      flattenPrimitiveToRtc(
        { positions, normals: new Float32Array([0, 1]) },
        identity,
        identity,
      ),
    ).toThrow(/normals/i);
  });
});
