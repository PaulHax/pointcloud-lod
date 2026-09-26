import { describe, expect, it } from "vitest";

import {
  IDENTITY,
  affineProblem,
  invert,
  multiply,
  sameMatrix,
  similarityScale,
  transformPoint,
  transformVector,
  translatedMatrix,
  validateAffineMatrix,
} from "./mat4";

// prettier-ignore
const translation = (x: number, y: number, z: number): number[] => [
  1, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 1, 0,
  x, y, z, 1,
];

// prettier-ignore
const scaling = (x: number, y: number, z: number): number[] => [
  x, 0, 0, 0,
  0, y, 0, 0,
  0, 0, z, 0,
  0, 0, 0, 1,
];

// Rotation by 90 degrees about z, uniform scale 2, translation (5, 6, 7).
// prettier-ignore
const SIMILARITY = [
  0, 2, 0, 0,
  -2, 0, 0, 0,
  0, 0, 2, 0,
  5, 6, 7, 1,
];

describe("multiply", () => {
  it("applies the right operand first", () => {
    const scaledThenMoved = multiply(translation(1, 2, 3), scaling(2, 2, 2));
    expect(transformPoint(scaledThenMoved, [1, 1, 1])).toEqual([3, 4, 5]);
    const movedThenScaled = multiply(scaling(2, 2, 2), translation(1, 2, 3));
    expect(transformPoint(movedThenScaled, [1, 1, 1])).toEqual([4, 6, 8]);
  });

  it("leaves a matrix unchanged under the identity", () => {
    expect(multiply(IDENTITY, SIMILARITY)).toEqual(SIMILARITY);
    expect(multiply(SIMILARITY, IDENTITY)).toEqual(SIMILARITY);
  });
});

describe("invert", () => {
  it("undoes the matrix it inverts", () => {
    const inverse = invert(SIMILARITY)!;
    const product = multiply(SIMILARITY, inverse);
    product.forEach((value, index) =>
      expect(value).toBeCloseTo(IDENTITY[index]!),
    );
  });

  it("refuses singular and non-finite matrices", () => {
    expect(invert(scaling(1, 0, 1))).toBeNull();
    expect(invert([...IDENTITY.slice(0, 15), Number.NaN])).toBeNull();
  });
});

describe("points and directions", () => {
  it("moves points but not directions", () => {
    expect(transformPoint(SIMILARITY, [1, 0, 0])).toEqual([5, 8, 7]);
    expect(transformVector(SIMILARITY, [1, 0, 0])).toEqual([0, 2, 0]);
  });

  it("folds a tile origin into the last column", () => {
    expect(translatedMatrix(SIMILARITY, [1, 2, 3])).toEqual(
      multiply(SIMILARITY, translation(1, 2, 3)),
    );
  });
});

describe("sameMatrix", () => {
  it("compares element by element, with null equal only to null", () => {
    expect(sameMatrix(SIMILARITY, [...SIMILARITY])).toBe(true);
    expect(sameMatrix(SIMILARITY, translation(5, 6, 7))).toBe(false);
    expect(sameMatrix(null, null)).toBe(true);
    expect(sameMatrix(null, IDENTITY)).toBe(false);
    expect(sameMatrix(IDENTITY, IDENTITY.slice(0, 15))).toBe(false);
  });
});

describe("similarityScale", () => {
  it("reads the uniform scale of a rotation, scale and translation", () => {
    expect(similarityScale(SIMILARITY)).toBeCloseTo(2);
  });

  it("refuses anisotropic, projective and malformed matrices", () => {
    expect(similarityScale(scaling(1, 2, 1))).toBeNull();
    expect(similarityScale([...IDENTITY.slice(0, 15), 2])).toBeNull();
    expect(similarityScale(scaling(0, 0, 0))).toBeNull();
    expect(similarityScale([1, 0, 0])).toBeNull();
  });
});

describe("affineProblem", () => {
  it("names the first rule a matrix breaks", () => {
    expect(affineProblem(SIMILARITY)).toBeNull();
    // A singular matrix is still affine: invertibility is each caller's rule.
    expect(affineProblem(scaling(0, 0, 0))).toBeNull();
    expect(affineProblem([...IDENTITY.slice(0, 15), Number.NaN])).toBe(
      "finite",
    );
    expect(affineProblem(IDENTITY.slice(0, 12))).toBe("finite");
    expect(affineProblem(undefined)).toBe("finite");
    expect(affineProblem([...IDENTITY.slice(0, 15), 2])).toBe("affine");
  });
});

describe("validateAffineMatrix", () => {
  it("returns a copy of a usable matrix", () => {
    const nearlyAffine = [...SIMILARITY];
    nearlyAffine[3] = 1e-13;
    const checked = validateAffineMatrix(nearlyAffine, "placement");
    expect(checked).toEqual(nearlyAffine);
    expect(checked).not.toBe(nearlyAffine);
  });

  it("throws a TypeError naming the matrix for each broken rule", () => {
    const broken: readonly [ArrayLike<number>, RegExp][] = [
      [[...IDENTITY.slice(0, 15), Number.POSITIVE_INFINITY], /finite/],
      [IDENTITY.slice(0, 9), /finite/],
      [[...IDENTITY.slice(0, 11), 1e-9, 0, 0, 0, 1], /affine/],
      [scaling(1, 0, 1), /invertible/],
      // A determinant of 1e-18 is below the floor even though it is not zero.
      [scaling(1e-6, 1e-6, 1e-6), /invertible/],
    ];
    for (const [matrix, message] of broken) {
      expect(() => validateAffineMatrix(matrix, "placement")).toThrow(
        TypeError,
      );
      expect(() => validateAffineMatrix(matrix, "placement")).toThrow(
        new RegExp(`placement must .*${message.source}`),
      );
    }
  });
});
