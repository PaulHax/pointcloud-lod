import { describe, expect, expectTypeOf, it } from "vitest";

import { sceneEnuPoint, type SceneEnuPoint } from "./frames";

describe("scene ENU frame values", () => {
  it("requires the finite constructor and does not accept a bare tuple", () => {
    expect(sceneEnuPoint(1, 2, 3)).toEqual([1, 2, 3]);
    expect(() => sceneEnuPoint(1, Number.NaN, 3)).toThrow(/finite/);
    expectTypeOf<[number, number, number]>().not.toExtend<SceneEnuPoint>();
    expectTypeOf(sceneEnuPoint(1, 2, 3)).toExtend<SceneEnuPoint>();
  });
});
