import { describe, expect, expectTypeOf, it } from "vitest";

import { scenePoint, type ScenePoint } from "./frames";

describe("scene coordinates frame values", () => {
  it("requires the finite constructor and does not accept a bare tuple", () => {
    expect(scenePoint(1, 2, 3)).toEqual([1, 2, 3]);
    expect(() => scenePoint(1, Number.NaN, 3)).toThrow(/finite/);
    expectTypeOf<[number, number, number]>().not.toExtend<ScenePoint>();
    expectTypeOf(scenePoint(1, 2, 3)).toExtend<ScenePoint>();
  });
});
