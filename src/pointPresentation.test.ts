import { describe, expect, it } from "vitest";

import {
  autoDiameterCssPx,
  checkPresentation,
  normalizePresentation,
  samePresentation,
} from "./pointPresentation";

describe("checkPresentation", () => {
  it("defaults to a fixed 2 px diameter and resolves Auto bounds", () => {
    expect(normalizePresentation(undefined)).toEqual({
      mode: "fixed",
      diameterCssPx: 2,
    });
    expect(normalizePresentation({ mode: "auto", userScale: 1 })).toEqual({
      mode: "auto",
      userScale: 1,
      minDiameterCssPx: 1.5,
      maxDiameterCssPx: 4,
    });
  });

  it("reports an unusable contract instead of normalizing it", () => {
    expect(checkPresentation({ mode: "fixed", diameterCssPx: 0 })).toEqual({
      error: expect.stringContaining("diameterCssPx"),
    });
    expect(
      checkPresentation({
        mode: "auto",
        userScale: 1,
        minDiameterCssPx: 5,
        maxDiameterCssPx: 2,
      }),
    ).toHaveProperty("error");
    expect(() =>
      normalizePresentation({ mode: "auto", userScale: Number.NaN }),
    ).toThrow(/userScale/);
  });

  it("compares normalized contracts field by field", () => {
    const auto = {
      mode: "auto",
      userScale: 1,
      minDiameterCssPx: 1.5,
      maxDiameterCssPx: 4,
    } as const;
    expect(samePresentation(auto, { ...auto })).toBe(true);
    expect(samePresentation(auto, { ...auto, userScale: 2 })).toBe(false);
    expect(samePresentation(auto, normalizePresentation(undefined))).toBe(
      false,
    );
  });
});

describe("autoDiameterCssPx", () => {
  const presentation = {
    mode: "auto",
    userScale: 2,
    minDiameterCssPx: 1.5,
    maxDiameterCssPx: 4,
  } as const;

  it("scales the spacing inside its bounds", () => {
    expect(autoDiameterCssPx(presentation, 3)).toBe(6);
  });

  it("clamps the spacing before scaling it", () => {
    expect(autoDiameterCssPx(presentation, 0.1)).toBe(3);
    expect(autoDiameterCssPx(presentation, 100)).toBe(8);
  });
});
