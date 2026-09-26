/**
 * The point presentation contract: a fixed CSS-pixel diameter, or an Auto
 * diameter sized from the projected spacing the selection leaves on screen.
 */

import { finitePositive } from "./numeric";

export type FixedPointPresentation = {
  readonly mode: "fixed";
  readonly diameterCssPx: number;
};

export type AutoPointPresentation = {
  readonly mode: "auto";
  /** Multiplier for the Auto diameter; 1 matches the projected spacing. */
  readonly userScale: number;
  /** Bounds for the unscaled density-aware diameter. */
  readonly minDiameterCssPx?: number;
  readonly maxDiameterCssPx?: number;
};

export type PointPresentation = FixedPointPresentation | AutoPointPresentation;

/** A checked presentation: Auto bounds are always resolved. */
export type NormalizedPresentation =
  | FixedPointPresentation
  | Required<AutoPointPresentation>;

const DEFAULT_PRESENTATION: FixedPointPresentation = {
  mode: "fixed",
  diameterCssPx: 2,
};
const DEFAULT_AUTO_MIN_DIAMETER_CSS_PX = 1.5;
const DEFAULT_AUTO_MAX_DIAMETER_CSS_PX = 4;

/** What Auto draws before a selection has measured any spacing. */
export const INITIAL_AUTO_DIAMETER_CSS_PX = 2;

type PresentationCheck =
  | { readonly presentation: NormalizedPresentation }
  | { readonly error: string };

/**
 * One validation body for both the throwing and the ignoring boundary.
 * Undefined is the default presentation.
 */
export const checkPresentation = (
  value: PointPresentation | undefined,
): PresentationCheck => {
  const presentation = value ?? DEFAULT_PRESENTATION;
  if (presentation.mode === "fixed") {
    if (!finitePositive(presentation.diameterCssPx)) {
      return {
        error: `Fixed diameterCssPx must be finite and > 0, got ${presentation.diameterCssPx}`,
      };
    }
    return {
      presentation: {
        mode: "fixed",
        diameterCssPx: presentation.diameterCssPx,
      },
    };
  }
  const min = presentation.minDiameterCssPx ?? DEFAULT_AUTO_MIN_DIAMETER_CSS_PX;
  const max = presentation.maxDiameterCssPx ?? DEFAULT_AUTO_MAX_DIAMETER_CSS_PX;
  if (
    !finitePositive(presentation.userScale) ||
    !finitePositive(min) ||
    !Number.isFinite(max) ||
    max < min
  ) {
    return {
      error:
        "Auto userScale/minDiameterCssPx/maxDiameterCssPx must be finite, positive, and ordered",
    };
  }
  return {
    presentation: {
      mode: "auto",
      userScale: presentation.userScale,
      minDiameterCssPx: min,
      maxDiameterCssPx: max,
    },
  };
};

export const normalizePresentation = (
  value: PointPresentation | undefined,
): NormalizedPresentation => {
  const checked = checkPresentation(value);
  if ("error" in checked) throw new Error(checked.error);
  return checked.presentation;
};

export const samePresentation = (
  left: NormalizedPresentation,
  right: NormalizedPresentation,
): boolean =>
  left.mode === "fixed"
    ? right.mode === "fixed" && left.diameterCssPx === right.diameterCssPx
    : right.mode === "auto" &&
      left.userScale === right.userScale &&
      left.minDiameterCssPx === right.minDiameterCssPx &&
      left.maxDiameterCssPx === right.maxDiameterCssPx;

/** The Auto diameter: the spacing clamped to the bounds, then scaled. */
export const autoDiameterCssPx = (
  presentation: Required<AutoPointPresentation>,
  spacingCssPx: number,
): number =>
  presentation.userScale *
  Math.min(
    presentation.maxDiameterCssPx,
    Math.max(presentation.minDiameterCssPx, spacingCssPx),
  );
