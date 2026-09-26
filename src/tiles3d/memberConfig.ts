import { validateAffineMatrix } from "../mat4";
import { integerAtLeast } from "../numeric";
import {
  DEFAULT_MAXIMUM_SCREEN_SPACE_ERROR_PX,
  DEFAULT_TILES3D_CACHE_BYTES,
  DEFAULT_TILES3D_CONCURRENCY,
  DEFAULT_VERTICAL_EXAGGERATION,
  DEFAULT_VERTICAL_PIVOT_Z,
  type Tiles3dMemberConfig,
} from "./memberTypes";

/** A validated config with every defaulted option resolved. */
export type ResolvedTiles3dConfig = Tiles3dMemberConfig &
  Required<
    Pick<
      Tiles3dMemberConfig,
      | "maximumScreenSpaceErrorPx"
      | "cacheBytes"
      | "concurrency"
      | "verticalExaggeration"
      | "verticalPivotZ"
      | "geometricErrorScale"
    >
  >;

export const validateTiles3dMemberConfig = (
  config: Tiles3dMemberConfig,
): ResolvedTiles3dConfig => {
  if (
    typeof config.endpoint !== "string" ||
    config.endpoint.length === 0 ||
    config.endpoint.endsWith("/")
  ) {
    throw new TypeError(
      "tiles endpoint must be non-empty and must not end with '/'",
    );
  }
  if (typeof config.revision !== "string" || config.revision.length === 0) {
    throw new TypeError("tiles revision must be non-empty");
  }
  validateAffineMatrix(config.tilesetToScene, "tilesetToScene");
  const maximumScreenSpaceErrorPx =
    config.maximumScreenSpaceErrorPx ?? DEFAULT_MAXIMUM_SCREEN_SPACE_ERROR_PX;
  if (
    !Number.isFinite(maximumScreenSpaceErrorPx) ||
    maximumScreenSpaceErrorPx <= 0
  ) {
    throw new RangeError("maximumScreenSpaceErrorPx must be finite and > 0");
  }
  const concurrency = integerAtLeast(
    "concurrency",
    config.concurrency ?? DEFAULT_TILES3D_CONCURRENCY,
    1,
  );
  const cacheBytes = config.cacheBytes ?? DEFAULT_TILES3D_CACHE_BYTES;
  if (!Number.isFinite(cacheBytes) || cacheBytes < 0) {
    throw new RangeError("cacheBytes must be finite and >= 0");
  }
  const verticalExaggeration =
    config.verticalExaggeration === undefined
      ? DEFAULT_VERTICAL_EXAGGERATION
      : config.verticalExaggeration;
  if (!Number.isFinite(verticalExaggeration) || verticalExaggeration <= 0) {
    throw new RangeError("verticalExaggeration must be finite and > 0");
  }
  const verticalPivotZ =
    config.verticalPivotZ === undefined
      ? DEFAULT_VERTICAL_PIVOT_Z
      : config.verticalPivotZ;
  if (!Number.isFinite(verticalPivotZ)) {
    throw new RangeError("verticalPivotZ must be finite");
  }
  const geometricErrorScale = config.geometricErrorScale ?? "maximum";
  if (
    geometricErrorScale !== "maximum" &&
    geometricErrorScale !== "horizontal"
  ) {
    throw new RangeError(
      "geometricErrorScale must be 'maximum' or 'horizontal'",
    );
  }
  return {
    ...config,
    maximumScreenSpaceErrorPx,
    concurrency,
    cacheBytes,
    verticalExaggeration,
    verticalPivotZ,
    geometricErrorScale,
  };
};
