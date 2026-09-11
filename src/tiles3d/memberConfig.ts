import { integerAtLeast } from "../numeric";
import {
  DEFAULT_MAXIMUM_SCREEN_SPACE_ERROR_PX,
  DEFAULT_INTERACTION_RETENTION_MAX_TRIANGLES,
  DEFAULT_INTERACTION_RETENTION_MAX_ACTORS,
  DEFAULT_TILES3D_CACHE_BYTES,
  DEFAULT_TILES3D_CONCURRENCY,
  DEFAULT_VERTICAL_EXAGGERATION,
  DEFAULT_VERTICAL_PIVOT_Z,
  type Tiles3dMemberConfig,
} from "./memberTypes";

const AFFINE_ENTRY_ABS_TOL = 1e-12;
const AFFINE_DETERMINANT_FLOOR = 1e-15;

export const finiteAffineMatrix = (
  matrix: readonly number[],
  label: string,
): readonly number[] => {
  if (
    !matrix ||
    matrix.length !== 16 ||
    Array.from(matrix).some((value) => !Number.isFinite(value))
  ) {
    throw new TypeError(`${label} must contain 16 finite numbers`);
  }
  if (
    Math.abs(matrix[3]!) > AFFINE_ENTRY_ABS_TOL ||
    Math.abs(matrix[7]!) > AFFINE_ENTRY_ABS_TOL ||
    Math.abs(matrix[11]!) > AFFINE_ENTRY_ABS_TOL ||
    Math.abs(matrix[15]! - 1) > AFFINE_ENTRY_ABS_TOL
  ) {
    throw new TypeError(`${label} must be an affine column-major matrix`);
  }
  const determinant =
    matrix[0]! * (matrix[5]! * matrix[10]! - matrix[9]! * matrix[6]!) -
    matrix[4]! * (matrix[1]! * matrix[10]! - matrix[9]! * matrix[2]!) +
    matrix[8]! * (matrix[1]! * matrix[6]! - matrix[5]! * matrix[2]!);
  if (
    !Number.isFinite(determinant) ||
    Math.abs(determinant) <= AFFINE_DETERMINANT_FLOOR
  ) {
    throw new TypeError(`${label} must be invertible`);
  }
  return [...matrix];
};

export const validateTiles3dMemberConfig = (
  config: Tiles3dMemberConfig,
): Tiles3dMemberConfig => {
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
  finiteAffineMatrix(config.tilesetToScene, "tilesetToScene");
  const maximum =
    config.maximumScreenSpaceErrorPx ?? DEFAULT_MAXIMUM_SCREEN_SPACE_ERROR_PX;
  if (!Number.isFinite(maximum) || maximum <= 0) {
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
    interactionRetentionMaxTriangles: integerAtLeast(
      "interactionRetentionMaxTriangles",
      config.interactionRetentionMaxTriangles ??
        DEFAULT_INTERACTION_RETENTION_MAX_TRIANGLES,
      0,
    ),
    interactionRetentionMaxActors: integerAtLeast(
      "interactionRetentionMaxActors",
      config.interactionRetentionMaxActors ??
        DEFAULT_INTERACTION_RETENTION_MAX_ACTORS,
      0,
    ),
    concurrency,
    cacheBytes,
    verticalExaggeration,
    verticalPivotZ,
    geometricErrorScale,
  };
};
