/**
 * The places both streamed-scene examples fly to, and the georeferencing that
 * puts a mesh and a point cloud in one frame.
 *
 * Each place fixes a scene origin: a local ENU frame whose Z is NAP, the Dutch
 * height datum both datasets are levelled to. From that one origin the mesh
 * gets its `tilesetToScene` and the cloud gets its model matrix, which is the
 * only way two members can share a camera.
 */

import { createEcefToEnuTransform, type Mat4 } from "../../../src/tiles3d/rtc";
import { PLACE_LABELS } from "./sceneCatalog";

/** 3DBAG LoD2.2, CC BY 4.0, no key, CORS-enabled. */
export const BAG3D_ENDPOINT =
  "https://data.3dbag.nl/v20250903/cesium3dtiles/lod22";

const AHN4 =
  "https://open-lidar-data.s3.eu-central-1.amazonaws.com/data/NL/AHN/AHN4_2020-2022/copc";

export type Place = {
  readonly label: string;
  /**
   * Scene origin in Amersfoort / RD New (EPSG:28992), metres. Also the origin
   * of the ENU scene frame, so scene coordinates stay small near the view.
   */
  readonly rdOrigin: readonly [number, number];
  readonly longitude: number;
  readonly latitude: number;
  /**
   * Ellipsoidal height of NAP zero at the origin. Using it as the ENU origin
   * height makes scene Z read as NAP, which is what the lidar Z already is.
   */
  readonly napZeroHeight: number;
  /** Meridian convergence of RD at the origin, radians. */
  readonly convergence: number;
  /** RD point scale factor at the origin (grid metres per true metre). */
  readonly gridScale: number;
  /** AHN4 map sheet covering the origin. */
  readonly copcUrl: string;
  /** Opening camera distance from the origin, metres. */
  readonly viewDistance: number;
};

/**
 * Derived with PROJ 9 and the Dutch grids: `EPSG:7415 -> EPSG:4979` for the
 * origin's geographic position and NAP-zero height, and the RD projection's
 * own factors at that position for convergence and scale.
 */
export const PLACES: readonly Place[] = [
  {
    label: PLACE_LABELS.rotterdam,
    rdOrigin: [92_600, 436_500],
    longitude: 4.4802686,
    latitude: 51.9134896,
    napZeroHeight: 43.603,
    convergence: (-0.715364 * Math.PI) / 180,
    gridScale: 0.999936159,
    copcUrl: `${AHN4}/C_37HN1.copc.laz`,
    viewDistance: 1400,
  },
  {
    label: PLACE_LABELS.delft,
    rdOrigin: [84_100, 447_000],
    longitude: 4.3545773,
    latitude: 52.0068349,
    napZeroHeight: 43.509,
    convergence: (-0.814977 * Math.PI) / 180,
    gridScale: 0.999940362,
    copcUrl: `${AHN4}/C_37EN1.copc.laz`,
    viewDistance: 1100,
  },
  {
    label: PLACE_LABELS.amsterdam,
    rdOrigin: [121_000, 486_300],
    longitude: 4.8880409,
    latitude: 52.3635352,
    napZeroHeight: 42.998,
    convergence: (-0.395081 * Math.PI) / 180,
    gridScale: 0.999918305,
    copcUrl: `${AHN4}/C_25GN1.copc.laz`,
    viewDistance: 1300,
  },
];

/** ECEF to the place's ENU scene frame — the mesh member's placement. */
export const ecefToEnu = (place: Place): Mat4 =>
  createEcefToEnuTransform(
    place.longitude,
    place.latitude,
    place.napZeroHeight,
  );

/** Tile ECEF to the place's ENU scene frame; content-axis correction is internal. */
export const contentToScene = ecefToEnu;

/**
 * RD New (x, y, NAP z) to the same ENU frame — the point cloud's model matrix.
 *
 * RD is conformal, so locally it is a rotation by the meridian convergence and
 * a uniform scale, which is exactly the similarity the LOD controller's
 * screen-space error math requires. What the similarity cannot carry is the
 * curvature the mesh keeps: the cloud is flat where the mesh bends away, so
 * ground and buildings separate by d²/2R — 17 cm a kilometre out, a metre at
 * three. The demo flies the origin, where that is invisible; a survey tool
 * would place both members in a projected frame instead.
 */
export const rdToScene = (place: Place): number[] => {
  const [x0, y0] = place.rdOrigin;
  const scale = 1 / place.gridScale;
  const cos = Math.cos(place.convergence) * scale;
  const sin = Math.sin(place.convergence) * scale;
  // Column-major: the columns are the images of RD x, RD y and NAP z. Z takes
  // the same scale as the horizontal axes even though NAP heights are already
  // true metres, because a controller needs one uniform scale and 64 ppm is
  // half a centimetre on the tallest building in the country.
  return [
    cos,
    -sin,
    0,
    0,
    sin,
    cos,
    0,
    0,
    0,
    0,
    scale,
    0,
    -(cos * x0 + sin * y0),
    sin * x0 - cos * y0,
    0,
    1,
  ];
};
