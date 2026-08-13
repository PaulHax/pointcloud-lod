import { describe, expect, it } from "vitest";

import {
  transformPoint,
  wgs84ToEcef,
  type Mat4,
} from "../../../src/tiles3d/rtc";
import { PLACES, contentToScene, ecefToEnu, rdToScene } from "./places";

const ROTTERDAM = PLACES[0]!;

/**
 * PROJ 9 with the Dutch grids: the same physical point stated in RD New with a
 * NAP height (what the lidar carries) and in WGS84 with an ellipsoidal height
 * (what the mesh carries, via ECEF).
 */
const GROUND_TRUTH = [
  {
    rd: [92600, 436500, 0.0],
    longitude: 4.48026861,
    latitude: 51.91348964,
    height: 43.6029,
  },
  {
    rd: [93600, 436500, 12.5],
    longitude: 4.49480119,
    latitude: 51.91360081,
    height: 56.1014,
  },
  {
    rd: [92600, 437500, -3.0],
    longitude: 4.4800872,
    latitude: 51.92247696,
    height: 40.5925,
  },
  {
    rd: [90100, 434000, 45.0],
    longitude: 4.44440893,
    latitude: 51.89073559,
    height: 88.6347,
  },
] as const;

describe("scene placement", () => {
  it("puts the RD lidar frame and the ECEF mesh frame on the same points", () => {
    const meshPlacement = ecefToEnu(ROTTERDAM);
    const cloudPlacement = rdToScene(ROTTERDAM) as unknown as Mat4;
    for (const truth of GROUND_TRUTH) {
      const viaMesh = transformPoint(
        meshPlacement,
        wgs84ToEcef(truth.longitude, truth.latitude, truth.height),
      );
      const viaCloud = transformPoint(cloudPlacement, [
        truth.rd[0],
        truth.rd[1],
        truth.rd[2],
      ]);
      // Horizontally the two agree to centimetres. Vertically they cannot: a
      // similarity has no curvature, so the flat cloud rises above the curved
      // mesh by d²/2R — a metre only once you are kilometres from the origin.
      expect(viaCloud[0]).toBeCloseTo(viaMesh[0], 1);
      expect(viaCloud[1]).toBeCloseTo(viaMesh[1], 1);
      const distance = Math.hypot(viaMesh[0], viaMesh[1]);
      expect(Math.abs(viaCloud[2] - viaMesh[2])).toBeLessThanOrEqual(
        0.05 + (distance * distance) / 12_000_000,
      );
    }
  });

  it("places the origin of every place at the scene origin", () => {
    for (const place of PLACES) {
      const origin = transformPoint(rdToScene(place) as unknown as Mat4, [
        place.rdOrigin[0],
        place.rdOrigin[1],
        0,
      ]);
      expect(origin[0]).toBeCloseTo(0, 6);
      expect(origin[1]).toBeCloseTo(0, 6);
      expect(origin[2]).toBeCloseTo(0, 6);
    }
  });

  it("rotates Y-up glTF content into the Z-up scene frame", () => {
    // A point one metre above the ellipsoid at the origin, stated the way 3D
    // Tiles glTF content states it: Y is up, and +Z runs south.
    const ecef = wgs84ToEcef(
      ROTTERDAM.longitude,
      ROTTERDAM.latitude,
      ROTTERDAM.napZeroHeight + 1,
    );
    const yUp: [number, number, number] = [ecef[0], ecef[2], -ecef[1]];
    const placed = transformPoint(contentToScene(ROTTERDAM), yUp);
    expect(placed[0]).toBeCloseTo(0, 6);
    expect(placed[1]).toBeCloseTo(0, 6);
    expect(placed[2]).toBeCloseTo(1, 6);
  });

  it("keeps the cloud placement a similarity, which LOD selection requires", () => {
    const matrix = rdToScene(ROTTERDAM);
    const columns = [
      [matrix[0]!, matrix[1]!, matrix[2]!],
      [matrix[4]!, matrix[5]!, matrix[6]!],
      [matrix[8]!, matrix[9]!, matrix[10]!],
    ];
    const lengths = columns.map((column) => Math.hypot(...column));
    for (const length of lengths) expect(length).toBeCloseTo(lengths[0]!, 12);
    const [ax, ay] = columns[0]!;
    const [bx, by] = columns[1]!;
    expect(ax! * bx! + ay! * by!).toBeCloseTo(0, 12);
  });
});
