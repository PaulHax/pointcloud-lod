declare const SCENE_ENU: unique symbol;

/** Canonical, unexaggerated scene-ENU coordinates. */
export type SceneEnuPoint = readonly [number, number, number] & {
  readonly [SCENE_ENU]: true;
};

export const sceneEnuPoint = (
  x: number,
  y: number,
  z: number,
): SceneEnuPoint => {
  if (![x, y, z].every(Number.isFinite)) {
    throw new TypeError("scene point must contain three finite coordinates");
  }
  return [x, y, z] as unknown as SceneEnuPoint;
};
