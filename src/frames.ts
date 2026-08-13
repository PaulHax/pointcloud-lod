declare const SCENE_POINT: unique symbol;

/** Canonical, unexaggerated scene coordinates. */
export type ScenePoint = readonly [number, number, number] & {
  readonly [SCENE_POINT]: true;
};

export const scenePoint = (x: number, y: number, z: number): ScenePoint => {
  if (![x, y, z].every(Number.isFinite)) {
    throw new TypeError("scene point must contain three finite coordinates");
  }
  return [x, y, z] as unknown as ScenePoint;
};
