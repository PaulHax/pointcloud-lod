/**
 * Draw-time state written without marking vtk.js objects modified.
 *
 * vtk.js regenerates a mapper's shader source whenever the actor, the
 * actor's property or the mapper is newer than that source. Point size and a
 * point tile's draw count change no shader source: vtk.js sets point size as
 * a uniform in every draw and reads the count when it issues the draw.
 * Writing them through their setters would still rebuild and rehash the
 * shader of every tile they touch, about 0.1 ms each, so an automatic
 * point-size or density change across a few hundred tiles added 20 ms or
 * more to one frame.
 */

/** The part of a vtk.js object these helpers write through. */
type QuietlySettable<Values> = {
  set(values: Values, noWarning: boolean, noFunction: boolean): boolean;
};

/** Set an actor's point size in CSS pixels, leaving its shaders valid. */
export const setActorPointSize = (
  actor: { getProperty(): QuietlySettable<{ readonly pointSize: number }> },
  pointSize: number,
): void => {
  actor.getProperty().set({ pointSize }, true, true);
};

/**
 * Set how many of a tile's points its mapper draws, leaving its shaders
 * valid. `count` is a whole number of points, at most the tile's.
 */
export const setMapperPointCount = (
  mapper: QuietlySettable<{ readonly maximumPointCount: number }>,
  count: number,
): void => {
  mapper.set({ maximumPointCount: count }, true, true);
};
