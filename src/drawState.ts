/**
 * Draw-time state written without marking vtk.js objects modified.
 *
 * vtk.js regenerates a mapper's shader source whenever the actor, or the
 * actor's property, is newer than that source. Point size changes no shader
 * source: vtk.js sets it as a uniform in every draw. Writing it through
 * `setPointSize` would still rebuild and rehash the shader of every tile it
 * touches, about 0.1 ms each, so an automatic point-size change across a few
 * hundred tiles added 20 to 40 ms to one frame.
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
