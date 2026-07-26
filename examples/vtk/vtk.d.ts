/**
 * Ambient declarations for the vtk.js surfaces this example uses directly.
 * The fork ships no deep-import types, and the library's own shim
 * (`src/vtk.d.ts`) covers only what the renderer adapter touches.
 */

declare module "@kitware/vtk.js/Rendering/Profiles/Geometry";

declare module "@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow" {
  const vtkFullScreenRenderWindow: { newInstance(initialValues?: object): any };
  export default vtkFullScreenRenderWindow;
}
