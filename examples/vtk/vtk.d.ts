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

declare module "@kitware/vtk.js/Interaction/Style/InteractorStyleManipulator" {
  const vtkInteractorStyleManipulator: {
    newInstance(initialValues?: object): any;
    dollyToPosition(
      factor: number,
      position: { x: number; y: number },
      renderer: any,
      interactor: any,
    ): void;
  };
  export default vtkInteractorStyleManipulator;
}

declare module "@kitware/vtk.js/Interaction/Manipulators/MouseCameraTrackballPanManipulator" {
  const manipulator: { newInstance(initialValues?: object): any };
  export default manipulator;
}

declare module "@kitware/vtk.js/Interaction/Manipulators/MouseCameraTrackballRotateManipulator" {
  const manipulator: { newInstance(initialValues?: object): any };
  export default manipulator;
}

declare module "@kitware/vtk.js/Interaction/Manipulators/CompositeMouseManipulator" {
  const manipulator: {
    extend(publicAPI: object, model: object, initialValues?: object): void;
  };
  export default manipulator;
}

declare module "@kitware/vtk.js/Interaction/Manipulators/CompositeCameraManipulator" {
  const manipulator: {
    extend(publicAPI: object, model: object, initialValues?: object): void;
  };
  export default manipulator;
}

declare module "@kitware/vtk.js/macros" {
  const macro: { obj(publicAPI: object, model: object): void };
  export default macro;
}
