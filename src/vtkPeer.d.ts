/** Minimal ambient declarations for the vtk.js peer's deep imports. */
declare module "@kitware/vtk.js/Rendering/Core/Actor" {
  const vtkActor: { newInstance(initialValues?: object): any };
  export default vtkActor;
}

declare module "@kitware/vtk.js/Rendering/Core/PointGaussianMapper" {
  const vtkPointGaussianMapper: { newInstance(initialValues?: object): any };
  export default vtkPointGaussianMapper;
}

declare module "@kitware/vtk.js/Common/DataModel/PolyData" {
  const vtkPolyData: { newInstance(initialValues?: object): any };
  export default vtkPolyData;
}

declare module "@kitware/vtk.js/Common/Core/DataArray" {
  const vtkDataArray: { newInstance(initialValues?: object): any };
  export default vtkDataArray;
}

declare module "@kitware/vtk.js/Rendering/Core/Mapper" {
  const vtkMapper: { newInstance(initialValues?: object): any };
  export default vtkMapper;
}

declare module "@kitware/vtk.js/Rendering/Core/Texture" {
  const vtkTexture: { newInstance(initialValues?: object): any };
  export default vtkTexture;
}

declare module "@kitware/vtk.js/Interaction/Style/InteractorStyleManipulator" {
  const vtkInteractorStyleManipulator: {
    newInstance(initialValues?: object): any;
  };
  export default vtkInteractorStyleManipulator;
}

declare module "@kitware/vtk.js/Interaction/Manipulators/CompositeMouseManipulator" {
  const vtkCompositeMouseManipulator: {
    extend(publicAPI: any, model: any, initialValues?: object): void;
  };
  export default vtkCompositeMouseManipulator;
}

declare module "@kitware/vtk.js/Interaction/Manipulators/CompositeCameraManipulator" {
  const vtkCompositeCameraManipulator: {
    extend(publicAPI: any, model: any, initialValues?: object): void;
  };
  export default vtkCompositeCameraManipulator;
}

declare module "@kitware/vtk.js/macros" {
  const macro: { obj(publicAPI: any, model: any): void };
  export default macro;
}
