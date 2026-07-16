/**
 * Minimal ambient declarations for the '@kitware/vtk.js' peer dependency.
 * The peer ships no such deep-import types to non-installed consumers; the
 * adapter uses only these surfaces, and tests stub the same specifiers.
 */

declare module '@kitware/vtk.js/Rendering/Core/Actor' {
  const vtkActor: { newInstance(initialValues?: object): any };
  export default vtkActor;
}

declare module '@kitware/vtk.js/Rendering/Core/PointGaussianMapper' {
  const vtkPointGaussianMapper: { newInstance(initialValues?: object): any };
  export default vtkPointGaussianMapper;
}

declare module '@kitware/vtk.js/Common/DataModel/PolyData' {
  const vtkPolyData: { newInstance(initialValues?: object): any };
  export default vtkPolyData;
}

declare module '@kitware/vtk.js/Common/Core/DataArray' {
  const vtkDataArray: { newInstance(initialValues?: object): any };
  export default vtkDataArray;
}
