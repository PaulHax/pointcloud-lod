import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

const stub = (name: string): string =>
  fileURLToPath(new URL(`./test/stubs/${name}.ts`, import.meta.url));

export default defineConfig({
  build: {
    lib: {
      entry: 'src/index.ts',
      formats: ['es'],
      fileName: 'index',
    },
    rollupOptions: {
      // Never bundle the vtk.js peer or the copc reader; consumers resolve them.
      external: [/^@kitware\/vtk\.js/, 'copc'],
    },
    sourcemap: true,
  },
  test: {
    // The adapter's vtk.js deep imports resolve to recording stubs so the
    // suite runs without the peer installed; real-GL behavior is covered by
    // downstream integration tests.
    alias: {
      '@kitware/vtk.js/Rendering/Core/Actor': stub('vtkActor'),
      '@kitware/vtk.js/Rendering/Core/PointGaussianMapper': stub(
        'vtkPointGaussianMapper',
      ),
      '@kitware/vtk.js/Common/DataModel/PolyData': stub('vtkPolyData'),
      '@kitware/vtk.js/Common/Core/DataArray': stub('vtkDataArray'),
    },
  },
});
