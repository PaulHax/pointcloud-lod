import { defineConfig } from 'vite';

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
});
