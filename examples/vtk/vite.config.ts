import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const require = createRequire(import.meta.url);
const exampleRoot = dirname(fileURLToPath(import.meta.url));

const vtkJsDir = process.env.VTK_JS_DIR
  ? resolve(process.env.VTK_JS_DIR)
  : dirname(require.resolve("@kitware/vtk.js/package.json"));
const glMatrixDir = dirname(
  require.resolve("gl-matrix/package.json", { paths: [vtkJsDir] }),
);
const lazPerfWasm = resolve(
  dirname(require.resolve("laz-perf/package.json")),
  "lib/web/laz-perf.wasm",
);

export default defineConfig({
  root: exampleRoot,
  plugins: [
    {
      name: "laz-perf-wasm",
      configureServer(server) {
        server.middlewares.use("/laz-perf.wasm", (_request, response) => {
          response.setHeader("Content-Type", "application/wasm");
          response.end(readFileSync(lazPerfWasm));
        });
      },
      generateBundle() {
        this.emitFile({
          type: "asset",
          fileName: "laz-perf.wasm",
          source: readFileSync(lazPerfWasm),
        });
      },
    },
  ],
  resolve: {
    alias: {
      "@kitware/vtk.js": vtkJsDir,
      "gl-matrix": glMatrixDir,
    },
    dedupe: ["@kitware/vtk.js"],
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: true,
    target: "esnext",
  },
});
