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

// The mesh members decode in a classic worker, so the streamed-scene pages
// need the built artifact rather than the TypeScript source Vite would
// otherwise bundle as a module. `npm run build` produces it.
const decodeWorker = resolve(
  exampleRoot,
  "../../dist/tiles3dDecodeWorker.classic.js",
);

// The same codec libraries the packaged worker stages, served under the path
// the pages inject. Reading them from the dependency keeps a codec update from
// silently leaving the examples on a stale copy.
const codecLibraries = (specifier: string): string =>
  resolve(dirname(fileURLToPath(import.meta.resolve(specifier))), "libs");

const codecs = new Map(
  [
    ["draco_wasm_wrapper.js", "@loaders.gl/draco"],
    ["draco_decoder.wasm", "@loaders.gl/draco"],
    ["basis_encoder.js", "@loaders.gl/textures"],
    ["basis_encoder.wasm", "@loaders.gl/textures"],
  ].map(([name, packageName]) => [
    name!,
    resolve(codecLibraries(packageName!), name!),
  ]),
);

const readDecodeWorker = (): Buffer => {
  try {
    return readFileSync(decodeWorker);
  } catch (cause) {
    throw new Error(
      "The streamed-scene examples need dist/tiles3dDecodeWorker.classic.js — run `npm run build` first.",
      { cause },
    );
  }
};

const servedAssets = (): Map<string, () => Buffer> =>
  new Map([
    ["/laz-perf.wasm", () => readFileSync(lazPerfWasm)],
    ["/tiles3d-decode-worker.js", readDecodeWorker],
    ...[...codecs].map(
      ([name, source]) =>
        [`/tiles3d-codecs/${name}`, () => readFileSync(source)] as const,
    ),
  ]);

const contentType = (path: string): string =>
  path.endsWith(".wasm") ? "application/wasm" : "text/javascript";

export default defineConfig({
  root: exampleRoot,
  plugins: [
    {
      name: "example-runtime-assets",
      configureServer(server) {
        for (const [path, read] of servedAssets()) {
          server.middlewares.use(path, (_request, response) => {
            response.setHeader("Content-Type", contentType(path));
            response.end(read());
          });
        }
      },
      generateBundle() {
        for (const [path, read] of servedAssets()) {
          this.emitFile({
            type: "asset",
            fileName: path.slice(1),
            source: read(),
          });
        }
      },
    },
  ],
  resolve: {
    // vtk.js's Vite ESM build is flat (`.../Camera.js`) while its public
    // subpaths are extensionless (`.../Camera`). Resolve those entry points
    // explicitly; imports inside the built modules are already relative .js
    // paths and need no special handling.
    alias: [
      {
        find: /^@kitware\/vtk\.js\/(.+)$/,
        replacement: `${vtkJsDir}/$1.js`,
      },
      {
        find: "@kitware/vtk.js",
        replacement: `${vtkJsDir}/index.js`,
      },
      { find: "gl-matrix", replacement: glMatrixDir },
    ],
    dedupe: ["@kitware/vtk.js"],
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: true,
    target: "esnext",
    // Names the entry chunk for scripts/verifyChain.mjs, so the chain check
    // reads the bundler's own record instead of guessing the output layout.
    manifest: true,
    rollupOptions: {
      input: {
        complete: resolve(exampleRoot, "index.html"),
        instrumented: resolve(exampleRoot, "complete/index.html"),
        simple: resolve(exampleRoot, "simple/index.html"),
        mesh: resolve(exampleRoot, "mesh/index.html"),
        combined: resolve(exampleRoot, "combined/index.html"),
      },
    },
  },
});
