import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const dependencyDist = (specifier: string): string =>
  dirname(fileURLToPath(import.meta.resolve(specifier)));

const codecSources = new Map([
  [
    "draco_wasm_wrapper.js",
    resolve(dependencyDist("@loaders.gl/draco"), "libs/draco_wasm_wrapper.js"),
  ],
  [
    "draco_decoder.wasm",
    resolve(dependencyDist("@loaders.gl/draco"), "libs/draco_decoder.wasm"),
  ],
  [
    "basis_encoder.js",
    resolve(dependencyDist("@loaders.gl/textures"), "libs/basis_encoder.js"),
  ],
  [
    "basis_encoder.wasm",
    resolve(dependencyDist("@loaders.gl/textures"), "libs/basis_encoder.wasm"),
  ],
]);

export default defineConfig({
  plugins: [
    {
      name: "disable-worker-remote-codec-defaults",
      renderChunk(code) {
        return {
          code: code
            .replaceAll(
              "https://unpkg.com/@loaders.gl",
              "offline-dependency-disabled:loaders.gl",
            )
            .replaceAll(
              "https://www.gstatic.com/draco/versioned/decoders/",
              "offline-dependency-disabled:draco/",
            )
            .replaceAll(
              "https://raw.githubusercontent.com/google/draco/",
              "offline-dependency-disabled:draco-encoder/",
            ),
          map: null,
        };
      },
    },
    {
      name: "stage-owned-tiles3d-codecs",
      writeBundle(outputOptions) {
        const outputDirectory = resolve(
          outputOptions.dir ?? "dist",
          "tiles3d-codecs",
        );
        mkdirSync(outputDirectory, { recursive: true });
        for (const [name, source] of codecSources) {
          copyFileSync(source, resolve(outputDirectory, name));
        }
      },
    },
  ],
  build: {
    emptyOutDir: false,
    lib: {
      entry: "src/tiles3dDecodeWorker.ts",
      formats: ["iife"],
      name: "Tiles3dDecodeWorker",
      fileName: () => "tiles3dDecodeWorker.classic.js",
    },
    sourcemap: true,
  },
});
