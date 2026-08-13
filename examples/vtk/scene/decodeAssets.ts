/**
 * Decode assets the page owns and serves.
 *
 * Tile decoding runs in a classic worker, and the Draco and Basis runtimes it
 * may need are injected rather than fetched from a CDN — the library never
 * reaches for a network dependency the host did not name. 3DBAG content is
 * meshopt-compressed and needs neither, but a tileset typed into the URL bar
 * may carry both, so both are wired.
 *
 * `examples/vtk/vite.config.ts` serves these paths from the built package and
 * from the loaders.gl codec libraries.
 */

import { DecodeWorkerPool } from "../../../src";
import type { DecodeWasmUrls } from "../../../src/tiles3d/decode/types";

const absolute = (path: string): string =>
  new URL(path, window.location.href).href;

export const DECODE_WORKER_URL = "/tiles3d-decode-worker.js";

export const decodeWasmUrls = (): DecodeWasmUrls => ({
  draco: {
    wrapperUrl: absolute("/tiles3d-codecs/draco_wasm_wrapper.js"),
    wasmUrl: absolute("/tiles3d-codecs/draco_decoder.wasm"),
  },
  basis: {
    encoderUrl: absolute("/tiles3d-codecs/basis_encoder.js"),
    wasmUrl: absolute("/tiles3d-codecs/basis_encoder.wasm"),
  },
});

export const createDecodeWorkers = (): DecodeWorkerPool =>
  new DecodeWorkerPool({ workerUrl: absolute(DECODE_WORKER_URL) });
