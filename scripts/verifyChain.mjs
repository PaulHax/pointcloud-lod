// Prove the standalone example was built from the vtk.js the adapter requires.
//
// The adapter needs `vtkPointGaussianMapper`, its OpenGL override, and the
// Geometry profile that registers them. All three are silent failures: a bundle
// missing them builds fine and draws nothing. So assert them in the built
// bundle, against a vtk.js checkout pinned to the commit in vtkjs-fork.env.
//
//   VTK_JS_DIR=/path/to/vtk-js/dist/esm node scripts/verifyChain.mjs

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { createRequire } from "node:module";
import { readFileSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");

const fail = (message) => {
  console.error(`verifyChain: FAIL: ${message}`);
  process.exit(1);
};

const readPin = () => {
  const text = readFileSync(join(root, "vtkjs-fork.env"), "utf8");
  const pin = Object.fromEntries(
    text
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith("#"))
      .map((line) => {
        const index = line.indexOf("=");
        return [line.slice(0, index).trim(), line.slice(index + 1).trim()];
      }),
  );
  if (!/^[0-9a-f]{40}$/.test(pin.VTKJS_FORK_COMMIT ?? "")) {
    fail(
      `VTKJS_FORK_COMMIT must be a full 40-character sha, got '${pin.VTKJS_FORK_COMMIT}'`,
    );
  }
  return pin;
};

const gitOutput = (cwd, ...args) => {
  try {
    return execFileSync("git", args, {
      cwd,
      stdio: ["ignore", "pipe", "ignore"],
    })
      .toString()
      .trim();
  } catch {
    return null;
  }
};

const checkVtkJs = (pin) => {
  const dir = process.env.VTK_JS_DIR
    ? resolve(process.env.VTK_JS_DIR)
    : dirname(require.resolve("@kitware/vtk.js/package.json"));
  const source = statSync(dir, { throwIfNoEntry: false }) ? dir : null;
  if (!source) fail(`vtk.js directory not found: ${dir}`);
  const toplevel = gitOutput(source, "rev-parse", "--show-toplevel");
  if (!toplevel) {
    fail(
      `${source} is not inside a git checkout, so the vtk.js commit it was built from cannot be proven`,
    );
  }
  const head = gitOutput(toplevel, "rev-parse", "HEAD");
  if (head !== pin.VTKJS_FORK_COMMIT) {
    fail(
      `vtk.js is at ${head}, vtkjs-fork.env pins ${pin.VTKJS_FORK_COMMIT}. Check out the pinned commit and rebuild 'npm run build:esm', or change the pin on purpose.`,
    );
  }
  return head;
};

// Every released @kitware/vtk.js lacks vtkPointGaussianMapper, so a declared
// range is a promise npm could satisfy with a build the adapter cannot use.
const checkNoVtkJsRange = () => {
  const manifest = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));
  for (const field of [
    "dependencies",
    "devDependencies",
    "peerDependencies",
    "optionalDependencies",
  ]) {
    const spec = manifest[field]?.["@kitware/vtk.js"];
    if (spec) {
      fail(
        `package.json declares ${field}['@kitware/vtk.js'] = '${spec}'; no released version carries vtkPointGaussianMapper, so the range would resolve to a build the adapter cannot use`,
      );
    }
  }
};

// The bundle is minified: class names are mangled, but the strings vtk.js
// registers itself with survive. Quote style differs per bundler.
const registers = (text, className) =>
  new RegExp(`classHierarchy\\.push\\(\\s*["'\`]${className}["'\`]`).test(text);

const checkExampleBundle = () => {
  const output = join(root, "examples", "vtk", "dist");
  // Vite's own manifest names the entry chunk, so this check stays correct
  // whatever `base` and `build.assetsDir` the example config uses, and it
  // ignores the worker and vendor chunks emitted beside it.
  const manifestPath = join(output, ".vite", "manifest.json");
  const manifest = statSync(manifestPath, { throwIfNoEntry: false })
    ? JSON.parse(readFileSync(manifestPath, "utf8"))
    : {};
  const entries = Object.values(manifest).filter((chunk) => chunk.isEntry);
  const completeEntry = entries.find((chunk) => chunk.src === "index.html");
  const simpleEntry = entries.find(
    (chunk) => chunk.src === "simple/index.html",
  );
  if (!completeEntry || !simpleEntry) {
    fail(
      `complete or simple example entry is missing from ${manifestPath} — run 'npm run example:build' first`,
    );
  }

  const entryBundle = (entry) => {
    const files = new Set();
    const visit = (chunk) => {
      if (files.has(chunk.file)) return;
      files.add(chunk.file);
      for (const imported of chunk.imports ?? []) visit(manifest[imported]);
    };
    visit(entry);
    return Buffer.concat(
      [...files].map((name) => readFileSync(join(output, name))),
    );
  };

  const requiredFeatures = (text) => ({
    // The mapper the renderer adapter instantiates.
    vtkPointGaussianMapper: registers(text, "vtkPointGaussianMapper"),
    // Without the OpenGL override the mapper draws nothing.
    vtkOpenGLPointGaussianMapper: registers(
      text,
      "vtkOpenGLPointGaussianMapper",
    ),
    // Both arrive with the Geometry profile the example must import.
    vtkOpenGLPolyDataMapper: registers(text, "vtkOpenGLPolyDataMapper"),
    vtkOpenGLActor: registers(text, "vtkOpenGLActor"),
    // World-space point sizing is the fork feature the pinned commit adds.
    worldSize:
      /["'`]scaleFactor["'`]\s*,\s*["'`]circle["'`]\s*,\s*["'`]worldSize["'`]/.test(
        text,
      ),
    // Progressive density depends on this core mapper API reaching the bundle;
    // an older/stale vtk.js build still carries every class above but would
    // fail only when the first tile actor applies its draw cap.
    maximumPointCount: text.includes("maximumPointCount"),
  });
  for (const [name, entry] of [
    ["complete", completeEntry],
    ["simple", simpleEntry],
  ]) {
    const required = requiredFeatures(entryBundle(entry).toString("utf8"));
    const missing = Object.entries(required)
      .filter(([, present]) => !present)
      .map(([feature]) => feature);
    if (missing.length) {
      fail(
        `${name} example is missing ${missing.join(", ")} — built against a vtk.js without the point-gaussian work, or the Geometry profile import was dropped`,
      );
    }
  }
  const file = join(output, completeEntry.file);
  const bytes = entryBundle(completeEntry);
  return {
    file,
    sha256: createHash("sha256").update(bytes).digest("hex"),
    bytes: bytes.length,
    present: Object.keys(requiredFeatures(bytes.toString("utf8"))),
  };
};

const pin = readPin();
const commit = checkVtkJs(pin);
checkNoVtkJsRange();
const bundle = checkExampleBundle();

console.log(`  vtk.js         : ${commit} (${pin.VTKJS_FORK_BRANCH})`);
console.log(`  example bundle : ${bundle.file}`);
console.log(`  bundle sha256  : ${bundle.sha256} (${bundle.bytes} bytes)`);
console.log(`  bundle carries : ${bundle.present.join(", ")}`);
console.log("CHAIN VERIFY PASS: example built from the pinned vtk.js commit");
