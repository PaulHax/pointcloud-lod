import { readdir, readFile } from "node:fs/promises";
import { extname, join, relative, resolve } from "node:path";
import { isDeepStrictEqual } from "node:util";

const generatedRoot = resolve(process.argv[2] ?? "");
const checkedRoot = resolve(
  process.argv[3] ?? new URL("./tiles3d-implicit/", import.meta.url).pathname,
);

const collect = async (root, directory = root, result = new Map()) => {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) {
      await collect(root, path, result);
    } else {
      result.set(relative(root, path), await readFile(path));
    }
  }
  return result;
};

const generated = await collect(generatedRoot);
const checked = await collect(checkedRoot);
const paths = [...new Set([...generated.keys(), ...checked.keys()])].sort();
const differences = [];

for (const path of paths) {
  const actual = generated.get(path);
  const expected = checked.get(path);
  if (actual === undefined || expected === undefined) {
    differences.push(
      `${path}: ${actual === undefined ? "missing generated" : "missing checked"}`,
    );
  } else if (extname(path) === ".json") {
    if (!isDeepStrictEqual(JSON.parse(actual), JSON.parse(expected))) {
      differences.push(`${path}: JSON differs`);
    }
  } else if (!actual.equals(expected)) {
    differences.push(`${path}: bytes differ`);
  }
}

if (differences.length > 0) {
  throw new Error(
    `checked terrane fixture is stale:\n${differences.join("\n")}`,
  );
}

console.log(`terrane fixture matches current producer (${paths.length} files)`);
