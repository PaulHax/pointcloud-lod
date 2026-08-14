import { execFile } from "node:child_process";
import { mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

import gltfValidator from "gltf-validator";
import { afterAll, describe, expect, it } from "vitest";

const execute = promisify(execFile);
const fixtureRoot = fileURLToPath(
  new URL("./tiles3d-implicit/", import.meta.url),
);
const temporaryRoot = await mkdtemp(
  resolve(tmpdir(), `pointcloud-lod-implicit-${process.pid}-`),
);

afterAll(async () => {
  await rm(temporaryRoot, { recursive: true, force: true });
});

const filesWithExtension = async (root, extension, relative = "") => {
  const files = [];
  for (const entry of await readdir(resolve(root, relative), {
    withFileTypes: true,
  })) {
    const path = join(relative, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await filesWithExtension(root, extension, path)));
    } else if (extname(entry.name) === extension) {
      files.push(resolve(root, path));
    }
  }
  return files.sort();
};

const validateTileset = async (tilesetFile, label) => {
  const optionsFile = resolve(temporaryRoot, `${label}-options.json`);
  const reportFile = resolve(temporaryRoot, `${label}-report.json`);
  await writeFile(optionsFile, '{"validateContentData":false}\n', "utf8");
  const validator = fileURLToPath(
    new URL("../../node_modules/.bin/3d-tiles-validator", import.meta.url),
  );
  await execute(
    validator,
    [
      "--tilesetFile",
      tilesetFile,
      "--optionsFile",
      optionsFile,
      "--reportFile",
      reportFile,
    ],
    { cwd: dirname(tilesetFile) },
  );
  const report = JSON.parse(await readFile(reportFile, "utf8"));
  expect(report, JSON.stringify(report.issues ?? [], null, 2)).toMatchObject({
    numErrors: 0,
    numWarnings: 0,
    numInfos: 0,
  });
  expect(report.issues ?? []).toEqual([]);
};

describe("the producer-owned implicit terrain fixture", () => {
  it("passes the independent explicit and implicit 3D Tiles validators", async () => {
    await validateTileset(resolve(fixtureRoot, "tileset.json"), "implicit");
    await validateTileset(
      resolve(fixtureRoot, "explicit/tileset.json"),
      "explicit",
    );
  }, 120_000);

  it("passes every generated terrain content through glTF Validator", async () => {
    const contents = await filesWithExtension(fixtureRoot, ".glb");
    expect(contents.length).toBeGreaterThan(4);
    for (const content of contents) {
      const report = await gltfValidator.validateBytes(
        new Uint8Array(await readFile(content)),
        { uri: content, writeTimestamp: false },
      );
      expect(
        report.issues.numErrors,
        `${content}: ${JSON.stringify(report.issues.messages, null, 2)}`,
      ).toBe(0);
      expect(report.issues.numWarnings).toBe(0);
    }
  });
});
