#!/usr/bin/env node

import { Document, NodeIO } from "@gltf-transform/core";
import { ALL_EXTENSIONS } from "@gltf-transform/extensions";
import { draco } from "@gltf-transform/functions";
import draco3d from "draco3dgltf";
import { ktx2 } from "ktx2-encoder/gltf-transform";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import sharp from "sharp";

export const FIXTURE_SEED = 0x3d711e5;
export const FIXTURE_CODEC_TILES = Object.freeze({
  draco: "level-1-0",
  ktx2: "level-2-1-0",
});
export const DEFAULT_FIXTURE_DIRECTORY = resolve(
  fileURLToPath(new URL("..", import.meta.url)),
  "artifacts/fixtures/tiles3d-1.1",
);

const CONTENT_DIRECTORY = "content";
const ECEF_LONGITUDE_DEGREES = -77.0353;
const ECEF_LATITUDE_DEGREES = 38.8895;
const ECEF_ALTITUDE_METERS = 42;

let ioPromise;

async function nodeIO() {
  if (!ioPromise) {
    ioPromise = Promise.all([
      draco3d.createDecoderModule(),
      draco3d.createEncoderModule(),
    ]).then(([decoder, encoder]) =>
      new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({
        "draco3d.decoder": decoder,
        "draco3d.encoder": encoder,
      }),
    );
  }
  return ioPromise;
}

function accessor(document, name, type, array) {
  return document
    .createAccessor(name)
    .setType(type)
    .setArray(array)
    .setBuffer(document.getRoot().listBuffers()[0]);
}

function quadPrimitive(document, name, x0, x1, y0, y1, height, material) {
  const positions = new Float32Array([
    x0,
    height,
    -y0,
    x1,
    height,
    -y0,
    x1,
    height,
    -y1,
    x0,
    height,
    -y1,
  ]);
  // glTF content is Y-up. The 3D Tiles decode transform maps these coordinates
  // to the intended tile-local (x, y, height) horizontal plane with +Z normals.
  const normals = new Float32Array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]);
  const uvs = new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]);
  const indices = new Uint16Array([0, 1, 2, 0, 2, 3]);

  return document
    .createPrimitive(name)
    .setAttribute(
      "POSITION",
      accessor(document, `${name}-positions`, "VEC3", positions),
    )
    .setAttribute(
      "NORMAL",
      accessor(document, `${name}-normals`, "VEC3", normals),
    )
    .setAttribute("TEXCOORD_0", accessor(document, `${name}-uvs`, "VEC2", uvs))
    .setIndices(accessor(document, `${name}-indices`, "SCALAR", indices))
    .setMaterial(material);
}

async function checkerTexture(seed) {
  const size = 16;
  const pixels = new Uint8Array(size * size * 4);
  const accent = [
    32 + ((seed * 17) % 32),
    192 + ((seed * 29) % 56),
    192 + ((seed * 43) % 56),
  ];

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const offset = (y * size + x) * 4;
      const bright = ((x >> 2) + (y >> 2) + seed) % 2 === 0;
      pixels[offset] = bright ? accent[0] : 28;
      pixels[offset + 1] = bright ? accent[1] : 36;
      pixels[offset + 2] = bright ? accent[2] : 48;
      pixels[offset + 3] = 255;
    }
  }

  return new Uint8Array(
    await sharp(pixels, { raw: { width: size, height: size, channels: 4 } })
      .png({ compressionLevel: 9, adaptiveFiltering: false })
      .toBuffer(),
  );
}

async function decodeImage(buffer) {
  const { data, info } = await sharp(buffer)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return {
    data: new Uint8Array(data),
    width: info.width,
    height: info.height,
  };
}

async function tileDocument({ name, size, height, textureSeed, compression }) {
  const document = new Document();
  document.createBuffer("tile-buffer");

  const image = await checkerTexture(textureSeed);
  const texture = document
    .createTexture(`${name}-checker`)
    .setMimeType("image/png")
    .setImage(image);
  const firstMaterial = document
    .createMaterial(`${name}-material-a`)
    .setBaseColorFactor([1, 1, 1, 1])
    .setBaseColorTexture(texture)
    .setRoughnessFactor(0.82)
    .setMetallicFactor(0);
  const secondMaterial = document
    .createMaterial(`${name}-material-b`)
    .setBaseColorFactor([0.92, 0.96, 1, 1])
    .setBaseColorTexture(texture)
    .setRoughnessFactor(0.68)
    .setMetallicFactor(0);

  const mesh = document
    .createMesh(`${name}-mesh`)
    .addPrimitive(
      quadPrimitive(
        document,
        `${name}-west`,
        -size / 2,
        0,
        -size / 2,
        size / 2,
        0,
        firstMaterial,
      ),
    )
    .addPrimitive(
      quadPrimitive(
        document,
        `${name}-east`,
        0,
        size / 2,
        -size / 2,
        size / 2,
        height,
        secondMaterial,
      ),
    );
  const node = document.createNode(`${name}-node`).setMesh(mesh);
  document.createScene("Tile scene").addChild(node);

  if (compression === "draco") {
    await document.transform(draco({ method: "edgebreaker" }));
  } else if (compression === "ktx2") {
    await document.transform(
      ktx2({
        enableDebug: false,
        generateMipmap: true,
        imageDecoder: decodeImage,
        isKTX2File: true,
        isPerceptual: true,
        isSetKTX2SRGBTransferFunc: true,
        isUASTC: true,
        needSupercompression: false,
      }),
    );
  }

  return document;
}

function translation(x, y, z = 0) {
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1];
}

function ecefFromWgs84(longitudeDegrees, latitudeDegrees, altitudeMeters) {
  const longitude = (longitudeDegrees * Math.PI) / 180;
  const latitude = (latitudeDegrees * Math.PI) / 180;
  const semimajor = 6378137;
  const eccentricitySquared = 6.69437999014e-3;
  const sinLatitude = Math.sin(latitude);
  const primeVertical =
    semimajor / Math.sqrt(1 - eccentricitySquared * sinLatitude * sinLatitude);
  return [
    (primeVertical + altitudeMeters) * Math.cos(latitude) * Math.cos(longitude),
    (primeVertical + altitudeMeters) * Math.cos(latitude) * Math.sin(longitude),
    (primeVertical * (1 - eccentricitySquared) + altitudeMeters) * sinLatitude,
  ];
}

function rootEcefTransform() {
  const longitude = (ECEF_LONGITUDE_DEGREES * Math.PI) / 180;
  const latitude = (ECEF_LATITUDE_DEGREES * Math.PI) / 180;
  const [x, y, z] = ecefFromWgs84(
    ECEF_LONGITUDE_DEGREES,
    ECEF_LATITUDE_DEGREES,
    ECEF_ALTITUDE_METERS,
  );

  const east = [-Math.sin(longitude), Math.cos(longitude), 0];
  const north = [
    -Math.sin(latitude) * Math.cos(longitude),
    -Math.sin(latitude) * Math.sin(longitude),
    Math.cos(latitude),
  ];
  const up = [
    Math.cos(latitude) * Math.cos(longitude),
    Math.cos(latitude) * Math.sin(longitude),
    Math.sin(latitude),
  ];

  return [...east, 0, ...north, 0, ...up, 0, x, y, z, 1];
}

function box(halfWidth, halfHeight, halfDepth) {
  return [0, 0, halfDepth, halfWidth, 0, 0, 0, halfHeight, 0, 0, 0, halfDepth];
}

function contentURI(name) {
  return `${CONTENT_DIRECTORY}/${name}.glb`;
}

function makeTile({ name, size, geometricError, transform, children = [] }) {
  const tile = {
    boundingVolume: { box: box(size / 2, size / 2, 8) },
    geometricError,
    refine: "REPLACE",
    content: { uri: contentURI(name) },
  };
  if (transform) tile.transform = transform;
  if (children.length > 0) tile.children = children;
  return tile;
}

function fixtureTiles() {
  const definitions = [
    {
      name: "level-0-root",
      size: 112,
      height: 3,
      textureSeed: FIXTURE_SEED,
      compression: "none",
    },
  ];
  const levelOne = [];
  const offsets = [
    [-28, -28],
    [28, -28],
    [-28, 28],
    [28, 28],
  ];

  for (let quadrant = 0; quadrant < offsets.length; quadrant += 1) {
    const levelOneName = `level-1-${quadrant}`;
    const children = [];
    definitions.push({
      name: levelOneName,
      size: 52,
      height: 4 + quadrant,
      textureSeed: FIXTURE_SEED + quadrant + 1,
      compression:
        levelOneName === FIXTURE_CODEC_TILES.draco ? "draco" : "none",
    });

    for (let leaf = 0; leaf < 2; leaf += 1) {
      const leafName = `level-2-${quadrant}-${leaf}`;
      definitions.push({
        name: leafName,
        size: 24,
        height: 6 + quadrant + leaf,
        textureSeed: FIXTURE_SEED + quadrant * 2 + leaf + 5,
        compression: leafName === FIXTURE_CODEC_TILES.ktx2 ? "ktx2" : "none",
      });
      children.push(
        makeTile({
          name: leafName,
          size: 24,
          geometricError: 0,
          transform: translation(leaf === 0 ? -13 : 13, 0, leaf),
        }),
      );
    }

    levelOne.push(
      makeTile({
        name: levelOneName,
        size: 52,
        geometricError: 12,
        transform: translation(
          offsets[quadrant][0],
          offsets[quadrant][1],
          quadrant,
        ),
        children,
      }),
    );
  }

  const tileset = {
    asset: {
      version: "1.1",
      tilesetVersion: `seed-${FIXTURE_SEED.toString(16)}`,
    },
    geometricError: 48,
    root: {
      ...makeTile({
        name: "level-0-root",
        size: 112,
        geometricError: 48,
        children: levelOne,
      }),
      transform: rootEcefTransform(),
    },
  };

  return { definitions, tileset };
}

/**
 * Fresh explicit-hierarchy document shared by source/traversal tests without
 * paying the codec-generation cost. Content-authoring definitions stay local.
 */
export function createTiles3dFixtureTileset() {
  return fixtureTiles().tileset;
}

export async function generateTiles3dFixture(
  outputDirectory = DEFAULT_FIXTURE_DIRECTORY,
) {
  const target = resolve(outputDirectory);
  await rm(target, { recursive: true, force: true });
  await mkdir(resolve(target, CONTENT_DIRECTORY), { recursive: true });

  const { definitions, tileset } = fixtureTiles();
  const io = await nodeIO();
  for (const definition of definitions) {
    const document = await tileDocument(definition);
    await io.write(resolve(target, contentURI(definition.name)), document);
  }

  await writeFile(
    resolve(target, "tileset.json"),
    `${JSON.stringify(tileset, null, 2)}\n`,
    "utf8",
  );
  return target;
}

const invokedPath = process.argv[1] ? resolve(process.argv[1]) : "";
if (invokedPath === fileURLToPath(import.meta.url)) {
  const outputDirectory = process.argv[2]
    ? resolve(process.argv[2])
    : DEFAULT_FIXTURE_DIRECTORY;
  await generateTiles3dFixture(outputDirectory);
  process.stdout.write(`${outputDirectory}\n`);
}
