import type { DecodedPrimitive, DecodedTileContent } from "./types";

export type ContentBuffers = {
  readonly geometry: ReadonlySet<ArrayBuffer>;
  readonly textures: ReadonlySet<ArrayBuffer>;
};

/**
 * The distinct backing buffers a decoded tile owns, split into the two budget
 * buckets. One walk serves both byte accounting and the worker transfer list.
 */
export const collectContentBuffers = (
  primitives: readonly DecodedPrimitive[],
): ContentBuffers => {
  const geometry = new Set<ArrayBuffer>();
  const textures = new Set<ArrayBuffer>();
  for (const primitive of primitives) {
    for (const array of [
      primitive.positions,
      primitive.normals,
      primitive.uvs,
      primitive.indices,
    ]) {
      if (array?.buffer instanceof ArrayBuffer) geometry.add(array.buffer);
    }
    const texture = primitive.material.baseColorTexture;
    if (texture?.kind === "rgba") {
      if (texture.rgba.buffer instanceof ArrayBuffer) {
        textures.add(texture.rgba.buffer);
      }
    } else if (texture?.kind === "compressed") {
      for (const level of texture.levels) {
        if (level.data.buffer instanceof ArrayBuffer) {
          textures.add(level.data.buffer);
        }
      }
    }
  }
  return { geometry, textures };
};

export const buildTransferList = (
  content: DecodedTileContent,
): ArrayBuffer[] => {
  const { geometry, textures } = collectContentBuffers(content.primitives);
  // A buffer shared by both buckets must still be transferred exactly once.
  return [...new Set([...geometry, ...textures])];
};
