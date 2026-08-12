import type { DecodedTileContent } from "./types";

export const buildTransferList = (
  content: DecodedTileContent,
): ArrayBuffer[] => {
  const buffers = new Set<ArrayBuffer>();
  for (const primitive of content.primitives) {
    for (const array of [
      primitive.positions,
      primitive.normals,
      primitive.uvs,
      primitive.indices,
    ]) {
      if (array?.buffer instanceof ArrayBuffer) buffers.add(array.buffer);
    }
    const texture = primitive.material.baseColorTexture;
    if (texture?.kind === "rgba") {
      if (texture.rgba.buffer instanceof ArrayBuffer) {
        buffers.add(texture.rgba.buffer);
      }
    } else if (texture?.kind === "compressed") {
      for (const level of texture.levels) {
        if (level.data.buffer instanceof ArrayBuffer) {
          buffers.add(level.data.buffer);
        }
      }
    }
  }
  return [...buffers];
};
