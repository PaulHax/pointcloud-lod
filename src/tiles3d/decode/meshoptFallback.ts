/**
 * Repairs meshopt fallback-buffer offsets in tile content before it is
 * decoded, whether that content is a GLB or bare glTF JSON.
 *
 * `EXT_meshopt_compression` points each compressed buffer view at a *fallback*
 * buffer — storage that carries no bytes in the file and exists so a loader
 * without the extension has somewhere to look. Publishers exist that write
 * every compressed view at offset zero of one fallback buffer sized for them
 * all concatenated, which conformant renderers never notice because they
 * decode each view into storage of its own.
 *
 * loaders.gl decodes in place, into the fallback buffer at the view's own
 * offset, so a 152 KB vertex view lands on top of a 26 KB index view and the
 * tile draws as a fan of spikes radiating from whichever vertex the scrambled
 * indices happen to name. Nothing reports an error, which is why this runs in
 * the decode path rather than being left to whoever supplies the content: a
 * silently wrong mesh is the one failure a profile gate cannot catch.
 *
 * Giving each view a distinct offset — the layout the fallback buffer's size
 * says was intended — is enough, and it touches only the JSON chunk.
 */

const GLB_MAGIC = 0x46546c67;
const JSON_CHUNK = 0x4e4f534a;
const HEADER_BYTES = 12;
const CHUNK_HEADER_BYTES = 8;

type BufferView = {
  buffer?: number;
  byteOffset?: number;
  byteLength?: number;
  extensions?: { EXT_meshopt_compression?: unknown };
};

type GltfBuffer = {
  byteLength?: number;
  extensions?: Record<string, unknown>;
};

type Gltf = {
  buffers?: GltfBuffer[];
  bufferViews?: BufferView[];
};

const isFallbackBuffer = (buffer: GltfBuffer | undefined): boolean =>
  Boolean(
    (
      buffer?.extensions?.["EXT_meshopt_compression"] as
        | { fallback?: boolean }
        | undefined
    )?.fallback,
  );

/**
 * Views that share a fallback buffer and start at the same offset. Anything
 * already laid out distinctly is left exactly as authored.
 */
const needsRepair = (gltf: Gltf): boolean => {
  const seen = new Map<number, Set<number>>();
  for (const view of gltf.bufferViews ?? []) {
    if (!view.extensions?.EXT_meshopt_compression) continue;
    const buffer = gltf.buffers?.[view.buffer ?? -1];
    if (!buffer || !isFallbackBuffer(buffer)) continue;
    const offsets = seen.get(view.buffer!) ?? new Set<number>();
    if (offsets.has(view.byteOffset ?? 0)) return true;
    offsets.add(view.byteOffset ?? 0);
    seen.set(view.buffer!, offsets);
  }
  return false;
};

const padToFour = (value: number): number => (value + 3) & ~3;

const repairOffsets = (gltf: Gltf): Gltf => {
  const nextOffset = new Map<number, number>();
  const bufferViews = (gltf.bufferViews ?? []).map((view) => {
    const buffer = gltf.buffers?.[view.buffer ?? -1];
    if (
      !view.extensions?.EXT_meshopt_compression ||
      !buffer ||
      !isFallbackBuffer(buffer)
    ) {
      return view;
    }
    const index = view.buffer!;
    const byteOffset = nextOffset.get(index) ?? 0;
    // Padded, because an accessor may only start on a multiple of its
    // component size and a compressed view's length is arbitrary — the
    // 90,318-byte index view in a 3DBAG tile put the float attributes that
    // follow it two bytes off, and the decode failed outright.
    nextOffset.set(index, padToFour(byteOffset + (view.byteLength ?? 0)));
    return { ...view, byteOffset };
  });
  // The fallback buffer was sized for those views concatenated without
  // padding, so the alignment above can push the last one past its declared
  // end. The buffer carries no bytes in the file; its length is only the
  // storage a loader allocates to decode into.
  const buffers = (gltf.buffers ?? []).map((buffer, index) => {
    const needed = nextOffset.get(index) ?? 0;
    return isFallbackBuffer(buffer) && needed > (buffer.byteLength ?? 0)
      ? { ...buffer, byteLength: needed }
      : buffer;
  });
  return { ...gltf, bufferViews, buffers };
};

/**
 * A tile whose content is bare glTF JSON rather than a GLB container. The
 * decoder accepts both, and a fallback-buffer collision authored in one
 * decodes to exactly the same silently wrong mesh as in the other.
 */
const repairJsonAsset = (content: ArrayBuffer): ArrayBuffer => {
  let gltf: unknown;
  try {
    // Matched to what the decoder itself will accept, padding included, so
    // this never rewrites a payload the decoder would have rejected.
    let text = new TextDecoder("utf-8", { fatal: true }).decode(
      new Uint8Array(content),
    );
    while (text.endsWith("\0") || text.endsWith(" ")) text = text.slice(0, -1);
    gltf = JSON.parse(text);
  } catch {
    return content;
  }
  if (!gltf || typeof gltf !== "object" || Array.isArray(gltf)) return content;
  if (!needsRepair(gltf as Gltf)) return content;

  const encoded = new TextEncoder().encode(
    JSON.stringify(repairOffsets(gltf as Gltf)),
  );
  return encoded.buffer.slice(
    encoded.byteOffset,
    encoded.byteOffset + encoded.byteLength,
  ) as ArrayBuffer;
};

/**
 * Returns the content unchanged unless it carries colliding meshopt views, so
 * a well-formed asset pays one JSON parse and nothing else.
 */
export const repairMeshoptFallbackOffsets = (
  content: ArrayBuffer,
): ArrayBuffer => {
  if (content.byteLength < HEADER_BYTES + CHUNK_HEADER_BYTES) {
    return repairJsonAsset(content);
  }
  const view = new DataView(content);
  if (view.getUint32(0, true) !== GLB_MAGIC) return repairJsonAsset(content);
  const jsonLength = view.getUint32(HEADER_BYTES, true);
  if (view.getUint32(HEADER_BYTES + 4, true) !== JSON_CHUNK) return content;

  const jsonStart = HEADER_BYTES + CHUNK_HEADER_BYTES;
  const jsonEnd = jsonStart + jsonLength;
  if (jsonEnd > content.byteLength) return content;

  let gltf: Gltf;
  try {
    gltf = JSON.parse(
      new TextDecoder().decode(new Uint8Array(content, jsonStart, jsonLength)),
    );
  } catch {
    return content;
  }
  if (!needsRepair(gltf)) return content;

  const encoded = new TextEncoder().encode(JSON.stringify(repairOffsets(gltf)));
  const paddedLength = padToFour(encoded.byteLength);
  const rest = new Uint8Array(content, jsonEnd);
  const output = new Uint8Array(jsonStart + paddedLength + rest.byteLength);
  const outputView = new DataView(output.buffer);

  output.set(new Uint8Array(content, 0, jsonStart));
  outputView.setUint32(0, GLB_MAGIC, true);
  outputView.setUint32(8, output.byteLength, true);
  outputView.setUint32(HEADER_BYTES, paddedLength, true);
  outputView.setUint32(HEADER_BYTES + 4, JSON_CHUNK, true);
  output.set(encoded, jsonStart);
  output.fill(0x20, jsonStart + encoded.byteLength, jsonStart + paddedLength);
  output.set(rest, jsonStart + paddedLength);
  return output.buffer;
};
