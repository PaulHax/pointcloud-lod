/**
 * Repairs meshopt fallback-buffer offsets in a GLB before it is decoded.
 *
 * `EXT_meshopt_compression` points each compressed buffer view at a *fallback*
 * buffer — storage that carries no bytes in the file and exists so a loader
 * without the extension has somewhere to look. 3DBAG writes both of its
 * compressed views at offset zero of one fallback buffer sized for the two
 * concatenated, which conformant renderers never notice because they decode
 * each view into storage of its own.
 *
 * loaders.gl decodes in place, into the fallback buffer at the view's own
 * offset, so the 152 KB vertex view lands on top of the 26 KB index view and
 * the tile draws as a fan of spikes radiating from whichever vertex the
 * scrambled indices happen to name.
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
    nextOffset.set(index, byteOffset + (view.byteLength ?? 0));
    return { ...view, byteOffset };
  });
  return { ...gltf, bufferViews };
};

const padToFour = (value: number): number => (value + 3) & ~3;

/**
 * Returns the GLB unchanged unless it carries colliding meshopt views, so a
 * well-formed asset pays one JSON parse and nothing else.
 */
export const repairMeshoptFallbackOffsets = (glb: ArrayBuffer): ArrayBuffer => {
  if (glb.byteLength < HEADER_BYTES + CHUNK_HEADER_BYTES) return glb;
  const view = new DataView(glb);
  if (view.getUint32(0, true) !== GLB_MAGIC) return glb;
  const jsonLength = view.getUint32(HEADER_BYTES, true);
  if (view.getUint32(HEADER_BYTES + 4, true) !== JSON_CHUNK) return glb;

  const jsonStart = HEADER_BYTES + CHUNK_HEADER_BYTES;
  const jsonEnd = jsonStart + jsonLength;
  if (jsonEnd > glb.byteLength) return glb;

  let gltf: Gltf;
  try {
    gltf = JSON.parse(
      new TextDecoder().decode(new Uint8Array(glb, jsonStart, jsonLength)),
    );
  } catch {
    return glb;
  }
  if (!needsRepair(gltf)) return glb;

  const encoded = new TextEncoder().encode(JSON.stringify(repairOffsets(gltf)));
  const paddedLength = padToFour(encoded.byteLength);
  const rest = new Uint8Array(glb, jsonEnd);
  const output = new Uint8Array(jsonStart + paddedLength + rest.byteLength);
  const outputView = new DataView(output.buffer);

  output.set(new Uint8Array(glb, 0, jsonStart));
  outputView.setUint32(0, GLB_MAGIC, true);
  outputView.setUint32(8, output.byteLength, true);
  outputView.setUint32(HEADER_BYTES, paddedLength, true);
  outputView.setUint32(HEADER_BYTES + 4, JSON_CHUNK, true);
  output.set(encoded, jsonStart);
  output.fill(0x20, jsonStart + encoded.byteLength, jsonStart + paddedLength);
  output.set(rest, jsonStart + paddedLength);
  return output.buffer;
};
