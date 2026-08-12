import type { DecodeTextureTarget, TextureCapabilities } from "./types";

const TARGET_PRIORITY: readonly Exclude<DecodeTextureTarget, "rgba">[] = [
  "astc-4x4",
  "bc7",
  "etc2-rgba8",
  "s3tc-dxt5",
];

export const capabilityTarget = (
  capabilities: TextureCapabilities,
): DecodeTextureTarget => {
  if (
    typeof capabilities.capabilityKey !== "string" ||
    capabilities.capabilityKey.length === 0
  ) {
    throw new Error("texture capabilityKey must be non-empty");
  }
  const supported = new Set(capabilities.compressedFormats);
  return TARGET_PRIORITY.find((format) => supported.has(format)) ?? "rgba";
};

export const loadersBasisFormat = (
  target: DecodeTextureTarget,
): "astc-4x4" | "bc7-m5" | "etc2" | "bc3" | "rgba32" => {
  switch (target) {
    case "astc-4x4":
      return "astc-4x4";
    case "bc7":
      // Mode 5 handles alpha. The opaque-only mode is not safe without an
      // explicit alpha proof from the source DFD.
      return "bc7-m5";
    case "etc2-rgba8":
      return "etc2";
    case "s3tc-dxt5":
      return "bc3";
    case "rgba":
      return "rgba32";
  }
};
