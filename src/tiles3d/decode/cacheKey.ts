import type { DecodeCacheIdentity } from "./types";

const encodePart = (value: string): string =>
  `${new TextEncoder().encode(value).byteLength}:${value}`;

export const buildDecodeCacheKey = (identity: DecodeCacheIdentity): string => {
  if (!identity.contentUrl || !identity.revision || !identity.capabilityKey) {
    throw new Error("decode cache identity fields must be non-empty");
  }
  return [
    "tiles3d-decode-v1",
    encodePart(identity.revision),
    encodePart(identity.contentUrl),
    encodePart(identity.capabilityKey),
  ].join("|");
};
