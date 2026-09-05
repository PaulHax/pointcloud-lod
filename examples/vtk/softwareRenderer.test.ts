import { describe, expect, it } from "vitest";

import { isSoftwareRenderer } from "./softwareRenderer";

describe("software renderer detection", () => {
  it("recognizes common software WebGL renderers without flagging hardware", () => {
    expect(isSoftwareRenderer("ANGLE (Google, Vulkan SwiftShader)")).toBe(true);
    expect(isSoftwareRenderer("Mesa/X.org", "llvmpipe (LLVM 19.1.1)")).toBe(
      true,
    );
    expect(isSoftwareRenderer("NVIDIA Corporation", "NVIDIA RTX 4090")).toBe(
      false,
    );
  });
});
