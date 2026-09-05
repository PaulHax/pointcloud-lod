import { describe, expect, it, vi } from "vitest";

import { createMemoryPool } from "./memoryPool";
import { createStreamedMemberFactoryRegistry } from "./memberFactoryRegistry";
import { createSubmissionScheduler } from "./submissionScheduler";
import type { StreamedMember, StreamedMemberContext } from "./streamedMember";

const context = (): StreamedMemberContext => ({
  renderer: {},
  scheduleRender: vi.fn(),
  memory: createMemoryPool({ totalBytes: 1 }),
  workers: {
    size: 0,
    decode: () => {
      throw new Error("unused");
    },
  },
  submissions: createSubmissionScheduler({
    scheduleRender: vi.fn(),
    maxTimeMsPerFrame: 100,
    now: () => 0,
  }),
  textureCapabilities: { capabilityKey: "none", compressedFormats: [] },
  devicePixelRatio: 1,
});

const member = { dispose: vi.fn() } as unknown as StreamedMember;

describe("createStreamedMemberFactoryRegistry", () => {
  it("creates by kind and rejects ambiguous or unknown registrations", () => {
    const registry = createStreamedMemberFactoryRegistry();
    const factory = vi.fn(() => member);
    const registration = registry.register("pointCloud", factory);
    const memberContext = context();
    const config = { pointCount: 10 };
    expect(registry.create("pointCloud", memberContext, config)).toBe(member);
    expect(factory).toHaveBeenCalledWith(memberContext, config);
    expect(() => registry.register("pointCloud", factory)).toThrow(
      /already registered/,
    );
    registration.release();
    expect(() => registry.create("pointCloud", memberContext, config)).toThrow(
      /unknown streamed member kind/,
    );
  });
});
