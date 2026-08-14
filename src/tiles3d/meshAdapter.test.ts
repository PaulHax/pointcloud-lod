import { beforeEach, describe, expect, it, vi } from "vitest";

import { createSubmissionScheduler } from "../submissionScheduler";
import type { DecodedTexture, DecodedTileContent } from "./decode";
import { createMeshAdapter } from "./meshAdapter";
import {
  actorInstances,
  failNextTexturePayload,
  mapperInstances,
  polyDataInstances,
  resetStubs,
  textureInstances,
} from "../../test/stubs/vtkStub";

const compressed = {
  kind: "compressed" as const,
  format: "astc-4x4" as const,
  width: 4,
  height: 4,
  colorSpace: "srgb" as const,
  levels: [{ width: 4, height: 4, data: new Uint8Array(16) }],
  sampler: { magFilter: 9729, minFilter: 9987, wrapS: 10497, wrapT: 10497 },
  capabilityKey: "astc",
} satisfies DecodedTexture;

const content = (texture: DecodedTexture = compressed): DecodedTileContent => ({
  origin: [10, 20, 30],
  primitives: [0, 1].map(() => ({
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    uvs: new Float32Array([0, 0, 1, 0, 0, 1]),
    indices: new Uint16Array([0, 1, 2]),
    material: {
      baseColorFactor: [0.5, 0.6, 0.7, 0.8],
      baseColorTexture: texture,
      raw: {
        version: 1,
        kind: "gltf-material",
        alphaMode: "BLEND",
        alphaCutoff: 0.5,
        doubleSided: false,
        unlit: false,
        metallicFactor: 1,
        roughnessFactor: 1,
        emissiveFactor: [0, 0, 0],
      },
    },
  })),
  // Worker ownership is the original typed arrays (102 bytes per primitive).
  // Adapter residency expands vtk cell records separately to 224 bytes.
  byteEstimate: { geometry: 204, textures: 16 },
});

const alphaContent = (
  alphaMode: "OPAQUE" | "MASK" | "BLEND",
  alphaCutoff: number,
  factorAlpha: number,
): DecodedTileContent => {
  const result = content();
  result.primitives.splice(1);
  result.primitives[0]!.material.raw.alphaMode = alphaMode;
  result.primitives[0]!.material.raw.alphaCutoff = alphaCutoff;
  result.primitives[0]!.material.baseColorFactor[3] = factorAlpha;
  return result;
};

describe("vtk mesh adapter", () => {
  beforeEach(resetStubs);

  it("admits multi-primitive tiles atomically and binds shared compressed textures once", () => {
    const renderer = { addActor: vi.fn(), removeActor: vi.fn() };
    const scheduleRender = vi.fn();
    const scheduler = createSubmissionScheduler({
      scheduleRender,
      maxBytesPerFrame: 128,
      maxTimeMsPerFrame: 100,
      now: () => 0,
    });
    const adapter = createMeshAdapter({
      renderer,
      scheduleRender,
      submissions: scheduler,
    });
    const admitted = vi.fn();
    adapter.submitTile("tile", content(), admitted);

    scheduler.prepareFrame();
    expect(renderer.addActor).not.toHaveBeenCalled();
    scheduler.prepareFrame();
    expect(renderer.addActor).toHaveBeenCalledTimes(2);
    expect(admitted).toHaveBeenCalledOnce();
    expect(textureInstances).toHaveLength(1);
    // Linear on both paths: the RGBA fallback cannot request an sRGB internal
    // format, so asking for one here would make the same texture render
    // darker on a compression-capable context than on a software one.
    expect(textureInstances[0]?.compressedData).toMatchObject({
      format: "astc-4x4",
      srgb: false,
    });
    expect(polyDataInstances[0]?.points).toBeInstanceOf(Float32Array);
    expect(polyDataInstances[0]?.polys).toEqual(new Uint32Array([3, 0, 1, 2]));
    expect(mapperInstances).toHaveLength(2);

    adapter.setDrawnTiles(["tile"]);
    expect(actorInstances.every((actor) => actor.visibility)).toBe(true);
    expect(adapter.stats()).toMatchObject({
      submittedTiles: 1,
      submittedTileIds: ["tile"],
      submittedPrimitives: 2,
      residentGeometryBytes: 224,
      logicalGeometryUploadBytes: 224,
      logicalTextureUploadBytes: 16,
      logicalUploadBytes: 240,
      submittedTextureBytes: 16,
      pooledTextureBytes: 0,
      residentTextureBytes: 16,
      drawnTiles: 1,
      drawnTileIds: ["tile"],
      drawnTextureBytes: 16,
      drawnTriangles: 2,
      textureRepresentation: "compressed",
      textureFormats: ["astc-4x4"],
    });
    expect(actorInstances[0]).toMatchObject({
      color: [0.5, 0.6, 0.7],
      opacity: 0.8,
      textures: [textureInstances[0]],
    });
    expect(textureInstances[0]).toMatchObject({
      sampler: {
        magFilter: "linear",
        minFilter: "linear-mipmap-linear",
        wrapS: "repeat",
        wrapT: "repeat",
      },
      flipY: false,
    });
  });

  it("realizes MASK as an opaque-pass alpha test before final color output", () => {
    const scheduler = createSubmissionScheduler({
      scheduleRender: vi.fn(),
      maxBytesPerFrame: 1024,
    });
    const adapter = createMeshAdapter({
      renderer: { addActor: vi.fn(), removeActor: vi.fn() },
      scheduleRender: vi.fn(),
      submissions: scheduler,
    });
    adapter.submitTile("mask", alphaContent("MASK", 0.37, 0.8));
    scheduler.prepareFrame();

    expect(actorInstances[0]).toMatchObject({
      forceOpaque: true,
      forceTranslucent: false,
      opacity: 0.8,
    });
    const replacement =
      mapperInstances[0]?.viewSpecificProperties?.OpenGL
        ?.ShaderReplacements?.[0];
    expect(replacement).toMatchObject({
      shaderType: "Fragment",
      originalValue: "//VTK::Light::Impl",
      replaceAll: false,
      replaceFirst: true,
    });
    expect(replacement?.replacementValue).toContain("opacity < 0.37");
    expect(replacement?.replacementValue).toContain("discard");

    const fragment = replacement?.replacementValue.replace(
      "//VTK::Light::Impl",
      "gl_FragData[0] = vec4(diffuseColor, opacity);",
    );
    expect(fragment?.indexOf("discard")).toBeLessThan(
      fragment?.indexOf("gl_FragData") ?? -1,
    );
    adapter.dispose();
  });

  it("disables vtk lighting for KHR_materials_unlit primitives", () => {
    const scheduler = createSubmissionScheduler({
      scheduleRender: vi.fn(),
      maxBytesPerFrame: 1024,
    });
    const adapter = createMeshAdapter({
      renderer: { addActor: vi.fn(), removeActor: vi.fn() },
      scheduleRender: vi.fn(),
      submissions: scheduler,
    });
    const unlit = content();
    unlit.primitives.splice(1);
    unlit.primitives[0]!.material.raw.unlit = true;
    adapter.submitTile("unlit", unlit);
    scheduler.prepareFrame();

    expect(actorInstances[0]?.lighting).toBe(false);
    adapter.dispose();
  });

  it("keeps OPAQUE texture alpha out of blending and BLEND in the translucent pass", () => {
    const scheduler = createSubmissionScheduler({
      scheduleRender: vi.fn(),
      maxBytesPerFrame: 1024,
    });
    const adapter = createMeshAdapter({
      renderer: { addActor: vi.fn(), removeActor: vi.fn() },
      scheduleRender: vi.fn(),
      submissions: scheduler,
    });
    adapter.submitTile("opaque", alphaContent("OPAQUE", 0.73, 0.25));
    scheduler.prepareFrame();
    adapter.submitTile("blend", alphaContent("BLEND", 0.19, 0.25));
    scheduler.prepareFrame();

    expect(actorInstances[0]).toMatchObject({
      forceOpaque: true,
      forceTranslucent: false,
      opacity: 1,
    });
    expect(
      mapperInstances[0]?.viewSpecificProperties?.OpenGL
        ?.ShaderReplacements?.[0]?.replacementValue,
    ).toContain("opacity = 1.0");
    expect(actorInstances[1]).toMatchObject({
      forceOpaque: false,
      forceTranslucent: true,
      opacity: 0.25,
    });
    expect(mapperInstances[1]?.viewSpecificProperties).toEqual({});
    adapter.dispose();
  });

  it("uses actual expanded RGBA bytes and applies anchor after the RTC origin", () => {
    const renderer = { addActor: vi.fn(), removeActor: vi.fn() };
    const scheduler = createSubmissionScheduler({
      scheduleRender: vi.fn(),
      maxBytesPerFrame: 1024,
    });
    const rgba = {
      kind: "rgba" as const,
      rgba: new Uint8Array(4 * 4 * 4),
      width: 4,
      height: 4,
      colorSpace: "srgb" as const,
      sampler: compressed.sampler,
    };
    const adapter = createMeshAdapter({
      renderer,
      scheduleRender: vi.fn(),
      submissions: scheduler,
    });
    adapter.setBaseMatrix([
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 100, 200, 300, 1,
    ]);
    adapter.submitTile("rgba", content(rgba));
    scheduler.prepareFrame();
    expect(actorInstances[0]?.userMatrix?.slice(12, 15)).toEqual([
      110, 220, 330,
    ]);
    expect(adapter.stats()).toMatchObject({
      submittedTextureBytes: 64,
      pooledTextureBytes: 0,
      residentTextureBytes: 64,
      drawnTextureBytes: 0,
      textureRepresentation: "rgba",
      textureFormats: ["rgba"],
    });
  });

  it("cancels partial admissions without attaching actors or leaking vtk resources", () => {
    const renderer = { addActor: vi.fn(), removeActor: vi.fn() };
    const scheduler = createSubmissionScheduler({
      scheduleRender: vi.fn(),
      maxBytesPerFrame: 128,
    });
    const adapter = createMeshAdapter({
      renderer,
      scheduleRender: vi.fn(),
      submissions: scheduler,
    });
    adapter.submitTile("tile", content());
    scheduler.prepareFrame();
    expect(adapter.cancelTile("tile")).toBe(true);
    scheduler.prepareFrame();
    expect(renderer.addActor).not.toHaveBeenCalled();
    expect(actorInstances.every((actor) => actor.deleted)).toBe(true);
    expect(adapter.stats().residentBytes).toBe(0);
  });

  it("stages one primitive larger than the scheduler cap without an oversized job", () => {
    const renderer = { addActor: vi.fn(), removeActor: vi.fn() };
    const scheduler = createSubmissionScheduler({
      scheduleRender: vi.fn(),
      maxBytesPerFrame: 128,
      maxTimeMsPerFrame: 100,
      now: () => 0,
    });
    const adapter = createMeshAdapter({
      renderer,
      scheduleRender: vi.fn(),
      submissions: scheduler,
    });
    const large = content();
    const original = large.primitives[0]!;
    const triangleCount = 6;
    large.primitives = [
      {
        ...original,
        positions: new Float32Array(
          Array.from({ length: triangleCount }, () => [
            -0.5, -0.5, 0, 0.5, -0.5, 0, 0, 0.5, 0,
          ]).flat(),
        ),
        normals: new Float32Array(
          Array.from({ length: triangleCount * 3 }, () => [0, 0, 1]).flat(),
        ),
        uvs: new Float32Array(
          Array.from({ length: triangleCount }, () => [
            0, 0, 1, 0, 0, 1,
          ]).flat(),
        ),
        indices: undefined,
      },
    ];
    expect(() => adapter.submitTile("large", large)).not.toThrow();
    expect(scheduler.stats().queuedJobs).toBeGreaterThan(2);
    while (scheduler.hasPending()) {
      scheduler.prepareFrame();
      expect(scheduler.stats().lastFrameAdmittedBytes).toBeLessThanOrEqual(128);
      if (scheduler.hasPending())
        expect(renderer.addActor).not.toHaveBeenCalled();
    }
    expect(renderer.addActor).toHaveBeenCalledTimes(6);
  });

  it("retires/removes/deletes exact submitted actors and never renders synchronously", () => {
    const renderer = { addActor: vi.fn(), removeActor: vi.fn() };
    const scheduleRender = vi.fn();
    const scheduler = createSubmissionScheduler({
      scheduleRender,
      maxBytesPerFrame: 1024,
    });
    const adapter = createMeshAdapter({
      renderer,
      scheduleRender,
      submissions: scheduler,
    });
    adapter.submitTile("tile", content());
    scheduler.prepareFrame();
    adapter.retireTile("tile");
    expect(renderer.removeActor).toHaveBeenCalledTimes(2);
    expect(adapter.stats()).toMatchObject({ pooledTiles: 1 });
    adapter.setResourceCeilingBytes(0);
    expect(actorInstances.every((actor) => actor.deleted)).toBe(true);
    expect(mapperInstances.every((mapper) => mapper.deleted)).toBe(true);
    expect(polyDataInstances.every((polyData) => polyData.deleted)).toBe(true);
    expect(textureInstances.every((texture) => texture.deleted)).toBe(true);
    expect((renderer as { render?: unknown }).render).toBeUndefined();
    adapter.dispose();
  });

  it("reuses retired submitted chunks without retaining the decoded payload", () => {
    const renderer = { addActor: vi.fn(), removeActor: vi.fn() };
    const scheduler = createSubmissionScheduler({
      scheduleRender: vi.fn(),
      maxBytesPerFrame: 1024,
    });
    const adapter = createMeshAdapter({
      renderer,
      scheduleRender: vi.fn(),
      submissions: scheduler,
    });
    const payload = content();
    adapter.submitTile("tile", payload);
    scheduler.prepareFrame();
    adapter.setDrawnTiles(["tile"]);
    const submitted = adapter.submittedTiles()[0]!;
    expect(submitted.primitives).toHaveLength(2);
    expect(submitted.primitives[0]?.positions).toBe(
      polyDataInstances[0]?.points,
    );
    expect(submitted.primitives[0]?.positions).not.toBe(
      payload.primitives[0]?.positions,
    );
    payload.primitives[0]!.positions.fill(99);
    expect(submitted.primitives[0]?.positions[0]).toBe(0);
    const createdActors = [...actorInstances];
    adapter.retireTile("tile");
    expect(adapter.stats()).toMatchObject({
      submittedTiles: 0,
      pooledTiles: 1,
      residentBytes: 240,
    });
    const redecoded = content();
    adapter.submitTile("tile", redecoded);
    expect(actorInstances).toEqual(createdActors);
    adapter.setDrawnTiles(["tile"]);
    expect(adapter.submittedTiles()[0]?.primitives[0]?.positions[0]).toBe(0);
    expect(adapter.stats()).toMatchObject({
      submittedTiles: 1,
      pooledTiles: 0,
      logicalUploadBytes: 240,
    });
    adapter.dispose();
  });

  it("passes asymmetric, mirrored, and mixed filters through the exact sampler seam", () => {
    const renderer = { addActor: vi.fn(), removeActor: vi.fn() };
    const onError = vi.fn();
    const scheduler = createSubmissionScheduler({
      scheduleRender: vi.fn(),
      maxBytesPerFrame: 1024,
    });
    const adapter = createMeshAdapter({
      renderer,
      scheduleRender: vi.fn(),
      submissions: scheduler,
      onError,
    });
    const sampler = {
      magFilter: 9728,
      minFilter: 9987,
      wrapS: 10497,
      wrapT: 33648,
    } satisfies DecodedTexture["sampler"];
    adapter.submitTile("mixed", content({ ...compressed, sampler }));
    scheduler.prepareFrame();
    expect(onError).not.toHaveBeenCalled();
    expect(renderer.addActor).toHaveBeenCalledTimes(2);
    expect(textureInstances[0]?.sampler).toEqual({
      magFilter: "nearest",
      minFilter: "linear-mipmap-linear",
      wrapS: "repeat",
      wrapT: "mirrored-repeat",
    });
    expect(textureInstances[0]?.flipY).toBe(false);
    adapter.dispose();
  });

  it("keeps MASK picking conservative when raster min/mag or mip sampling is unknowable", () => {
    const renderer = { addActor: vi.fn(), removeActor: vi.fn() };
    const scheduler = createSubmissionScheduler({
      scheduleRender: vi.fn(),
      maxBytesPerFrame: 1024,
    });
    const adapter = createMeshAdapter({
      renderer,
      scheduleRender: vi.fn(),
      submissions: scheduler,
    });
    const texture: Extract<DecodedTexture, { kind: "rgba" }> = {
      kind: "rgba" as const,
      rgba: new Uint8Array([255, 255, 255, 0, 255, 255, 255, 255]),
      width: 2,
      height: 1,
      colorSpace: "srgb" as const,
      sampler: {
        magFilter: 9729,
        minFilter: 9987,
        wrapS: 33071,
        wrapT: 33071,
      },
    };
    const masked = alphaContent("MASK", 0.5, 1);
    masked.primitives[0]!.material.baseColorTexture = texture;
    expect(adapter.submitTile("mipped", masked)).toBe("queued");
    scheduler.prepareFrame();
    adapter.setDrawnTiles(["mipped"]);
    expect(adapter.submittedTiles()[0]?.primitives[0]?.alphaMask).toEqual({
      kind: "unknown",
    });
    // The alpha plane is discarded and therefore is not charged as residency.
    expect(adapter.stats().residentGeometryBytes).toBe(112);

    adapter.clearTiles();
    texture.sampler = { ...texture.sampler, minFilter: 9729 };
    expect(adapter.submitTile("base-linear", masked)).toBe("queued");
    scheduler.prepareFrame();
    adapter.setDrawnTiles(["base-linear"]);
    expect(adapter.submittedTiles()[0]?.primitives[0]?.alphaMask?.kind).toBe(
      "known",
    );
    expect(adapter.stats().residentGeometryBytes).toBe(114);
    adapter.dispose();
  });

  it("maps an exact nearest+clamp sampler without broadening either axis or filter", () => {
    const renderer = { addActor: vi.fn(), removeActor: vi.fn() };
    const scheduler = createSubmissionScheduler({
      scheduleRender: vi.fn(),
      maxBytesPerFrame: 1024,
    });
    const adapter = createMeshAdapter({
      renderer,
      scheduleRender: vi.fn(),
      submissions: scheduler,
    });
    const nearest = content({
      ...compressed,
      sampler: { magFilter: 9728, minFilter: 9984, wrapS: 33071, wrapT: 33071 },
    });
    expect(adapter.submitTile("nearest", nearest)).toBe("queued");
    scheduler.prepareFrame();
    expect(textureInstances[0]?.sampler).toEqual({
      magFilter: "nearest",
      minFilter: "nearest-mipmap-nearest",
      wrapS: "clamp-to-edge",
      wrapT: "clamp-to-edge",
    });
    expect(textureInstances[0]?.flipY).toBe(false);
    adapter.dispose();
  });

  it("clears submitted actors even when they are outside the exact drawn pick set", () => {
    const renderer = { addActor: vi.fn(), removeActor: vi.fn() };
    const scheduler = createSubmissionScheduler({
      scheduleRender: vi.fn(),
      maxBytesPerFrame: 1024,
    });
    const adapter = createMeshAdapter({
      renderer,
      scheduleRender: vi.fn(),
      submissions: scheduler,
    });
    adapter.submitTile("hidden-fallback", content());
    scheduler.prepareFrame();
    adapter.setDrawnTiles([]);
    expect(adapter.submittedTiles()).toEqual([]);
    expect(adapter.stats().submittedTiles).toBe(1);
    adapter.clearTiles();
    expect(adapter.stats()).toMatchObject({
      submittedTiles: 0,
      residentBytes: 0,
    });
    expect(renderer.removeActor).toHaveBeenCalledTimes(2);
    expect(actorInstances.every((actor) => actor.deleted)).toBe(true);
    adapter.dispose();
  });

  it("deletes vtk resources when a scheduler realization setter throws", () => {
    const renderer = { addActor: vi.fn(), removeActor: vi.fn() };
    const onError = vi.fn();
    const scheduler = createSubmissionScheduler({
      scheduleRender: vi.fn(),
      maxBytesPerFrame: 1024,
    });
    const adapter = createMeshAdapter({
      renderer,
      scheduleRender: vi.fn(),
      submissions: scheduler,
      onError,
    });
    failNextTexturePayload("compressed");
    expect(adapter.submitTile("throwing", content())).toBe("queued");
    scheduler.prepareFrame();
    expect(onError).toHaveBeenCalledWith(
      expect.objectContaining({
        message: "injected compressed payload failure",
      }),
    );
    expect(textureInstances).toHaveLength(1);
    expect(textureInstances[0]?.deleted).toBe(true);
    expect(renderer.addActor).not.toHaveBeenCalled();
    expect(adapter.stats()).toMatchObject({
      pendingTiles: 0,
      submittedTiles: 0,
      residentBytes: 0,
    });
    adapter.dispose();
  });

  it("rolls back every actor when transactional renderer attachment throws", () => {
    const renderer = {
      addActor: vi
        .fn()
        .mockImplementationOnce(() => {})
        .mockImplementationOnce(() => {
          throw new Error("injected renderer attachment failure");
        }),
      removeActor: vi.fn(),
    };
    const onError = vi.fn();
    const scheduler = createSubmissionScheduler({
      scheduleRender: vi.fn(),
      maxBytesPerFrame: 1024,
    });
    const adapter = createMeshAdapter({
      renderer,
      scheduleRender: vi.fn(),
      submissions: scheduler,
      onError,
    });
    expect(adapter.submitTile("throwing-attach", content())).toBe("queued");
    scheduler.prepareFrame();
    expect(onError).toHaveBeenCalledWith(
      expect.objectContaining({
        message: "injected renderer attachment failure",
      }),
    );
    expect(renderer.removeActor).toHaveBeenCalledTimes(2);
    expect(actorInstances.every((actor) => actor.deleted)).toBe(true);
    expect(adapter.stats()).toMatchObject({
      pendingTiles: 0,
      submittedTiles: 0,
      pooledTiles: 0,
      residentBytes: 0,
    });
    expect(adapter.retireTile("throwing-attach")).toBe(true);
    expect(adapter.retireTile("throwing-attach")).toBe(false);
    adapter.dispose();
  });

  it("rolls back and releases an exact pooled reuse when reattachment throws", () => {
    const renderer = { addActor: vi.fn(), removeActor: vi.fn() };
    const onError = vi.fn();
    const scheduler = createSubmissionScheduler({
      scheduleRender: vi.fn(),
      maxBytesPerFrame: 1024,
    });
    const adapter = createMeshAdapter({
      renderer,
      scheduleRender: vi.fn(),
      submissions: scheduler,
      onError,
    });
    const decoded = content();
    adapter.submitTile("reuse", decoded);
    scheduler.prepareFrame();
    adapter.retireTile("reuse");
    renderer.addActor.mockImplementationOnce(() => {
      throw new Error("injected pooled attachment failure");
    });
    expect(adapter.submitTile("reuse", decoded)).toBe("failed");
    expect(onError).toHaveBeenCalledWith(
      expect.objectContaining({
        message: "injected pooled attachment failure",
      }),
    );
    expect(adapter.stats()).toMatchObject({
      submittedTiles: 0,
      pooledTiles: 0,
      residentBytes: 0,
    });
    expect(actorInstances.every((actor) => actor.deleted)).toBe(true);
    adapter.dispose();
  });
});
