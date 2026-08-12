import { describe, expect, it, vi } from "vitest";

import type { CameraView } from "./camera";
import { createMemoryPool } from "./memoryPool";
import { ROOT_KEY } from "./octree";
import {
  createPointCloudMember,
  type PointCloudMemberConfig,
  type PointCloudMemberStats,
} from "./pointCloudMember";
import { createStreamedSceneCoordinator } from "./streamedSceneCoordinator";
import { createSubmissionScheduler } from "./submissionScheduler";
import type { StreamedMemberContext } from "./streamedMember";
import type { TileSource } from "./tileSource";

const settle = async (): Promise<void> => {
  await Promise.resolve();
  await Promise.resolve();
  await new Promise((resolve) => setTimeout(resolve, 0));
};

const VIEW: CameraView = {
  projection: "perspective",
  viewProj: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
  position: [0, 0, -1],
  fovY: Math.PI / 2,
  viewportWidthCssPx: 100,
  viewportHeightCssPx: 100,
};

const source = (pointCount = 1_000_000): TileSource => ({
  metadata: () => ({ pointCount }),
  nodes: async () => [
    {
      key: ROOT_KEY,
      pointCount,
      bounds: { min: [-1, -1, -0.5], max: [1, 1, 0.5] },
      spacing: 1,
      children: [],
    },
  ],
  loadTile: async () => ({
    origin: [0, 0, 0],
    positions: new Float32Array(pointCount * 3),
    pointCount,
  }),
});

const makeContext = (memory = createMemoryPool({ totalBytes: 64_000_000 })) => {
  const renderer = {
    addActor: vi.fn(),
    removeActor: vi.fn(),
  };
  const scheduleRender = vi.fn();
  const context: StreamedMemberContext = {
    renderer,
    scheduleRender,
    memory,
    workers: {
      size: 0,
      decode: () => {
        throw new Error("unused");
      },
    },
    submissions: createSubmissionScheduler({ scheduleRender }),
    textureCapabilities: { capabilityKey: "none", compressedFormats: [] },
    devicePixelRatio: 1,
    onWorkChange: vi.fn(),
  };
  return { context, renderer };
};

const config = (
  overrides: Partial<PointCloudMemberConfig> = {},
): PointCloudMemberConfig => ({
  source: source(),
  pointCount: 1_000_000,
  presentation: { mode: "fixed", diameterCssPx: 2 },
  adaptive: true,
  adaptiveOptions: { minBudget: 1, maxBudget: 1_000_000 },
  selectionDelayMs: 0,
  ...overrides,
});

describe("createPointCloudMember", () => {
  it("maps configured maximum and normalized quality to point budget", () => {
    const { context } = makeContext();
    const member = createPointCloudMember(context, config());
    member.applyAllocation({
      qualityFraction: 0.5,
      memoryBudgetBytes: 32_000_000,
      regime: "stationary",
    });
    expect(member.stats()).toMatchObject({
      minimumPointBudget: 1,
      configuredPointCeiling: 1_000_000,
      fullPointCeiling: 1_000_000,
      controller: { pointBudget: 500_000 },
    });
    member.dispose();
  });

  it("reports an unconfigured adaptive point ceiling on the point member", () => {
    const { context } = makeContext();
    const member = createPointCloudMember(
      context,
      config({ adaptiveOptions: { minBudget: 10 } }),
    );
    expect(member.stats()).toMatchObject({
      minimumPointBudget: 10,
      configuredPointCeiling: null,
    });
    member.dispose();
  });

  it("maps the byte ceiling with measured/fallback bytes per point", () => {
    const { context } = makeContext();
    const member = createPointCloudMember(context, config());
    member.applyAllocation({
      qualityFraction: 0.5,
      memoryBudgetBytes: 400_000 * 16,
      regime: "stationary",
    });
    expect((member.stats() as any).controller.memoryCeilingPoints).toBe(
      400_000,
    );
    expect((member.stats() as any).controller.pointBudget).toBe(200_000);
    member.dispose();
  });

  it("retains stationary selection and thins density while moving", () => {
    const { context } = makeContext();
    const member = createPointCloudMember(context, config());
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 32_000_000,
      regime: "stationary",
    });
    member.applyAllocation({
      qualityFraction: 0.25,
      memoryBudgetBytes: 32_000_000,
      regime: "moving",
    });
    expect((member.stats() as any).controller).toMatchObject({
      pointBudget: 1_000_000,
      densityFraction: 0.25,
    });
    member.dispose();
  });

  it("caps normalized allocation at current camera demand", async () => {
    const { context } = makeContext();
    const member = createPointCloudMember(
      context,
      config({ source: source(100_000), pointCount: 1_000_000 }),
    );
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 32_000_000,
      regime: "stationary",
    });
    member.setCamera(VIEW);
    await settle();
    expect(member.governorInputs().qualityDemand).toBeCloseTo(0.1);
    member.applyAllocation({
      qualityFraction: 0.5,
      memoryBudgetBytes: 32_000_000,
      regime: "stationary",
    });
    expect((member.stats() as any).controller.pointBudget).toBe(100_000);
    member.dispose();
  });

  it("delegates submitted-point picking and positive occlusion depth", async () => {
    const { context } = makeContext();
    const member = createPointCloudMember(
      context,
      config({ source: source(1), pointCount: 1 }),
    );
    member.applyAllocation({
      qualityFraction: 1,
      memoryBudgetBytes: 1024,
      regime: "stationary",
    });
    member.setCamera(VIEW);
    await settle();
    const pick = member.pick(VIEW, 50, 50);
    expect(pick).toMatchObject({ status: "hit", rayDepth: 1 });
    expect(member.occlusionDepth(VIEW, 50, 50)).toEqual({
      status: "hit",
      rayDepth: 1,
    });
    member.dispose();
  });

  it("never self-registers with the page memory pool", () => {
    const memory = createMemoryPool({ totalBytes: 1_000 });
    const { context } = makeContext(memory);
    const member = createPointCloudMember(context, config());
    expect(memory.memberCount()).toBe(0);
    member.dispose();
  });

  it("integrates coordinator quality, memory, live DPR, and config lifecycle", async () => {
    const memory = createMemoryPool({ totalBytes: 16_000 });
    const scheduleRender = vi.fn();
    const renderer = { addActor: vi.fn(), removeActor: vi.fn() };
    const coordinator = createStreamedSceneCoordinator({
      scheduleRender,
      memory,
    });
    const initial = config({ adaptive: false, pointBudget: 1_000 });
    const member = createPointCloudMember(
      coordinator.context(renderer, 1),
      initial,
    );
    const registration = coordinator.register(member, {
      id: "cloud",
      qualityManaged: false,
    });
    registration.setCamera(VIEW);
    await settle();
    let stats = member.stats() as PointCloudMemberStats;
    expect(memory.memberCount()).toBe(1);
    expect(stats.allocation).toMatchObject({
      qualityFraction: 1,
      memoryBudgetBytes: 16_000,
    });
    expect(stats.controller).toMatchObject({
      pointBudget: 1_000,
      memoryCeilingPoints: 1_000,
    });

    registration.setDevicePixelRatio(2);
    registration.setConfig({
      ...initial,
      presentation: { mode: "fixed", diameterCssPx: 5 },
      refinementCutoffPx: 2,
    });
    stats = member.stats() as PointCloudMemberStats;
    expect(stats.renderer.devicePixelRatio).toBe(2);
    expect(stats.controller.presentation).toMatchObject({
      diameterCssPx: 5,
    });
    expect(stats.controller.refinementCutoffPx).toBe(2);

    registration.setActive(false);
    expect(memory.memberCount()).toBe(0);
    expect((member.stats() as PointCloudMemberStats).controller.active).toBe(
      false,
    );
    coordinator.dispose();
  });
});
