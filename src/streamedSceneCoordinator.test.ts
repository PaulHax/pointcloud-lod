import { describe, expect, it, vi } from "vitest";

import type { CameraView, Mat16 } from "./camera";
import { createMemoryPool } from "./memoryPool";
import { createStreamedSceneCoordinator } from "./streamedSceneCoordinator";
import type {
  Allocation,
  GovernorInputs,
  Importance,
  MemberPickResult,
  OcclusionResult,
} from "./streamedMember";

const VIEW: CameraView = {
  projection: "perspective",
  viewProj: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
  position: [0, 0, 0],
  fovY: Math.PI / 2,
  viewportWidthCssPx: 100,
  viewportHeightCssPx: 100,
};

const makeMember = (overrides: Partial<GovernorInputs> = {}) => {
  const inputs: GovernorInputs = {
    projectedImportance: 1 as Importance,
    qualityDemand: 1,
    work: { operations: 0, progressSerial: 0 },
    physicalTileOperations: 0,
    physicalHierarchyOperations: 0,
    residentBytes: 0,
    ...overrides,
  };
  const allocations: Allocation[] = [];
  return {
    allocations,
    setCamera: vi.fn<(view: CameraView) => void>(),
    setModelMatrix: vi.fn<(matrix: Mat16 | null) => void>(),
    setDevicePixelRatio: vi.fn<(devicePixelRatio: number) => void>(),
    setActive: vi.fn<(active: boolean) => void>(),
    setConfig: vi.fn<(config: object) => void>(),
    beginInteraction: vi.fn<() => void>(),
    endInteraction: vi.fn<() => void>(),
    prepareFrame: vi.fn<() => void>(),
    onStall: vi.fn<(error: Error) => void>(),
    governorInputs: () => inputs,
    applyAllocation: (allocation: Allocation) => allocations.push(allocation),
    pick: (): MemberPickResult => ({ status: "miss" }),
    occlusionDepth: (): OcclusionResult => ({ status: "clear" }),
    stats: () => ({}),
    dispose: vi.fn<() => void>(),
  };
};

describe("createStreamedSceneCoordinator", () => {
  it("owns one memory registration per active member and fans out byte shares", () => {
    const memory = createMemoryPool({ totalBytes: 900 });
    const coordinator = createStreamedSceneCoordinator({
      scheduleRender: vi.fn(),
      memory,
    });
    const a = makeMember();
    const b = makeMember();
    const aRegistration = coordinator.register(a, { qualityManaged: true });
    coordinator.register(b, { qualityManaged: true });
    expect(memory.memberCount()).toBe(2);
    expect(a.allocations.at(-1)?.memoryBudgetBytes).toBe(450);
    expect(b.allocations.at(-1)?.memoryBudgetBytes).toBe(450);
    aRegistration.setActive(false);
    expect(memory.memberCount()).toBe(1);
    expect(b.allocations.at(-1)?.memoryBudgetBytes).toBe(900);
  });

  it("survives members that report work from inside applyAllocation", async () => {
    // Applying an allocation makes the member queue work, and queueing work is
    // a work change, which is another refresh. Two members sharing one quality
    // budget can trade it indefinitely; before the refresh loop was made
    // iterative this recursed until the stack ran out, which is how a combined
    // point-cloud and 3D Tiles scene died a few seconds into a gesture.
    const coordinator = createStreamedSceneCoordinator({
      scheduleRender: vi.fn(),
      memory: createMemoryPool({ totalBytes: 1_000 }),
    });
    const context = coordinator.context({} as never);

    let serial = 0;
    const reentrant = (demand: () => number) => {
      const member = makeMember();
      return {
        ...member,
        // Every allocation moves this member's demand, so the allocator never
        // reaches a fixed point and the loop has to be the thing that stops.
        governorInputs: () => ({
          ...member.governorInputs(),
          qualityDemand: demand(),
          work: { operations: 1, progressSerial: (serial += 1) },
        }),
        applyAllocation: (allocation: Allocation) => {
          member.applyAllocation(allocation);
          context.onWorkChange?.();
        },
      };
    };

    let flip = 0;
    const a = reentrant(() => ((flip += 1) % 2 === 0 ? 0.1 : 1));
    const b = reentrant(() => ((flip += 1) % 2 === 0 ? 1 : 0.1));
    coordinator.register(a, { qualityManaged: true });
    coordinator.register(b, { qualityManaged: true });

    expect(() => context.onWorkChange?.()).not.toThrow();
    // It stopped rather than running away, and each member still holds a real
    // allocation rather than being left mid-update.
    expect(a.allocations.length).toBeGreaterThan(0);
    expect(b.allocations.length).toBeGreaterThan(0);
    expect(a.allocations.length).toBeLessThan(100);
    expect(b.allocations.length).toBeLessThan(100);
    // And it says that it stopped early, so a scene that never settles is a
    // number someone can look at rather than silence.
    expect(coordinator.stats().exhaustedRefreshes).toBeGreaterThan(0);
  });

  it("water-fills normalized quality across managed members only", () => {
    const coordinator = createStreamedSceneCoordinator({
      scheduleRender: vi.fn(),
      memory: createMemoryPool({ totalBytes: 300 }),
    });
    const capped = makeMember({ qualityDemand: 0.1 });
    const open = makeMember();
    const fixed = makeMember();
    coordinator.register(capped, { qualityManaged: true });
    coordinator.register(open, { qualityManaged: true });
    coordinator.register(fixed, { qualityManaged: false });
    expect(capped.allocations.at(-1)?.qualityFraction).toBe(0.1);
    expect(open.allocations.at(-1)?.qualityFraction).toBe(1);
    expect(fixed.allocations.at(-1)?.qualityFraction).toBe(1);
  });

  it("feeds the governor fraction directly into view allocation", () => {
    const coordinator = createStreamedSceneCoordinator({
      scheduleRender: vi.fn(),
      memory: createMemoryPool({ totalBytes: 300 }),
      governor: { minSamples: 1, cooldownMs: 0, hysteresis: 0 },
    });
    const adaptive = makeMember();
    const fixed = makeMember();
    coordinator.register(adaptive, { qualityManaged: true });
    coordinator.register(fixed, { qualityManaged: false });
    coordinator.recordHostFrame({ hostFrameMs: 66, now: 0 });
    expect(coordinator.stats().viewQualityFraction).toBeCloseTo(0.5);
    expect(adaptive.allocations.at(-1)?.qualityFraction).toBeCloseTo(0.5);
    expect(fixed.allocations.at(-1)?.qualityFraction).toBe(1);
  });

  it("uses the first active adaptive member's targets in stable registry order", () => {
    const coordinator = createStreamedSceneCoordinator({
      scheduleRender: vi.fn(),
      memory: createMemoryPool({ totalBytes: 300 }),
    });
    const first = coordinator.register(makeMember(), {
      id: "first",
      qualityManaged: true,
      qualityTargets: { stationaryTargetMs: 41, interactionTargetMs: 17 },
    });
    coordinator.register(makeMember(), {
      id: "second",
      qualityManaged: true,
      qualityTargets: { stationaryTargetMs: 55, interactionTargetMs: 23 },
    });
    expect(coordinator.stats()).toMatchObject({
      targetOverrideMemberId: "first",
      governor: { targetFrameTimeMs: 41 },
    });
    first.setActive(false);
    expect(coordinator.stats()).toMatchObject({
      targetOverrideMemberId: "second",
      governor: { targetFrameTimeMs: 55 },
    });
  });

  it("lets the first adaptive point override targets after targetless members", () => {
    const coordinator = createStreamedSceneCoordinator({
      scheduleRender: vi.fn(),
      memory: createMemoryPool({ totalBytes: 300 }),
    });
    coordinator.register(makeMember(), {
      id: "tiles-first",
      qualityManaged: true,
    });
    coordinator.register(makeMember(), {
      id: "adaptive-point",
      qualityManaged: true,
      qualityTargets: { stationaryTargetMs: 45 },
    });
    expect(coordinator.stats()).toMatchObject({
      targetOverrideMemberId: "adaptive-point",
      governor: { targetFrameTimeMs: 45 },
    });
  });

  it("lets the host own global frame targets", () => {
    const coordinator = createStreamedSceneCoordinator({
      scheduleRender: vi.fn(),
      memory: createMemoryPool({ totalBytes: 300 }),
    });
    coordinator.register(makeMember(), {
      id: "point",
      qualityManaged: true,
      qualityTargets: { stationaryTargetMs: 45, interactionTargetMs: 18 },
    });
    coordinator.setQualityTargets({
      stationaryTargetMs: 27,
      interactionTargetMs: 12,
    });
    expect(coordinator.stats()).toMatchObject({
      targetOverrideMemberId: null,
      governor: { targetFrameTimeMs: 27 },
    });
    coordinator.beginInteraction();
    expect(coordinator.stats().governor.targetFrameTimeMs).toBe(12);
  });

  it("keeps camera/model/DPR/config member-specific and drains shared submissions", () => {
    const order: string[] = [];
    const coordinator = createStreamedSceneCoordinator({
      scheduleRender: vi.fn(),
      memory: createMemoryPool({ totalBytes: 300 }),
    });
    const member = makeMember();
    const other = makeMember();
    member.prepareFrame.mockImplementation(() => order.push("member"));
    const registration = coordinator.register(member);
    coordinator.register(other);
    coordinator.context({}, 1).submissions.enqueue({
      bytes: 1,
      run: () => order.push("submission"),
    });
    registration.setCamera(VIEW);
    registration.setCamera({ ...VIEW });
    registration.setModelMatrix(null);
    registration.setModelMatrix(null);
    registration.setDevicePixelRatio(2);
    registration.setDevicePixelRatio(2);
    registration.setConfig({ revision: "next" });
    coordinator.noteRenderedCameras(new Map([["renderer-a", VIEW]]));
    coordinator.beginInteraction();
    coordinator.endInteraction();
    coordinator.prepareFrame(1);
    coordinator.prepareFrame(1);
    expect(order).toEqual(["member", "submission"]);
    expect(member.setCamera).toHaveBeenCalledWith(VIEW);
    expect(member.setCamera).toHaveBeenCalledOnce();
    expect(member.setModelMatrix).toHaveBeenCalledWith(null);
    expect(member.setModelMatrix).toHaveBeenCalledOnce();
    expect(member.setDevicePixelRatio).toHaveBeenCalledWith(2);
    expect(member.setDevicePixelRatio).toHaveBeenCalledOnce();
    expect(member.setConfig).toHaveBeenCalledWith({ revision: "next" });
    expect(other.setCamera).not.toHaveBeenCalled();
    expect(other.setModelMatrix).not.toHaveBeenCalled();
    expect(other.setDevicePixelRatio).not.toHaveBeenCalled();
    expect(member.beginInteraction).toHaveBeenCalledOnce();
    expect(member.endInteraction).toHaveBeenCalledOnce();
    coordinator.dispose();
    expect(member.dispose).toHaveBeenCalledOnce();
  });

  it("replays held interaction depth to members registered mid-gesture", () => {
    const coordinator = createStreamedSceneCoordinator({
      scheduleRender: vi.fn(),
      memory: createMemoryPool({ totalBytes: 300 }),
    });
    coordinator.beginInteraction();
    coordinator.beginInteraction();
    const member = makeMember();
    coordinator.register(member, { qualityManaged: true });
    expect(member.beginInteraction).toHaveBeenCalledTimes(2);
    expect(member.endInteraction).not.toHaveBeenCalled();
    coordinator.endInteraction();
    expect(member.endInteraction).toHaveBeenCalledOnce();
    coordinator.endInteraction();
    expect(member.endInteraction).toHaveBeenCalledTimes(2);
    coordinator.dispose();
  });

  it("compares camera snapshots rather than aliases restated by the host", () => {
    const coordinator = createStreamedSceneCoordinator({
      scheduleRender: vi.fn(),
      memory: createMemoryPool({ totalBytes: 300 }),
    });
    const member = makeMember();
    const registration = coordinator.register(member);
    const mutable = {
      ...VIEW,
      position: [...VIEW.position] as [number, number, number],
      viewProj: Array.from(VIEW.viewProj) as Mat16,
    };
    registration.setCamera(mutable);
    registration.setCamera(mutable);
    mutable.position[0] = 1;
    registration.setCamera(mutable);
    expect(member.setCamera).toHaveBeenCalledTimes(2);
    coordinator.dispose();
  });

  it("propagates retry-aware member workPending through the coordinator context", () => {
    let workPending = true;
    const member = makeMember();
    member.governorInputs = () => ({
      projectedImportance: 1 as Importance,
      qualityDemand: 1,
      work: { operations: workPending ? 1 : 0, progressSerial: 0 },
      physicalTileOperations: 0,
      physicalHierarchyOperations: 0,
      residentBytes: 0,
    });
    const coordinator = createStreamedSceneCoordinator({
      scheduleRender: vi.fn(),
      memory: createMemoryPool({ totalBytes: 300 }),
    });
    coordinator.register(member, { qualityManaged: true });
    expect(coordinator.stats().governor.activity.workPending).toBe(true);
    workPending = false;
    coordinator.context({}).onWorkChange?.();
    expect(coordinator.stats().governor.activity.workPending).toBe(false);
  });

  it("includes fixed-member work in adaptive sample eligibility", () => {
    const coordinator = createStreamedSceneCoordinator({
      scheduleRender: vi.fn(),
      memory: createMemoryPool({ totalBytes: 300 }),
    });
    coordinator.register(makeMember(), { qualityManaged: true });
    coordinator.register(
      makeMember({ work: { operations: 1, progressSerial: 0 } }),
      { qualityManaged: false },
    );
    expect(coordinator.stats().governor.activity).toMatchObject({
      workPending: true,
      measurementEligible: false,
    });
  });

  it("quarantines a member with unchanged progress without freezing the healthy view", () => {
    let now = 0;
    const hung = makeMember({
      work: { operations: 1, progressSerial: 7 },
      physicalTileOperations: 1,
    });
    const coordinator = createStreamedSceneCoordinator({
      scheduleRender: vi.fn(),
      memory: createMemoryPool({ totalBytes: 300 }),
      stallWindowMs: 4_000,
      now: () => now,
    });
    coordinator.register(makeMember(), {
      id: "healthy",
      qualityManaged: true,
    });
    coordinator.register(hung, { id: "hung", qualityManaged: false });
    expect(coordinator.stats().governor.activity.measurementEligible).toBe(
      false,
    );

    now = 4_001;
    coordinator.context({}).onWorkChange?.();
    expect(coordinator.stats()).toMatchObject({
      stalledMembers: ["hung"],
      governor: { activity: { measurementEligible: true, workPending: false } },
    });
    expect(hung.onStall).toHaveBeenCalledOnce();

    coordinator.context({}).onWorkChange?.();
    expect(hung.onStall).toHaveBeenCalledOnce();
    coordinator.dispose();
  });

  it("shares one page pool across independent view coordinators", () => {
    const memory = createMemoryPool({ totalBytes: 1_000 });
    const first = createStreamedSceneCoordinator({
      scheduleRender: vi.fn(),
      memory,
    });
    const second = createStreamedSceneCoordinator({
      scheduleRender: vi.fn(),
      memory,
    });
    const firstMember = makeMember();
    const secondMember = makeMember();
    first.register(firstMember);
    second.register(secondMember);
    expect(memory.memberCount()).toBe(2);
    expect(firstMember.allocations.at(-1)?.memoryBudgetBytes).toBe(500);
    expect(secondMember.allocations.at(-1)?.memoryBudgetBytes).toBe(500);
    first.dispose();
    expect(secondMember.allocations.at(-1)?.memoryBudgetBytes).toBe(1_000);
    second.dispose();
  });
});
