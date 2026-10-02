import { afterAll, expect, it } from "vitest";
import { closeBrowser, openExample } from "./harness";

afterAll(closeBrowser);

it("uploads and grows real COPC prefixes without replacing resident buffers", async () => {
  const session = await openExample({
    preparePage: async (page) => {
      await page.addInitScript(() => {
        const state = {
          bytes: 0,
          writes: 0,
          partialWrites: 0,
          mismatches: 0,
          allocations: 0,
        };
        (window as any).pointUploads = state;
        const prototype = WebGL2RenderingContext.prototype;
        const full = prototype.bufferData;
        prototype.bufferData = function (...args: any[]) {
          const data = args[1];
          if (args[0] === this.ARRAY_BUFFER) state.allocations += 1;
          if (args[0] === this.ARRAY_BUFFER && typeof data !== "number") {
            state.bytes += data.byteLength;
            state.writes += 1;
          }
          return (full as any).apply(this, args);
        };
        const partial = prototype.bufferSubData;
        prototype.bufferSubData = function (...args: any[]) {
          const result = (partial as any).apply(this, args);
          if (args[0] === this.ARRAY_BUFFER) {
            const data = args[2];
            const expected = new Uint8Array(
              data.buffer,
              data.byteOffset,
              data.byteLength,
            );
            const actual = new Uint8Array(data.byteLength);
            this.getBufferSubData(args[0], args[1], actual);
            state.partialWrites += 1;
            state.bytes += expected.length;
            state.writes += 1;
            if (expected.some((value, index) => value !== actual[index]))
              state.mismatches += 1;
          }
          return result;
        };
      });
    },
  });
  try {
    await session.setBudgetMode("fixed");
    await session.setDensityFraction(0.25);
    await session.settle();
    await session.page.evaluate(() => {
      const uploads = (window as any).pointUploads;
      uploads.bytes =
        uploads.writes =
        uploads.partialWrites =
        uploads.mismatches =
        uploads.allocations =
          0;
    });
    await session.load(`${session.origin}/fixtures/fixture.copc.laz`);
    expect(session.failures).toEqual([]);
    expect((await session.stats()).controller).not.toBeNull();
    const quarter = await session.settle();
    const before = await session.page.evaluate(() => ({
      ...(window as any).pointUploads,
    }));
    expect(quarter.adapter!.drawnPoints).toBeGreaterThan(0);
    expect(before.bytes).toBe(quarter.adapter!.drawnPoints * 16);
    expect(before.bytes).toBeLessThan(quarter.adapter!.submittedPoints * 16);
    expect(before.partialWrites).toBeGreaterThan(0);
    expect(before.mismatches).toBe(0);

    const keys = await session.keys();
    const served = session.served.length;
    await session.setDensityFraction(0.5);
    const half = await session.settle();
    const after = await session.page.evaluate(() => ({
      ...(window as any).pointUploads,
    }));
    expect(after.allocations).toBe(before.allocations);
    expect(after.bytes - before.bytes).toBe(
      (half.adapter!.drawnPoints - quarter.adapter!.drawnPoints) * 16,
    );
    expect(after.mismatches).toBe(0);
    expect(await session.keys()).toEqual(keys);
    expect(session.served.length).toBe(served);
    expect(session.failures).toEqual([]);
    console.info(
      "COPC_ADMISSION_EVIDENCE",
      JSON.stringify({
        submittedPoints: quarter.adapter!.submittedPoints,
        quarterPoints: quarter.adapter!.drawnPoints,
        halfPoints: half.adapter!.drawnPoints,
        initialBytes: before.bytes,
        appendedBytes: after.bytes - before.bytes,
        allocations: before.allocations,
      }),
    );
  } finally {
    await session.close();
  }
});
