import { describe, expect, it, vi } from "vitest";

import { createGpuFrameTimer, type GpuTimerResult } from "./gpuTimer";

const makeContext = () => {
  const query = {} as WebGLQuery;
  let available = false;
  let disjoint = false;
  const gl = {
    QUERY_RESULT_AVAILABLE: 1,
    QUERY_RESULT: 2,
    getExtension: vi.fn(() => ({
      TIME_ELAPSED_EXT: 3,
      GPU_DISJOINT_EXT: 4,
    })),
    createQuery: vi.fn(() => query),
    beginQuery: vi.fn(),
    endQuery: vi.fn(),
    deleteQuery: vi.fn(),
    getParameter: vi.fn(() => disjoint),
    getQueryParameter: vi.fn((_query: WebGLQuery, parameter: number) =>
      parameter === 1 ? available : 12_500_000,
    ),
  } as unknown as WebGL2RenderingContext;
  return {
    gl,
    query,
    setAvailable: (value: boolean) => {
      available = value;
    },
    setDisjoint: (value: boolean) => {
      disjoint = value;
    },
  };
};

describe("createGpuFrameTimer", () => {
  it("collects a TIME_ELAPSED result without reading it early", () => {
    const context = makeContext();
    const results: GpuTimerResult[] = [];
    const polls: (() => void)[] = [];
    const timer = createGpuFrameTimer(context.gl, {
      onResult: (result) => results.push(result),
      schedulePoll: (poll) => polls.push(poll),
      cancelPoll: vi.fn(),
    });

    expect(timer.begin()).toBe(1);
    timer.end();
    expect(results).toEqual([]);

    polls.shift()!();
    expect(results).toEqual([]);
    context.setAvailable(true);
    polls.shift()!();

    expect(results).toEqual([{ id: 1, status: "valid", gpuMs: 12.5 }]);
    expect(context.gl.deleteQuery).toHaveBeenCalledWith(context.query);
  });

  it("rejects every pending query when the context reports disjoint", () => {
    const context = makeContext();
    const results: GpuTimerResult[] = [];
    const timer = createGpuFrameTimer(context.gl, {
      onResult: (result) => results.push(result),
      schedulePoll: () => 1,
      cancelPoll: vi.fn(),
    });

    timer.begin();
    timer.end();
    context.setDisjoint(true);
    timer.poll();

    expect(results).toEqual([{ id: 1, status: "disjoint", gpuMs: null }]);
  });

  it("is inert when the extension is unavailable", () => {
    const context = makeContext();
    vi.mocked(context.gl.getExtension).mockReturnValue(null);
    const timer = createGpuFrameTimer(context.gl, {
      onResult: vi.fn(),
    });

    expect(timer.supported).toBe(false);
    expect(timer.begin()).toBeNull();
    timer.end();
    timer.poll();
  });

  it("does not emit an orphan result when a query cannot begin", () => {
    const context = makeContext();
    const onResult = vi.fn();
    vi.mocked(context.gl.beginQuery).mockImplementation(() => {
      throw new Error("context lost");
    });
    const timer = createGpuFrameTimer(context.gl, {
      onResult,
      schedulePoll: vi.fn(),
      cancelPoll: vi.fn(),
    });

    expect(timer.begin()).toBeNull();
    expect(onResult).not.toHaveBeenCalled();
    expect(context.gl.deleteQuery).toHaveBeenCalledWith(context.query);
  });
});
