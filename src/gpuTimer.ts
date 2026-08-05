export type GpuTimerResult = {
  readonly id: number;
  readonly status: "valid" | "disjoint" | "error";
  readonly gpuMs: number | null;
};

export type GpuFrameTimer = {
  readonly supported: boolean;
  /** Begin one non-overlapping TIME_ELAPSED query. */
  begin(): number | null;
  /** End the active query, if any, and enqueue asynchronous readback. */
  end(): void;
  /** Poll without blocking. Pending results schedule another poll. */
  poll(): void;
  dispose(): void;
};

type TimerQueryExtension = {
  readonly TIME_ELAPSED_EXT: number;
  readonly GPU_DISJOINT_EXT: number;
};

export type GpuFrameTimerOptions = {
  readonly onResult: (result: GpuTimerResult) => void;
  readonly schedulePoll?: (poll: () => void) => unknown;
  readonly cancelPoll?: (handle: unknown) => void;
};

type PendingQuery = {
  readonly id: number;
  readonly query: WebGLQuery;
};

/**
 * Asynchronous WebGL2 GPU timing. Query results are never read until the
 * driver reports them available, so timing cannot serialize the render loop.
 */
export const createGpuFrameTimer = (
  gl: WebGL2RenderingContext | null,
  options: GpuFrameTimerOptions,
): GpuFrameTimer => {
  const extension = gl?.getExtension(
    "EXT_disjoint_timer_query_webgl2",
  ) as TimerQueryExtension | null;
  const schedulePoll =
    options.schedulePoll ??
    ((poll: () => void): unknown => requestAnimationFrame(() => poll()));
  const cancelPoll =
    options.cancelPoll ??
    ((handle: unknown): void => cancelAnimationFrame(handle as number));
  const pending = new Map<number, WebGLQuery>();
  let active: PendingQuery | null = null;
  let pollHandle: unknown = null;
  let nextId = 0;
  let disposed = false;

  const supported = gl !== null && extension !== null;

  const deleteQuery = (query: WebGLQuery): void => {
    try {
      gl?.deleteQuery(query);
    } catch {
      // Context loss can make cleanup fail; the context owns the resource now.
    }
  };

  const emit = (
    id: number,
    status: GpuTimerResult["status"],
    gpuMs: number | null = null,
  ): void => options.onResult({ id, status, gpuMs });

  /** Retire one pending query: forget it, free it, and report its result. */
  const settle = (
    id: number,
    query: WebGLQuery,
    status: GpuTimerResult["status"],
    gpuMs: number | null = null,
  ): void => {
    pending.delete(id);
    deleteQuery(query);
    emit(id, status, gpuMs);
  };

  const ensurePoll = (): void => {
    if (disposed || pending.size === 0 || pollHandle !== null) return;
    pollHandle = schedulePoll(() => {
      pollHandle = null;
      timer.poll();
    });
  };

  const timer: GpuFrameTimer = {
    supported,

    begin() {
      if (disposed || !supported || active !== null) return null;
      const query = gl.createQuery();
      if (query === null) return null;
      const id = ++nextId;
      try {
        gl.beginQuery(extension.TIME_ELAPSED_EXT, query);
        active = { id, query };
        return id;
      } catch {
        deleteQuery(query);
        return null;
      }
    },

    end() {
      if (disposed || active === null || !supported) return;
      const ended = active;
      active = null;
      try {
        gl.endQuery(extension.TIME_ELAPSED_EXT);
        pending.set(ended.id, ended.query);
        ensurePoll();
      } catch {
        deleteQuery(ended.query);
        emit(ended.id, "error");
      }
    },

    poll() {
      if (disposed || !supported || pending.size === 0) return;
      try {
        if (gl.getParameter(extension.GPU_DISJOINT_EXT) === true) {
          for (const [id, query] of pending) settle(id, query, "disjoint");
          return;
        }
      } catch {
        for (const [id, query] of pending) settle(id, query, "error");
        return;
      }

      for (const [id, query] of pending) {
        try {
          if (gl.getQueryParameter(query, gl.QUERY_RESULT_AVAILABLE) !== true) {
            continue;
          }
          const nanoseconds: unknown = gl.getQueryParameter(
            query,
            gl.QUERY_RESULT,
          );
          if (
            typeof nanoseconds === "number" &&
            Number.isFinite(nanoseconds) &&
            nanoseconds >= 0
          ) {
            settle(id, query, "valid", nanoseconds / 1_000_000);
          } else {
            settle(id, query, "error");
          }
        } catch {
          settle(id, query, "error");
        }
      }
      ensurePoll();
    },

    dispose() {
      if (disposed) return;
      disposed = true;
      if (pollHandle !== null) cancelPoll(pollHandle);
      pollHandle = null;
      if (active !== null) {
        try {
          gl?.endQuery(extension?.TIME_ELAPSED_EXT ?? 0);
        } catch {
          // A lost context has already discarded the active query.
        }
        deleteQuery(active.query);
        active = null;
      }
      for (const query of pending.values()) deleteQuery(query);
      pending.clear();
    },
  };

  return timer;
};
