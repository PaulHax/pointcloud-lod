/**
 * Catches a page whose main thread stops answering.
 *
 * A frozen page cannot answer the harness's own evaluations, so without this a
 * run that hangs only ends at the test timeout, with nothing to say where it
 * was stuck. The watchdog asks the page a trivial question every second from
 * its own DevTools session; once it has gone unanswered for `frozenMs`, it
 * pauses the page (the debugger can interrupt a busy loop), records the
 * stack, and closes the page so every pending evaluation fails at once.
 */

import type { Page } from "playwright";

export type Freeze = {
  readonly afterMs: number;
  readonly stack: readonly string[];
};

export type Watchdog = {
  /** The freeze that closed the page, if one did. */
  freeze(): Freeze | null;
  stop(): Promise<void>;
};

const POLL_MS = 1_000;
const ANSWER_MS = 3_000;

export const watchResponsiveness = async (
  page: Page,
  { frozenMs = 12_000 }: { readonly frozenMs?: number } = {},
): Promise<Watchdog> => {
  const cdp = await page.context().newCDPSession(page);
  await cdp.send("Debugger.enable");
  let stopped = false;
  let frozen: Freeze | null = null;
  let unansweredSince: number | null = null;

  const answers = (): Promise<boolean> =>
    Promise.race([
      cdp.send("Runtime.evaluate", { expression: "0" }).then(
        () => true,
        () => true,
      ),
      new Promise<boolean>((resolve) =>
        setTimeout(() => resolve(false), ANSWER_MS),
      ),
    ]);

  const capture = (): Promise<readonly string[]> =>
    new Promise((resolve) => {
      cdp.once("Debugger.paused", (event) => {
        resolve(
          event.callFrames.slice(0, 20).map((frame) => {
            const file = frame.url.slice(frame.url.lastIndexOf("/") + 1);
            return `${frame.functionName || "(anonymous)"} ${file}:${frame.location.lineNumber + 1}`;
          }),
        );
      });
      void cdp.send("Debugger.pause").catch(() => resolve([]));
    });

  const loop = (async () => {
    while (!stopped) {
      await new Promise((resolve) => setTimeout(resolve, POLL_MS));
      if (stopped) break;
      if (await answers()) {
        unansweredSince = null;
        continue;
      }
      unansweredSince ??= Date.now() - ANSWER_MS;
      const afterMs = Date.now() - unansweredSince;
      if (afterMs < frozenMs) continue;
      frozen = { afterMs, stack: await capture() };
      stopped = true;
      await page.close().catch(() => {});
    }
  })();

  return {
    freeze: () => frozen,
    stop: async () => {
      stopped = true;
      await loop;
      await cdp.detach().catch(() => {});
    },
  };
};

export const describeFreeze = (freeze: Freeze): string =>
  `page main thread froze for ${Math.round(freeze.afterMs / 1000)} s; ` +
  `paused at:\n  ${freeze.stack.join("\n  ")}`;
