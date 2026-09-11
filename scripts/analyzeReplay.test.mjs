import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, it } from "vitest";

it("reports verified settlement and leaves unobserved churn blank in the CLI", () => {
  const directory = mkdtempSync(join(tmpdir(), "replay-analysis-"));
  const output = join(directory, "analysis.json");
  try {
    for (const repeat of [1, 2]) {
      const coordinator = {
        governor: {
          regime: "stationary",
          needsFrame: repeat === 1,
          activity: { workPending: false },
        },
        members: [{ active: true, qualityManaged: true }],
        submissions: { queuedJobs: 0 },
      };
      const marker = (atMs, label, state = {}) => ({
        type: "state",
        atMs,
        reason: `marker:${label}`,
        state,
      });
      writeFileSync(
        join(directory, `${repeat}.json`),
        JSON.stringify({
          schemaVersion: 1,
          recording: { name: "points" },
          config: { name: "baseline" },
          repeat,
          network: { mode: "replay", transferredBytes: 0 },
          fidelity: {
            dispatch: { meanLatenessMs: 0, maxLatenessMs: 0, lateEvents: 0 },
          },
          trace: {
            environment: {
              webgl: { renderer: "test", softwareRenderer: false },
            },
            events: [
              marker(0, "replay-start"),
              {
                type: "frame",
                atMs: 1,
                rafIntervalMs: 16,
                state: {
                  coordinator: { governor: { regime: "interaction" } },
                  members: [],
                },
              },
              marker(10, "replay-ended"),
              marker(20, "settled-after-replay", {
                coordinator,
                members: [{ kind: "pointCloud", drawnPoints: 8_101_165 }],
              }),
            ],
          },
        }),
      );
    }
    const report = execFileSync(
      process.execPath,
      [
        fileURLToPath(new URL("./analyzeReplay.mjs", import.meta.url)),
        directory,
        "--json",
        output,
      ],
      { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] },
    );
    const rows = JSON.parse(readFileSync(output, "utf8"));
    expect(rows[0].settled).toMatchObject({
      confirmed: false,
      drawnPoints: null,
    });
    expect(rows[0].convergence.msAfterGesture).toBeNull();
    expect(rows[1].settled).toMatchObject({
      confirmed: true,
      drawnPoints: 8_101_165,
    });
    expect(rows[1].convergence.msAfterGesture).toBe(10);
    expect(report).toContain("| baseline | 1 | — | — | — |");
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
