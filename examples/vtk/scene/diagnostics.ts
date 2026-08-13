/**
 * The streamed-scene diagnostics panel.
 *
 * It shows the numbers that were invisible when the mixed-view defects
 * happened: what share of the view each member claims, what quality the
 * governor gave it back, whether a member has decided its budget is
 * irreducible, and what is queued for submission behind the frame you are
 * looking at. In a two-member scene those columns side by side are the demo.
 */

import type { StreamedSceneCoordinatorStats } from "../../../src";
import type { PointCloudMemberStats } from "../../../src/pointCloudMember";
import type { Tiles3dMemberStats } from "../../../src/tiles3d/memberTypes";

export type MemberStats = Tiles3dMemberStats | PointCloudMemberStats;

export type MemberRow = {
  /** Must match the id the member was registered under. */
  readonly id: string;
  readonly label: string;
  readonly stats: MemberStats;
};

export type DiagnosticsInput = {
  readonly coordinator: StreamedSceneCoordinatorStats;
  readonly rows: readonly MemberRow[];
  readonly frameMs: number;
  readonly rendererName: string;
  readonly textureFormat: string;
};

const megabytes = (bytes: number): string =>
  `${(bytes / 1048576).toFixed(1)} MB`;
const percent = (fraction: number): string => `${(fraction * 100).toFixed(0)}%`;
const count = (value: number): string => value.toLocaleString("en-US");

const isMesh = (stats: MemberStats): stats is Tiles3dMemberStats =>
  stats.kind === "tiles3d";

type Field = readonly [label: string, value: string];

const meshFields = (stats: Tiles3dMemberStats): Field[] => [
  ["source", stats.sourceState],
  ["screen-space error", `${stats.effectiveScreenSpaceErrorPx.toFixed(1)} px`],
  ["selected tiles", count(stats.selectedTiles)],
  ["drawn tiles", count(stats.renderer.drawnTiles)],
  ["drawn triangles", count(stats.renderer.drawnTriangles)],
  ["resident geometry", megabytes(stats.renderer.residentGeometryBytes)],
  ["resident texture", megabytes(stats.renderer.residentTextureBytes)],
  ["queued submission", megabytes(stats.submissions.queuedBytes)],
  [
    "decode queue",
    `${stats.queue?.active ?? 0} active · ${stats.queue?.queued ?? 0} queued`,
  ],
  ["irreducible budget", stats.irreducibleBudget ? "yes — nothing fits" : "no"],
  [
    "errors",
    stats.errorCount === 0
      ? "none"
      : `${stats.errorCount}: ${stats.lastError ?? ""}`,
  ],
];

const pointFields = (stats: PointCloudMemberStats): Field[] => [
  ["mode", stats.adaptive ? "adaptive" : "fixed"],
  ["point budget", count(stats.controller.pointBudget)],
  ["drawn points", count(stats.renderer.submittedPoints)],
  ["drawn tiles", count(stats.renderer.submittedTiles)],
  ["resident", megabytes(stats.renderer.gpuResidentBytes)],
  ["decoded", megabytes(stats.controller.decodedBytes)],
  ["in flight", count(stats.controller.inFlight)],
];

const fieldsHtml = (fields: readonly Field[]): string =>
  fields
    .map(
      ([label, value]) =>
        `<div class="field"><span>${label}</span><b>${value}</b></div>`,
    )
    .join("");

const escape = (value: string): string =>
  value.replace(/[<&>]/g, (character) =>
    character === "<" ? "&lt;" : character === ">" ? "&gt;" : "&amp;",
  );

export const renderDiagnostics = (
  container: HTMLElement,
  input: DiagnosticsInput,
): void => {
  const { coordinator } = input;
  const memberById = new Map(
    coordinator.members.map((member) => [member.id, member]),
  );

  const columns = input.rows
    .map((row) => {
      const shared = memberById.get(row.id);
      const share: Field[] = shared
        ? [
            [
              "share of view",
              percent(shared.governorInputs.projectedImportance),
            ],
            ["quality allocated", percent(shared.allocation.qualityFraction)],
            [
              "memory allowance",
              megabytes(shared.allocation.memoryBudgetBytes),
            ],
            [
              "work pending",
              shared.governorInputs.work.operations > 0 ? "yes" : "no",
            ],
          ]
        : [];
      const own = isMesh(row.stats)
        ? meshFields(row.stats)
        : pointFields(row.stats);
      return `<section class="member"><h3>${escape(row.label)}</h3>${fieldsHtml(
        [...share, ...own],
      )}</section>`;
    })
    .join("");

  container.innerHTML = `
    <section class="view">
      <h3>View</h3>
      ${fieldsHtml([
        ["frame", `${input.frameMs.toFixed(1)} ms`],
        ["regime", coordinator.governor.regime],
        ["view quality", percent(coordinator.viewQualityFraction)],
        ["submission queue", megabytes(coordinator.submissions.queuedBytes)],
        [
          "admitted last frame",
          megabytes(coordinator.submissions.lastFrameAdmittedBytes),
        ],
        ["texture format", input.textureFormat],
        ["renderer", escape(input.rendererName)],
      ])}
    </section>
    ${columns}`;
};
