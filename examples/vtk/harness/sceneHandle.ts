/**
 * What a page has to offer before it can be measured.
 *
 * The harness is observation: telemetry, gesture capture and replay, and the
 * automation surface a benchmark drives. None of it belongs in the code being
 * measured, so instead of the explorer reaching into a recorder, the explorer
 * hands over a handle and the harness does the rest. That inverts the
 * dependency — nothing under `scene/` imports anything under `harness/` — and
 * it is why the whole harness can be a dynamic import that an unmeasured
 * session never loads.
 *
 * Everything here is a getter rather than a snapshot: a handle is built once,
 * at start-up, and read repeatedly for the life of the page, so a scene whose
 * datasets change while the harness is attached stays described by it.
 */

import type { PointCloudMemberStats } from "../../../src/pointCloudMember";
import type { Tiles3dMemberStats } from "../../../src/tiles3d/memberTypes";
import type { SceneHost } from "../scene/host";

/** How a dataset reports what it is busy with, in the panel's own words. */
export type HarnessActivity = {
  readonly state: string;
  readonly label: string;
  readonly detail: string;
};

/**
 * The controls a benchmark may move, written as the optional set they are: a
 * point cloud has no screen-space error and a mesh has no point budget. The
 * harness turns a missing one into a named error rather than a silent no-op,
 * because a configuration sweep that quietly skipped a setting would report
 * the baseline twice.
 */
export type HarnessDatasetControls = {
  setVisible(visible: boolean): void;
  setScreenSpaceErrorPx?(px: number): void;
  setBudgetMode?(mode: "adaptive" | "fixed"): void;
  setFixedPointBudget?(points: number): void;
  setMaximumPoints?(points: number | null): void;
  setPointSize?(mode: "auto" | "fixed", value: number): void;
};

export type HarnessDataset = {
  readonly id: string;
  readonly kind: "points" | "tiles";
  readonly label: string;
  readonly controls: HarnessDatasetControls;
  stats(): PointCloudMemberStats | Tiles3dMemberStats;
  activity(): HarnessActivity;
};

export type SceneHandle = {
  /** Names the page shape a trace came from, so two are never averaged. */
  readonly preset: string;
  readonly host: SceneHost;
  /** The element a gesture is aimed at; recorded coordinates are relative to it. */
  readonly viewer: HTMLElement;
  datasets(): readonly HarnessDataset[];
  /** Datasets still loading. Zero, with every dataset settled, is converged. */
  loading(): number;
  error(): string | null;
  qualityTargets(): {
    readonly interactionTargetMs: number;
    readonly stationaryTargetMs: number;
  };
  setQualityTargets(targets: {
    readonly interactionTargetMs?: number;
    readonly stationaryTargetMs?: number;
  }): void;
  /** Frame the whole scene, as the page's own reset control does. */
  resetCamera(): void;
};
