import type { SceneHost } from "./host";

const WIDTH = 320;
const HEIGHT = 80;
const MAX_FPS = 120;
const CAPACITY = 90;
const FILTER_WEIGHT = 0.15;
const IDLE_MS = 250;

const format = (fps: number): string =>
  `${fps < 100 ? fps.toFixed(1) : Math.round(fps).toLocaleString()} fps`;

const yFor = (fps: number): number =>
  HEIGHT * (1 - Math.min(Math.max(fps, 0), MAX_FPS) / MAX_FPS);

export type FrameRateMonitor = {
  setTargetFrameMs(frameMs: number | null): void;
};

/** Frame cadence display shared by coordinator-backed scene examples. */
export const createFrameRateMonitor = (
  host: SceneHost,
  container: HTMLElement,
): FrameRateMonitor => {
  container.innerHTML = `
    <div class="frame-rate-heading">
      <h2>Frame rate</h2>
      <div class="frame-rate-values">
        <span><output data-role="current">— fps</output><small>current</small></span>
        <span><output data-role="average">— fps</output><small>average</small></span>
      </div>
    </div>
    <div class="frame-rate-plot">
      <svg viewBox="0 0 ${WIDTH} ${HEIGHT}" preserveAspectRatio="none" role="img" aria-label="No frame-rate samples yet">
        <line class="frame-rate-grid" x1="0" y1="20" x2="320" y2="20" />
        <line class="frame-rate-grid" x1="0" y1="40" x2="320" y2="40" />
        <line class="frame-rate-grid" x1="0" y1="60" x2="320" y2="60" />
        <path data-role="area" d="" />
        <polyline data-role="line" points="" />
        <line data-role="target-line" x1="0" y1="40" x2="320" y2="40" hidden />
        <circle data-role="latest" cx="320" cy="80" r="2.5" hidden />
      </svg>
      <span class="frame-rate-scale frame-rate-scale-top">120</span>
      <span class="frame-rate-scale frame-rate-scale-bottom">0</span>
    </div>
    <div class="frame-rate-caption">
      <span data-role="target">No target</span>
      <span>Smoothed · 90 frames</span>
    </div>`;

  const element = <T extends Element>(role: string): T => {
    const found = container.querySelector<T>(`[data-role="${role}"]`);
    if (!found) throw new Error(`The frame-rate card is missing ${role}`);
    return found;
  };
  const chart = container.querySelector<SVGSVGElement>("svg")!;
  const current = element<HTMLOutputElement>("current");
  const average = element<HTMLOutputElement>("average");
  const area = element<SVGPathElement>("area");
  const line = element<SVGPolylineElement>("line");
  const latest = element<SVGCircleElement>("latest");
  const targetLine = element<SVGLineElement>("target-line");
  const target = element<HTMLElement>("target");
  const samples: number[] = [];
  let previousAt: number | null = null;
  let filteredIntervalMs: number | null = null;
  let lastActiveAt: number | null = null;
  let idle = true;

  const plot = (fps: number): void => {
    samples.push(fps);
    if (samples.length > CAPACITY) samples.shift();
    const firstSlot = CAPACITY - samples.length;
    const plotted = samples.map((sample, index) => ({
      x: ((firstSlot + index) / (CAPACITY - 1)) * WIDTH,
      y: yFor(sample),
    }));
    const points = plotted
      .map(({ x, y }) => `${x.toFixed(1)},${y.toFixed(1)}`)
      .join(" ");
    const first = plotted[0]!;
    const newest = plotted[plotted.length - 1]!;
    line.setAttribute("points", points);
    area.setAttribute(
      "d",
      `M ${first.x.toFixed(1)} ${HEIGHT} L ${points.replaceAll(",", " ")} L ${newest.x.toFixed(1)} ${HEIGHT} Z`,
    );
    latest.setAttribute("cx", newest.x.toFixed(1));
    latest.setAttribute("cy", newest.y.toFixed(1));
    latest.removeAttribute("hidden");
  };

  host.onPresentation((presentedAt) => {
    lastActiveAt = presentedAt;
    idle = false;
    const prior = previousAt;
    previousAt = presentedAt;
    if (prior === null) return;
    const intervalMs = presentedAt - prior;
    if (
      !Number.isFinite(intervalMs) ||
      intervalMs <= 0 ||
      intervalMs > IDLE_MS
    ) {
      filteredIntervalMs = null;
      return;
    }
    filteredIntervalMs =
      filteredIntervalMs === null
        ? intervalMs
        : filteredIntervalMs * (1 - FILTER_WEIGHT) + intervalMs * FILTER_WEIGHT;
    const currentFps = 1000 / intervalMs;
    const averageFps = 1000 / filteredIntervalMs;
    plot(averageFps);
    current.value = format(currentFps);
    average.value = format(averageFps);
    chart.setAttribute(
      "aria-label",
      `${format(currentFps)} current, ${format(averageFps)} average, plotted over the latest ${samples.length} rendered frames`,
    );
  });

  setInterval(() => {
    if (
      idle ||
      lastActiveAt === null ||
      performance.now() - lastActiveAt < IDLE_MS
    )
      return;
    idle = true;
    previousAt = null;
    filteredIntervalMs = null;
    plot(0);
    current.value = "0.0 fps";
    average.value = "0.0 fps";
    chart.setAttribute(
      "aria-label",
      `0.0 fps, idle, plotted over the latest ${samples.length} samples`,
    );
  }, 100);

  return {
    setTargetFrameMs(frameMs) {
      if (frameMs === null || !Number.isFinite(frameMs) || frameMs <= 0) {
        targetLine.setAttribute("hidden", "");
        target.textContent = "No target";
        return;
      }
      const fps = 1000 / frameMs;
      const y = yFor(fps).toFixed(1);
      targetLine.setAttribute("y1", y);
      targetLine.setAttribute("y2", y);
      targetLine.removeAttribute("hidden");
      target.textContent = `Target ${format(fps)}`;
    },
  };
};
