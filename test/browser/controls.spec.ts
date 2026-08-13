/**
 * User-facing instruments and camera controls in the assembled example.
 *
 * These checks use pointer and wheel input rather than the page's camera
 * handles. The failure modes live in the interactor path: frame presentation
 * can differ from render cost, and a dolly can lose numerical separation
 * between the eye and its focus only after many real wheel steps.
 */

import { afterAll, describe, expect, it } from "vitest";

import {
  cameraDistance,
  cameraPitchDegrees,
  closeBrowser,
  cross,
  dot,
  MULTIPAGE_CLOUD,
  openExample,
  subtract,
  type CameraReading,
  type ExampleSession,
} from "./harness";

const cameraYawDegrees = (reading: CameraReading): number => {
  const offset = subtract(reading.position, reading.focalPoint);
  return (Math.atan2(offset[1]!, offset[0]!) * 180) / Math.PI;
};

const signedAngleDifference = (after: number, before: number): number =>
  ((((after - before) % 360) + 540) % 360) - 180;

const relativeGap = (left: number, right: number): number =>
  Math.abs(left - right) /
  Math.max(Math.abs(left), Math.abs(right), Number.MIN_VALUE);

const normalised = (vector: readonly number[]): number[] => {
  const length = Math.hypot(...vector);
  return vector.map((value) => value / length);
};

const projectPoint = (
  camera: CameraReading,
  point: readonly number[],
  box: { x: number; y: number; width: number; height: number },
): { x: number; y: number } => {
  const forward = normalised(subtract(camera.focalPoint, camera.position));
  const right = normalised(cross(forward, camera.viewUp));
  const up = normalised(cross(right, forward));
  const eyeToPoint = subtract(point, camera.position);
  const depth = dot(eyeToPoint, forward);
  const halfHeight = camera.parallelProjection
    ? camera.parallelScale
    : depth * Math.tan((camera.viewAngle * Math.PI) / 360);
  const x = dot(eyeToPoint, right) / (halfHeight * (box.width / box.height));
  const y = dot(eyeToPoint, up) / halfHeight;
  return {
    x: box.x + ((x + 1) * box.width) / 2,
    y: box.y + ((1 - y) * box.height) / 2,
  };
};

const focalPlanePoint = (
  camera: CameraReading,
  local: { x: number; y: number },
  box: { width: number; height: number },
): number[] => {
  const forward = normalised(subtract(camera.focalPoint, camera.position));
  const right = normalised(cross(forward, camera.viewUp));
  const up = normalised(cross(right, forward));
  const focalDepth = dot(subtract(camera.focalPoint, camera.position), forward);
  const halfHeight = camera.parallelProjection
    ? camera.parallelScale
    : focalDepth * Math.tan((camera.viewAngle * Math.PI) / 360);
  const horizontal = (2 * local.x) / box.width - 1;
  const vertical = 1 - (2 * local.y) / box.height;
  return camera.focalPoint.map(
    (value, axis) =>
      value +
      right[axis]! * horizontal * halfHeight * (box.width / box.height) +
      up[axis]! * vertical * halfHeight,
  );
};

const wheelAtViewerCenter = async (
  session: ExampleSession,
  steps: number,
  pauseMs = 0,
  xRatio = 0.5,
  yRatio = 0.5,
): Promise<void> => {
  const box = await session.page.locator("#viewer").boundingBox();
  if (box === null) throw new Error("the viewer has no box to wheel over");
  await session.page.mouse.move(
    box.x + box.width * xRatio,
    box.y + box.height * yRatio,
  );
  for (let index = 0; index < steps; index += 1) {
    await session.page.mouse.wheel(0, -100);
    if (pauseMs > 0) await session.page.waitForTimeout(pauseMs);
  }
};

describe("example controls", () => {
  it("exposes Fixed size and Auto scale as live point-presentation controls", async () => {
    const session = await openExample({ cloud: MULTIPAGE_CLOUD.urlPath });
    try {
      expect(
        (await session.page.locator("[role=tooltip]").allTextContents()).map(
          (text) => text.replace(/\s+/g, " ").trim(),
        ),
      ).toEqual([
        "Adaptive changes the visible-point budget to meet frame-time targets; Fixed uses the configured point count.",
        "Sets the frame-time target used to tune quality while the camera is moving.",
        "Sets the frame-time target used to tune quality after camera movement stops.",
        "Caps the adaptive visible-point budget; leave blank to use only the memory-derived ceiling.",
        "Auto derives point diameter from projected density; Fixed uses a constant CSS-pixel diameter.",
        "Auto scale multiplies the density-derived point diameter; Fixed size sets its CSS-pixel diameter directly.",
      ]);
      expect(await session.page.locator(".info-tip").count()).toBe(6);
      const budgetInfo = session.page.locator(".info-tip").first();
      await budgetInfo.hover();
      await expect
        .poll(() =>
          session.page
            .locator("#budget-mode-help")
            .evaluate((tooltip) => getComputedStyle(tooltip).opacity),
        )
        .toBe("1");
      const infoBox = await budgetInfo.boundingBox();
      const tooltipBox = await session.page
        .locator("#budget-mode-help")
        .boundingBox();
      expect(infoBox).not.toBeNull();
      expect(tooltipBox).not.toBeNull();
      expect(tooltipBox!.y + tooltipBox!.height).toBeLessThanOrEqual(
        infoBox!.y,
      );
      expect(
        await session.page.locator("#budget-mode").evaluate((budget) => {
          const adaptive = document.querySelector("#adaptive-controls")!;
          const pointSize = document.querySelector("#point-size-mode")!;
          return {
            adaptiveFollowsBudget: Boolean(
              budget.compareDocumentPosition(adaptive) &
              Node.DOCUMENT_POSITION_FOLLOWING,
            ),
            pointSizeFollowsAdaptive: Boolean(
              adaptive.compareDocumentPosition(pointSize) &
              Node.DOCUMENT_POSITION_FOLLOWING,
            ),
          };
        }),
      ).toEqual({
        adaptiveFollowsBudget: true,
        pointSizeFollowsAdaptive: true,
      });
      expect(await session.page.locator("a[href='./simple/']").count()).toBe(0);
      expect((await session.stats()).controller?.presentation).toMatchObject({
        config: {
          mode: "auto",
          userScale: 0.5,
          minDiameterCssPx: 1.5,
          maxDiameterCssPx: 4,
        },
      });
      expect(
        await session.page.locator("#point-size-label").textContent(),
      ).toBe("Auto scale");
      expect(
        await session.page.locator("#point-size-value").textContent(),
      ).toBe("0.50×");
      await expect
        .poll(() =>
          session.page.locator("#point-size").evaluate((input) => ({
            min: (input as HTMLInputElement).min,
            max: (input as HTMLInputElement).max,
            step: (input as HTMLInputElement).step,
          })),
        )
        .toEqual({ min: "0.25", max: "2", step: "0.05" });

      await session.page.locator("#point-size").evaluate((input) => {
        const range = input as HTMLInputElement;
        range.value = "1.5";
        range.dispatchEvent(new Event("input", { bubbles: true }));
      });
      expect((await session.stats()).controller?.presentation.config).toEqual({
        mode: "auto",
        userScale: 1.5,
        minDiameterCssPx: 1.5,
        maxDiameterCssPx: 4,
      });

      await session.page.locator("#point-size-mode").selectOption("fixed");
      expect(
        await session.page.locator("#point-size-label").textContent(),
      ).toBe("Size (CSS px)");
      expect((await session.stats()).controller?.presentation).toMatchObject({
        config: { mode: "fixed", diameterCssPx: 2 },
        diameterCssPx: 2,
      });
      expect(
        await session.page.locator("#point-size-value").textContent(),
      ).toBe("2.00 px");
      await expect
        .poll(() =>
          session.page.locator("#point-size").evaluate((input) => ({
            min: (input as HTMLInputElement).min,
            max: (input as HTMLInputElement).max,
            step: (input as HTMLInputElement).step,
          })),
        )
        .toEqual({ min: "0.25", max: "8", step: "0.25" });

      await session.page.locator("#point-size").evaluate((input) => {
        const range = input as HTMLInputElement;
        range.value = "3.5";
        range.dispatchEvent(new Event("input", { bubbles: true }));
      });
      expect((await session.stats()).controller?.presentation).toMatchObject({
        config: { mode: "fixed", diameterCssPx: 3.5 },
        diameterCssPx: 3.5,
      });

      await session.page.locator("#point-size-mode").selectOption("auto");
      expect((await session.stats()).controller?.presentation.config).toEqual({
        mode: "auto",
        userScale: 1.5,
        minDiameterCssPx: 1.5,
        maxDiameterCssPx: 4,
      });
      expect(
        await session.page.locator("#point-size-value").textContent(),
      ).toBe("1.50×");
      expect(session.failures).toEqual([]);
    } finally {
      await session.close();
    }
  });

  it("keeps status out of the header and Reset View visibly actionable", async () => {
    const session = await openExample({ cloud: MULTIPAGE_CLOUD.urlPath });
    try {
      const before = await session.page.evaluate(() => {
        const panel = document.querySelector<HTMLElement>(".panel")!;
        const header = document.querySelector<HTMLElement>(".title-row")!;
        const reset = document.querySelector<HTMLElement>("#reset-view")!;
        const style = getComputedStyle(reset);
        return {
          header: header.getBoundingClientRect().toJSON(),
          noHorizontalOverflow: panel.scrollWidth <= panel.clientWidth,
          reset: {
            tag: reset.tagName,
            background: style.backgroundColor,
            border: style.borderStyle,
            cursor: style.cursor,
          },
        };
      });

      expect(await session.page.locator("#regime").count()).toBe(0);
      expect(before.noHorizontalOverflow).toBe(true);
      expect(before.reset).toMatchObject({
        tag: "BUTTON",
        border: "solid",
        cursor: "pointer",
      });
      expect(before.reset.background).not.toBe("rgba(0, 0, 0, 0)");
      expect(
        await session.page.locator("#stats dt").first().textContent(),
      ).toBe("status");

      const box = await session.page.locator("#viewer").boundingBox();
      if (box === null) throw new Error("the viewer has no box to interact in");
      await session.page.mouse.move(
        box.x + box.width / 2,
        box.y + box.height / 2,
      );
      await session.page.mouse.down();
      await session.page.mouse.move(
        box.x + box.width / 2 + 40,
        box.y + box.height / 2,
      );
      await session.page.waitForFunction(
        () => document.querySelector("#stats dd")?.textContent === "moving",
      );
      expect(
        await session.page.locator("#stats dd").first().textContent(),
      ).toBe("moving");

      const during = await session.page.evaluate(() => {
        const panel = document.querySelector<HTMLElement>(".panel")!;
        const header = document.querySelector<HTMLElement>(".title-row")!;
        return {
          header: header.getBoundingClientRect().toJSON(),
          noHorizontalOverflow: panel.scrollWidth <= panel.clientWidth,
        };
      });
      await session.page.mouse.up();

      expect(during.header.width).toBe(before.header.width);
      expect(during.header.height).toBe(before.header.height);
      expect(during.noHorizontalOverflow).toBe(true);
      expect(session.failures).toEqual([]);
    } finally {
      await session.close();
    }
  });

  it("applies the first drag move after the pointer has been still", async () => {
    const session = await openExample({ cloud: MULTIPAGE_CLOUD.urlPath });
    try {
      await session.setBudgetMode("fixed");
      const box = await session.page.locator("#viewer").boundingBox();
      if (box === null) throw new Error("the viewer has no box to pan in");
      const x = box.x + box.width / 2;
      const y = box.y + box.height / 2;

      await session.page.mouse.move(x, y);
      // vtk.js starts a new mouse-move burst after 200 ms of quiet. This is
      // the ordinary click-after-looking case.
      await session.page.waitForTimeout(250);
      const before = await session.readCamera();
      await session.page.mouse.down();
      await session.page.mouse.move(x + 80, y);
      await session.frame();
      await session.page.mouse.up();
      const after = await session.readCamera();

      expect(after.position).not.toEqual(before.position);
      expect(after.focalPoint).not.toEqual(before.focalPoint);
      expect(session.failures).toEqual([]);
    } finally {
      await session.close();
    }
  });

  it("keeps an off-centre focal-plane point under the cursor through a long pan", async () => {
    const session = await openExample({ cloud: MULTIPAGE_CLOUD.urlPath });
    try {
      await session.setBudgetMode("fixed");
      const box = await session.page.locator("#viewer").boundingBox();
      if (box === null) throw new Error("the viewer has no box to pan in");
      const local = { x: box.width * 0.25, y: box.height * 0.22 };
      const start = { x: box.x + local.x, y: box.y + local.y };
      const anchor = focalPlanePoint(await session.readCamera(), local, box);
      const end = { x: start.x, y: box.y + box.height * 0.78 };
      const stepCount = 16;
      await session.drag(
        Array.from({ length: stepCount }, () => ({
          dx: (end.x - start.x) / stepCount,
          dy: (end.y - start.y) / stepCount,
        })),
        { start },
      );
      const projected = projectPoint(await session.readCamera(), anchor, box);
      expect(projected.x).toBeCloseTo(end.x, 0);
      expect(projected.y).toBeCloseTo(end.y, 0);
      expect(session.failures).toEqual([]);
    } finally {
      await session.close();
    }
  });

  it("stops orbit pitch before either world-up pole", async () => {
    const session = await openExample({ cloud: MULTIPAGE_CLOUD.urlPath });
    try {
      await session.setBudgetMode("fixed");
      const initial = await session.readCamera();
      const initialDistance = cameraDistance(initial);
      const box = await session.page.locator("#viewer").boundingBox();
      if (box === null) throw new Error("the viewer has no box to orbit in");
      const x = box.x + box.width / 2;
      const y = box.y + box.height / 2;

      await session.page.mouse.move(x, y);
      await session.page.mouse.down({ button: "right" });
      await session.page.mouse.move(x, y + 100);
      await session.frame();
      const upper = await session.readCamera();
      expect(cameraPitchDegrees(upper)).toBeCloseTo(89, 8);

      // More travel into the pole holds the same orientation, but backing the
      // pointer away responds on its first move instead of sticking until it
      // crosses the whole discarded overshoot.
      await session.page.mouse.move(x, y + 180);
      await session.frame();
      const heldUpper = await session.readCamera();
      expect(cameraPitchDegrees(heldUpper)).toBeCloseTo(89, 8);
      await session.page.mouse.move(x, y + 170);
      await session.frame();
      const awayFromUpper = await session.readCamera();
      expect(cameraPitchDegrees(awayFromUpper)).toBeLessThan(86);
      await session.page.mouse.up({ button: "right" });

      await session.page.locator("#reset-view").click();
      await session.frame();
      await session.page.mouse.move(x, y);
      await session.page.mouse.down({ button: "right" });
      await session.page.mouse.move(x, y - 350);
      await session.frame();
      await session.page.mouse.up({ button: "right" });
      const lower = await session.readCamera();
      expect(cameraPitchDegrees(lower)).toBeCloseTo(-89, 8);
      expect(relativeGap(cameraDistance(lower), initialDistance)).toBeLessThan(
        1e-12,
      );
      expect(session.failures).toEqual([]);
    } finally {
      await session.close();
    }
  });

  it("keeps a centered pivot throughout a long orbit", async () => {
    const session = await openExample({ cloud: MULTIPAGE_CLOUD.urlPath });
    try {
      await session.setBudgetMode("fixed");
      const initial = await session.readCamera();
      await wheelAtViewerCenter(session, 8, 0, 0.8, 0.25);
      const afterZoom = await session.readCamera();
      expect(afterZoom.focalPoint).not.toEqual(initial.focalPoint);

      const box = await session.page.locator("#viewer").boundingBox();
      if (box === null) throw new Error("the viewer has no box to orbit in");
      const x = box.x + box.width / 2;
      const y = box.y + box.height / 2;
      await session.page.mouse.move(x, y);
      await session.page.mouse.down({ button: "right" });
      const atOrbitStart = await session.readCamera();
      const path = [atOrbitStart];
      const steps = 24;
      for (let step = 1; step <= steps; step += 1) {
        await session.page.mouse.move(x + (box.width / 2) * (step / steps), y);
        await session.frame();
        path.push(await session.readCamera());
      }
      await session.page.mouse.up({ button: "right" });
      const afterOrbit = path.at(-1)!;
      const orbitRadius = cameraDistance(atOrbitStart);
      let pathYawDegrees = 0;

      for (let index = 1; index < path.length; index += 1) {
        const previous = path[index - 1]!;
        const current = path[index]!;
        const yawStep = signedAngleDifference(
          cameraYawDegrees(current),
          cameraYawDegrees(previous),
        );
        pathYawDegrees += yawStep;

        // The camera's focal point defines the screen-centre ray. Keeping it
        // fixed while preserving a positive radius proves that the eye moves
        // around the centred pivot rather than through or beyond it.
        expect(current.focalPoint).toEqual(atOrbitStart.focalPoint);
        expect(cameraDistance(current)).toBeGreaterThan(0);
        expect(relativeGap(cameraDistance(current), orbitRadius)).toBeLessThan(
          1e-12,
        );
        expect(yawStep).toBeLessThan(0);
        expect(Math.abs(yawStep)).toBeLessThan(10);
      }

      expect(afterOrbit.focalPoint).toEqual(atOrbitStart.focalPoint);
      expect(Math.abs(pathYawDegrees)).toBeGreaterThan(120);
      expect(afterOrbit.position).not.toEqual(atOrbitStart.position);
      expect(session.failures).toEqual([]);
    } finally {
      await session.close();
    }
  });

  it("reports presented FPS and stops wheel zoom before the focal point", async () => {
    const session = await openExample({ cloud: MULTIPAGE_CLOUD.urlPath });
    try {
      await session.setBudgetMode("fixed");
      const initial = await session.readCamera();
      const initialDistance = cameraDistance(initial);
      const initialDirection = subtract(initial.focalPoint, initial.position);

      // Pace the opening steps so Chromium presents several distinct frames.
      // Both values and the graph should then describe cadence, while the
      // "last frame" diagnostic describes render cost.
      await wheelAtViewerCenter(session, 12, 20);
      await session.page.waitForFunction(
        () =>
          !document
            .querySelector("#frame-rate-value")
            ?.textContent?.includes("—"),
      );
      const currentFps = await session.page
        .locator("#frame-rate-value")
        .textContent();
      const averageFps = await session.page
        .locator("#frame-rate-filtered-value")
        .textContent();
      expect(currentFps).toMatch(/^\d+(?:\.\d)? fps$/);
      expect(averageFps).toMatch(/^\d+(?:\.\d)? fps$/);
      expect(
        await session.page.locator("#frame-rate-line").getAttribute("points"),
      ).not.toBe("");

      // Enough input to pass vtk.js's 1e-20 fallback without the example's
      // scene-relative stop. The camera must approach, then hold, on the near
      // side of its focus.
      // Zoom toward an off-centre point so the focal point moves. The next
      // pan must use that live point, not the framing centre from page load.
      await wheelAtViewerCenter(session, 180, 0, 0.82, 0.2);
      const atLimit = await session.readCamera();
      const limitedDistance = cameraDistance(atLimit);
      expect(limitedDistance).toBeGreaterThan(0);
      expect(limitedDistance).toBeLessThan(initialDistance * 1e-5);
      expect(
        dot(subtract(atLimit.focalPoint, atLimit.position), initialDirection),
      ).toBeGreaterThan(0);

      await wheelAtViewerCenter(session, 20, 0, 0.82, 0.2);
      const held = await session.readCamera();
      expect(relativeGap(cameraDistance(held), limitedDistance)).toBeLessThan(
        1e-7,
      );

      // Dragging right pans the scene with the pointer: the camera translates
      // left along its screen-right axis, and eye-to-focus distance is
      // unchanged. Crossing the focus reverses this sign.
      await session.until(
        "the wheel interaction to end before panning",
        (stats) => stats.controller?.interactionDepth === 0,
      );
      const beforePan = await session.readCamera();
      const direction = subtract(beforePan.focalPoint, beforePan.position);
      const screenRight = cross(direction, beforePan.viewUp);
      await session.drag([{ dx: 80, dy: 0 }]);
      const afterPan = await session.readCamera();
      const pan = subtract(afterPan.focalPoint, beforePan.focalPoint);
      expect(dot(pan, screenRight)).toBeLessThan(0);
      expect(
        relativeGap(cameraDistance(afterPan), cameraDistance(beforePan)),
      ).toBeLessThan(1e-7);

      await session.settle();
      await session.page.waitForFunction(
        () =>
          document.querySelector("#frame-rate-value")?.textContent ===
            "0.0 fps" &&
          document.querySelector("#frame-rate-filtered-value")?.textContent ===
            "0.0 fps",
      );
      expect(session.failures).toEqual([]);
    } finally {
      await session.close();
    }
  });
});

afterAll(closeBrowser);
