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
  closeBrowser,
  MULTIPAGE_CLOUD,
  openExample,
  type CameraReading,
  type ExampleSession,
} from "./harness";

const difference = (to: number[], from: number[]): number[] =>
  to.map((value, axis) => value - from[axis]!);

const dot = (left: number[], right: number[]): number =>
  left.reduce((sum, value, axis) => sum + value * right[axis]!, 0);

const cross = (left: number[], right: number[]): number[] => [
  left[1]! * right[2]! - left[2]! * right[1]!,
  left[2]! * right[0]! - left[0]! * right[2]!,
  left[0]! * right[1]! - left[1]! * right[0]!,
];

const cameraDistance = (reading: CameraReading): number =>
  Math.hypot(...difference(reading.position, reading.focalPoint));

const cameraPitchDegrees = (reading: CameraReading): number => {
  const offset = difference(reading.position, reading.focalPoint);
  return (Math.asin(offset[2]! / Math.hypot(...offset)) * 180) / Math.PI;
};

const relativeGap = (left: number, right: number): number =>
  Math.abs(left - right) /
  Math.max(Math.abs(left), Math.abs(right), Number.MIN_VALUE);

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
      await session.page.waitForTimeout(150);
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
      // the ordinary click-after-looking case that used to discard its first
      // movement event.
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

  it("orbits around a centered pivot after an off-center zoom", async () => {
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
      await session.page.mouse.move(x + 100, y);
      await session.frame();
      await session.page.mouse.up({ button: "right" });
      const afterOrbit = await session.readCamera();
      const beforeEye = difference(
        atOrbitStart.position,
        atOrbitStart.focalPoint,
      );
      const afterEye = difference(afterOrbit.position, afterOrbit.focalPoint);

      expect(afterOrbit.focalPoint).toEqual(atOrbitStart.focalPoint);
      expect(
        relativeGap(cameraDistance(afterOrbit), cameraDistance(atOrbitStart)),
      ).toBeLessThan(1e-12);
      expect(dot(beforeEye, afterEye)).toBeGreaterThan(0);
      expect(cross(beforeEye, afterEye)[2]).toBeLessThan(0);
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
      const initialDirection = difference(initial.focalPoint, initial.position);

      // Pace the opening steps so Chromium presents several distinct frames.
      // Both values and the graph should then describe cadence, while the
      // existing "last frame" diagnostic continues to describe render cost.
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
        dot(difference(atLimit.focalPoint, atLimit.position), initialDirection),
      ).toBeGreaterThan(0);

      await wheelAtViewerCenter(session, 20, 0, 0.82, 0.2);
      const held = await session.readCamera();
      expect(relativeGap(cameraDistance(held), limitedDistance)).toBeLessThan(
        1e-7,
      );

      // Dragging right pans the scene with the pointer: the camera translates
      // left along its screen-right axis, and eye-to-focus distance is
      // unchanged. Crossing the focus reverses this sign.
      await session.page.waitForTimeout(250);
      const beforePan = await session.readCamera();
      const direction = difference(beforePan.focalPoint, beforePan.position);
      const screenRight = cross(direction, beforePan.viewUp);
      await session.drag([{ dx: 80, dy: 0 }]);
      const afterPan = await session.readCamera();
      const pan = difference(afterPan.focalPoint, beforePan.focalPoint);
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
