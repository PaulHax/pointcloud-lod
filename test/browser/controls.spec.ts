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
  FIXTURE_CLOUD,
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
): Promise<void> => {
  const box = await session.page.locator("#viewer").boundingBox();
  if (box === null) throw new Error("the viewer has no box to wheel over");
  await session.page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  for (let index = 0; index < steps; index += 1) {
    await session.page.mouse.wheel(0, -100);
    if (pauseMs > 0) await session.page.waitForTimeout(pauseMs);
  }
};

describe("example controls", () => {
  it("stops orbit pitch before either world-up pole", async () => {
    const session = await openExample({ cloud: FIXTURE_CLOUD.urlPath });
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

  it("reports presented FPS and stops wheel zoom before the focal point", async () => {
    const session = await openExample({ cloud: FIXTURE_CLOUD.urlPath });
    try {
      await session.setBudgetMode("fixed");
      const initial = await session.readCamera();
      const initialDistance = cameraDistance(initial);

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
      await wheelAtViewerCenter(session, 180);
      const atLimit = await session.readCamera();
      const limitedDistance = cameraDistance(atLimit);
      expect(limitedDistance).toBeGreaterThan(0);
      expect(limitedDistance).toBeLessThan(initialDistance * 1e-5);

      await wheelAtViewerCenter(session, 20);
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
