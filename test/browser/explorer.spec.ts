/**
 * The explorer's own controls.
 *
 * The rest of the browser suite drives `instrumented/`, which is one point
 * cloud and every low-level knob. This drives the page a visitor actually
 * lands on, through the same harness the benchmark uses — headless here,
 * because nothing in it is a measurement.
 */

import { afterAll, describe, expect, it } from "vitest";

import { closeBenchmarkBrowser, openScene } from "./sceneHarness";

afterAll(closeBenchmarkBrowser);

/** Each view-quality control, and what its help says it does. */
const VIEW_QUALITY_HELP = [
  "The view governor adjusts rendering quality across the loaded datasets to keep interaction responsive and refine the view when it settles.",
  "Frame-time target while the camera is moving. Lower values favor responsiveness over detail.",
  "Frame-time target after camera motion stops. A larger value allows more detail per frame while the view refines.",
];

describe("explorer controls", () => {
  it("explains every view-quality control, by pointer and by keyboard", async () => {
    const session = await openScene({ path: "/index.html", headless: true });
    try {
      // The same words three ways: the hover title, the accessible name, and
      // the text the stylesheet paints. A tip that carries only one of them
      // is help somebody cannot reach.
      expect(
        await session.page.locator(".info-tip").evaluateAll((tips) =>
          tips.map((tip) => ({
            title: tip.getAttribute("title"),
            label: tip.getAttribute("aria-label"),
            painted: tip.getAttribute("data-tooltip"),
            focusable: tip.getAttribute("tabindex") === "0",
          })),
        ),
      ).toEqual(
        VIEW_QUALITY_HELP.map((help) => ({
          title: help,
          label: help,
          painted: help,
          focusable: true,
        })),
      );
      expect(session.failures).toEqual([]);
    } finally {
      await session.close();
    }
  });

  it("says nothing is loaded until a dataset is added", async () => {
    const session = await openScene({ path: "/index.html", headless: true });
    try {
      expect(await session.page.locator("#message").textContent()).toBe(
        "Add a point cloud or 3D Tiles dataset.",
      );
      expect(await session.datasets()).toEqual([]);
      expect(
        await session.page
          .locator("#governor-activity")
          .getAttribute("aria-label"),
      ).toBe("Add a dataset to begin streaming");
      expect(session.failures).toEqual([]);
    } finally {
      await session.close();
    }
  });
});
