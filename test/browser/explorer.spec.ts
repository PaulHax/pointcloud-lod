/**
 * The explorer's own controls.
 *
 * The rest of the browser suite drives `instrumented/`, which is one point
 * cloud and every low-level knob. This drives the page a visitor actually
 * lands on, through the same harness the benchmark uses — headless here,
 * because nothing in it is a measurement.
 */

import { afterAll, describe, expect, it } from "vitest";
import { readFile } from "node:fs/promises";
import { join } from "node:path";

import { closeBenchmarkBrowser, openScene, FIXTURES } from "./sceneHarness";

afterAll(closeBenchmarkBrowser);

/** Each view-quality control, and what its help says it does. */
const VIEW_QUALITY_HELP = [
  "The view governor adjusts rendering quality across the loaded datasets to keep interaction responsive and refine the view when it settles.",
  "Frame-time target while the camera is moving. Lower values favor responsiveness over detail.",
  "Frame-time target after camera motion stops. A larger value allows more detail per frame while the view refines.",
];

describe("explorer controls", () => {
  it("loads a kilometre-scale cloud from the URL and after replacing a distant dataset", async () => {
    // Scale the small COPC fixture's coordinate system, including its info
    // VLR, without changing compressed point integers or hierarchy offsets.
    const large = await readFile(join(FIXTURES, "fixture.copc.laz"));
    for (let offset = 131; offset < 227; offset += 8) {
      large.writeDoubleLE(large.readDoubleLE(offset) * 100, offset);
    }
    const infoOffset = large.readUInt16LE(94) + 54;
    for (let offset = infoOffset; offset < infoOffset + 40; offset += 8) {
      large.writeDoubleLE(large.readDoubleLE(offset) * 100, offset);
    }
    const session = await openScene({ path: "/index.html", headless: true });
    try {
      await session.page.context().route(
        (url) => url.pathname === "/large.copc.laz",
        async (route) => {
          const range = route
            .request()
            .headers()
            .range?.match(/bytes=(\d+)-(\d+)/);
          const start = range ? Number(range[1]) : 0;
          const end = range
            ? Math.min(Number(range[2]), large.length - 1)
            : large.length - 1;
          await route.fulfill({
            status: range ? 206 : 200,
            headers: {
              "content-type": "application/octet-stream",
              "accept-ranges": "bytes",
              ...(range
                ? { "content-range": `bytes ${start}-${end}/${large.length}` }
                : {}),
            },
            body: large.subarray(start, end + 1),
          });
        },
      );
      const waitForPoints = async () => {
        await expect
          .poll(
            async () => {
              const stats = await session.stats();
              const renderer = stats.members[0]?.stats.renderer as
                | { drawnPoints: number }
                | undefined;
              return (
                stats.loading === 0 &&
                stats.members.length === 1 &&
                (renderer?.drawnPoints ?? 0) > 0
              );
            },
            { timeout: 15_000 },
          )
          .toBe(true);
      };
      await session.page.goto(
        `${session.origin}/?harness=1&url=/large.copc.laz`,
      );
      await waitForPoints();
      await session.page.reload();
      await waitForPoints();

      // Start elsewhere with visible actors, then use the actual replacement
      // dialog. The old actors must not dictate the new cloud's clip planes.
      await session.page.goto(
        `${session.origin}/?harness=1&url=/fixtures/fixture.copc.laz`,
      );
      await waitForPoints();
      await session.page
        .locator(".dataset-heading select")
        .selectOption("custom");
      await session.page.locator("#source-type").selectOption("url");
      await session.page
        .locator("#dataset-url")
        .fill(`${session.origin}/large.copc.laz`);
      await session.page.locator("#submit-dataset").click();
      await expect
        .poll(async () => (await session.datasets())[0]?.label)
        .toBe("large.copc.laz");
      await waitForPoints();
      expect(session.failures).toEqual([]);
    } finally {
      await session.close();
    }
  });

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
