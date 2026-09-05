/**
 * Whether a WebGL renderer string names a software rasterizer.
 *
 * Outside `harness/` because the scene host itself asks: a software context
 * reports compressed texture formats it would transcode on the CPU, so the
 * page a visitor loads has to know before it picks a format.
 */
export const isSoftwareRenderer = (
  ...descriptions: readonly (string | null)[]
): boolean =>
  descriptions.some((description) =>
    /swiftshader|llvmpipe|softpipe|lavapipe|software rasterizer|software renderer/i.test(
      description ?? "",
    ),
  );
