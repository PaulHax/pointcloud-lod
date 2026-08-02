/** Number of points in the nested prefix drawn at this density. */
export const pointPrefixCount = (
  pointCount: number,
  densityFraction: number,
): number =>
  densityFraction <= 0 || pointCount <= 0
    ? 0
    : Math.max(1, Math.floor(pointCount * densityFraction));

/**
 * Random thinning by a fraction f increases projected sample spacing by
 * 1/sqrt(f): point count is an area density for screen-space coverage.
 */
export const projectedSpacingScale = (densityFraction: number): number =>
  densityFraction > 0 ? 1 / Math.sqrt(densityFraction) : Infinity;
