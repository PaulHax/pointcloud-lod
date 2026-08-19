/**
 * Whether this page was asked to be measurable.
 *
 * Its own module, with no imports, because it is checked *before* the harness
 * is loaded — a page that had to import the harness to ask would have already
 * paid for it. Everything else under `harness/` is reachable only through the
 * dynamic import this guards.
 */
export const harnessRequested = (): boolean => {
  const query = new URLSearchParams(window.location.search);
  const asked = (name: string): boolean => {
    const value = query.get(name);
    return value !== null && value !== "0" && value !== "false";
  };
  return asked("record") || asked("telemetry") || asked("harness");
};
