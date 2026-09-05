/**
 * Consumer callbacks are observers: an application's `onError` or `onContent`
 * throwing must not abandon the fetch accounting, retry state or renderer
 * cleanup that was mid-flight when it was notified. Applications surface their
 * own callback failures.
 */
export const safeCall = (callback: (() => void) | undefined): void => {
  try {
    callback?.();
  } catch {
    // Deliberately swallowed: see the module doc.
  }
};
