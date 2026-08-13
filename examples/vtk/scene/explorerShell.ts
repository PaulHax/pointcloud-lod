export type ExplorerPreset = "points" | "tiles" | "combined";

const copy = (
  preset: ExplorerPreset,
): { title: string; description: string } =>
  preset === "points"
    ? {
        title: "pointcloud-lod · point cloud",
        description:
          "Stream point clouds and 3D Tiles into one governed vtk.js view. Each dataset keeps its own rendering controls.",
      }
    : preset === "tiles"
      ? {
          title: "pointcloud-lod · 3D Tiles",
          description:
            "Explore streamed 3D Tiles, then add point clouds or more tilesets to the same governed vtk.js view.",
        }
      : {
          title: "pointcloud-lod · combined",
          description:
            "AHN4 lidar and 3DBAG buildings share one memory pool, submission scheduler, frame target, and camera.",
        };

export const renderExplorerShell = (preset: ExplorerPreset): void => {
  const { title, description } = copy(preset);
  document.body.innerHTML = `
    <section class="panel" aria-label="Dataset explorer controls">
      <header class="title-row">
        <h1>${title}</h1>
        <div class="title-actions">
          <button id="reset-view" type="button" class="reset-view">Reset view</button>
        </div>
      </header>

      <p class="description">${description}</p>
      <section id="frame-rate" class="frame-rate-card" aria-label="Frame rate"></section>

      <details class="dataset-card global-card" open>
        <summary>
          <span class="dataset-heading">
            <span class="heading-line">
              <strong>View governor</strong>
              <span class="info-tip" tabindex="0" title="The view governor adjusts rendering quality across the loaded datasets to keep interaction responsive and refine the view when it settles." aria-label="The view governor adjusts rendering quality across the loaded datasets to keep interaction responsive and refine the view when it settles." data-tooltip="The view governor adjusts rendering quality across the loaded datasets to keep interaction responsive and refine the view when it settles.">i</span>
              <span id="governor-activity" class="activity-indicator" data-state="loading" role="status" aria-label="Starting">
                <span class="activity-spinner" aria-hidden="true"></span>
                <span data-role="activity-label">Starting</span>
              </span>
            </span>
            <small>Balances detail and frame time</small>
          </span>
        </summary>
        <div class="dataset-body">
          <div class="field-grid">
            <div>
              <label for="moving-target">Moving target <span class="info-tip" tabindex="0" title="Frame-time target while the camera is moving. Lower values favor responsiveness over detail." aria-label="Frame-time target while the camera is moving. Lower values favor responsiveness over detail." data-tooltip="Frame-time target while the camera is moving. Lower values favor responsiveness over detail.">i</span></label>
              <div class="input-with-unit">
                <input id="moving-target" type="number" min="1" step="1" value="16" />
                <span>ms</span>
              </div>
            </div>
            <div>
              <label for="stationary-target">Settled target <span class="info-tip" tabindex="0" title="Frame-time target after camera motion stops. A larger value allows more detail per frame while the view refines." aria-label="Frame-time target after camera motion stops. A larger value allows more detail per frame while the view refines." data-tooltip="Frame-time target after camera motion stops. A larger value allows more detail per frame while the view refines.">i</span></label>
              <div class="input-with-unit">
                <input id="stationary-target" type="number" min="1" step="1" value="33" />
                <span>ms</span>
              </div>
            </div>
          </div>
          <details class="governor-diagnostics">
            <summary>Advanced diagnostics</summary>
            <div id="stats" aria-label="Shared streaming diagnostics"></div>
          </details>
        </div>
      </details>

      <div class="dataset-toolbar">
        <h2>Datasets</h2>
        <button id="open-add-dataset" type="button">Add dataset</button>
      </div>
      <div id="dataset-list" class="dataset-list"></div>

      <output id="message" role="status">Loading…</output>
    </section>
    <main id="viewer"></main>

    <dialog id="add-dataset-dialog" aria-labelledby="add-dataset-heading">
      <form id="add-dataset-form">
        <div class="dialog-heading">
          <div>
            <h2 id="add-dataset-heading">Add dataset</h2>
            <p id="add-dataset-description">Add another independently controlled member to this view.</p>
          </div>
          <button id="close-add-dataset" class="icon-button" type="button" aria-label="Close">×</button>
        </div>

        <div class="field-grid">
          <div>
            <label for="dataset-type">Data type</label>
            <select id="dataset-type">
              <option value="points">Point cloud (COPC)</option>
              <option value="tiles">3D Tiles 1.1</option>
            </select>
          </div>
          <div>
            <label for="source-type">Source</label>
            <select id="source-type">
              <option value="preset">Hosted preset</option>
              <option value="url">URL</option>
              <option value="local">Local files</option>
            </select>
          </div>
        </div>

        <div id="preset-source-fields">
          <label for="dataset-preset">Preset</label>
          <select id="dataset-preset"></select>
        </div>
        <div id="url-source-fields" hidden>
          <label for="dataset-url">URL</label>
          <input id="dataset-url" type="url" spellcheck="false" placeholder="https://example.com/data.copc.laz" />
        </div>
        <div id="local-source-fields" hidden>
          <label for="dataset-files">Local data</label>
          <input id="dataset-files" type="file" />
          <p id="local-source-hint" class="field-hint"></p>
        </div>

        <label for="dataset-name">Display name <span class="optional">optional</span></label>
        <input id="dataset-name" type="text" placeholder="Derived from source" />
        <output id="add-dataset-error" class="form-error"></output>
        <div class="dialog-actions">
          <button id="cancel-add-dataset" class="secondary-button" type="button">Cancel</button>
          <button id="submit-dataset" type="submit">Add dataset</button>
        </div>
      </form>
    </dialog>
  `;
};
