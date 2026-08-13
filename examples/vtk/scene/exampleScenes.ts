import { HOSTED_POINT_CLOUDS, PLACE_LABELS } from "./sceneCatalog";

export { HOSTED_POINT_CLOUDS } from "./sceneCatalog";

export type ExampleSceneKind = "points" | "tiles" | "combined";

type SceneSelection = {
  readonly kind: ExampleSceneKind;
  readonly value: string;
};

export type ExampleSceneSelect = {
  setCurrent(selection: SceneSelection): void;
};

type ExampleSceneSelectOptions = {
  readonly kind: ExampleSceneKind;
  readonly currentValue: string;
  readonly onSelect: (value: string) => void;
};

const optionValue = ({ kind, value }: SceneSelection): string =>
  `${kind}:${value}`;

const selectionFrom = (value: string): SceneSelection | null => {
  const separator = value.indexOf(":");
  if (separator < 0) return null;
  const kind = value.slice(0, separator);
  if (kind !== "points" && kind !== "tiles" && kind !== "combined") {
    return null;
  }
  return { kind, value: value.slice(separator + 1) };
};

const addGroup = (
  select: HTMLSelectElement,
  label: string,
  kind: ExampleSceneKind,
  choices: readonly { readonly label: string; readonly value: string }[],
): void => {
  const group = document.createElement("optgroup");
  group.label = label;
  for (const choice of choices) {
    const option = document.createElement("option");
    option.value = optionValue({ kind, value: choice.value });
    option.textContent = choice.label;
    group.append(option);
  }
  select.append(group);
};

const navigateTo = (selection: SceneSelection): void => {
  const currentPath = window.location.pathname;
  const exampleRoot =
    currentPath.endsWith("/mesh/") ||
    currentPath.endsWith("/combined/") ||
    currentPath.endsWith("/complete/")
      ? new URL("../", window.location.href)
      : new URL("./", window.location.href);
  const target = new URL(
    selection.kind === "points"
      ? "./"
      : `./${selection.kind === "tiles" ? "mesh" : "combined"}/`,
    exampleRoot,
  );
  if (selection.kind === "points")
    target.searchParams.set("url", selection.value);
  else target.searchParams.set("place", selection.value);
  window.location.assign(target);
};

/** One scene picker shared by the point, 3D Tiles and combined examples. */
export const installExampleSceneSelect = (
  select: HTMLSelectElement,
  options: ExampleSceneSelectOptions,
): ExampleSceneSelect => {
  select.replaceChildren();

  const custom = document.createElement("option");
  custom.value = "";
  custom.textContent = "Custom point cloud…";
  select.append(custom);

  addGroup(
    select,
    "Point clouds",
    "points",
    HOSTED_POINT_CLOUDS.map(({ label, url }) => ({ label, value: url })),
  );
  const places = Object.values(PLACE_LABELS).map((label) => ({
    label,
    value: label,
  }));
  addGroup(select, "3D Tiles", "tiles", places);
  addGroup(select, "3D Tiles + point cloud", "combined", places);

  const setCurrent = (selection: SceneSelection): void => {
    const value = optionValue(selection);
    select.value = Array.from(select.options).some(
      (candidate) => candidate.value === value,
    )
      ? value
      : "";
  };

  setCurrent({ kind: options.kind, value: options.currentValue });
  select.addEventListener("change", () => {
    const selection = selectionFrom(select.value);
    if (!selection) return;
    if (selection.kind === options.kind) options.onSelect(selection.value);
    else navigateTo(selection);
  });

  return { setCurrent };
};
