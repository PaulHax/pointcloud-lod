import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

import type { Tiles3dMemberConfig } from "./memberTypes";
import { validateTiles3dMemberConfig } from "./memberConfig";

type SourceDocument = {
  readonly endpoint: string;
  readonly revision: string;
  readonly tiles3d: Omit<Tiles3dMemberConfig, "endpoint" | "revision">;
};

type ContractCase = {
  readonly name: string;
  readonly reason?: string;
  readonly document: SourceDocument;
};

const corpus = JSON.parse(
  readFileSync(
    new URL(
      "../../test/fixtures/tiles3d-source-contract.json",
      import.meta.url,
    ),
    "utf8",
  ),
) as {
  readonly valid: readonly ContractCase[];
  readonly invalid: readonly ContractCase[];
};

const configFrom = (source: SourceDocument): Tiles3dMemberConfig => ({
  endpoint: source.endpoint,
  revision: source.revision,
  ...source.tiles3d,
});

describe("shared Tiles3DSource wire contract", () => {
  for (const fixture of corpus.valid) {
    it(`accepts ${fixture.name}`, () => {
      expect(() =>
        validateTiles3dMemberConfig(configFrom(fixture.document)),
      ).not.toThrow();
    });
  }

  for (const fixture of corpus.invalid) {
    it(`rejects ${fixture.name}`, () => {
      expect(() =>
        validateTiles3dMemberConfig(configFrom(fixture.document)),
      ).toThrow(new RegExp(fixture.reason!, "i"));
    });
  }
});
