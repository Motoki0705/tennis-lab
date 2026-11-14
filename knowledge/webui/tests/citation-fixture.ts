import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { stringify } from "yaml";

export const citationFixtureRoot = fileURLToPath(
  new URL("../.citation-e2e/", import.meta.url),
);
export const citationFixtureSource = fileURLToPath(
  new URL("../../../tests/fixtures/knowledge/citation_map/", import.meta.url),
);
export const fixtureIds = {
  A: "cite-78f78b45b14c940f",
  B: "cite-15ff6ca63fce212e",
  C: "cite-520c4b649c4b20f6",
  D: "cite-297fb8cf588b14c9",
  E: "cite-9278de4b09038c2d",
};
export async function prepareCitationFixture() {
  await fs.rm(citationFixtureRoot, { recursive: true, force: true });
  await fs.mkdir(path.join(citationFixtureRoot, "nodes"), { recursive: true });
  await fs.mkdir(
    path.join(citationFixtureRoot, "Papers", "paper-2024-fixture-d"),
    { recursive: true },
  );
  await fs.writeFile(
    path.join(citationFixtureRoot, "summary.md"),
    "# Citation UI fixture\n",
  );
  await fs.cp(
    citationFixtureSource,
    path.join(citationFixtureRoot, "citation_map"),
    { recursive: true },
  );
  await fs.writeFile(
    path.join(
      citationFixtureRoot,
      "Papers",
      "paper-2024-fixture-d",
      "paper.md",
    ),
    `---\n${stringify({ id: "paper-2024-fixture-d", type: "paper", title: "Registered Fixture D", year: 2024, authors: ["Fixture Author"], tasks: ["citation_fixture"], source: "https://example.org/fixture-d", summary: "A fictional registered paper for UI verification.", reason: "Human selected fixture for testing external-ID navigation.", external_ids: { doi: "10.5555/fixture-d" } })}---\nTest reading notes.\n`,
  );
}
if (process.argv[1] === fileURLToPath(import.meta.url))
  await prepareCitationFixture();
