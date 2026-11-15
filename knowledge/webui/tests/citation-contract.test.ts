import assert from "node:assert/strict";
import test from "node:test";
import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseDocument, stringify } from "yaml";
import {
  loadCitations,
  parseCitationDocument,
  validateCitationLibrary,
} from "../src/lib/citation-documents";
import type { CitationLibrary } from "../src/lib/citation-types";
import { rankCitations, searchReferences } from "../src/lib/citations";

const fixture = fileURLToPath(
  new URL("../../../tests/fixtures/knowledge/citation_map/", import.meta.url),
);
type Documents = Record<
  "papers" | "searches",
  Record<string, Record<string, unknown>>
>;
type Field = { path: (string | number)[]; value?: unknown; remove?: boolean };
const examples: {
  name: string;
  accept: boolean;
  edits: Field[];
  expect?: Field[];
}[] = JSON.parse(
  await fs.readFile(
    path.join(fixture, "..", "citation-boundaries.json"),
    "utf8",
  ),
).libraries;
const documents: Documents = { papers: {}, searches: {} };
for (const kind of ["papers", "searches"] as const) {
  for (const file of (await fs.readdir(path.join(fixture, kind))).filter((f) =>
    f.endsWith(".md"),
  )) {
    const raw = await fs.readFile(path.join(fixture, kind, file), "utf8");
    const match = /^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/.exec(raw);
    assert.ok(match);
    const meta = parseDocument(match[1], { version: "1.1" }).toJS();
    documents[kind][kind === "papers" ? meta.title : meta.id] = meta;
  }
}
function field(value: unknown, keys: (string | number)[]): unknown {
  for (const key of keys) {
    assert.ok(
      value !== null && typeof value === "object" && Object.hasOwn(value, key),
      `missing fixture path ${keys.join("/")}`,
    );
    value = (value as Record<string | number, unknown>)[key];
  }
  return value;
}
function editedDocuments(edits: Field[]): Documents {
  const result = structuredClone(documents);
  for (const edit of edits) {
    const parent = field(result, edit.path.slice(0, -1)) as Record<
      string | number,
      unknown
    >;
    const key = edit.path[edit.path.length - 1];
    if (edit.remove) {
      assert.ok(Object.hasOwn(parent, key));
      if (Array.isArray(parent)) parent.splice(Number(key), 1);
      else delete parent[key];
    } else parent[key] = structuredClone(edit.value);
  }
  return result;
}
function parsedLibrary(raw: Documents): CitationLibrary {
  const library: CitationLibrary = {
    papers: [],
    searches: [],
    registeredIds: {},
  };
  for (const [kind, docs] of Object.entries(raw))
    for (const data of Object.values(docs)) {
      const doc = parseCitationDocument(
        `---\n${stringify(data, { version: "1.1" })}---\n`,
      );
      assert.equal(kind, doc.type === "citation_paper" ? "papers" : "searches");
      if (doc.type === "citation_paper") library.papers.push(doc);
      else library.searches.push(doc);
    }
  validateCitationLibrary(library);
  return library;
}
for (const row of examples)
  test(`shared library contract: ${row.name}`, () => {
    const raw = editedDocuments(row.edits);
    if (!row.accept) return assert.throws(() => parsedLibrary(raw));
    const parsed = parsedLibrary(raw);
    const observed = {
      papers: Object.fromEntries(parsed.papers.map((p) => [p.title, p])),
      searches: Object.fromEntries(parsed.searches.map((s) => [s.id, s])),
    };
    for (const expected of row.expect ?? [])
      assert.deepEqual(field(observed, expected.path), expected.value);
  });

test("saved WASB papers and both searches load without errors", async () => {
  const real = fileURLToPath(new URL("../../citation_map/", import.meta.url));
  const papers = (await fs.readdir(path.join(real, "papers"))).filter((f) =>
    f.endsWith(".md"),
  );
  const library = await loadCitations(real);
  assert.equal(library.papers.length, papers.length);
  assert.ok(library.papers.length >= 115);
  for (const id of [
    "search-wasb-comparison-depth1",
    "search-wasb-comparison-depth2",
  ]) {
    const search = library.searches.find((s) => s.id === id);
    assert.ok(search);
    assert.equal(search.root_id, "cite-baef928f0aefd5a6");
    assert.equal(search.status, "complete");
  }
  const depth2 = library.searches.find(
    (s) => s.id === "search-wasb-comparison-depth2",
  )!;
  const ranking = rankCitations(
    searchReferences(library, depth2),
    "comparison",
  );
  assert.equal(ranking.length, 10);
  assert.deepEqual(
    ranking
      .filter((r) => r.count === 2)
      .map((r) => r.paper.id)
      .sort(),
    ["cite-8f0c4c4fb0a47be0", "cite-aece796df0c08e14", "cite-bee887c491b4ab99"],
  );
  assert.ok(ranking.every((r) => [1, 2].includes(r.count)));
});
