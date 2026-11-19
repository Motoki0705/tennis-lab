import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import test, { type TestContext } from "node:test";
import { fileURLToPath } from "node:url";
import { stringify } from "yaml";
import { loadCitations, readCitations } from "../src/lib/citation-library";
import { CITATION_ROLES } from "../src/lib/citation-types";
import type {
  CitationLibrary,
  CitationPaper,
  CitationSearch,
} from "../src/lib/citation-types";
import {
  citationRelations,
  rankCitations,
  searchReferences,
} from "../src/lib/citations";

const fixture = fileURLToPath(
  new URL("../../../tests/fixtures/knowledge/citation_map/", import.meta.url),
);
const baseline = await loadCitations(fixture);
const paper = (name: string) =>
  baseline.papers.find((p) => p.title === `Fixture ${name}`)!;
const mixed = "search-fixture-mixed-depth2";
const depth1 = "search-fixture-comparison-depth1";
const pending = "search-fixture-pending-identification";
const cached: CitationSearch = {
  ...baseline.searches.find((s) => s.id === mixed)!,
  id: "search-cached-b",
  root_id: paper("B").id,
  max_depth: 0,
  generations: [{ depth: 0, paper_ids: [paper("B").id], expanded_ids: [] }],
  issues: [],
};
const cachedBaseline = {
  ...baseline,
  searches: [...baseline.searches, cached],
};
const registeredD = {
  id: "paper-2024-fixture-d",
  externalIds: paper("D").external_ids,
};
const document = (data: unknown) =>
  `---\n${stringify(data, { version: "1.1" })}---\n`;
async function workspace(t: TestContext) {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "citation-isolation-"));
  t.after(() => fs.rm(dir, { recursive: true, force: true }));
  await fs.cp(fixture, dir, { recursive: true });
  return dir;
}
function snapshot(library: CitationLibrary, id: string) {
  const search = library.searches.find((s) => s.id === id);
  assert.ok(search, `viewable search ${id}`);
  const rows = searchReferences(library, search);
  return {
    relations: citationRelations(rows, "").map((r) => [
      r.id,
      r.roles,
      r.references,
    ]),
    rankings: CITATION_ROLES.map((role) =>
      rankCitations(rows, role).map((r) => [
        r.paper.id,
        r.count,
        r.sources.map((s) => [s.paper.id, s.references]),
      ]),
    ),
  };
}
async function replacePaper(dir: string, data: CitationPaper) {
  await fs.writeFile(path.join(dir, "papers", `${data.id}.md`), document(data));
}
function alias(identity: string, ids: Record<string, string>): CitationPaper {
  return {
    ...paper("D"),
    id: `cite-${createHash("sha256").update(identity).digest("hex").slice(0, 16)}`,
    identity_key: identity,
    external_ids: ids,
    references: [],
  };
}

test("an invalid search reports its file and reason without changing any healthy map, ranking or Papers join", async (t) => {
  const dir = await workspace(t);
  const id = "search-invalid-status";
  await fs.writeFile(
    path.join(dir, "searches", `${id}.md`),
    document({
      ...baseline.searches.find((s) => s.id === mixed),
      id,
      status: "invalid",
    }),
  );
  const loaded = await loadCitations(dir, [registeredD]);
  assert.equal(loaded.searches.length, baseline.searches.length);
  for (const search of baseline.searches)
    assert.deepEqual(
      snapshot(loaded, search.id),
      snapshot(baseline, search.id),
    );
  assert.deepEqual(loaded.registeredIds, { [paper("D").id]: registeredD.id });
  assert.deepEqual(
    loaded.searchErrors.map((s) => s.id),
    [id],
  );
  assert.match(loaded.searchErrors[0].errors[0].reason, /search status/);
  assert.ok(
    loaded.documentErrors.some((e) =>
      e.files.includes(`citation_map/searches/${id}.md`),
    ),
  );
});

test("an invalid target blocks dependent searches but not a search whose unexpanded cache mentions it", async (t) => {
  const dir = await workspace(t);
  await fs.writeFile(
    path.join(dir, "searches", `${cached.id}.md`),
    document(cached),
  );
  await replacePaper(dir, { ...paper("D"), source: "https:example.org/paper" });
  const loaded = await loadCitations(dir, [registeredD]);
  assert.deepEqual(
    snapshot(loaded, cached.id),
    snapshot(cachedBaseline, cached.id),
  );
  assert.deepEqual(snapshot(loaded, pending), snapshot(baseline, pending));
  assert.ok(loaded.searchErrors.some((s) => s.id === mixed));
  assert.ok(
    loaded.searchErrors.some(
      (s) => s.id === "search-fixture-comparison-depth2",
    ),
  );
  assert.ok(
    loaded.searchErrors
      .find((s) => s.id === mixed)!
      .errors.some((e) =>
        e.files.includes(`citation_map/papers/${paper("D").id}.md`),
      ),
  );
  assert.ok(!loaded.papers.some((p) => p.id === paper("D").id));
  assert.deepEqual(loaded.registeredIds, {});
});

test("expanded sources depend on every resolved target, including roles outside the search filter", async (t) => {
  const dir = await workspace(t);
  await replacePaper(dir, { ...paper("C"), source: "invalid" });
  const loaded = await loadCitations(dir);
  assert.ok(loaded.searchErrors.some((s) => s.id === depth1));
  assert.deepEqual(snapshot(loaded, pending), snapshot(baseline, pending));
});

test("paper reference inconsistencies belong to their source document", async (t) => {
  const dir = await workspace(t);
  const b = structuredClone(paper("B"));
  b.references[0].target_id = "cite-0000000000000000";
  await replacePaper(dir, b);
  const loaded = await loadCitations(dir);
  assert.ok(
    loaded.documentErrors.some(
      (e) =>
        e.files.includes(`citation_map/papers/${b.id}.md`) &&
        /unknown target_id/.test(e.reason),
    ),
  );
  assert.ok(loaded.searchErrors.some((s) => s.id === depth1));
  assert.deepEqual(snapshot(loaded, pending), snapshot(baseline, pending));
});

for (const brokenReference of [false, true])
  test(`external-ID collisions invalidate every participant, including preexisting reference errors: ${brokenReference}`, async (t) => {
    const dir = await workspace(t);
    await fs.writeFile(
      path.join(dir, "searches", `${cached.id}.md`),
      document(cached),
    );
    const duplicate = alias("url:https://example.org/alias-d", {
      ...paper("D").external_ids,
      url: "https://example.org/alias-d",
    });
    if (brokenReference) {
      duplicate.text_status = "available";
      duplicate.summary = "Fixture reading with an invalid reference target.";
      duplicate.references = [structuredClone(paper("B").references[0])];
      duplicate.references[0].target_id = "cite-0000000000000000";
    }
    await replacePaper(dir, duplicate);
    const loaded = await loadCitations(dir, [registeredD]);
    const error = loaded.documentErrors.find((e) =>
      /duplicate literature/.test(e.reason),
    );
    assert.ok(error);
    if (brokenReference)
      assert.ok(
        loaded.documentErrors.some((e) => /unknown target_id/.test(e.reason)),
      );
    for (const id of [paper("D").id, duplicate.id]) {
      assert.ok(error.files.includes(`citation_map/papers/${id}.md`));
      assert.ok(!loaded.papers.some((p) => p.id === id));
    }
    assert.deepEqual(loaded.registeredIds, {});
    assert.deepEqual(
      snapshot(loaded, cached.id),
      snapshot(cachedBaseline, cached.id),
    );
    assert.deepEqual(snapshot(loaded, pending), snapshot(baseline, pending));
  });

test("Papers conflicts reject the citation participant and retain unrelated valid links", async (t) => {
  const dir = await workspace(t);
  const badRegistered = {
    id: "paper-2024-conflict-b",
    externalIds: { arxiv: "2401.00002", doi: "10.5555/conflict" },
  };
  const loaded = await loadCitations(dir, [badRegistered, registeredD]);
  const error = loaded.documentErrors.find((e) =>
    /conflicting external IDs/.test(e.reason),
  );
  assert.ok(error);
  assert.ok(error.files.includes(`citation_map/papers/${paper("B").id}.md`));
  assert.ok(error.files.includes(`Papers/${badRegistered.id}/paper.md`));
  assert.ok(!loaded.papers.some((p) => p.id === paper("B").id));
  assert.deepEqual(loaded.registeredIds, { [paper("D").id]: registeredD.id });
  assert.ok(loaded.searchErrors.some((s) => s.id === mixed));
  assert.deepEqual(snapshot(loaded, pending), snapshot(baseline, pending));
});

test("a registered bridge invalidates both citation aliases instead of choosing one", async (t) => {
  const dir = await workspace(t);
  const d = paper("D");
  const other = alias("url:https://example.org/disjoint-alias", {
    url: "https://example.org/disjoint-alias",
  });
  await replacePaper(dir, other);
  const bridge = {
    id: "paper-2024-bridge",
    externalIds: { ...d.external_ids, ...other.external_ids },
  };
  const loaded = await loadCitations(dir, [bridge]);
  const error = loaded.documentErrors.find((e) =>
    e.files.includes(`Papers/${bridge.id}/paper.md`),
  );
  assert.ok(error);
  for (const id of [d.id, other.id]) {
    assert.ok(error.files.includes(`citation_map/papers/${id}.md`));
    assert.ok(!loaded.papers.some((p) => p.id === id));
  }
  assert.deepEqual(loaded.registeredIds, {});
  assert.deepEqual(snapshot(loaded, pending), snapshot(baseline, pending));
});

test("collisions outside every search remain explicit and do not alter healthy results", async (t) => {
  const dir = await workspace(t);
  const a = alias("doi:10.5555/outside", { doi: "10.5555/outside" });
  const b = alias("url:https://example.org/outside", {
    doi: "10.5555/outside",
    url: "https://example.org/outside",
  });
  await replacePaper(dir, a);
  await replacePaper(dir, b);
  const loaded = await loadCitations(dir, [registeredD]);
  assert.equal(loaded.documentErrors.length, 1);
  assert.equal(loaded.searchErrors.length, 0);
  for (const search of baseline.searches)
    assert.deepEqual(
      snapshot(loaded, search.id),
      snapshot(baseline, search.id),
    );
  assert.deepEqual(loaded.registeredIds, { [paper("D").id]: registeredD.id });
});

test("unrelated duplicate Papers are reported without poisoning citation searches", async (t) => {
  const dir = await workspace(t);
  const loaded = await loadCitations(dir, [
    registeredD,
    { id: "paper-2024-outside-a", externalIds: { doi: "10.5555/outside" } },
    { id: "paper-2024-outside-b", externalIds: { doi: "10.5555/outside" } },
  ]);
  assert.equal(loaded.documentErrors.length, 1);
  assert.equal(loaded.searchErrors.length, 0);
  for (const search of baseline.searches)
    assert.deepEqual(
      snapshot(loaded, search.id),
      snapshot(baseline, search.id),
    );
  assert.deepEqual(loaded.registeredIds, { [paper("D").id]: registeredD.id });
});

test("a path/id disagreement invalidates both claimed identities", async (t) => {
  const dir = await workspace(t);
  await fs.writeFile(
    path.join(dir, "papers", `${paper("A").id}.md`),
    document(paper("D")),
  );
  const loaded = await loadCitations(dir, [registeredD]);
  assert.ok(
    !loaded.papers.some((p) => [paper("A").id, paper("D").id].includes(p.id)),
  );
  assert.deepEqual(loaded.registeredIds, {});
  assert.ok(loaded.documentErrors.some((e) => /match path/.test(e.reason)));
  assert.deepEqual(snapshot(loaded, pending), snapshot(baseline, pending));
});

test("search traversal errors are attributed to that search document", async (t) => {
  const dir = await workspace(t);
  const id = "search-invalid-traversal";
  await fs.writeFile(
    path.join(dir, "searches", `${id}.md`),
    document({
      ...baseline.searches.find((s) => s.id === mixed),
      id,
      max_depth: 0,
    }),
  );
  const loaded = await loadCitations(dir);
  assert.deepEqual(
    loaded.searchErrors.map((s) => s.id),
    [id],
  );
  assert.ok(
    loaded.documentErrors.some(
      (e) =>
        e.files.includes(`citation_map/searches/${id}.md`) &&
        /expanded_ids/.test(e.reason),
    ),
  );
  for (const search of baseline.searches)
    assert.deepEqual(
      snapshot(loaded, search.id),
      snapshot(baseline, search.id),
    );
});

test("unexpanded generation members still require valid paper documents", async (t) => {
  const dir = await workspace(t);
  await fs.writeFile(
    path.join(dir, "searches", `${cached.id}.md`),
    document(cached),
  );
  await replacePaper(dir, { ...paper("B"), source: "invalid" });
  const loaded = await loadCitations(dir);
  const failed = loaded.searchErrors.find((s) => s.id === cached.id);
  assert.ok(failed);
  assert.ok(
    failed.errors.some((e) =>
      e.files.includes(`citation_map/papers/${paper("B").id}.md`),
    ),
  );
  assert.deepEqual(snapshot(loaded, pending), snapshot(baseline, pending));
});

test("missing dependency files affect only searches that reference them", async (t) => {
  const dir = await workspace(t);
  const id = "search-missing-root";
  const missing = "cite-0000000000000000";
  await fs.writeFile(
    path.join(dir, "searches", `${id}.md`),
    document({
      ...cached,
      id,
      root_id: missing,
      generations: [{ depth: 0, paper_ids: [missing], expanded_ids: [] }],
    }),
  );
  const loaded = await loadCitations(dir);
  assert.deepEqual(
    loaded.searchErrors.map((s) => s.id),
    [id],
  );
  assert.ok(
    loaded.documentErrors.some(
      (e) =>
        e.files.includes(`citation_map/papers/${missing}.md`) &&
        /missing citation paper/.test(e.reason),
    ),
  );
  for (const search of baseline.searches)
    assert.deepEqual(
      snapshot(loaded, search.id),
      snapshot(baseline, search.id),
    );
});

test("unattributable document identities still reject the whole library", async (t) => {
  const dir = await workspace(t);
  await fs.writeFile(
    path.join(dir, "papers", "unknown.md"),
    document(paper("D")),
  );
  const result = await readCitations([], dir);
  assert.equal(result.ok, false);
  if (!result.ok) assert.match(result.error, /unknown\.md/);
});
