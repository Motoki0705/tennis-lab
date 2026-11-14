import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { stringify } from "yaml";
import {
  normalizeIdentifier,
  joinRegisteredPapers,
  loadCitations,
  parseCitationDocument,
  readCitations,
  validateCitationLibrary,
} from "../src/lib/citation-documents";
import {
  citationRelations,
  rankCitations,
  searchReferences,
} from "../src/lib/citations";
import type { CitationPaper, CitationSearch } from "../src/lib/citation-types";

const fixture = fileURLToPath(
  new URL("../../../tests/fixtures/knowledge/citation_map/", import.meta.url),
);
const library = await loadCitations(fixture);
const byTitle = (name: string) =>
  library.papers.find((p) => p.title === `Fixture ${name}`)!;
const document = (data: unknown) =>
  `---\n${stringify(data, { version: "1.1" })}---\n`;
const root = byTitle("A");
const mixed = library.searches.find(
  (s) => s.id === "search-fixture-mixed-depth2",
)!;
const boundaries: {
  timestamps: { name: string; value: string; accept: boolean }[];
  source_urls: { name: string; value: string; accept: boolean }[];
  external_ids: {
    name: string;
    namespace: string;
    value: string;
    normalized: string | null;
  }[];
} = JSON.parse(
  await fs.readFile(
    path.join(fixture, "..", "citation-boundaries.json"),
    "utf8",
  ),
);

for (const row of boundaries.timestamps)
  test(`shared boundary timestamp: ${row.name}`, () => {
    const raw = document({ ...mixed, created_at: row.value });
    if (!row.accept) return assert.throws(() => parseCitationDocument(raw));
    const parsed = parseCitationDocument(raw) as CitationSearch;
    assert.equal(parsed.created_at, row.value);
    validateCitationLibrary({ ...library, searches: [parsed] });
  });

for (const row of boundaries.source_urls)
  test(`shared boundary source URL: ${row.name}`, () => {
    const raw = document({ ...root, source: row.value });
    if (!row.accept) return assert.throws(() => parseCitationDocument(raw));
    const parsed = parseCitationDocument(raw) as CitationPaper;
    assert.equal(parsed.source, row.value);
    validateCitationLibrary({
      ...library,
      papers: library.papers.map((p) => (p.id === parsed.id ? parsed : p)),
    });
  });

for (const row of boundaries.external_ids)
  test(`shared boundary external ID: ${row.name}`, () => {
    if (row.normalized === null)
      assert.throws(() => normalizeIdentifier(row.namespace, row.value));
    else
      assert.equal(
        normalizeIdentifier(row.namespace, row.value),
        row.normalized,
      );
    function parsed(value: string): CitationPaper {
      const identity_key = `${row.namespace}:${value}`;
      return parseCitationDocument(
        document({
          ...root,
          references: [],
          identity_key,
          id: `cite-${createHash("sha256").update(identity_key).digest("hex").slice(0, 16)}`,
          external_ids: { [row.namespace]: value },
        }),
      ) as CitationPaper;
    }
    if (row.normalized !== null) {
      const canonical = parsed(row.normalized);
      assert.deepEqual(canonical.external_ids, {
        [row.namespace]: row.normalized,
      });
      validateCitationLibrary({
        papers: [canonical],
        searches: [],
        registeredIds: {},
      });
    }
    if (row.normalized === null || row.normalized !== row.value)
      assert.throws(() => parsed(row.value));
    else
      assert.deepEqual(parsed(row.value).external_ids, {
        [row.namespace]: row.value,
      });
  });

test("shared documents join canonical target metadata and preserve all converging evidence", () => {
  assert.equal(library.papers.length, 5);
  assert.equal(library.searches.length, 3);
  const rows = searchReferences(library, mixed);
  const comparison = rankCitations(rows, "comparison");
  const d = comparison.find((r) => r.paper.title === "Fixture D")!;
  assert.deepEqual(d.sources.map((s) => s.paper.title).sort(), [
    "Fixture B",
    "Fixture C",
  ]);
  assert.equal(d.paper.source, "https://example.org/fixture-d");
  assert.deepEqual(d.paper.external_ids, { doi: "10.5555/fixture-d" });
  assert.equal(
    d.sources[0].references[0].evidence[0],
    "Section 3, paragraph 1; Table 1",
  );
  const b = comparison.find((r) => r.paper.title === "Fixture B")!;
  assert.equal(b.count, 1);
  assert.deepEqual(
    b.sources[0].references.map((r) => r.ref_id),
    ["ref-1", "ref-6"],
  );
  assert.equal(
    citationRelations(rows, "comparison").filter(
      (r) => r.source.id === root.id && r.target?.id === byTitle("B").id,
    ).length,
    1,
  );
  assert.equal(rows.find((r) => r.reference.ref_id === "ref-4")!.target, null);
});

// Hand counts from tests/fixtures/knowledge/README.md; cached C must not leak.
for (const [searchId, expected] of [
  [
    "search-fixture-mixed-depth2",
    {
      comparison: { B: 1, D: 2, E: 1 },
      foundation: { B: 1, C: 1, D: 1 },
      component: { D: 1 },
      related: { D: 1 },
      benchmark: { E: 1 },
    },
  ],
  [
    "search-fixture-comparison-depth2",
    {
      comparison: { B: 1, D: 1, E: 1 },
      foundation: { B: 1, C: 1 },
      component: { D: 1 },
      related: { D: 1 },
      benchmark: { E: 1 },
    },
  ],
  [
    "search-fixture-comparison-depth1",
    {
      comparison: { B: 1, E: 1 },
      foundation: { B: 1, C: 1 },
      component: { D: 1 },
      related: { D: 1 },
      benchmark: { E: 1 },
    },
  ],
] as const) {
  test(`${searchId}: every role uses distinct sources within expanded_ids`, () => {
    const search = library.searches.find((s) => s.id === searchId)!;
    const rows = searchReferences(library, search);
    for (const role of [
      "comparison",
      "foundation",
      "component",
      "related",
      "benchmark",
    ] as const) {
      assert.deepEqual(
        Object.fromEntries(
          rankCitations(rows, role).map((r) => [
            r.paper.title.slice(-1),
            r.count,
          ]),
        ),
        expected[role],
      );
    }
    assert.equal(
      searchReferences(
        { ...library, searches: [...library.searches, search] },
        search,
      ).length,
      rows.length,
    );
  });
}

test("fetch failure, unidentified target, unclassifiable and pending are independent", () => {
  assert.equal(byTitle("E").text_status, "failed");
  assert.equal(byTitle("D").text_status, "not_requested");
  assert.equal(root.references[3].identification_status, "unresolved");
  assert.equal(root.references[4].identification_status, "resolved");
  assert.equal(root.references[4].decision_status, "unclassifiable");
  const copy = structuredClone(root);
  copy.references[3].decision_status = "pending";
  copy.references[3].roles = ["comparison"];
  assert.throws(() => parseCitationDocument(document(copy)), /empty roles/);
  copy.references[3].decision_status = "classified";
  copy.references[3].evidence = ["Section 2"];
  assert.equal(
    (parseCitationDocument(document(copy)) as CitationPaper).references[3]
      .identification_status,
    "unresolved",
  );
  copy.references[3].decision_status = "pending";
  copy.references[3].roles = [];
  assert.equal(
    (parseCitationDocument(document(copy)) as CitationPaper).references[3]
      .decision_status,
    "pending",
  );
});

const invalidPapers: [string, (p: Record<string, any>) => void, RegExp][] = [
  [
    "unknown key",
    (p) => {
      p.extra = true;
    },
    /expected keys/,
  ],
  [
    "wrong schema",
    (p) => {
      p.schema_version = true;
    },
    /schema_version/,
  ],
  [
    "wrong type",
    (p) => {
      p.type = "paper";
    },
    /type must/,
  ],
  [
    "unstable identity",
    (p) => {
      p.identity_key = "doi:10.5555/other";
    },
    /identity_key/,
  ],
  [
    "bad title",
    (p) => {
      p.title = " ";
    },
    /title/,
  ],
  [
    "boolean year",
    (p) => {
      p.year = true;
    },
    /year/,
  ],
  [
    "nonlist authors",
    (p) => {
      p.authors = "Author";
    },
    /authors/,
  ],
  [
    "unknown namespace",
    (p) => {
      p.external_ids.isbn = "123";
    },
    /namespace/,
  ],
  [
    "nonnormalized id",
    (p) => {
      p.external_ids.doi = "https://doi.org/10.5555/FIXTURE-A";
    },
    /normalized/,
  ],
  [
    "empty ids",
    (p) => {
      p.external_ids = {};
    },
    /at least one/,
  ],
  [
    "unsafe source",
    (p) => {
      p.source = "javascript:alert(1)";
    },
    /HTTP/,
  ],
  [
    "unknown text status",
    (p) => {
      p.text_status = "ok";
    },
    /text_status/,
  ],
  [
    "empty available summary",
    (p) => {
      p.summary = "";
    },
    /summary/,
  ],
  [
    "failed with references",
    (p) => {
      p.text_status = "failed";
      p.text_reason = "HTTP 403";
    },
    /cannot have references/,
  ],
  [
    "bad ref key",
    (p) => {
      p.references[0].extra = "x";
    },
    /expected keys/,
  ],
  [
    "duplicate ref",
    (p) => {
      p.references.push(structuredClone(p.references[0]));
    },
    /duplicate ref_id/,
  ],
  [
    "target status disagreement",
    (p) => {
      p.references[0].target_id = null;
    },
    /disagree/,
  ],
  [
    "unknown role",
    (p) => {
      p.references[0].roles = ["new"];
    },
    /roles/,
  ],
  [
    "duplicate role",
    (p) => {
      p.references[0].roles = ["comparison", "comparison"];
    },
    /duplicate/,
  ],
  [
    "no classified evidence",
    (p) => {
      p.references[0].evidence = [];
    },
    /requires roles/,
  ],
  [
    "empty reason",
    (p) => {
      p.references[0].reason = " ";
    },
    /reason/,
  ],
  [
    "unknown decision",
    (p) => {
      p.references[0].decision_status = "ok";
    },
    /decision_status/,
  ],
];
for (const [name, change, error] of invalidPapers)
  test(`reject malformed paper: ${name}`, () => {
    const copy = structuredClone(root);
    change(copy);
    assert.throws(() => parseCitationDocument(document(copy)), error);
  });

test("duplicate YAML keys, aliases and missing frontmatter are rejected", () => {
  const raw = document(root);
  assert.throws(
    () =>
      parseCitationDocument(
        raw.replace(
          "schema_version: 1",
          "schema_version: 1\nschema_version: 1",
        ),
      ),
    /invalid citation YAML/,
  );
  assert.throws(
    () =>
      parseCitationDocument(
        raw
          .replace("title: Fixture A", "title: &title Fixture A")
          .replace(
            "summary: Fictional A for testing citation traversal; no real research claims.",
            "summary: *title",
          ),
      ),
    /aliases/,
  );
  assert.throws(() => parseCitationDocument("title: Fixture A"), /frontmatter/);
  assert.throws(
    () =>
      parseCitationDocument(
        raw.replace("title: Fixture A", "title: !custom Fixture A"),
      ),
    /invalid citation YAML/,
  );
  assert.throws(
    () =>
      parseCitationDocument(
        raw.replace("schema_version: 1", "schema_version: 1.0"),
      ),
    /integer YAML scalar/,
  );
});

for (const [name, change] of [
  [
    "unknown field",
    (s: any) => {
      s.extra = 1;
    },
  ],
  [
    "invalid id",
    (s: any) => {
      s.id = "search-Invalid";
    },
  ],
  [
    "no timezone",
    (s: any) => {
      s.created_at = "2026-11-14T00:00:00";
    },
  ],
  [
    "invalid calendar date",
    (s: any) => {
      s.created_at = "2026-02-30T00:00:00Z";
    },
  ],
  [
    "negative depth",
    (s: any) => {
      s.max_depth = -1;
    },
  ],
  [
    "empty roles",
    (s: any) => {
      s.roles = [];
    },
  ],
  [
    "empty generations",
    (s: any) => {
      s.generations = [];
    },
  ],
  [
    "duplicate issue",
    (s: any) => {
      s.issues.push(structuredClone(s.issues[0]));
    },
  ],
] as const)
  test(`reject malformed search: ${name}`, () => {
    const copy = structuredClone(mixed);
    change(copy);
    assert.throws(() => parseCitationDocument(document(copy)));
  });

for (const [name, change, error] of [
  [
    "unknown target",
    (s: CitationSearch) => {
      s.root_id = "cite-0000000000000000";
    },
    /traversal|unknown/,
  ],
  [
    "skipped generation",
    (s: CitationSearch) => {
      s.generations[1].depth = 2;
    },
    /consecutive/,
  ],
  [
    "unexpanded source",
    (s: CitationSearch) => {
      s.generations[1].expanded_ids = [];
    },
    /expanded_ids/,
  ],
  [
    "extra target",
    (s: CitationSearch) => {
      s.generations[1].paper_ids.push(byTitle("D").id);
    },
    /traversal/,
  ],
  [
    "incomplete traversal",
    (s: CitationSearch) => {
      s.generations.pop();
    },
    /missing/,
  ],
  [
    "missing issue",
    (s: CitationSearch) => {
      s.issues.pop();
    },
    /issues disagree/,
  ],
] as const)
  test(`reject inconsistent search: ${name}`, () => {
    const copy = structuredClone(mixed);
    change(copy);
    assert.throws(
      () => validateCitationLibrary({ ...library, searches: [copy] }),
      error,
    );
  });

test("external IDs link registered papers, refusing conflicts and ambiguous or duplicate aliases", () => {
  const b = byTitle("B");
  assert.deepEqual(
    joinRegisteredPapers(library.papers, [
      { id: "paper-2024-b", externalIds: { arxiv: "2401.00002" } },
    ]),
    { [b.id]: "paper-2024-b" },
  );
  assert.deepEqual(
    joinRegisteredPapers(library.papers, [
      { id: "paper-2024-b", externalIds: {} },
    ]),
    {},
  );
  assert.throws(
    () =>
      joinRegisteredPapers(library.papers, [
        {
          id: "paper-2024-b",
          externalIds: { arxiv: "2401.00002", doi: "10.5555/conflict" },
        },
      ]),
    /conflicting external IDs/,
  );
  assert.throws(
    () =>
      joinRegisteredPapers(library.papers, [
        { id: "paper-2024-b", externalIds: { doi: "10.5555/fixture-b" } },
        { id: "paper-2024-b2", externalIds: { arxiv: "2401.00002" } },
      ]),
    /ambiguous/,
  );
  assert.throws(
    () =>
      joinRegisteredPapers(library.papers, [
        { id: "paper-2024-b", externalIds: b.external_ids },
        { id: "paper-2024-b2", externalIds: { arxiv: "2401.00002" } },
      ]),
    /duplicate literature/,
  );
  assert.throws(
    () =>
      joinRegisteredPapers(
        [...library.papers, { ...b, id: "cite-0000000000000000" }],
        [],
      ),
    /duplicate literature/,
  );
  // Disjoint citation aliases cannot silently merge through a registered entry.
  const aliasD = {
    ...byTitle("D"),
    external_ids: { url: "https://example.org/d" },
  };
  assert.throws(
    () =>
      joinRegisteredPapers(
        [root, aliasD],
        [
          {
            id: "paper-2024-bridge",
            externalIds: {
              doi: "10.5555/fixture-a",
              url: "https://example.org/d",
            },
          },
        ],
      ),
    /duplicate literature through Papers/,
  );
});

test("broken reference targets and unread expanded papers fail cross-document validation", () => {
  const broken = structuredClone(library);
  broken.papers.find((p) => p.id === root.id)!.references[0].target_id =
    "cite-0000000000000000";
  assert.throws(() => validateCitationLibrary(broken), /unknown target_id/);
  const unread = structuredClone(library);
  unread.papers.find((p) => p.id === byTitle("E").id)!.text_status =
    "not_requested";
  assert.throws(
    () => validateCitationLibrary(unread),
    /expanded paper.*not_requested/,
  );
});

test("absent storage is empty, while file/read/schema failures remain errors", async (t) => {
  const temp = await fs.mkdtemp(path.join(os.tmpdir(), "citation-load-"));
  t.after(() => fs.rm(temp, { recursive: true, force: true }));
  assert.equal(
    (await loadCitations(path.join(temp, "absent"))).searches.length,
    0,
  );
  const dir = path.join(temp, "citation_map");
  await fs.cp(fixture, dir, { recursive: true });
  assert.equal((await loadCitations(dir)).searches.length, 3);
  const file = path.join(dir, "papers", `${root.id}.md`);
  await fs.writeFile(file, "---\nbroken: true\n---\n");
  await assert.rejects(loadCitations(dir), /schema_version/);
  const state = await readCitations([], dir);
  assert.equal(state.ok, false);
  if (!state.ok) assert.match(state.error, /schema_version/);
  await fs.writeFile(file, Buffer.from([0xff, 0xfe]));
  await assert.rejects(loadCitations(dir), /encoded data/);
  await fs.rm(file);
  await fs.symlink("/nonexistent-citation", file);
  await assert.rejects(loadCitations(dir), /regular Markdown/);
  await fs.rm(file);
  await fs.writeFile(path.join(dir, "papers", "attached.pdf"), "pdf");
  await assert.rejects(loadCitations(dir), /regular Markdown/);
  const blocker = path.join(temp, "not-a-directory");
  await fs.writeFile(blocker, "text");
  await assert.rejects(loadCitations(path.join(blocker, "child")), /ENOTDIR/);
  const failedRead = await readCitations([], path.join(blocker, "child"));
  assert.equal(failedRead.ok, false);
  if (!failedRead.ok) assert.match(failedRead.error, /ENOTDIR/);
  const rootLink = path.join(temp, "dangling");
  await fs.symlink(path.join(temp, "missing"), rootLink);
  await assert.rejects(
    loadCitations(rootLink),
    /regular citation_map directory/,
  );
});
