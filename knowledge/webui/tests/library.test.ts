import assert from "node:assert/strict";
import test from "node:test";
import path from "node:path";
import { getGraph } from "../src/lib/nodes";
import { renderMarkdown } from "../src/lib/content";
import { EMPTY_FILTERS, filterNodes, metricsCsv } from "../src/lib/explorer";

test("real library loads task nodes, stable ids, papers and linked summary", async () => {
  const graph = await getGraph();
  assert.ok(graph.nodes.length >= 200);
  assert.ok(graph.nodes.filter((n) => n.type === "run").length >= 173);
  assert.ok(new Set(graph.nodes.map((n) => n.task)).size >= 7);
  assert.equal(new Set(graph.nodes.map((n) => n.id)).size, graph.nodes.length);
  assert.ok(
    graph.summaryHtml.includes(
      "/?node=run-plcs-accad-gvhmr-meiji-1000-v1-train",
    ),
  );
  const paper = graph.papers.find((p) => p.id === "paper-2024-gvhmr")!;
  assert.equal(paper.pdfUrl, "/api/papers/paper-2024-gvhmr");
  assert.ok(graph.nodes.filter((n) => n.papers.includes(paper.id)).length >= 3);
  const node = graph.nodes.find(
    (n) => n.id === "run-plcs-accad-gvhmr-meiji-1000-v1-train",
  )!;
  assert.equal(node.date, "2026-09-15");
});

test("filters combine task, query and paper without mixing task sequences", async () => {
  const graph = await getGraph();
  const filtered = filterNodes(graph.nodes, {
    ...EMPTY_FILTERS,
    task: "plcs",
    query: "GVHMR",
    paper: "paper-2024-gvhmr",
    order: "oldest",
  });
  assert.ok(filtered.length >= 3);
  assert.ok(filtered.every((n) => n.task === "plcs"));
  assert.equal(filtered[0].date, "2026-09-14");
  const future = {
    ...filtered[0],
    id: "run-future",
    sequence: 500,
    date: "2020-01-01",
  };
  assert.equal(
    filterNodes([future, ...filtered], { ...EMPTY_FILTERS, order: "oldest" })[0]
      .id,
    future.id,
  );
  assert.equal(
    filterNodes(graph.nodes, { ...EMPTY_FILTERS, task: "missing" }).length,
    0,
  );
});

test("markdown strips scripts and javascript URLs while resolving local graph links", () => {
  const file = path.resolve("../summary.md");
  const target = path.resolve("../nodes/plcs/000001-run-test.md");
  const html = renderMarkdown(
    '<script>alert(1)</script>\n\n[bad](javascript:alert) [good](nodes/plcs/000001-run-test.md) <img src="x" onerror="alert(1)">',
    file,
    new Map([[target, "/?node=run-test"]]),
  );
  assert.ok(
    !html.includes("<script") &&
      !html.includes("javascript:") &&
      !html.includes("onerror"),
  );
  assert.ok(html.includes('href="/?node=run-test"'));
});

test("CSV retains metric union and escapes formula injection", async () => {
  const graph = await getGraph();
  const node = { ...graph.nodes[0], metrics: { x: "=1+1", y: 'a,"b' } };
  const csv = metricsCsv([node]);
  assert.ok(csv.includes('"\'=1+1"'));
  assert.ok(csv.includes('"a,""b"'));
});
