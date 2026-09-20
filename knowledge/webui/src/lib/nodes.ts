import { promises as fs } from "node:fs";
import path from "node:path";

import matter from "gray-matter";
import {
  KNOWLEDGE_DIR,
  NODES_DIR,
  RUNS_DIR,
  PAPERS_DIR,
  markdownFiles,
  renderMarkdown,
} from "./content";
import type { Paper, JsonValue } from "./types";

import type {
  KnowledgeEdge,
  KnowledgeGraph,
  KnowledgeNode,
  Relation,
} from "./types";

async function hasCurves(id: string): Promise<boolean> {
  try {
    await fs.access(path.join(RUNS_DIR, id, "curves.png"));
    return true;
  } catch {
    return false;
  }
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((v) => String(v));
}

function asRelations(value: unknown): Relation[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter(
      (v): v is Record<string, unknown> => typeof v === "object" && v !== null,
    )
    .map((v) => ({
      to: String(v.to ?? ""),
      rel: v.rel ? String(v.rel) : undefined,
    }))
    .filter((r) => r.to);
}

async function parseFile(
  file: string,
  targets: Map<string, string>,
): Promise<KnowledgeNode> {
  const raw = await fs.readFile(file, "utf-8");
  const { data, content } = matter(raw);
  const id = String(data.id ?? path.basename(file, ".md"));
  const type = data.type === "group" ? "group" : "run";
  return {
    id,
    type,
    task: String(data.task),
    sequence: Number(data.sequence),
    recordedAt: isoDay(data.recorded_at),
    dateSource: data.date_source,
    papers: asStringArray(data.papers),
    searchText: content.toLowerCase(),
    file: path.relative(KNOWLEDGE_DIR, file),
    title: String(data.title ?? id),
    issue: data.issue as number | number[] | undefined,
    provider: data.provider ? String(data.provider) : undefined,
    date: data.date ? isoDay(data.date) : undefined,
    status: data.status ? String(data.status) : undefined,
    config: (data.config as Record<string, JsonValue>) ?? undefined,
    metrics: (data.metrics as Record<string, JsonValue>) ?? undefined,
    artifacts: (data.artifacts as Record<string, JsonValue>) ?? undefined,
    parents: asStringArray(data.parents),
    members: asStringArray(data.members),
    relations: asRelations(data.relations),
    tags: asStringArray(data.tags),
    bodyHtml: renderMarkdown(content.trim(), file, targets),
    curvesUrl: (await hasCurves(id)) ? `/api/curves/${id}` : undefined,
  };
}

function buildEdges(nodes: KnowledgeNode[]): KnowledgeEdge[] {
  const known = new Set(nodes.map((n) => n.id));
  const edges: KnowledgeEdge[] = [];
  for (const n of nodes) {
    for (const p of n.parents) {
      if (known.has(p)) {
        edges.push({
          id: `${p}->${n.id}`,
          source: p,
          target: n.id,
          kind: "parent",
        });
      }
    }
    if (n.type === "group") {
      for (const m of n.members) {
        if (known.has(m)) {
          edges.push({
            id: `${n.id}~${m}`,
            source: n.id,
            target: m,
            kind: "member",
          });
        }
      }
    }
    for (const r of n.relations) {
      if (known.has(r.to)) {
        edges.push({
          id: `${n.id}=>${r.to}`,
          source: n.id,
          target: r.to,
          kind: "relation",
          label: r.rel,
        });
      }
    }
  }
  return edges;
}

function isoDay(value: unknown): string {
  return value instanceof Date
    ? value.toISOString().slice(0, 10)
    : String(value ?? "");
}

export async function getGraph(): Promise<KnowledgeGraph> {
  const entries = await markdownFiles(NODES_DIR);
  const targets = new Map<string, string>();
  const identities = new Set<string>();
  const sequences = new Set<string>();
  for (const file of entries) {
    const { data } = matter(await fs.readFile(file, "utf8"));
    const seqKey = `${data.task}/${data.sequence}`;
    if (
      !/^(run|group)-[a-z0-9-]+$/.test(data.id) ||
      !/^[a-z][a-z0-9_]*$/.test(data.task) ||
      !Number.isInteger(data.sequence) ||
      data.sequence < 1 ||
      data.sequence > 999999 ||
      path.basename(file) !==
        `${String(data.sequence).padStart(6, "0")}-${data.id}.md` ||
      path.basename(path.dirname(file)) !== data.task ||
      identities.has(data.id) ||
      sequences.has(seqKey)
    ) {
      throw new Error(
        `Invalid node identity or duplicate sequence: ${file}. Run kg_validate.py.`,
      );
    }
    identities.add(data.id);
    sequences.add(seqKey);
    targets.set(path.resolve(file), `/?node=${encodeURIComponent(data.id)}`);
  }
  const paperFiles = (await markdownFiles(PAPERS_DIR)).filter(
    (f) => path.basename(f) === "paper.md",
  );
  for (const file of paperFiles) {
    const { data } = matter(await fs.readFile(file, "utf8"));
    targets.set(path.resolve(file), `/?paper=${encodeURIComponent(data.id)}`);
    targets.set(
      path.resolve(path.dirname(file), "paper.pdf"),
      `/api/papers/${data.id}`,
    );
  }
  const papers: Paper[] = await Promise.all(
    paperFiles.map(async (file) => {
      const { data, content } = matter(await fs.readFile(file, "utf8"));
      return {
        id: data.id,
        title: data.title,
        year: data.year,
        authors: asStringArray(data.authors),
        tasks: asStringArray(data.tasks),
        source: data.source,
        license: data.license,
        pdfUrl: `/api/papers/${data.id}`,
        bodyHtml: renderMarkdown(content, file, targets),
      };
    }),
  );
  const nodes = await Promise.all(
    entries.map((file) => parseFile(file, targets)),
  );
  nodes.sort((a, b) => a.task.localeCompare(b.task) || a.sequence - b.sequence);
  const summaryFile = path.join(KNOWLEDGE_DIR, "summary.md");
  const summaryHtml = renderMarkdown(
    await fs.readFile(summaryFile, "utf8"),
    summaryFile,
    targets,
  );
  return { nodes, edges: buildEdges(nodes), papers, summaryHtml };
}
