import "server-only";

import { promises as fs } from "node:fs";
import path from "node:path";

import matter from "gray-matter";
import { marked } from "marked";

import type {
  KnowledgeEdge,
  KnowledgeGraph,
  KnowledgeNode,
  Relation,
} from "./types";

// knowledge/nodes lives one directory above this Next.js app (knowledge/webui).
const NODES_DIR =
  process.env.KNOWLEDGE_NODES_DIR ?? path.resolve(process.cwd(), "..", "nodes");

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((v) => String(v));
}

function asRelations(value: unknown): Relation[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((v): v is Record<string, unknown> => typeof v === "object" && v !== null)
    .map((v) => ({ to: String(v.to ?? ""), rel: v.rel ? String(v.rel) : undefined }))
    .filter((r) => r.to);
}

async function parseFile(file: string): Promise<KnowledgeNode> {
  const raw = await fs.readFile(file, "utf-8");
  const { data, content } = matter(raw);
  const id = String(data.id ?? path.basename(file, ".md"));
  const type = data.type === "group" ? "group" : "run";
  return {
    id,
    type,
    title: String(data.title ?? id),
    issue: data.issue as number | number[] | undefined,
    provider: data.provider ? String(data.provider) : undefined,
    date: data.date ? String(data.date) : undefined,
    status: data.status ? String(data.status) : undefined,
    config: (data.config as Record<string, unknown>) ?? undefined,
    metrics: (data.metrics as Record<string, number | string>) ?? undefined,
    artifacts: (data.artifacts as Record<string, string>) ?? undefined,
    parents: asStringArray(data.parents),
    members: asStringArray(data.members),
    relations: asRelations(data.relations),
    tags: asStringArray(data.tags),
    bodyHtml: marked.parse(content.trim(), { async: false }) as string,
  };
}

function buildEdges(nodes: KnowledgeNode[]): KnowledgeEdge[] {
  const known = new Set(nodes.map((n) => n.id));
  const edges: KnowledgeEdge[] = [];
  for (const n of nodes) {
    for (const p of n.parents) {
      if (known.has(p)) {
        edges.push({ id: `${p}->${n.id}`, source: p, target: n.id, kind: "parent" });
      }
    }
    if (n.type === "group") {
      for (const m of n.members) {
        if (known.has(m)) {
          edges.push({ id: `${n.id}~${m}`, source: n.id, target: m, kind: "member" });
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

export async function getGraph(): Promise<KnowledgeGraph> {
  let entries: string[] = [];
  try {
    entries = (await fs.readdir(NODES_DIR)).filter((f) => f.endsWith(".md"));
  } catch {
    return { nodes: [], edges: [] };
  }
  const nodes = await Promise.all(
    entries.sort().map((f) => parseFile(path.join(NODES_DIR, f))),
  );
  return { nodes, edges: buildEdges(nodes) };
}
