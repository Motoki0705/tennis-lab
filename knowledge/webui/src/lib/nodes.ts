import "server-only";

import { promises as fs } from "node:fs";
import path from "node:path";

import matter from "gray-matter";
import { marked } from "marked";

import type {
  KnowledgeEdge,
  KnowledgeGraph,
  KnowledgeNode,
  NodeType,
  Relation,
  Source,
} from "./types";

const NODES_DIR =
  process.env.KNOWLEDGE_NODES_DIR ?? path.resolve(process.cwd(), "..", "nodes");
const RUNS_DIR =
  process.env.KNOWLEDGE_RUNS_DIR ?? path.resolve(process.cwd(), "..", "runs");

const NODE_TYPES = new Set<NodeType>(["run", "group", "paper", "proposal"]);

async function hasCurves(id: string): Promise<boolean> {
  try {
    await fs.access(path.join(RUNS_DIR, id, "curves.png"));
    return true;
  } catch {
    return false;
  }
}

function asNodeType(value: unknown, file: string): NodeType {
  const nodeType = String(value ?? "") as NodeType;
  if (!NODE_TYPES.has(nodeType)) {
    throw new Error(`${file}: unknown knowledge node type '${String(value ?? "")}'`);
  }
  return nodeType;
}

function asStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => String(item));
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return undefined;
  return value as Record<string, unknown>;
}

function asStringRecord(value: unknown): Record<string, string> | undefined {
  const record = asRecord(value);
  if (!record) return undefined;
  return Object.fromEntries(Object.entries(record).map(([key, item]) => [key, String(item)]));
}

function asExternalIds(value: unknown): Record<string, string | null> | undefined {
  const record = asRecord(value);
  if (!record) return undefined;
  return Object.fromEntries(
    Object.entries(record).map(([key, item]) => [key, item == null ? null : String(item)]),
  );
}

function asRelations(value: unknown): Relation[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null)
    .map((item) => ({
      to: String(item.to ?? ""),
      rel: item.rel ? String(item.rel) : undefined,
    }))
    .filter((relation) => relation.to);
}

function asSources(value: unknown): Source[] {
  if (!Array.isArray(value)) return [];
  return value.filter(
    (item): item is Source => typeof item === "object" && item !== null && !Array.isArray(item),
  );
}

async function parseFile(file: string): Promise<KnowledgeNode> {
  const raw = await fs.readFile(file, "utf-8");
  const { data, content } = matter(raw);
  const id = String(data.id ?? path.basename(file, ".md"));
  const type = asNodeType(data.type, path.basename(file));
  return {
    id,
    type,
    title: String(data.title ?? id),
    issue: data.issue as number | number[] | undefined,
    provider: data.provider ? String(data.provider) : undefined,
    curator: data.curator ? String(data.curator) : undefined,
    date: data.date ? String(data.date) : undefined,
    status: data.status ? String(data.status) : undefined,
    externalIds: asExternalIds(data.external_ids),
    publishedAt: data.published_at ? String(data.published_at) : undefined,
    reviewedAt: data.reviewed_at ? String(data.reviewed_at) : undefined,
    evidenceLevel: data.evidence_level ? String(data.evidence_level) : undefined,
    task: data.task ? String(data.task) : undefined,
    tasks: asStringArray(data.tasks),
    repoPaths: asStringArray(data.repo_paths),
    sources: asSources(data.sources),
    hypothesis: asRecord(data.hypothesis),
    evaluation: asRecord(data.evaluation),
    evidenceRuns: asStringArray(data.evidence_runs),
    config: asRecord(data.config),
    metrics: data.metrics as Record<string, number | string> | undefined,
    artifacts: asStringRecord(data.artifacts),
    parents: asStringArray(data.parents),
    members: asStringArray(data.members),
    relations: asRelations(data.relations),
    tags: asStringArray(data.tags),
    bodyHtml: marked.parse(content.trim(), { async: false }) as string,
    curvesUrl:
      type === "run" && (await hasCurves(id)) ? `/api/curves/${id}` : undefined,
  };
}

function buildEdges(nodes: KnowledgeNode[]): KnowledgeEdge[] {
  const known = new Set(nodes.map((node) => node.id));
  const edges: KnowledgeEdge[] = [];
  for (const node of nodes) {
    for (const parent of node.parents) {
      if (known.has(parent)) {
        edges.push({
          id: `${parent}->${node.id}`,
          source: parent,
          target: node.id,
          kind: "parent",
        });
      }
    }
    if (node.type === "group") {
      for (const member of node.members) {
        if (known.has(member)) {
          edges.push({
            id: `${node.id}~${member}`,
            source: node.id,
            target: member,
            kind: "member",
          });
        }
      }
    }
    for (const relation of node.relations) {
      if (known.has(relation.to)) {
        edges.push({
          id: `${node.id}=>${relation.to}`,
          source: node.id,
          target: relation.to,
          kind: "relation",
          label: relation.rel,
        });
      }
    }
  }
  return edges;
}

export async function getGraph(): Promise<KnowledgeGraph> {
  let entries: string[] = [];
  try {
    entries = (await fs.readdir(NODES_DIR)).filter((file) => file.endsWith(".md"));
  } catch {
    return { nodes: [], edges: [] };
  }
  const nodes = await Promise.all(
    entries.sort().map((file) => parseFile(path.join(NODES_DIR, file))),
  );
  return { nodes, edges: buildEdges(nodes) };
}
