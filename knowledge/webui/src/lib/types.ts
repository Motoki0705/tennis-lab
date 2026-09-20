export type NodeType = "run" | "group";
export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

export type Relation = { to: string; rel?: string };

export interface KnowledgeNode {
  id: string;
  type: NodeType;
  title: string;
  task: string;
  sequence: number;
  recordedAt: string;
  dateSource?: string;
  papers: string[];
  searchText: string;
  file: string;
  issue?: number | number[];
  provider?: string;
  date?: string;
  status?: string;
  config?: Record<string, JsonValue>;
  metrics?: Record<string, JsonValue>;
  artifacts?: Record<string, JsonValue>;
  parents: string[];
  members: string[];
  relations: Relation[];
  tags: string[];
  bodyHtml: string;
  /** Set when knowledge/runs/<id>/curves.png exists; served via /api/curves/<id>. */
  curvesUrl?: string;
}

export type EdgeKind = "parent" | "member" | "relation";

export interface KnowledgeEdge {
  id: string;
  source: string;
  target: string;
  kind: EdgeKind;
  label?: string;
}

export interface KnowledgeGraph {
  nodes: KnowledgeNode[];
  edges: KnowledgeEdge[];
  papers: Paper[];
  summaryHtml: string;
}

export interface Paper {
  id: string;
  title: string;
  year: number;
  authors: string[];
  tasks: string[];
  source: string;
  license: string;
  pdfUrl: string;
  bodyHtml: string;
}
