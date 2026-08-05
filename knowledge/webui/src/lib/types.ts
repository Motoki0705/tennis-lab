export type NodeType = "run" | "group" | "paper" | "proposal";

export type Relation = { to: string; rel?: string };
export type Source = { kind?: string; url?: string; [key: string]: unknown };

export interface KnowledgeNode {
  id: string;
  type: NodeType;
  title: string;
  issue?: number | number[];
  provider?: string;
  curator?: string;
  date?: string;
  status?: string;
  externalIds?: Record<string, string | null>;
  publishedAt?: string;
  reviewedAt?: string;
  evidenceLevel?: string;
  task?: string;
  tasks: string[];
  repoPaths: string[];
  sources: Source[];
  hypothesis?: Record<string, unknown>;
  evaluation?: Record<string, unknown>;
  evidenceRuns: string[];
  config?: Record<string, unknown>;
  metrics?: Record<string, number | string>;
  artifacts?: Record<string, string>;
  parents: string[];
  members: string[];
  relations: Relation[];
  tags: string[];
  bodyHtml: string;
  /** Set only for run nodes with knowledge/runs/<id>/curves.png. */
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
}
