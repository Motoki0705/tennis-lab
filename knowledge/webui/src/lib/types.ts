export type NodeType = "run" | "group";

export type Relation = { to: string; rel?: string };

export interface KnowledgeNode {
  id: string;
  type: NodeType;
  title: string;
  issue?: number | number[];
  provider?: string;
  date?: string;
  status?: string;
  config?: Record<string, unknown>;
  metrics?: Record<string, number | string>;
  artifacts?: Record<string, string>;
  parents: string[];
  members: string[];
  relations: Relation[];
  tags: string[];
  bodyHtml: string;
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
