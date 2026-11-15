export const ROLE_LABELS = {
  foundation: "基盤アーキテクチャ・モデル",
  component: "個別技術・構成要素",
  comparison: "比較対象",
  related: "直接関連する研究・機能",
  benchmark: "ベンチマーク・データセット",
} as const;
export type CitationRole = keyof typeof ROLE_LABELS;
export const CITATION_ROLES = Object.keys(ROLE_LABELS) as CitationRole[];
export const TEXT_LABELS = {
  available: "取得済み",
  failed: "本文取得失敗",
  not_requested: "未展開・未読",
} as const;
export const DECISION_LABELS = {
  classified: "判定済み",
  unclassifiable: "役割判定不能",
  pending: "役割未判定",
} as const;
export const IDENTIFICATION_LABELS = {
  resolved: "同定済み",
  unresolved: "未同定",
  pending: "照会待ち",
  unverified: "一意性未確認",
} as const;
export const SEARCH_LABELS = {
  complete: "探索完了",
  interrupted: "探索中断",
} as const;

export interface ReferenceLookup {
  title: string;
  year: number | null;
  authors: string[];
  links: string[];
}

export interface CitationReference {
  ref_id: string;
  citation: string;
  target_id: string | null;
  identification_status: keyof typeof IDENTIFICATION_LABELS;
  roles: CitationRole[];
  decision_status: keyof typeof DECISION_LABELS;
  reason: string;
  evidence: string[];
  lookup?: ReferenceLookup | null;
}
export interface CitationPaper {
  schema_version: 1;
  type: "citation_paper";
  id: string;
  identity_key: string;
  title: string;
  year: number | null;
  authors: string[];
  external_ids: Record<string, string>;
  source: string;
  summary: string;
  text_status: keyof typeof TEXT_LABELS;
  text_reason: string;
  references: CitationReference[];
}
export interface CitationSearch {
  schema_version: 1;
  type: "citation_search";
  id: string;
  created_at: string;
  root_id: string;
  roles: CitationRole[];
  max_depth: number;
  generations: { depth: number; paper_ids: string[]; expanded_ids: string[] }[];
  issues: { paper_id: string; ref_id: string | null }[];
  status: keyof typeof SEARCH_LABELS;
  stop_reason: string;
}
export interface CitationLibrary {
  papers: CitationPaper[];
  searches: CitationSearch[];
  /** Derived from external IDs; never stored in the source documents. */
  registeredIds: Record<string, string>;
}
export type CitationLoadResult =
  | { ok: true; library: CitationLibrary }
  | { ok: false; error: string };
