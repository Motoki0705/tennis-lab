import type {
  CitationLibrary,
  CitationPaper,
  CitationReference,
  CitationRole,
  CitationSearch,
} from "./citation-types";

export interface ScopedReference {
  source: CitationPaper;
  target: CitationPaper | null;
  reference: CitationReference;
}

/** Only expanded sources belong to a search, even if other papers are cached. */
export function searchReferences(
  library: CitationLibrary,
  search: CitationSearch,
): ScopedReference[] {
  const byId = new Map(library.papers.map((p) => [p.id, p]));
  const expanded = new Set(search.generations.flatMap((g) => g.expanded_ids));
  return [...expanded].flatMap((id) => {
    const source = byId.get(id)!;
    return source.references.map((reference) => ({
      source,
      target:
        reference.target_id === null ? null : byId.get(reference.target_id)!,
      reference,
    }));
  });
}

export interface CitationRelation {
  id: string;
  source: CitationPaper;
  target: CitationPaper | null;
  roles: CitationRole[];
  references: CitationReference[];
}

/** Merge repeated bibliography rows for drawing, retaining every evidence row. */
export function citationRelations(
  rows: ScopedReference[],
  role: CitationRole | "",
): CitationRelation[] {
  const groups = new Map<string, CitationRelation>();
  for (const { source, target, reference } of rows) {
    if (role && !reference.roles.includes(role)) continue;
    const id = `${source.id}/${target?.id ?? reference.ref_id}`;
    let group = groups.get(id);
    if (!group) {
      group = { id, source, target, roles: [], references: [] };
      groups.set(id, group);
    }
    group.references.push(reference);
    group.roles = [...new Set([...group.roles, ...reference.roles])];
  }
  return [...groups.values()];
}

export interface CitationRank {
  paper: CitationPaper;
  count: number;
  sources: { paper: CitationPaper; references: CitationReference[] }[];
}

/** One count per distinct source/target/role, never per ref row or search run. */
export function rankCitations(
  rows: ScopedReference[],
  role: CitationRole,
): CitationRank[] {
  const targets = new Map<string, CitationRank>();
  for (const { source, target, reference } of rows) {
    if (!target || !reference.roles.includes(role)) continue;
    let rank = targets.get(target.id);
    if (!rank) {
      rank = { paper: target, count: 0, sources: [] };
      targets.set(target.id, rank);
    }
    let citing = rank.sources.find((s) => s.paper.id === source.id);
    if (!citing) {
      citing = { paper: source, references: [] };
      rank.sources.push(citing);
      rank.count += 1;
    }
    citing.references.push(reference);
  }
  return [...targets.values()].sort(
    (a, b) => b.count - a.count || a.paper.title.localeCompare(b.paper.title),
  );
}
