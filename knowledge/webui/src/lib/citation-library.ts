import { promises as fs } from "node:fs";
import path from "node:path";
import { KNOWLEDGE_DIR } from "./content";
import {
  externalIds,
  joinRegisteredPapers,
  parseCitationDocument,
  validateCitationPaperReferences,
  validateCitationSearch,
} from "./citation-documents";
import type {
  CitationDocumentError,
  CitationLoadResult,
  CitationPaper,
  CitationReadLibrary,
  CitationSearch,
} from "./citation-types";
import type { Paper } from "./types";

type Kind = "papers" | "searches";
type Source = {
  id: string;
  file: string;
  kind: Kind;
  doc?: CitationPaper | CitationSearch;
  errors: Set<CitationDocumentError>;
};
type Identity = {
  id: string;
  file: string;
  area: "citation" | "registered";
  ids: Record<string, string>;
  source?: Source;
};
const message = (error: unknown) =>
  error instanceof Error ? error.message : String(error);

/** Include every participant, including aliases joined indirectly via Papers. */
function identityGroups(entries: Identity[]): Identity[][] {
  const parents = entries.map((_, i) => i);
  function root(i: number): number {
    if (parents[i] !== i) parents[i] = root(parents[i]);
    return parents[i];
  }
  const aliases = new Map<string, number>();
  for (const [i, entry] of entries.entries()) {
    const keys = [
      JSON.stringify([entry.area, entry.id]),
      ...Object.entries(entry.ids).map(([ns, value]) =>
        JSON.stringify(["external", ns, value]),
      ),
    ];
    for (const key of keys) {
      const previous = aliases.get(key);
      if (previous !== undefined) parents[root(i)] = root(previous);
      aliases.set(key, i);
    }
  }
  const groups = new Map<number, Identity[]>();
  for (const [i, entry] of entries.entries()) {
    const group = groups.get(root(i)) ?? [];
    group.push(entry);
    groups.set(root(i), group);
  }
  return [...groups.values()];
}

/** Strict documents, with browsing isolated to each search's dependencies. */
export async function loadCitations(
  directory = path.join(KNOWLEDGE_DIR, "citation_map"),
  registered: Pick<Paper, "id" | "externalIds">[] = [],
): Promise<CitationReadLibrary> {
  const library: CitationReadLibrary = {
    papers: [],
    searches: [],
    registeredIds: {},
    documentErrors: [],
    searchErrors: [],
  };
  const sources: Source[] = [];
  const errors = new Map<string, CitationDocumentError>();
  function report(files: string[], reason: string, affected: Source[] = []) {
    const sorted = [...new Set(files)].sort();
    const key = JSON.stringify([sorted, reason]);
    let error = errors.get(key);
    if (!error) {
      error = { files: sorted, reason };
      errors.set(key, error);
      library.documentErrors.push(error);
    }
    for (const source of affected) source.errors.add(error);
    return error;
  }
  const stat = await fs
    .lstat(directory)
    .catch((error: NodeJS.ErrnoException) => {
      if (error.code === "ENOENT") return null;
      throw error;
    });
  if (!stat) return library;
  if (!stat.isDirectory() || stat.isSymbolicLink())
    throw new Error(`${directory}: expected regular citation_map directory`);
  const folders = await fs.readdir(directory, { withFileTypes: true });
  for (const entry of folders.sort((a, b) => a.name.localeCompare(b.name))) {
    const folder = path.join(directory, entry.name);
    if (entry.name === "README.md" && entry.isFile()) continue;
    if (
      !["papers", "searches"].includes(entry.name) ||
      !entry.isDirectory() ||
      entry.isSymbolicLink()
    )
      throw new Error(`${folder}: expected papers/ or searches/ directory`);
    const kind = entry.name as Kind;
    const items = await fs.readdir(folder, { withFileTypes: true });
    for (const item of items.sort((a, b) => a.name.localeCompare(b.name))) {
      const file = path.join(folder, item.name);
      const pattern =
        kind === "papers"
          ? /^cite-[0-9a-f]{16}\.md$/
          : /^search-[a-z0-9]+(?:-[a-z0-9]+)*\.md$/;
      // Without a trustworthy filename ID, an invalid file's impact is unknown.
      if (pattern.exec(item.name)?.[0] !== item.name)
        throw new Error(
          `${file}: only regular Markdown citation documents with a canonical filename are allowed; cannot determine affected searches`,
        );
      const source: Source = {
        id: item.name.slice(0, -3),
        file: `citation_map/${kind}/${item.name}`,
        kind,
        errors: new Set(),
      };
      sources.push(source);
      try {
        if (!item.isFile() || item.isSymbolicLink())
          throw new Error(
            "only regular Markdown citation documents are allowed",
          );
        source.doc = parseCitationDocument(
          new TextDecoder("utf-8", { fatal: true }).decode(
            await fs.readFile(file),
          ),
        );
      } catch (error) {
        report([source.file], message(error), [source]);
      }
    }
  }
  const byPath = new Map(sources.map((s) => [`${s.kind}/${s.id}`, s]));
  for (const source of sources) {
    if (!source.doc) continue;
    const kind = source.doc.type === "citation_paper" ? "papers" : "searches";
    if (source.id !== source.doc.id || source.kind !== kind) {
      const other = byPath.get(`${kind}/${source.doc.id}`);
      const affected = other ? [source, other] : [source];
      report(
        affected.map((s) => s.file),
        "document type/id must match path",
        affected,
      );
    }
  }
  const paperSources = sources.filter((s) => s.kind === "papers");
  const paperIds = new Set(paperSources.map((s) => s.id));
  for (const source of paperSources) {
    if (source.errors.size || source.doc?.type !== "citation_paper") continue;
    try {
      // Known but invalid targets are checked against each search's scope below.
      // Do not propagate their errors through unexpanded cached references.
      validateCitationPaperReferences(source.doc, paperIds);
    } catch (error) {
      report([source.file], message(error), [source]);
    }
  }

  // A different validation error must not hide a known identity collision.
  const identities: Identity[] = sources
    .filter((s) => s.doc?.type === "citation_paper")
    .map((s) => ({
      id: s.id,
      file: s.file,
      area: "citation",
      source: s,
      ids: (s.doc as CitationPaper).external_ids,
    }));
  for (const paper of registered) {
    if (/^paper-[a-z0-9]+(?:-[a-z0-9]+)*$/.exec(paper.id)?.[0] !== paper.id)
      throw new Error(
        `cannot attribute invalid Papers document id: ${paper.id}`,
      );
    const file = `Papers/${paper.id}/paper.md`;
    try {
      identities.push({
        id: paper.id,
        file,
        area: "registered",
        ids: externalIds(paper.externalIds, true),
      });
    } catch (error) {
      report([file], message(error));
    }
  }
  const validRegistered: Pick<Paper, "id" | "externalIds">[] = [];
  for (const group of identityGroups(identities)) {
    const namespaces = new Map<string, Set<string>>();
    for (const entry of group)
      for (const [ns, value] of Object.entries(entry.ids)) {
        const values = namespaces.get(ns) ?? new Set<string>();
        values.add(value);
        namespaces.set(ns, values);
      }
    const conflicts = [...namespaces]
      .filter(([, values]) => values.size > 1)
      .map(([ns]) => ns);
    const duplicate = ["citation", "registered"].some(
      (area) => group.filter((e) => e.area === area).length > 1,
    );
    if (conflicts.length || duplicate) {
      const reason = [
        ...(conflicts.length
          ? [`conflicting external IDs: ${conflicts.sort().join(", ")}`]
          : []),
        ...(duplicate ? ["ambiguous aliases / duplicate literature"] : []),
      ].join("; ");
      report(
        group.map((e) => e.file),
        reason,
        group.flatMap((e) => (e.source ? [e.source] : [])),
      );
    } else {
      for (const entry of group)
        if (entry.area === "registered")
          validRegistered.push({ id: entry.id, externalIds: entry.ids });
    }
  }
  library.papers = paperSources.flatMap((s) =>
    !s.errors.size && s.doc?.type === "citation_paper" ? [s.doc] : [],
  );
  // Recheck the derived join after every invalid participant has been removed.
  // An unexpected, unattributable join failure must still stop the whole reader.
  library.registeredIds = joinRegisteredPapers(library.papers, validRegistered);
  const papers = new Map(library.papers.map((p) => [p.id, p]));
  for (const source of sources.filter((s) => s.kind === "searches")) {
    if (!source.errors.size && source.doc?.type === "citation_search") {
      const search = source.doc;
      const dependencies = new Set([
        search.root_id,
        ...search.generations.flatMap((g) => [
          ...g.paper_ids,
          ...g.expanded_ids,
        ]),
      ]);
      for (const id of search.generations.flatMap((g) => g.expanded_ids)) {
        const doc = byPath.get(`papers/${id}`)?.doc;
        if (doc?.type === "citation_paper")
          for (const ref of doc.references)
            if (ref.target_id) dependencies.add(ref.target_id);
      }
      for (const id of dependencies) {
        const paper = byPath.get(`papers/${id}`);
        if (!paper)
          source.errors.add(
            report(
              [`citation_map/papers/${id}.md`],
              `missing citation paper ${id}`,
            ),
          );
        else for (const error of paper.errors) source.errors.add(error);
      }
      if (!source.errors.size) {
        try {
          validateCitationSearch(search, papers);
        } catch (error) {
          report([source.file], message(error), [source]);
        }
      }
      if (!source.errors.size) library.searches.push(search);
    }
    if (source.errors.size)
      library.searchErrors.push({
        id: source.id,
        file: source.file,
        errors: [...source.errors],
      });
  }
  library.searches.sort(
    (a, b) =>
      b.created_at.localeCompare(a.created_at) || a.id.localeCompare(b.id),
  );
  return library;
}

export async function readCitations(
  registered: Pick<Paper, "id" | "externalIds">[],
  directory?: string,
): Promise<CitationLoadResult> {
  try {
    return { ok: true, library: await loadCitations(directory, registered) };
  } catch (error) {
    return { ok: false, error: message(error) };
  }
}
