import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import { isIP } from "node:net";
import path from "node:path";
import { isAlias, isScalar, parseDocument, visit } from "yaml";
import { KNOWLEDGE_DIR } from "./content";
import { CITATION_ROLES } from "./citation-types";
import type {
  CitationLibrary,
  CitationLoadResult,
  CitationPaper,
  CitationReference,
  CitationSearch,
} from "./citation-types";
import type { Paper } from "./types";

function requireValue(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}
function mapping(value: unknown, keys?: string[]): Record<string, unknown> {
  requireValue(
    value !== null &&
      typeof value === "object" &&
      !Array.isArray(value) &&
      !(value instanceof Date),
    "expected a mapping",
  );
  const data = value as Record<string, unknown>;
  if (keys)
    requireValue(
      Object.keys(data).length === keys.length &&
        keys.every((key) => Object.hasOwn(data, key)),
      `expected keys: ${keys.join(", ")}`,
    );
  return data;
}
function text(value: unknown, name: string, empty = false): string {
  requireValue(
    typeof value === "string" && (empty || value.trim()),
    `${name}: expected ${empty ? "a string" : "nonempty text"}`,
  );
  return value;
}
function list(value: unknown, name: string): unknown[] {
  requireValue(Array.isArray(value), `${name}: expected a list`);
  return value;
}
function strings(value: unknown, name: string, empty = true): string[] {
  const values = list(value, name).map((x) => text(x, name));
  requireValue(
    (empty || values.length) && new Set(values).size === values.length,
    `${name}: empty or duplicate entries`,
  );
  return values;
}
function choice<T extends string>(
  value: unknown,
  choices: readonly T[],
  name: string,
): T {
  requireValue(
    typeof value === "string" && choices.includes(value as T),
    `${name}: expected ${choices.join(" / ")}`,
  );
  return value as T;
}
function roles(value: unknown, empty = true) {
  return strings(value, "roles", empty).map((v) =>
    choice(v, CITATION_ROLES, "roles"),
  );
}
function integer(value: unknown, name: string): number {
  requireValue(
    Number.isSafeInteger(value) && (value as number) >= 0,
    `${name}: expected a nonnegative integer`,
  );
  return value as number;
}
function paperId(value: unknown): string {
  requireValue(
    typeof value === "string" && /^cite-[0-9a-f]{16}$/.test(value),
    "invalid citation paper id",
  );
  return value;
}
function refId(value: unknown): string {
  requireValue(
    typeof value === "string" && /^ref-[0-9]+$/.test(value),
    "invalid ref_id",
  );
  return value;
}
function httpUrl(value: unknown): string {
  const source = text(value, "source");
  const parts = /^(https?):\/\/([^/?#]+)([^?#]*)(\?[^#]*)?(?:#.*)?$/i.exec(
    source,
  );
  requireValue(
    parts &&
      !/[\x00-\x20\x7f-\x9f\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff\\]/.test(
        source,
      ),
    "expected absolute HTTP(S) URL without repairs, credentials or whitespace",
  );
  const authority = parts[2];
  let hostPort: RegExpExecArray | null;
  if (authority.startsWith("[")) {
    hostPort = /^\[([0-9a-fA-F:.]+)\](?::([0-9]{1,5}))?$/.exec(authority);
    requireValue(hostPort && isIP(hostPort[1]) === 6, "invalid IPv6 URL host");
  } else {
    hostPort = /^([a-zA-Z0-9.-]+)(?::([0-9]{1,5}))?$/.exec(authority);
    requireValue(hostPort, "invalid URL authority");
    const host = hostPort[1];
    const dnsName = host.replace(/\.$/, "");
    const labels = dnsName.split(".");
    requireValue(
      dnsName.length <= 253 &&
        labels.every((label) =>
          /^[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$/.test(label),
        ),
      "invalid DNS URL host",
    );
    if (/^(?:[0-9]+|0[xX][0-9a-fA-F]+)$/.test(labels[labels.length - 1])) {
      requireValue(isIP(host) === 4, "expected standard IPv4 URL host");
    }
  }
  requireValue(
    hostPort[2] === undefined || Number(hostPort[2]) <= 65535,
    "invalid URL port",
  );
  return source;
}

// Match Python's str.strip()/Unicode whitespace; a BOM is not whitespace there.
const ID_SPACE =
  "[\\u0009-\\u000d\\u001c-\\u0020\\u0085\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]";

export function normalizeIdentifier(namespace: string, value: string): string {
  const v = text(value, "external_ids").replace(
    new RegExp(`^${ID_SPACE}+|${ID_SPACE}+$`, "g"),
    "",
  );
  let normalized: string;
  if (namespace === "doi") {
    normalized = v
      .replace(/^https?:\/\/(?:dx\.)?doi\.org\//i, "")
      .replace(new RegExp(`^doi:${ID_SPACE}*`, "i"), "")
      .replace(/(?:%[0-9a-f]{2})+/gi, (part) =>
        Buffer.from(part.replaceAll("%", ""), "hex").toString("utf8"),
      )
      .toLowerCase();
    requireValue(
      /^10\.[0-9]{4,9}\/[^?#]+$/.test(normalized) &&
        !new RegExp(ID_SPACE).test(normalized),
      "invalid doi external ID",
    );
  } else if (namespace === "arxiv") {
    normalized = v
      .replace(/^https?:\/\/(?:www\.)?arxiv\.org\/(?:abs|pdf)\//i, "")
      .replace(new RegExp(`^arxiv:${ID_SPACE}*`, "i"), "")
      .toLowerCase()
      .replace(/\.pdf$/, "")
      .replace(/v[1-9][0-9]*$/, "");
    requireValue(
      /^(?:[0-9]{4}\.[0-9]{4,5}|[a-z][a-z0-9.-]*\/[0-9]{7})$/.test(normalized),
      "invalid arxiv external ID",
    );
  } else if (namespace === "openalex") {
    normalized = v
      .replace(/^https?:\/\/(?:api\.)?openalex\.org\/(?:works\/)?/i, "")
      .toUpperCase();
    requireValue(
      /^W[1-9][0-9]*$/.test(normalized),
      "invalid openalex external ID",
    );
  } else if (namespace === "url") {
    httpUrl(v);
    const parts = /^(https?):\/\/([^/?#]+)([^?#]*)(\?[^#]*)?(?:#.*)?$/i.exec(
      v,
    )!;
    requireValue(parts, "invalid url external ID");
    normalized = `${parts[1].toLowerCase()}://${parts[2].toLowerCase()}${parts[3] || "/"}${parts[4] === "?" ? "" : (parts[4] ?? "")}`;
  } else throw new Error(`unknown external ID namespace: ${namespace}`);
  return normalized;
}

/** Saved values must already be canonical, as required by kg_identifiers.py. */
export function externalIds(
  value: unknown,
  allowEmpty = false,
): Record<string, string> {
  const data = mapping(value);
  requireValue(
    allowEmpty || Object.keys(data).length,
    "external_ids: at least one ID required",
  );
  const ids: Record<string, string> = {};
  for (const [namespace, raw] of Object.entries(data)) {
    const v = text(raw, "external_ids");
    const normalized = normalizeIdentifier(namespace, v);
    requireValue(
      normalized === v,
      `external_ids.${namespace}: value must be normalized`,
    );
    ids[namespace] = v;
  }
  return ids;
}

function timestamp(value: unknown): string {
  const created = text(value, "created_at");
  const match =
    /^([0-9]{4})-([0-9]{2})-([0-9]{2})T(?:[01][0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9](?:\.[0-9]{1,6})?)?(?:Z|[+-](?:[01][0-9]|2[0-3]):?[0-5][0-9])$/.exec(
      created,
    );
  requireValue(
    match && match[0] === created,
    "created_at must be ISO 8601 with timezone",
  );
  const [year, month, day] = match.slice(1).map(Number);
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  const days = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  requireValue(
    year >= 1 &&
      month >= 1 &&
      month <= 12 &&
      day >= 1 &&
      day <= days[month - 1],
    "created_at has an invalid calendar date",
  );
  return created;
}

function reference(value: unknown): CitationReference {
  const d = mapping(value, [
    "ref_id",
    "citation",
    "target_id",
    "identification_status",
    "roles",
    "decision_status",
    "reason",
    "evidence",
  ]);
  const target = d.target_id === null ? null : paperId(d.target_id);
  const identification = choice(
    d.identification_status,
    ["resolved", "unresolved"],
    "identification_status",
  );
  const decision = choice(
    d.decision_status,
    ["classified", "unclassifiable", "pending"],
    "decision_status",
  );
  const roleList = roles(d.roles);
  const evidence = strings(d.evidence, "evidence");
  requireValue(
    (identification === "resolved") === (target !== null),
    "identification_status and target_id disagree",
  );
  requireValue(
    decision === "classified"
      ? roleList.length && evidence.length
      : !roleList.length,
    "classified requires roles/evidence; unclassifiable/pending requires empty roles",
  );
  return {
    ref_id: refId(d.ref_id),
    citation: text(d.citation, "citation"),
    target_id: target,
    identification_status: identification,
    roles: roleList,
    decision_status: decision,
    reason: text(d.reason, "reason"),
    evidence,
  };
}

export function parseCitationDocument(
  raw: string,
): CitationPaper | CitationSearch {
  const match = /^---[ \t]*\r?\n([\s\S]*?)\r?\n---[ \t]*(?:\r?\n|$)/.exec(raw);
  requireValue(match, "citation document requires YAML frontmatter");
  const yaml = parseDocument(match[1], {
    version: "1.1",
    uniqueKeys: true,
    merge: false,
  });
  const yamlErrors = [...yaml.errors, ...yaml.warnings];
  requireValue(
    !yamlErrors.length,
    `invalid citation YAML: ${yamlErrors.map((e) => e.message).join("; ")}`,
  );
  visit(yaml, (_, node) => {
    requireValue(!isAlias(node), "YAML aliases are not allowed");
    if (isScalar(node) && typeof node.value === "number") {
      // JavaScript erases 1 vs 1.0; the persisted schema accepts integer scalars only.
      const source = node.source ?? "";
      requireValue(
        node.tag !== "tag:yaml.org,2002:float" &&
          (/^[+-]?0[xob]/i.test(source) || !/[.eE]/.test(source)),
        "expected integer YAML scalar",
      );
    }
  });
  const d = mapping(yaml.toJS());
  requireValue(d.schema_version === 1, "citation schema_version must be 1");
  if (d.type === "citation_paper") {
    mapping(d, [
      "schema_version",
      "type",
      "id",
      "identity_key",
      "title",
      "year",
      "authors",
      "external_ids",
      "source",
      "summary",
      "text_status",
      "text_reason",
      "references",
    ]);
    const ids = externalIds(d.external_ids);
    const identity = text(d.identity_key, "identity_key");
    const id = paperId(d.id);
    requireValue(
      Object.entries(ids).some(([k, v]) => `${k}:${v}` === identity) &&
        id ===
          `cite-${createHash("sha256").update(identity).digest("hex").slice(0, 16)}`,
      "citation id/identity_key must match an external ID",
    );
    requireValue(
      d.year === null ||
        (Number.isInteger(d.year) &&
          (d.year as number) >= 1000 &&
          (d.year as number) <= 9999),
      "year: expected 1000..9999 or null",
    );
    const status = choice(
      d.text_status,
      ["available", "failed", "not_requested"],
      "text_status",
    );
    const refs = list(d.references, "references").map(reference);
    requireValue(
      status === "available" || !refs.length,
      "failed/not_requested paper cannot have references",
    );
    requireValue(
      new Set(refs.map((r) => r.ref_id)).size === refs.length,
      "duplicate ref_id",
    );
    return {
      schema_version: 1,
      type: "citation_paper",
      id,
      identity_key: identity,
      title: text(d.title, "title"),
      year: d.year as number | null,
      authors: strings(d.authors, "authors"),
      external_ids: ids,
      source: httpUrl(d.source),
      summary: text(d.summary, "summary", status !== "available"),
      text_status: status,
      text_reason: text(d.text_reason, "text_reason", status !== "failed"),
      references: refs,
    };
  }
  requireValue(
    d.type === "citation_search",
    "type must be citation_paper or citation_search",
  );
  mapping(d, [
    "schema_version",
    "type",
    "id",
    "created_at",
    "root_id",
    "roles",
    "max_depth",
    "generations",
    "issues",
  ]);
  const id = text(d.id, "id");
  requireValue(
    /^search-[a-z0-9]+(?:-[a-z0-9]+)*$/.test(id),
    "search id must be search-kebab-slug",
  );
  const created = timestamp(d.created_at);
  const generations = list(d.generations, "generations").map((value) => {
    const g = mapping(value, ["depth", "paper_ids", "expanded_ids"]);
    return {
      depth: integer(g.depth, "depth"),
      paper_ids: strings(g.paper_ids, "paper_ids", false).map(paperId),
      expanded_ids: strings(g.expanded_ids, "expanded_ids").map(paperId),
    };
  });
  requireValue(generations.length, "generations: root generation required");
  const issues = list(d.issues, "issues").map((value) => {
    const p = mapping(value, ["paper_id", "ref_id"]);
    return {
      paper_id: paperId(p.paper_id),
      ref_id: p.ref_id === null ? null : refId(p.ref_id),
    };
  });
  requireValue(
    new Set(issues.map((p) => `${p.paper_id}/${p.ref_id}`)).size ===
      issues.length,
    "duplicate issue pointer",
  );
  return {
    schema_version: 1,
    type: "citation_search",
    id,
    created_at: created,
    root_id: paperId(d.root_id),
    roles: roles(d.roles, false),
    max_depth: integer(d.max_depth, "max_depth"),
    generations,
    issues,
  };
}

type Identity = { id: string; ids: Record<string, string> };
function resolveIdentity(
  ids: Record<string, string>,
  entries: Identity[],
): string | undefined {
  const matches = entries.filter((e) =>
    Object.entries(ids).some(([k, v]) => e.ids[k] === v),
  );
  requireValue(
    matches.length <= 1,
    "ambiguous literature aliases; multiple documents match external IDs",
  );
  const match = matches[0];
  if (match)
    for (const [k, v] of Object.entries(ids))
      requireValue(
        !Object.hasOwn(match.ids, k) || match.ids[k] === v,
        `conflicting external IDs for ${match.id}: ${k}`,
      );
  return match?.id;
}
function identityIndex(entries: Identity[]): Identity[] {
  const index: Identity[] = [];
  for (const entry of entries) {
    requireValue(
      !index.some((e) => e.id === entry.id),
      `duplicate document id: ${entry.id}`,
    );
    requireValue(
      !resolveIdentity(entry.ids, index),
      `duplicate literature: ${entry.id}`,
    );
    index.push(entry);
  }
  return index;
}
export function joinRegisteredPapers(
  papers: CitationPaper[],
  registered: Pick<Paper, "id" | "externalIds">[],
): Record<string, string> {
  identityIndex(
    papers.map((p) => ({ id: p.id, ids: externalIds(p.external_ids) })),
  );
  const index = identityIndex(
    registered.map((p) => ({
      id: p.id,
      ids: externalIds(p.externalIds, true),
    })),
  );
  const result: Record<string, string> = {};
  const reverse = new Set<string>();
  for (const paper of papers) {
    const match = resolveIdentity(paper.external_ids, index);
    if (!match) continue;
    requireValue(
      !reverse.has(match),
      `duplicate literature through Papers ${match}`,
    );
    result[paper.id] = match;
    reverse.add(match);
  }
  return result;
}

function equalSets(a: Set<string>, b: Set<string>) {
  return a.size === b.size && [...a].every((v) => b.has(v));
}
export function validateCitationLibrary(library: CitationLibrary): void {
  const papers = new Map(library.papers.map((p) => [p.id, p]));
  identityIndex(library.papers.map((p) => ({ id: p.id, ids: p.external_ids })));
  requireValue(
    new Set(library.searches.map((s) => s.id)).size === library.searches.length,
    "duplicate search id",
  );
  for (const p of papers.values())
    for (const r of p.references)
      requireValue(
        r.target_id === null || papers.has(r.target_id),
        `${p.id}/${r.ref_id}: unknown target_id ${r.target_id}`,
      );
  for (const search of library.searches) {
    let expected = new Set([search.root_id]);
    const seen = new Set<string>();
    const issues = new Set<string>();
    for (const [depth, g] of search.generations.entries()) {
      const reached = new Set(g.paper_ids);
      const expanded = new Set(g.expanded_ids);
      requireValue(
        g.depth === depth && depth <= search.max_depth,
        `${search.id}: generations must have consecutive depths within max_depth`,
      );
      requireValue(
        equalSets(reached, expected) &&
          ![...reached].some((id) => seen.has(id)),
        `${search.id}: generation ${depth} paper_ids disagree with selected-role traversal`,
      );
      requireValue(
        equalSets(expanded, depth < search.max_depth ? reached : new Set()),
        `${search.id}: expanded_ids disagree with max_depth`,
      );
      for (const id of reached) {
        requireValue(
          papers.has(id),
          `${search.id}: unknown citation paper ${id}`,
        );
        seen.add(id);
      }
      expected = new Set();
      for (const id of expanded) {
        const p = papers.get(id)!;
        requireValue(
          p.text_status !== "not_requested",
          `${search.id}: expanded paper ${id} has text_status not_requested`,
        );
        if (p.text_status === "failed") issues.add(`${id}/null`);
        for (const r of p.references) {
          if (
            r.identification_status === "unresolved" ||
            r.decision_status !== "classified"
          )
            issues.add(`${id}/${r.ref_id}`);
          if (
            r.target_id &&
            r.decision_status === "classified" &&
            r.roles.some((role) => search.roles.includes(role))
          )
            expected.add(r.target_id);
        }
      }
      for (const id of seen) expected.delete(id);
    }
    requireValue(
      !expected.size,
      `${search.id}: missing a selected-role generation`,
    );
    requireValue(
      equalSets(
        issues,
        new Set(search.issues.map((p) => `${p.paper_id}/${p.ref_id}`)),
      ),
      `${search.id}: issues disagree with expanded paper/reference states`,
    );
  }
}

export async function loadCitations(
  directory = path.join(KNOWLEDGE_DIR, "citation_map"),
  registered: Pick<Paper, "id" | "externalIds">[] = [],
): Promise<CitationLibrary> {
  const library: CitationLibrary = {
    papers: [],
    searches: [],
    registeredIds: {},
  };
  const stat = await fs
    .lstat(directory)
    .catch((error: NodeJS.ErrnoException) => {
      if (error.code === "ENOENT") return null;
      throw error;
    });
  if (!stat) return library;
  requireValue(
    stat.isDirectory() && !stat.isSymbolicLink(),
    `${directory}: expected regular citation_map directory`,
  );
  for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
    const folder = path.join(directory, entry.name);
    if (entry.name === "README.md" && entry.isFile()) continue;
    requireValue(
      ["papers", "searches"].includes(entry.name) &&
        entry.isDirectory() &&
        !entry.isSymbolicLink(),
      `${folder}: expected papers/ or searches/ directory`,
    );
    for (const item of await fs.readdir(folder, { withFileTypes: true })) {
      const file = path.join(folder, item.name);
      try {
        requireValue(
          item.isFile() && !item.isSymbolicLink() && item.name.endsWith(".md"),
          "only regular Markdown citation documents are allowed",
        );
        const doc = parseCitationDocument(
          new TextDecoder("utf-8", { fatal: true }).decode(
            await fs.readFile(file),
          ),
        );
        requireValue(
          item.name === `${doc.id}.md` &&
            entry.name ===
              (doc.type === "citation_paper" ? "papers" : "searches"),
          "document type/id must match path",
        );
        if (doc.type === "citation_paper") library.papers.push(doc);
        else library.searches.push(doc);
      } catch (error) {
        throw new Error(
          `${file}: ${error instanceof Error ? error.message : String(error)}`,
        );
      }
    }
  }
  validateCitationLibrary(library);
  library.registeredIds = joinRegisteredPapers(library.papers, registered);
  library.searches.sort(
    (a, b) =>
      b.created_at.localeCompare(a.created_at) || a.id.localeCompare(b.id),
  );
  return library;
}

export async function readCitations(
  registered: Paper[],
  directory?: string,
): Promise<CitationLoadResult> {
  try {
    return { ok: true, library: await loadCitations(directory, registered) };
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}
