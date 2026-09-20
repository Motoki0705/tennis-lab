import type { KnowledgeNode } from "./types";

export interface Filters {
  task: string;
  query: string;
  status: string;
  provider: string;
  tag: string;
  issue: string;
  paper: string;
  order: string;
}
export const EMPTY_FILTERS: Filters = {
  task: "",
  query: "",
  status: "",
  provider: "",
  tag: "",
  issue: "",
  paper: "",
  order: "newest",
};
export function filterNodes(
  nodes: KnowledgeNode[],
  filters: Filters,
): KnowledgeNode[] {
  const words = filters.query.toLowerCase().trim().split(/\s+/).filter(Boolean);
  return nodes
    .filter((n) => {
      const text = [
        n.id,
        n.title,
        n.searchText,
        ...n.tags,
        JSON.stringify(n.config ?? {}),
        JSON.stringify(n.metrics ?? {}),
      ]
        .join(" ")
        .toLowerCase();
      return (
        (!filters.task || n.task === filters.task) &&
        (!filters.status || n.status === filters.status) &&
        (!filters.provider || n.provider === filters.provider) &&
        (!filters.tag || n.tags.includes(filters.tag)) &&
        (!filters.issue ||
          [n.issue].flat().map(String).includes(filters.issue)) &&
        (!filters.paper || n.papers.includes(filters.paper)) &&
        words.every((word) => text.includes(word))
      );
    })
    .sort((a, b) => {
      const order =
        (a.date ?? a.recordedAt).localeCompare(b.date ?? b.recordedAt) ||
        a.task.localeCompare(b.task) ||
        a.sequence - b.sequence;
      return filters.order === "oldest" ? order : -order;
    });
}
export function metricsCsv(nodes: KnowledgeNode[]): string {
  const keys = Array.from(
    new Set(nodes.flatMap((n) => Object.keys(n.metrics ?? {}))),
  ).sort();
  const cell = (value: unknown) => {
    let text = formatValue(value, "");
    if (/^[=+@\-\t\r]/.test(text)) text = `'${text}`;
    return `"${text.replace(/"/g, '""')}"`;
  };
  return [
    ["id", "task", "date", "config", ...keys],
    ...nodes.map((n) => [
      n.id,
      n.task,
      n.date ?? "",
      JSON.stringify(n.config ?? {}),
      ...keys.map((k) => n.metrics?.[k] ?? ""),
    ]),
  ]
    .map((row) => row.map(cell).join(","))
    .join("\r\n");
}

/** Preserve structured measurements and booleans at every display/export boundary. */
export function formatValue(value: unknown, missing = "—"): string {
  if (value == null) return missing;
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}
