"use client";

import { Handle, Position, type NodeProps } from "reactflow";

import type { KnowledgeNode } from "@/lib/types";

export const PROVIDER_COLOR: Record<string, string> = {
  claude: "#d97757",
  codex: "#10a37f",
  gemini: "#4285f4",
  human: "#9b59b6",
  other: "#7f8c8d",
};

const STATUS_COLOR: Record<string, string> = {
  done: "#2ecc71",
  failed: "#e74c3c",
  running: "#f1c40f",
  planned: "#95a5a6",
};

function issueLabel(issue?: number | number[]): string | null {
  if (issue == null) return null;
  return Array.isArray(issue) ? issue.map((i) => `#${i}`).join(" ") : `#${issue}`;
}

/** Pick up to two headline metrics that exist, for the card preview. */
function headlineMetrics(metrics?: Record<string, number | string>): [string, string][] {
  if (!metrics) return [];
  const priority = [
    "ang_error_deg",
    "position_error_m",
    "best_val_miou",
    "loss",
    "total_loss",
  ];
  const out: [string, string][] = [];
  for (const k of priority) {
    if (k in metrics) out.push([k, String(metrics[k])]);
    if (out.length === 2) break;
  }
  return out;
}

export function RunNode({ data, selected }: NodeProps<KnowledgeNode>) {
  const accent = PROVIDER_COLOR[data.provider ?? "other"] ?? PROVIDER_COLOR.other;
  const issue = issueLabel(data.issue);
  const metrics = headlineMetrics(data.metrics);
  return (
    <div
      className="run-node"
      style={{ borderColor: selected ? "#fff" : accent, boxShadow: selected ? `0 0 0 2px ${accent}` : undefined }}
    >
      <Handle type="target" position={Position.Top} />
      <div className="run-node__head">
        <span className="run-node__provider" style={{ background: accent }}>
          {data.provider ?? "?"}
        </span>
        {issue && <span className="run-node__issue">{issue}</span>}
        {data.status && (
          <span
            className="run-node__status"
            style={{ color: STATUS_COLOR[data.status] ?? "#bbb" }}
            title={data.status}
          >
            ●
          </span>
        )}
      </div>
      <div className="run-node__title">{data.title}</div>
      {metrics.length > 0 && (
        <div className="run-node__metrics">
          {metrics.map(([k, v]) => (
            <span key={k} className="metric-chip">
              <b>{k}</b> {v}
            </span>
          ))}
        </div>
      )}
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

export function GroupNode({ data, selected }: NodeProps<KnowledgeNode>) {
  const issue = issueLabel(data.issue);
  return (
    <div className="group-node" style={{ boxShadow: selected ? "0 0 0 2px #f5d76e" : undefined }}>
      <Handle type="target" position={Position.Top} />
      <div className="group-node__label">GROUP {issue ?? ""}</div>
      <div className="group-node__title">{data.title}</div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

export const nodeTypes = { runNode: RunNode, groupNode: GroupNode };
