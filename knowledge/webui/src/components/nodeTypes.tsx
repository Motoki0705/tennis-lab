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

/** Pick the usual two headlines plus conditional canonical-pose metrics. */
function headlineMetrics(metrics?: Record<string, number | string>): [string, string][] {
  if (!metrics) return [];
  const angularErrorKey = "angular_error_deg" in metrics
    ? "angular_error_deg"
    : "ang_error_deg";
  const priority = [
    angularErrorKey,
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
  const canonicalMpjpeKey = "canonical_mpjpe_m" in metrics
    ? "canonical_mpjpe_m"
    : "canonical_mpjpe" in metrics
      ? "canonical_mpjpe"
      : null;
  if (canonicalMpjpeKey) {
    out.push(["canonical_mpjpe_m", String(metrics[canonicalMpjpeKey])]);
  }
  if ("canonical_pck_0.1m" in metrics) {
    out.push(["canonical_pck_0.1m", String(metrics["canonical_pck_0.1m"])]);
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
  // A group is a container that encloses its member runs (positioned inside by
  // the layout); it has no edges of its own, hence no handles.
  return (
    <div className={`group-node ${selected ? "group-node--selected" : ""}`}>
      <div className="group-node__header">
        <span className="group-node__label">GROUP {issue ?? ""}</span>
        <span className="group-node__title">{data.title}</span>
      </div>
    </div>
  );
}

export const nodeTypes = { runNode: RunNode, groupNode: GroupNode };
