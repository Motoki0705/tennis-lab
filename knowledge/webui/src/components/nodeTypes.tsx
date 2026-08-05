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

const TYPE_COLOR: Record<KnowledgeNode["type"], string> = {
  run: PROVIDER_COLOR.other,
  group: "#f5d76e",
  paper: "#4aa3df",
  proposal: "#af7ac5",
};

const STATUS_COLOR: Record<string, string> = {
  done: "#2ecc71",
  failed: "#e74c3c",
  running: "#f1c40f",
  planned: "#95a5a6",
  reviewed: "#4aa3df",
  superseded: "#95a5a6",
  withdrawn: "#e74c3c",
  candidate: "#95a5a6",
  ready: "#5dade2",
  "issue-open": "#f5b041",
  testing: "#f1c40f",
  supported: "#2ecc71",
  refuted: "#e74c3c",
  inconclusive: "#aab7b8",
  adopted: "#58d68d",
};

export function nodeAccent(node: KnowledgeNode): string {
  if (node.type === "run") {
    return PROVIDER_COLOR[node.provider ?? "other"] ?? PROVIDER_COLOR.other;
  }
  return TYPE_COLOR[node.type];
}

function issueLabel(issue?: number | number[]): string | null {
  if (issue == null) return null;
  return Array.isArray(issue) ? issue.map((item) => `#${item}`).join(" ") : `#${issue}`;
}

function headlineMetrics(metrics?: Record<string, number | string>): [string, string][] {
  if (!metrics) return [];
  const priority = [
    "ang_error_deg",
    "position_error_m",
    "best_val_miou",
    "loss",
    "total_loss",
  ];
  const output: [string, string][] = [];
  for (const key of priority) {
    if (key in metrics) output.push([key, String(metrics[key])]);
    if (output.length === 2) break;
  }
  return output;
}

function taskLabels(data: KnowledgeNode): string[] {
  if (data.task) return [data.task];
  return data.tasks.slice(0, 2);
}

export function RunNode({ data, selected }: NodeProps<KnowledgeNode>) {
  const accent = nodeAccent(data);
  const issue = issueLabel(data.issue);
  const metrics = data.type === "run" ? headlineMetrics(data.metrics) : [];
  const tasks = data.type === "run" ? [] : taskLabels(data);
  const identity = data.type === "run" ? (data.provider ?? "run") : data.type;
  return (
    <div
      className="run-node"
      style={{
        borderColor: selected ? "#fff" : accent,
        boxShadow: selected ? `0 0 0 2px ${accent}` : undefined,
      }}
    >
      <Handle type="target" position={Position.Top} />
      <div className="run-node__head">
        <span className="run-node__provider" style={{ background: accent }}>
          {identity}
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
          {metrics.map(([key, value]) => (
            <span key={key} className="metric-chip">
              <b>{key}</b> {value}
            </span>
          ))}
        </div>
      )}
      {tasks.length > 0 && (
        <div className="run-node__metrics">
          {tasks.map((task) => (
            <span key={task} className="metric-chip">
              {task}
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
    <div className={`group-node ${selected ? "group-node--selected" : ""}`}>
      <div className="group-node__header">
        <span className="group-node__label">GROUP {issue ?? ""}</span>
        <span className="group-node__title">{data.title}</span>
      </div>
    </div>
  );
}

export const nodeTypes = { runNode: RunNode, groupNode: GroupNode };
