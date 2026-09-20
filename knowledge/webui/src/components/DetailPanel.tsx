"use client";

import { useEffect } from "react";
import type { KnowledgeGraph, KnowledgeNode } from "@/lib/types";
import { formatValue } from "@/lib/explorer";
import { PROVIDER_COLOR } from "./nodeTypes";

function KeyVals({ obj }: { obj?: Record<string, unknown> }) {
  if (!obj || Object.keys(obj).length === 0) return null;
  return (
    <table className="kv">
      <tbody>
        {Object.entries(obj).map(([k, v]) => (
          <tr key={k}>
            <th>{k}</th>
            <td>{formatValue(v)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function DetailPanel({
  node,
  onClose,
  graph,
  onNavigate,
  onPaper,
}: {
  node: KnowledgeNode | null;
  onClose: () => void;
  graph: KnowledgeGraph;
  onNavigate: (id: string) => void;
  onPaper: (id: string) => void;
}) {
  useEffect(() => {
    const listener = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, [onClose]);
  if (!node) {
    return (
      <aside className="panel panel--empty">
        <p>ノードをクリックすると詳細が表示されます。</p>
      </aside>
    );
  }
  const accent =
    PROVIDER_COLOR[node.provider ?? "other"] ?? PROVIDER_COLOR.other;
  const issue = Array.isArray(node.issue)
    ? node.issue.map((i) => `#${i}`).join(" ")
    : node.issue != null
      ? `#${node.issue}`
      : null;
  return (
    <aside className="panel" aria-label="実験の詳細">
      <button className="panel__close" onClick={onClose} aria-label="close">
        ×
      </button>
      <div
        className="panel__type"
        style={{ color: node.type === "group" ? "#f5d76e" : accent }}
      >
        {node.type.toUpperCase()}
      </div>
      <p className="eyebrow">
        {node.task} / {String(node.sequence).padStart(6, "0")}
      </p>
      <h2 className="panel__title">{node.title}</h2>
      <div className="panel__meta">
        {issue && <span className="badge">{issue}</span>}
        {node.provider && (
          <span className="badge" style={{ background: accent }}>
            {node.provider}
          </span>
        )}
        {node.status && <span className="badge">{node.status}</span>}
        {node.date && <span className="badge badge--ghost">{node.date}</span>}
      </div>
      {node.tags.length > 0 && (
        <div className="panel__tags">
          {node.tags.map((t) => (
            <span key={t} className="tag">
              #{t}
            </span>
          ))}
        </div>
      )}

      <section>
        <h3>関連する実験</h3>
        {[
          ...node.parents.map((id) => ({ id, label: "前提" })),
          ...node.members.map((id) => ({ id, label: "メンバー" })),
          ...node.relations.map((r) => ({ id: r.to, label: r.rel ?? "関連" })),
          ...graph.nodes
            .filter((n) => n.parents.includes(node.id))
            .map((n) => ({ id: n.id, label: "後続" })),
        ].map((r, i) => (
          <button
            key={`${r.id}-${i}`}
            className="related-link"
            onClick={() => onNavigate(r.id)}
          >
            {r.label} · {graph.nodes.find((n) => n.id === r.id)?.title ?? r.id}{" "}
            →
          </button>
        ))}
      </section>
      <section>
        <h3>関連研究</h3>
        {node.papers.length ? (
          node.papers.map((id) => (
            <button
              className="related-link"
              key={id}
              onClick={() => onPaper(id)}
            >
              {graph.papers.find((p) => p.id === id)?.title ?? id} ↗
            </button>
          ))
        ) : (
          <p className="muted">論文参照はまだ登録されていません。</p>
        )}
      </section>
      <details>
        <summary>登録情報</summary>
        <p>{node.file}</p>
        <p>
          登録日: {node.recordedAt} / 日付の根拠:{" "}
          {node.dateSource ?? "registration"}
        </p>
      </details>
      {node.config && (
        <section>
          <h3>config</h3>
          <KeyVals obj={node.config} />
        </section>
      )}
      {node.metrics && (
        <section>
          <h3>metrics</h3>
          <KeyVals obj={node.metrics} />
        </section>
      )}
      {node.curvesUrl && (
        <section>
          <h3>学習曲線 / curves</h3>
          <a href={node.curvesUrl} target="_blank" rel="noreferrer">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              className="curves"
              src={node.curvesUrl}
              alt={`${node.id} train/val curves`}
            />
          </a>
        </section>
      )}
      {node.artifacts && (
        <section>
          <h3>artifacts</h3>
          <KeyVals obj={node.artifacts} />
        </section>
      )}
      {node.bodyHtml && (
        <section>
          <h3>考察 / Findings</h3>
          <div
            className="prose"
            dangerouslySetInnerHTML={{ __html: node.bodyHtml }}
          />
        </section>
      )}
    </aside>
  );
}
