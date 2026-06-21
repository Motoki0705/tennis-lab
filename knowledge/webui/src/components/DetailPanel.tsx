"use client";

import type { KnowledgeNode } from "@/lib/types";
import { PROVIDER_COLOR } from "./nodeTypes";

function KeyVals({ obj }: { obj?: Record<string, unknown> }) {
  if (!obj || Object.keys(obj).length === 0) return null;
  return (
    <table className="kv">
      <tbody>
        {Object.entries(obj).map(([k, v]) => (
          <tr key={k}>
            <th>{k}</th>
            <td>{String(v)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function DetailPanel({
  node,
  onClose,
}: {
  node: KnowledgeNode | null;
  onClose: () => void;
}) {
  if (!node) {
    return (
      <aside className="panel panel--empty">
        <p>ノードをクリックすると詳細が表示されます。</p>
      </aside>
    );
  }
  const accent = PROVIDER_COLOR[node.provider ?? "other"] ?? PROVIDER_COLOR.other;
  const issue = Array.isArray(node.issue)
    ? node.issue.map((i) => `#${i}`).join(" ")
    : node.issue != null
      ? `#${node.issue}`
      : null;
  return (
    <aside className="panel">
      <button className="panel__close" onClick={onClose} aria-label="close">
        ×
      </button>
      <div className="panel__type" style={{ color: node.type === "group" ? "#f5d76e" : accent }}>
        {node.type.toUpperCase()}
      </div>
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
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <a href={node.curvesUrl} target="_blank" rel="noreferrer">
            <img className="curves" src={node.curvesUrl} alt={`${node.id} train/val curves`} />
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
          <div className="prose" dangerouslySetInnerHTML={{ __html: node.bodyHtml }} />
        </section>
      )}
    </aside>
  );
}
