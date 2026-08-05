"use client";

import type { KnowledgeNode } from "@/lib/types";
import { nodeAccent } from "./nodeTypes";

function displayValue(value: unknown): string {
  if (typeof value === "object" && value !== null) {
    return JSON.stringify(value, null, 2);
  }
  return String(value);
}

function KeyVals({ obj }: { obj?: Record<string, unknown> }) {
  if (!obj || Object.keys(obj).length === 0) return null;
  return (
    <table className="kv">
      <tbody>
        {Object.entries(obj).map(([key, value]) => (
          <tr key={key}>
            <th>{key}</th>
            <td>
              <span style={{ whiteSpace: "pre-wrap" }}>{displayValue(value)}</span>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function List({ values }: { values: string[] }) {
  if (values.length === 0) return null;
  return (
    <ul>
      {values.map((value) => (
        <li key={value}>{value}</li>
      ))}
    </ul>
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
  const accent = nodeAccent(node);
  const issue = Array.isArray(node.issue)
    ? node.issue.map((item) => `#${item}`).join(" ")
    : node.issue != null
      ? `#${node.issue}`
      : null;
  const tasks = node.task ? [node.task] : node.tasks;
  return (
    <aside className="panel">
      <button className="panel__close" onClick={onClose} aria-label="close">
        ×
      </button>
      <div className="panel__type" style={{ color: accent }}>
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
        {node.curator && <span className="badge">curator: {node.curator}</span>}
        {node.status && <span className="badge">{node.status}</span>}
        {node.date && <span className="badge badge--ghost">{node.date}</span>}
        {node.evidenceLevel && (
          <span className="badge badge--ghost">{node.evidenceLevel}</span>
        )}
      </div>
      {node.tags.length > 0 && (
        <div className="panel__tags">
          {node.tags.map((tag) => (
            <span key={tag} className="tag">
              #{tag}
            </span>
          ))}
        </div>
      )}

      {tasks.length > 0 && (
        <section>
          <h3>tasks</h3>
          <List values={tasks} />
        </section>
      )}
      {node.repoPaths.length > 0 && (
        <section>
          <h3>repo paths</h3>
          <List values={node.repoPaths} />
        </section>
      )}
      {node.externalIds && (
        <section>
          <h3>external ids</h3>
          <KeyVals obj={node.externalIds} />
        </section>
      )}
      {node.hypothesis && (
        <section>
          <h3>hypothesis</h3>
          <KeyVals obj={node.hypothesis} />
        </section>
      )}
      {node.evaluation && (
        <section>
          <h3>evaluation</h3>
          <KeyVals obj={node.evaluation} />
        </section>
      )}
      {node.evidenceRuns.length > 0 && (
        <section>
          <h3>evidence runs</h3>
          <List values={node.evidenceRuns} />
        </section>
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
      {node.sources.length > 0 && (
        <section>
          <h3>sources</h3>
          <ul>
            {node.sources.map((source, index) => {
              const url = typeof source.url === "string" ? source.url : null;
              const label = typeof source.kind === "string" ? source.kind : `source ${index + 1}`;
              return <li key={`${label}-${url ?? index}`}>{url ? <a href={url}>{label}</a> : label}</li>;
            })}
          </ul>
        </section>
      )}
      {node.bodyHtml && (
        <section>
          <h3>内容 / Findings</h3>
          <div className="prose" dangerouslySetInnerHTML={{ __html: node.bodyHtml }} />
        </section>
      )}
    </aside>
  );
}
