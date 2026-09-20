"use client";
import { useEffect, useMemo, useState } from "react";
import type { KnowledgeGraph } from "@/lib/types";
import {
  EMPTY_FILTERS,
  filterNodes,
  metricsCsv,
  type Filters,
} from "@/lib/explorer";
import { DetailPanel } from "./DetailPanel";
import { GraphView } from "./GraphView";

type View = "timeline" | "graph" | "papers" | "summary" | "compare";
const VIEWS: [View, string][] = [
  ["timeline", "実験一覧"],
  ["graph", "知識グラフ"],
  ["papers", "Papers"],
  ["summary", "研究サマリー"],
  ["compare", "実験比較"],
];
const PAGE_SIZE = 30;
export function ResearchExplorer({ graph }: { graph: KnowledgeGraph }) {
  const [filters, setFilters] = useState<Filters>(EMPTY_FILTERS);
  const [view, setView] = useState<View>("timeline");
  const [selected, setSelected] = useState<string | null>(null);
  const [paperId, setPaperId] = useState("");
  const [compare, setCompare] = useState<string[]>([]);
  const [page, setPage] = useState(0);
  const [ready, setReady] = useState(false);
  useEffect(() => {
    function restore() {
      const params = new URLSearchParams(location.search);
      setFilters(
        Object.fromEntries(
          Object.entries(EMPTY_FILTERS).map(([key, value]) => [
            key,
            params.get(key === "paper" ? "paperFilter" : key) ?? value,
          ]),
        ) as unknown as Filters,
      );
      const requested = params.get("view");
      setView(
        VIEWS.some(([key]) => key === requested)
          ? (requested as View)
          : params.has("paper")
            ? "papers"
            : "timeline",
      );
      setSelected(params.get("node"));
      setPaperId(params.get("paper") ?? "");
      setReady(true);
    }
    restore();
    window.addEventListener("popstate", restore);
    return () => window.removeEventListener("popstate", restore);
  }, []);
  useEffect(() => {
    if (!ready) return;
    const params = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => {
      if (value && value !== EMPTY_FILTERS[key as keyof Filters])
        params.set(key === "paper" ? "paperFilter" : key, value);
    });
    params.set("view", view);
    if (selected) params.set("node", selected);
    if (paperId && view === "papers") params.set("paper", paperId);
    history.replaceState(null, "", `?${params}`);
  }, [filters, view, selected, paperId, ready]);
  const tasks = useMemo(
    () =>
      Array.from(
        new Set([
          ...graph.nodes.map((n) => n.task),
          ...graph.papers.flatMap((p) => p.tasks),
        ]),
      ).sort(),
    [graph],
  );
  const visible = useMemo(
    () => filterNodes(graph.nodes, filters),
    [graph, filters],
  );
  const node = graph.nodes.find((n) => n.id === selected) ?? null;
  const compared = graph.nodes.filter((n) => compare.includes(n.id));
  const metricKeys = Array.from(
    new Set(compared.flatMap((n) => Object.keys(n.metrics ?? {}))),
  ).sort();
  const paper = graph.papers.find((p) => p.id === paperId);
  const papers = graph.papers.filter(
    (p) =>
      (!filters.task || p.tasks.includes(filters.task)) &&
      `${p.title} ${p.authors.join(" ")} ${p.id}`
        .toLowerCase()
        .includes(filters.query.toLowerCase()),
  );
  function update(key: keyof Filters, value: string) {
    setFilters((f) => ({ ...f, [key]: value }));
    setPage(0);
  }
  function openPaper(id: string) {
    setPaperId(id);
    setView("papers");
    setSelected(null);
  }
  function exportCsv() {
    const url = URL.createObjectURL(
      new Blob(["\uFEFF" + metricsCsv(compared)], {
        type: "text/csv;charset=utf-8",
      }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = "experiment-comparison.csv";
    a.click();
    URL.revokeObjectURL(url);
  }
  const options = (key: "status" | "provider" | "tag" | "issue") =>
    Array.from(
      new Set(
        graph.nodes.flatMap((n) =>
          key === "tag"
            ? n.tags
            : key === "issue"
              ? [n.issue]
                  .flat()
                  .filter((x) => x != null)
                  .map(String)
              : n[key]
                ? [n[key]!]
                : [],
        ),
      ),
    ).sort();
  return (
    <div className="research-shell">
      <aside className="sidebar">
        <div className="brand-mark">
          TL<span>RESEARCH</span>
        </div>
        <p className="eyebrow">KNOWLEDGE LIBRARY</p>
        <h1>実験から、次の発見へ。</h1>
        <p className="sidebar-description">
          結果、判断、関連研究をタスクごとに辿る。
        </p>
        <div className="sidebar-label">
          TASKS <span>{tasks.length}</span>
        </div>
        <button
          className={`task-button ${!filters.task ? "active" : ""}`}
          onClick={() => update("task", "")}
        >
          すべてのタスク <b>{graph.nodes.length}</b>
        </button>
        {tasks.map((task) => (
          <button
            key={task}
            className={`task-button ${filters.task === task ? "active" : ""}`}
            onClick={() => update("task", task)}
          >
            {task.replaceAll("_", " ")}
            <b>{graph.nodes.filter((n) => n.task === task).length}</b>
          </button>
        ))}
        <div className="sidebar-footer">
          TENNIS LAB<span>一つの実験を、一つの知見に。</span>
        </div>
      </aside>
      <main className="workspace">
        <header className="workspace-header">
          <div>
            <p className="eyebrow">RESEARCH / {filters.task || "ALL TASKS"}</p>
            <h2>
              {filters.task
                ? filters.task.replaceAll("_", " ")
                : "研究ライブラリ"}
            </h2>
          </div>
          <div className="header-count">
            <strong>
              {graph.nodes.filter((n) => n.type === "run").length}
            </strong>{" "}
            experiments <span> / {graph.papers.length} papers</span>
          </div>
        </header>
        <nav className="view-tabs" aria-label="表示切替">
          {VIEWS.map(([key, title]) => (
            <button
              key={key}
              className={view === key ? "active" : ""}
              aria-pressed={view === key}
              onClick={() => {
                setView(key);
                setSelected(null);
              }}
            >
              {title}
              {key === "compare" && compare.length > 0
                ? ` (${compare.length})`
                : ""}
            </button>
          ))}
        </nav>
        {!["summary", "compare"].includes(view) && (
          <div className="search-toolbar">
            <input
              className="search-input"
              aria-label="検索"
              placeholder={
                view === "papers"
                  ? "論文名・著者を検索…"
                  : "実験名・ID・考察・設定を検索…"
              }
              value={filters.query}
              onChange={(e) => update("query", e.target.value)}
            />
            {view !== "papers" && (
              <>
                <details className="advanced-filters">
                  <summary>絞り込み</summary>
                  <div>
                    {(["status", "provider", "tag", "issue"] as const).map(
                      (key) => (
                        <label key={key}>
                          {key}
                          <select
                            aria-label={key}
                            value={filters[key]}
                            onChange={(e) => update(key, e.target.value)}
                          >
                            <option value="">すべて</option>
                            {options(key).map((value) => (
                              <option key={value}>{value}</option>
                            ))}
                          </select>
                        </label>
                      ),
                    )}
                    <label>
                      関連論文
                      <select
                        aria-label="関連論文"
                        value={filters.paper}
                        onChange={(e) => update("paper", e.target.value)}
                      >
                        <option value="">すべて</option>
                        {graph.papers.map((p) => (
                          <option key={p.id} value={p.id}>
                            {p.title}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                </details>
                <select
                  aria-label="並び順"
                  value={filters.order}
                  onChange={(e) => update("order", e.target.value)}
                >
                  <option value="newest">新しい順</option>
                  <option value="oldest">古い順</option>
                </select>
              </>
            )}
            <button
              className="text-button"
              onClick={() => {
                setFilters(EMPTY_FILTERS);
                setPage(0);
              }}
            >
              リセット
            </button>
          </div>
        )}
        <div className="content-area">
          {view === "timeline" && (
            <div className="timeline">
              <div className="section-caption">
                <span>{visible.length} 件のノード</span>
                <span>実験日順 · 日付不明は登録日 / 比較は最大4件</span>
              </div>
              {!visible.length && (
                <div className="empty-state">
                  <h3>該当する実験がありません</h3>
                  <p>検索語や絞り込み条件を変更してください。</p>
                </div>
              )}
              {visible
                .slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)
                .map((n) => (
                  <article className="experiment-row" key={n.id}>
                    <div className="sequence">
                      <span>{n.task}</span>
                      <strong>{String(n.sequence).padStart(6, "0")}</strong>
                      <time>{n.date ?? n.recordedAt}</time>
                      {!n.date && <small>登録日</small>}
                    </div>
                    <button
                      className="experiment-main"
                      onClick={() => setSelected(n.id)}
                    >
                      <span className="row-meta">
                        <span className={`status-dot ${n.status}`} />
                        {n.type === "group"
                          ? "GROUP"
                          : n.status?.toUpperCase()}{" "}
                        · {n.provider ?? "research"}
                        {n.issue != null
                          ? ` · #${[n.issue].flat().join(" / #")}`
                          : ""}
                      </span>
                      <h3>{n.title}</h3>
                      <span className="row-id">{n.id}</span>
                      <span className="row-tags">
                        {n.tags.slice(0, 4).map((t) => (
                          <span key={t}>{t}</span>
                        ))}
                        {n.papers.length > 0 && (
                          <span>↗ {n.papers.length} paper</span>
                        )}
                      </span>
                    </button>
                    <div className="row-right">
                      {Object.entries(n.metrics ?? {})
                        .slice(0, 2)
                        .map(([key, value]) => (
                          <div className="row-metric" key={key}>
                            <span>{key}</span>
                            <strong>
                              {typeof value === "number"
                                ? value.toLocaleString(undefined, {
                                    maximumFractionDigits: 4,
                                  })
                                : value}
                            </strong>
                          </div>
                        ))}
                      {n.type === "run" && (
                        <label className="compare-check">
                          <input
                            type="checkbox"
                            aria-label={`${n.id} を比較`}
                            checked={compare.includes(n.id)}
                            disabled={
                              !compare.includes(n.id) && compare.length >= 4
                            }
                            onChange={(e) =>
                              setCompare((ids) =>
                                e.target.checked
                                  ? [...ids, n.id]
                                  : ids.filter((id) => id !== n.id),
                              )
                            }
                          />{" "}
                          比較に追加
                        </label>
                      )}
                    </div>
                  </article>
                ))}
              <div className="pagination">
                <button
                  disabled={page === 0}
                  onClick={() => setPage((p) => p - 1)}
                >
                  前へ
                </button>
                <span>
                  {page + 1} /{" "}
                  {Math.max(1, Math.ceil(visible.length / PAGE_SIZE))}
                </span>
                <button
                  disabled={(page + 1) * PAGE_SIZE >= visible.length}
                  onClick={() => setPage((p) => p + 1)}
                >
                  次へ
                </button>
              </div>
            </div>
          )}
          {view === "graph" &&
            (visible.length ? (
              <GraphView
                graph={{ ...graph, nodes: visible }}
                onSelect={setSelected}
              />
            ) : (
              <div className="empty-state">該当するノードがありません。</div>
            ))}
          {view === "summary" && (
            <article
              className="summary-document prose"
              dangerouslySetInnerHTML={{ __html: graph.summaryHtml }}
            />
          )}
          {view === "papers" && (
            <div className="paper-library">
              <div className="section-caption">
                <span>RELATED RESEARCH</span>
                <span>{papers.length} papers</span>
              </div>
              <div className="paper-grid">
                {papers.map((p) => (
                  <button
                    className={`paper-card ${paperId === p.id ? "selected" : ""}`}
                    key={p.id}
                    onClick={() => setPaperId(p.id)}
                  >
                    <span className="eyebrow">PAPER / {p.year}</span>
                    <h3>{p.title}</h3>
                    <p>
                      {p.authors.slice(0, 2).join(", ")}
                      {p.authors.length > 2 ? " et al." : ""}
                    </p>
                    <span>
                      {
                        graph.nodes.filter((n) => n.papers.includes(p.id))
                          .length
                      }{" "}
                      関連ノード ↗
                    </span>
                  </button>
                ))}
              </div>
              {!papers.length && (
                <div className="empty-state">該当する論文がありません。</div>
              )}
              {paper && (
                <article className="paper-detail">
                  <h2>{paper.title}</h2>
                  <p>{paper.authors.join(", ")}</p>
                  <div className="paper-actions">
                    <a href={paper.pdfUrl} target="_blank" rel="noreferrer">
                      PDFを開く ↗
                    </a>
                    <a href={paper.source} target="_blank" rel="noreferrer">
                      原論文
                    </a>
                    <a href={paper.license} target="_blank" rel="noreferrer">
                      ライセンス
                    </a>
                  </div>
                  <div
                    className="prose"
                    dangerouslySetInnerHTML={{ __html: paper.bodyHtml }}
                  />
                  <h3>この研究を参照する実験</h3>
                  {graph.nodes
                    .filter((n) => n.papers.includes(paper.id))
                    .map((n) => (
                      <button
                        className="related-link"
                        key={n.id}
                        onClick={() => setSelected(n.id)}
                      >
                        {n.task} / {String(n.sequence).padStart(6, "0")} ·{" "}
                        {n.title} →
                      </button>
                    ))}
                </article>
              )}
            </div>
          )}
          {view === "compare" && (
            <div className="comparison">
              <div className="section-caption">
                <span>EXPERIMENT COMPARISON</span>
                <div>
                  <button onClick={exportCsv} disabled={!compared.length}>
                    CSVを書き出す
                  </button>{" "}
                  <button onClick={() => setCompare([])}>選択を解除</button>
                </div>
              </div>
              <p className="comparison-note">
                データ・split・seed・評価指標の定義を確認して比較してください。異なる条件の値に自動の順位付けは行いません。
              </p>
              {!compared.length ? (
                <div className="empty-state">
                  実験一覧で「比較に追加」を選んでください。
                </div>
              ) : (
                <div className="table-scroll">
                  <table className="comparison-table">
                    <thead>
                      <tr>
                        <th>条件 / 指標</th>
                        {compared.map((n) => (
                          <th key={n.id}>
                            <button onClick={() => setSelected(n.id)}>
                              {n.title}
                            </button>
                            <small>
                              {n.task} / {n.date ?? "日付不明"}
                            </small>
                            <button
                              className="text-button"
                              onClick={() =>
                                setCompare((ids) =>
                                  ids.filter((id) => id !== n.id),
                                )
                              }
                            >
                              解除
                            </button>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <th>config</th>
                        {compared.map((n) => (
                          <td key={n.id}>
                            <pre>{JSON.stringify(n.config, null, 2)}</pre>
                          </td>
                        ))}
                      </tr>
                      {metricKeys.map((key) => (
                        <tr key={key}>
                          <th>{key}</th>
                          {compared.map((n) => (
                            <td key={n.id}>{n.metrics?.[key] ?? "—"}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
          {selected && !node && (
            <div className="empty-state">
              参照されたノードが見つかりません。
              <button onClick={() => setSelected(null)}>閉じる</button>
            </div>
          )}
          <DetailPanel
            node={node}
            onClose={() => setSelected(null)}
            graph={graph}
            onNavigate={setSelected}
            onPaper={openPaper}
          />
        </div>
      </main>
    </div>
  );
}
