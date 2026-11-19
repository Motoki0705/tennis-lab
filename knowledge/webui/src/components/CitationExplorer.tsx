"use client";
import { useEffect, useRef } from "react";
import {
  CITATION_ROLES,
  DECISION_LABELS,
  IDENTIFICATION_LABELS,
  ROLE_LABELS,
  SEARCH_LABELS,
  TEXT_LABELS,
} from "@/lib/citation-types";
import type {
  CitationLibrary,
  CitationLoadResult,
  CitationPaper,
  CitationReference,
  CitationReadLibrary,
  CitationRole,
  CitationSearch,
} from "@/lib/citation-types";
import {
  citationRelations,
  identificationPending,
  rankCitations,
  searchReferences,
} from "@/lib/citations";
import { CitationMap } from "./CitationMap";

function CitationError({ children }: { children: React.ReactNode }) {
  return (
    <div className="citation-error" role="alert" aria-label="引用探索のエラー">
      <h3>引用探索を表示できません</h3>
      <div>{children}</div>
    </div>
  );
}
function CitationDiagnostics({
  library,
  onOpen,
}: {
  library: CitationReadLibrary;
  onOpen: (id: string) => void;
}) {
  if (!library.documentErrors.length) return null;
  return (
    <section
      className="citation-error"
      role="alert"
      aria-label="引用文書のエラー"
    >
      <h3>引用文書のエラー</h3>
      <p>
        影響を受けない探索は閲覧できます。エラーになった文書は表示・集計・Papers結合に使いません。
      </p>
      <ul>
        {library.documentErrors.map((error, i) => (
          <li key={i}>
            {error.files.map((file) => (
              <div key={file}>
                <code>{file}</code>
              </div>
            ))}
            <p>{error.reason}</p>
          </li>
        ))}
      </ul>
      {library.searchErrors.length > 0 && (
        <>
          <h4>閲覧できない探索 ({library.searchErrors.length})</h4>
          <ul>
            {library.searchErrors.map((search) => (
              <li key={search.id}>
                <button
                  className="related-link"
                  onClick={() => onOpen(search.id)}
                >
                  エラー: {search.id}
                </button>
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}
export function CitationSearchList({
  result,
  onOpen,
}: {
  result: CitationLoadResult;
  onOpen: (id: string) => void;
}) {
  if (!result.ok) return <CitationError>{result.error}</CitationError>;
  const { library } = result;
  return (
    <section className="citation-searches" aria-label="保存済み引用探索">
      <div className="section-caption">
        <h3>引用探索</h3>
        <span>
          閲覧可能 {library.searches.length} 件 · エラー{" "}
          {library.searchErrors.length} 件 · 閲覧専用
        </span>
      </div>
      <CitationDiagnostics library={library} onOpen={onOpen} />
      {!library.searches.length ? (
        <p className="empty-state">
          {library.documentErrors.length
            ? "閲覧可能な探索はありません。文書のエラーを確認してください。"
            : "探索なし — 保存された引用探索はありません。"}
        </p>
      ) : (
        <div className="citation-search-grid">
          {library.searches.map((s) => (
            <button
              key={s.id}
              className="citation-search-card"
              onClick={() => onOpen(s.id)}
            >
              <strong>
                起点: {library.papers.find((p) => p.id === s.root_id)!.title}
              </strong>
              <span>
                {s.roles.map((r) => ROLE_LABELS[r]).join(" / ")} · 深さ{" "}
                {s.max_depth}
              </span>
              <span className={`citation-search-state ${s.status}`}>
                {SEARCH_LABELS[s.status]}
              </span>
              {s.stop_reason && <span>中断理由: {s.stop_reason}</span>}
              <small>
                {s.id} · {s.created_at}
              </small>
              <span>引用マップを開く →</span>
            </button>
          ))}
        </div>
      )}
    </section>
  );
}

function Evidence({ reference }: { reference: CitationReference }) {
  return (
    <div className="citation-evidence">
      <div className="citation-badges">
        <b>{reference.ref_id}</b>
        <span>{IDENTIFICATION_LABELS[reference.identification_status]}</span>
        <span>{DECISION_LABELS[reference.decision_status]}</span>
        {reference.roles.map((r) => (
          <span key={r}>{ROLE_LABELS[r]}</span>
        ))}
      </div>
      <p className="citation-original">{reference.citation}</p>
      <dl>
        <dt>判断理由</dt>
        <dd>{reference.reason}</dd>
        <dt>根拠の位置</dt>
        <dd>
          {reference.evidence.length ? (
            <ul>
              {reference.evidence.map((e) => (
                <li key={e}>{e}</li>
              ))}
            </ul>
          ) : (
            "根拠の記録なし"
          )}
        </dd>
      </dl>
      {reference.lookup && (
        <details className="citation-lookup">
          <summary>照会に使う情報</summary>
          <dl>
            <dt>題名</dt>
            <dd>{reference.lookup.title || "未記録"}</dd>
            <dt>年 / 著者</dt>
            <dd>
              {reference.lookup.year ?? "年不明"} /{" "}
              {reference.lookup.authors.join(", ") || "著者不明"}
            </dd>
            <dt>照会元リンク</dt>
            <dd>
              {reference.lookup.links.length
                ? reference.lookup.links.map((link) => (
                    <div key={link}>
                      <a href={link} target="_blank" rel="noreferrer">
                        {link}
                      </a>
                    </div>
                  ))
                : "リンクの記録なし"}
            </dd>
          </dl>
        </details>
      )}
    </div>
  );
}

function CitationDetail({
  paper,
  library,
  search,
  role,
  onSelect,
  onClose,
  onPaper,
}: {
  paper: CitationPaper;
  library: CitationLibrary;
  search: CitationSearch;
  role: CitationRole | "";
  onSelect: (id: string) => void;
  onClose: () => void;
  onPaper: (id: string) => void;
}) {
  const heading = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    const previous = document.activeElement;
    return () => {
      if (previous instanceof HTMLElement && previous.isConnected)
        previous.focus({ preventScroll: true });
    };
  }, []);
  useEffect(() => {
    heading.current?.focus();
  }, [paper.id, role]);
  useEffect(() => {
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [onClose]);
  const rows = searchReferences(library, search).filter(
    (r) => !role || r.reference.roles.includes(role),
  );
  const incoming = rows.filter((r) => r.target?.id === paper.id);
  const outgoing = rows.filter((r) => r.source.id === paper.id);
  const expanded = search.generations.some((g) =>
    g.expanded_ids.includes(paper.id),
  );
  return (
    <aside className="citation-detail" aria-label="引用論文の詳細">
      <button className="text-button citation-close" onClick={onClose}>
        閉じる (Esc)
      </button>
      <p className="eyebrow">CITATION / {paper.year ?? "年不明"}</p>
      <h3 ref={heading} tabIndex={-1}>
        {paper.title}
      </h3>
      <p>{paper.authors.join(", ") || "著者不明"}</p>
      <div className="citation-badges">
        <span>{TEXT_LABELS[paper.text_status]}</span>
        <span>{expanded ? "この探索で展開" : "この探索では未展開"}</span>
      </div>
      {paper.text_reason && <p>{paper.text_reason}</p>}
      <p>{paper.summary || "研究要約の記録なし"}</p>
      <dl>
        {Object.entries(paper.external_ids).map(([key, value]) => (
          <div key={key}>
            <dt>{key}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
      <div className="paper-actions">
        <a href={paper.source} target="_blank" rel="noreferrer">
          一次資料を開く ↗
        </a>
        {library.registeredIds[paper.id] && (
          <button onClick={() => onPaper(library.registeredIds[paper.id])}>
            Papersの詳細へ →
          </button>
        )}
      </div>
      <p className="citation-note">
        {role
          ? `${ROLE_LABELS[role]}の引用を表示中。`
          : "すべての役割を表示中。"}{" "}
        範囲はこの探索の展開済み引用元です。
      </p>
      <section aria-label="引用元と判定根拠">
        <h4>引用元 {new Set(incoming.map((r) => r.source.id)).size} 論文</h4>
        {!incoming.length && <p>この条件での引用元なし。</p>}
        {incoming.map((r) => (
          <article
            className="citation-incoming"
            key={`${r.source.id}/${r.reference.ref_id}`}
          >
            <button
              className="related-link"
              onClick={() => onSelect(r.source.id)}
            >
              {r.source.title} → {paper.title}
            </button>
            <Evidence reference={r.reference} />
          </article>
        ))}
      </section>
      <section aria-label="この論文の参考文献">
        <h4>この論文が引用した文献</h4>
        {!expanded ? (
          <>
            <p>この探索では未展開です。</p>
            {identificationPending(paper) && (
              <>
                <p className="citation-note">
                  照会が完了していない参考文献です。展開・集計には含めず、役割の絞り込みにかかわらず表示します。
                </p>
                {paper.references
                  .filter((r) =>
                    ["pending", "unverified"].includes(r.identification_status),
                  )
                  .map((r) => (
                    <Evidence key={r.ref_id} reference={r} />
                  ))}
              </>
            )}
          </>
        ) : !outgoing.length ? (
          <p>この条件での参考文献なし。</p>
        ) : (
          outgoing.map((r) => (
            <article key={r.reference.ref_id}>
              {r.target && (
                <button
                  className="related-link"
                  onClick={() => onSelect(r.target!.id)}
                >
                  {r.target.title} →
                </button>
              )}
              <Evidence reference={r.reference} />
            </article>
          ))
        )}
      </section>
    </aside>
  );
}

export function CitationExplorer({
  result,
  searchId,
  roleValue,
  selected,
  mode,
  onSearch,
  onRole,
  onSelect,
  onMode,
  onPaper,
}: {
  result: CitationLoadResult;
  searchId: string;
  roleValue: string;
  selected: string;
  mode: string;
  onSearch: (id: string) => void;
  onRole: (role: string) => void;
  onSelect: (id: string) => void;
  onMode: (mode: string) => void;
  onPaper: (id: string) => void;
}) {
  if (!result.ok) return <CitationError>{result.error}</CitationError>;
  const library = result.library;
  if (!searchId)
    return (
      <div className="citation-explorer">
        <CitationSearchList result={result} onOpen={onSearch} />
      </div>
    );
  const search = library.searches.find((s) => s.id === searchId);
  if (!search) {
    const failed = library.searchErrors.find((s) => s.id === searchId);
    return (
      <div className="citation-explorer">
        <CitationSearchList result={result} onOpen={onSearch} />
        <CitationError>
          {failed
            ? `この探索はエラーのため閲覧できません: ${searchId}`
            : `指定された探索が見つかりません: ${searchId}`}
          {failed && (
            <>
              <p>
                <code>{failed.file}</code>
              </p>
              <ul>
                {failed.errors.map((error, i) => (
                  <li key={i}>
                    {error.files.join(", ")}: {error.reason}
                  </li>
                ))}
              </ul>
            </>
          )}
        </CitationError>
      </div>
    );
  }
  if (roleValue && !CITATION_ROLES.includes(roleValue as CitationRole))
    return <CitationError>不正な役割: {roleValue}</CitationError>;
  if (!["map", "ranking"].includes(mode))
    return <CitationError>不正な引用表示: {mode}</CitationError>;
  const role = roleValue as CitationRole | "";
  const root = library.papers.find((p) => p.id === search.root_id)!;
  const rows = searchReferences(library, search);
  const relations = citationRelations(rows, role);
  const scopeIds = new Set([
    search.root_id,
    ...rows.flatMap((r) => [r.source.id, ...(r.target ? [r.target.id] : [])]),
  ]);
  const paper = library.papers.find(
    (p) => p.id === selected && scopeIds.has(p.id),
  );
  const expanded = new Set(search.generations.flatMap((g) => g.expanded_ids));
  const reached = new Set(search.generations.flatMap((g) => g.paper_ids));
  const waitingPapers = library.papers.filter(
    (p) => reached.has(p.id) && !expanded.has(p.id) && identificationPending(p),
  );
  return (
    <div className="citation-explorer">
      <CitationDiagnostics library={library} onOpen={onSearch} />
      <div className="citation-controls">
        <label>
          保存済み探索
          <select
            aria-label="保存済み探索"
            value={search.id}
            onChange={(e) => onSearch(e.target.value)}
          >
            {library.searches.map((s) => (
              <option key={s.id} value={s.id}>
                {library.papers.find((p) => p.id === s.root_id)!.title} · {s.id}{" "}
                · {SEARCH_LABELS[s.status]}
              </option>
            ))}
          </select>
        </label>
        <label>
          引用の役割
          <select
            aria-label="引用の役割"
            value={role}
            onChange={(e) => onRole(e.target.value)}
          >
            <option value="">すべての役割</option>
            {CITATION_ROLES.map((r) => (
              <option key={r} value={r}>
                {ROLE_LABELS[r]}
              </option>
            ))}
          </select>
        </label>
      </div>
      <header className="citation-header">
        <p className="eyebrow">CITATION EXPLORATION · 閲覧専用</p>
        <h2>起点: {root.title}</h2>
        <p>
          展開条件: {search.roles.map((r) => ROLE_LABELS[r]).join(" / ")} ·
          最大深さ {search.max_depth} · 展開 {expanded.size} 論文
        </p>
        <p className="citation-note">
          {search.created_at} ·
          表示・集計は展開済み論文の参考文献。役割の絞り込みは保存済み探索の範囲を変更しません。
        </p>
      </header>
      <section
        className={`citation-search-state ${search.status}`}
        aria-label="探索の状態"
      >
        <strong>{SEARCH_LABELS[search.status]}</strong>
        {search.status === "interrupted" && (
          <>
            <p>中断理由: {search.stop_reason}</p>
            <p className="citation-note">
              展開を完了した引用元の参考文献を表示・集計しています。未展開・照会未完了の文献は集計に含めません。
            </p>
          </>
        )}
      </section>
      <nav className="citation-modes" aria-label="引用表示切替">
        <button aria-pressed={mode === "map"} onClick={() => onMode("map")}>
          引用マップ
        </button>
        <button
          aria-pressed={mode === "ranking"}
          onClick={() => onMode("ranking")}
        >
          役割別ランキング
        </button>
      </nav>
      <div className={`citation-layout ${selected ? "with-detail" : ""}`}>
        <div className="citation-results">
          {mode === "map" ? (
            <>
              <CitationMap
                root={root}
                relations={relations}
                selected={selected}
                onSelect={onSelect}
              />
              {!relations.length && (
                <p className="empty-state">
                  この条件での展開済み引用関係はありません。
                </p>
              )}
              <details className="citation-relations">
                <summary>引用関係の一覧 ({relations.length})</summary>
                {relations.map((rel) => (
                  <article key={rel.id}>
                    <button onClick={() => onSelect(rel.source.id)}>
                      {rel.source.title}
                    </button>
                    <span> → </span>
                    <button
                      onClick={() => onSelect(rel.target?.id ?? rel.source.id)}
                    >
                      {rel.target?.title ?? "未同定"}
                    </button>
                    <p>
                      {rel.roles.length
                        ? rel.roles.map((r) => ROLE_LABELS[r]).join(" / ")
                        : "役割未確定"}{" "}
                      · 参考文献{" "}
                      {rel.references.map((r) => r.ref_id).join(", ")}
                    </p>
                  </article>
                ))}
              </details>
            </>
          ) : (
            <div className="citation-rankings">
              <p className="citation-note">
                件数 =
                この探索で展開済みの論文のうち、その役割で引用した異なる論文数。同じ引用元の複数行は1件です。件数を選ぶと引用元・判断理由・根拠へ移れます。
              </p>
              {(role ? [role] : CITATION_ROLES).map((r) => (
                <section key={r} aria-label={`${ROLE_LABELS[r]}ランキング`}>
                  <h3>{ROLE_LABELS[r]}</h3>
                  <table>
                    <thead>
                      <tr>
                        <th>引用された論文</th>
                        <th>引用元の論文数</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rankCitations(rows, r).map((rank) => (
                        <tr key={rank.paper.id}>
                          <td>
                            <button onClick={() => onSelect(rank.paper.id)}>
                              {rank.paper.title}
                            </button>
                          </td>
                          <td>
                            <button
                              className="citation-count"
                              aria-label={`${rank.paper.title} の${ROLE_LABELS[r]} ${rank.count}件の根拠`}
                              onClick={() => {
                                onRole(r);
                                onSelect(rank.paper.id);
                              }}
                            >
                              {rank.count} 件 →
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {!rankCitations(rows, r).length && <p>該当する引用なし。</p>}
                </section>
              ))}
            </div>
          )}
          {waitingPapers.length > 0 && (
            <section className="citation-pending" aria-label="照会未完了の論文">
              <h3>照会未完了の論文 ({waitingPapers.length})</h3>
              <p className="citation-note">
                本文や役割の記録を保持し、同定が完了するまで展開・集計に含めません。役割の絞り込みにかかわらず表示します。
              </p>
              {waitingPapers.map((p) => (
                <article key={p.id}>
                  <button
                    className="related-link"
                    onClick={() => onSelect(p.id)}
                  >
                    {p.title}
                  </button>
                  <span className="citation-badges">
                    {TEXT_LABELS[p.text_status]} · この探索では未展開
                  </span>
                  {p.references
                    .filter((r) =>
                      ["pending", "unverified"].includes(
                        r.identification_status,
                      ),
                    )
                    .map((r) => (
                      <Evidence key={r.ref_id} reference={r} />
                    ))}
                </article>
              ))}
            </section>
          )}
          <section className="citation-issues" aria-label="探索の問題箇所">
            <h3>問題箇所 ({search.issues.length})</h3>
            <p className="citation-note">
              役割の絞り込みにかかわらず、展開済み引用元の全問題箇所を表示します。
            </p>
            {!search.issues.length && <p>記録された問題箇所なし。</p>}
            {search.issues.map((issue) => {
              const source = library.papers.find(
                (p) => p.id === issue.paper_id,
              )!;
              const ref = source.references.find(
                (r) => r.ref_id === issue.ref_id,
              );
              return (
                <article key={`${issue.paper_id}/${issue.ref_id}`}>
                  <button
                    className="related-link"
                    onClick={() => onSelect(source.id)}
                  >
                    {source.title}
                    {issue.ref_id ? ` / ${issue.ref_id}` : ""}
                  </button>
                  {ref ? (
                    <Evidence reference={ref} />
                  ) : (
                    <>
                      <strong>{TEXT_LABELS[source.text_status]}</strong>
                      <p>{source.text_reason}</p>
                    </>
                  )}
                </article>
              );
            })}
          </section>
        </div>
        {selected && !paper && (
          <CitationError>
            この探索の範囲に選択論文がありません: {selected}
            <button onClick={() => onSelect("")}>選択を解除</button>
          </CitationError>
        )}
        {paper && (
          <CitationDetail
            paper={paper}
            library={library}
            search={search}
            role={role}
            onSelect={onSelect}
            onClose={() => onSelect("")}
            onPaper={onPaper}
          />
        )}
      </div>
    </div>
  );
}
