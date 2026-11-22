親Issue #2384 配下の子Issue #2388 の実装担当として、専用worktreeで実装・検証・PR作成まで行ってください。

## 対象Issueと目的
- 担当子Issue: #2388（親 #2384）
- 目的: 採用したログ公開系の改善提案4件をまとめて実装し、`worker_logs`・`decision_logs` 公開の途中中断・失敗調査・GitHub反映遅延に対する耐障害性と可観測性を高める。

## 対象4項目と事前調査箇所（最新コードを再確認すること）
1. **`I-2157-worker-log-worktree-recovery`（元提案 #2219）**:
   - 箇所: `.agents/ops/worker_logs/publication.py` L260-L296 付近（`commit()` 内の worktree 作成・確認）。
   - 現象: `worktree add` が途中で中断すると git 登録が `locked (initializing)`・空 index（全ファイル削除状態）のまま残り、再実行時に `worktree_dirty` で止まり手修復が必要になる。
   - 要件: 作成途中で中断した専用 worktree（`locked (initializing)` かつ空 index 等、途中中断と確実に判定できる状態）を検出し、固定原文を変えずに安全に作り直して公開を続行する。作成中断以外の汚れ（未知の変更・未追跡ファイル・別ブランチ等）は従来どおり `worktree_dirty` / `worktree_changed` で厳格に拒否する。
2. **`I-1968-worker-log-failure-diagnostic`（元提案 #2137）**:
   - 箇所: `.agents/ops/worker_logs/publication.py` L600-L646 付近、および `blocked` 記録の再実行・status 表示箇所（`publication.py` / `service.py` / `reconciliation.py`）。
   - 現象: `PublicationError` 以外の例外が汎用 `publication_failed` に変換された際に `row["diagnostic"]` が空のまま残り、原因特定にコードトレースが必要だった。また `blocked` 記録で `--retry` が必要なことが status や CLI 再実行出力に表示されない。
   - 要件: 汎用 `publication_failed` に変換される失敗時にも、例外の種類と秘匿情報を含まない要約を private journal の `diagnostic` に保存する。また `--retry` が必要な `blocked` 記録では status と CLI 出力に `--retry` の必要（次の操作）を明示する。
3. **`I-1900-journal-unavailable-identity`（元提案 #2132）**:
   - 箇所: `.agents/ops/worker_logs/reconciliation.py` L276-L294 付近および `service.py` / `status`（`remaining_publications`）。
   - 現象: 一括照合（`publish-worker-log --all-parents` 等）で journal を読めなかった項目が `{"state": "blocked", "error": "journal_unavailable", "saved_to_main": False}` だけを返し、どの親・名前空間の記録か特定できなかった。
   - 要件: `journal_unavailable` の各行（一括照合出力・`remaining_publications`）に、判明する親番号・名前空間のパスまたは digest・読めなかった理由を含める。識別できない場合はその旨を明示し、空の識別子を残件なし扱いしない。
4. **`I-1900-decision-pr-head-lag`（元提案 #2132）**:
   - 箇所: `.agents/ops/decision_logs/publication.py` L426-L454（`reconcile_pr`）および `.agents/ops/worker_logs/publication.py` L331-L347（`pr()` の HEAD 照合）。
   - 現象: push 直後の GitHub PR HEAD 反映遅延で古い HEAD や空の HEAD が返ると即 `pr_identity_changed` で停止し、手動 `--retry` が必要になっていた。
   - 要件: 観測した HEAD が未知のとき、上限付き（短い待機・テストで clock/sleep を注入または高速化できる形）で再取得し、既知の revision に一致すれば続行する。上限後も一致しない未知 HEAD は従来どおり `pr_identity_changed` で止め、batch/row を改変しない。PR 同一性検査自体は一切緩和しない。

## 受入条件（#2388 の Acceptance checklist をすべて満たすこと）
1. `worktree add` の途中中断を再現して公開 CLI 再実行で公開が完了するテストと、作成中断以外の汚れは従来どおり `worktree_dirty` で拒否するテスト。
2. 汎用 `publication_failed` で止まった記録に原因要約が `diagnostic` として残るテストと、`--retry` が必要な `blocked` 記録で status / 再実行時に次操作が表示されるテスト。
3. journal を読めない名前空間を含む一括照合・`remaining_publications` のテストで、対象の名前空間（親番号・パス[local path]）と理由が特定でき、識別不能時も空識別子を残件なし扱いしないこと。
4. push 直後に古い HEAD や空の HEAD を返す fake GitHub で、`decision_logs`・`worker_logs` ともに上限付き再取得で既知 revision に一致すれば続行し、不一致なら `pr_identity_changed` で止めて batch を変えないテスト。

## 検証・PR作成・終了ルール
- 関連テスト（`tests/unit/agents_ops/test_worker_logs.py`, `tests/e2e/agents_ops/` の worker_logs / decision_logs 関連）＋新規テストを全通過させる。
- pre-commit（ruff / mypy）を通してコミット・push し、PR 本文に `Closes #2388`、変更概要、受入条件ごとの検証証拠、テスト結果を日本語で記載する。
- push・PR作成が終わったら、このターン内で CI 待ちループを行わず、最終メッセージ末尾に以下の形式で 1 行だけ出力して終了する（`<PR番号>` と `<40桁HEAD SHA>` は実際の値に置換）：
`PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":<PR番号>,"head":"<40桁HEAD SHA>","resume_on_success":false}}`
