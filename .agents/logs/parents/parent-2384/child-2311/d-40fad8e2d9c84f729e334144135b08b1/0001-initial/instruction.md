親Issue #2384 配下の子Issue #2311 の実装担当として、専用worktreeで実装・検証・PR作成まで行ってください。

## 対象Issueと目的
- 担当子Issue: #2311（親 #2384）
- 目的:
  1. (#2311) 証拠Issue作成直後の GitHub 一覧 API 反映遅延で `parent register`（および `handoff --accept`）が失敗し、Herdr 表示先確定・掲示板名同期・`assigned` 記録が飛ばされる不具合を修正する。
  2. (`I-2157-stop-keeps-approved-contract` / #2219) 改善候補から取り出した親で自身の作業を `stop --child <親番号>` しても、追記される「作業状況・停止事項」セクションによって承認済み要件 contract の照合が壊れないようにする。

## 事前調査で特定した対象箇所（最新コードを再確認すること）
1. `.agents/ops/parent/evidence.py` (`EvidenceIssues.ensure`, `find`):
   - `ensure()` L203-L225 で `POST [local path]` の応答から `found = self._binding(response, login)` を得た後、L222 で `if self.find(parent, login) != found:` を呼んでいる。
   - `find()` は `issues?state=all&labels=parent-evidence` の一覧 API を叩くため、GitHub の検索/一覧反映遅延により新規作成直後の Issue が一覧に現れず、`evidence issue creation uncertain; reconcile GitHub before any new POST` で失敗する（本親 #2384 の初回 `register` でも実際に再現した）。
   - 修正方針: POST 応答で得た `found.number` を個別 GET（`repos/{repo}[local path]`）で直接取得・`_binding()` 照合し、一覧に他 Issue との重複が無いことも確認して紐付け（`bound`）を確定できるようにする。重複 POST 防止・既存重複検出の安全契約は厳格に維持すること。
2. `.agents/ops/parent/session.py` (`register`, `handoff`) と `.agents/ops/parent/cli_impl.py` (L1160-L1220付近):
   - `register()` は `store.save()` で担当を保存した後に `self.post({"event": "assigned", ...})` を呼ぶ。ここで例外が出ると `cli_impl.py` の `parent_action_applied: True`・`board_sync`・`sync_display(..., confirm_parent=True)` が一切実行されず、Herdr 表示先が恒久的に未確認になる。
   - `handoff --accept` でも証拠Issue作成・投稿が担当保存後に走る場合は同じ構造になるため併せて点検すること。
   - 修正方針: 担当を保存した後に記録投稿や証拠確認が失敗した場合でも、`parent_action_applied: true` と `board_sync`・`display_sync`（`confirm_parent=True` による表示先確定を含む）を確実に適用・返却する（または失敗を明示しつつ表示先確定を完了させ、再試行手順を明示する）。
   - また、担当開始（`assigned`）記録が欠落した状態を検出でき、`register` 再実行または `reconcile` で重複なく 1 件だけ回収できること（さらに `assigned` 未記録のまま気づかず進まない検出・回収設計）。
3. `.agents/ops/issues/improvement.py` (`PROGRESS`, `contract`) および `session.py` (`stop`):
   - `session.py:1596-1609` の `stop` は対象 Issue 本文へ `作業状況・停止事項` セクションを追記する。取り出し親で親自身の作業を `stop --child <親番号>` すると親本文へ `作業状況・停止事項` が追加されるが、`issues/improvement.py:21-29` の `PROGRESS` に `作業状況・停止事項` が含まれていないため、直後の `create-child` 等が `requirements changed; human decision required` で拒否される。
   - `PROGRESS`（または `contract()` の進捗扱い）を整合させ、親自身の停止記録で承認済み要件 contract が壊れないようにする。同時に要件節（目的・スコープ・完了条件等）や未知節の変更検知は絶対に緩和しないこと。

## 受入条件（#2311 の Acceptance checklist をすべて満たすこと）
1. fake gh で POST 直後の一覧に新 Issue を含めない E2E テストで、個別 GET 照合により紐付けが確定して `register` が成功し、重複 POST が起きないことを確認する。
2. `post` が失敗する fake を使った E2E テストで、担当保存後の `register`（および `handoff --accept` 相当）が `parent_action_applied: true`・`board_sync`・`display_sync` を適切に処理し、その後の `herdr_labels sync` が成功することを確認する。
3. 記録投稿失敗後に `register` 再実行または `reconcile` で `assigned` 記録の欠落を検出し、重複なく 1 件だけ投稿されるテストを追加する。
4. 取り出し親で `stop --child <親番号>` の後に `create-child` が受理される E2E テストと、停止記録の追記で承認 contract が変わらない（かつ要件変更は拒否される）ユニットテストを追加する。

## 検証・PR作成・終了ルール
- 関連テスト（`tests/unit/agents_ops/`, `tests/e2e/agents_ops/` の parent / evidence / herdr / improvement 関連）＋新規テストを全通過させる。
- pre-commit（ruff / mypy）を通してコミット・push し、PR 本文に `Closes #2311`、変更概要、受入条件ごとの検証証拠、テスト結果を日本語で記載する。
- push・PR作成が終わったら、このターン内で CI 待ちループを行わず、最終メッセージ末尾に以下の形式で 1 行だけ出力して終了する（`<PR番号>` と `<40桁HEAD SHA>` は実際の値に置換）：
`PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":<PR番号>,"head":"<40桁HEAD SHA>","resume_on_success":false}}`
