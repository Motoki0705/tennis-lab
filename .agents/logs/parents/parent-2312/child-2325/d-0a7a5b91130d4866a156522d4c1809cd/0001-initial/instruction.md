親Issue #2312 の子Issue #2325「共有memory専用CLI（全4層add/update/delete/batch・機械検査・即自動マージ）とci.yml・SKILL・README更新」を実装してください。

## 厳守事項・制約
- **`.agents/ops/review/` 本体は一切変更しないこと**（不変）。
- `.agents/memory/ops-report/` 専用領域および `worker_logs` / `decision_logs` の公開契約を壊さないこと。
- 静かなフォールバック・後方互換シムを置かないこと。違反や曖昧な応答はすべて fail-closed で明示拒否すること。
- テストは `tests/README.md` の「テスト作成方針」に沿って設計してから実装すること。
- 途中終了せず、実装・全検証・pre-commit・commit・push・日本語PR作成（`Closes #2325`、非draft）までこのターンで完了させ、PR番号と完全なHEADを `PARENT_RESULT` の `wait.kind="ci"`（`resume_on_success: false`）で宣言して終了すること（CI待機はこのターンで行わない）。

## 実装要件

### 1. 共有memory専用CLIの公開・自動マージ機能（`.agents/ops/shared_memory/`）
#2324 で導入済みの `schema.py` / `markdown.py` / `render.py` / `tree.py` / `validate.py` / `cli.py` を拡張し、全4層（`workflow`・`decision`・`environment`・`gotcha`）を対象とする専用公開経路（`publish.py` 等）とサブコマンドを実装してください。設計パターンは `.agents/ops/worker_logs/publication.py` および `.agents/ops/decision_logs/publication_policy.py` の fail-closed 境界に合わせてください。

#### サブコマンドと操作モデル
- `PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory add --spec <FILE|-> [--repo PATH] [--remote origin] [--dry-run]`
- `PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory update --spec <FILE|-> [--repo PATH] [--remote origin] [--dry-run]`
- `PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory delete --layer <LAYER> --slug <SLUG> --reason <TEXT> [--repo PATH] [--remote origin] [--dry-run]`（`--spec` での指定も可）
- `PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory batch --spec <FILE|-> [--repo PATH] [--remote origin] [--dry-run]`
  - 複数の `add` / `update` / `delete` を1つのアトミックな変更として適用（後続 #2326 AutoDream の Phase 4 から Python API としても直接呼べるように、`apply_operations(...)` と `publish_operations(...)` 等の明確な公開APIを分離すること）。
  - 同一 `(layer, slug)` に対する重複・矛盾操作は適用前に明示拒否すること。
  - `add`: 既存エントリがあれば拒否。`layer`（4層のみ）・`slug`（kebab-case）・`title`・`summary`・`applies_to`・`source`・`created`（省略時今日）・`last_verified`（省略時今日）・`evidence`・`citations`・`body`（Markdown本文）から正規フロントマター付きエントリを生成。
  - `update`: 既存エントリがなければ拒否。変更するフィールド／本文を適用し、`reason` を記録。
  - `delete`: 既存エントリがなければ拒否。非空の `reason` を必須とする。

#### 公開・機械検査・即自動マージパイプライン（fail-closed）
1. **最新 `origin/main` の取得と隔離適用**:
   - リポジトリ解決は `issues.cli` / `worker_logs` 同等に remote URL から安全に解決（明示 `--github-repo` オプションも可）。
   - `git fetch <remote> main` で最新の `main` commit SHA を取得。
   - **呼び出し元の worktree やメインcheckoutの作業ツリー・HEAD・ブランチを一切変更しないこと**（一時ディレクトリに detached worktree または専用一時 index で適用し、`finally` で確実に削除・prune する）。
2. **決定的再生成と全機械検査（push・PR作成前）**:
   - 要求された操作を最新 `origin/main` ツリーに適用し、`render.render_documents()` で `.agents/memory/SUMMARY.md` と全4層の `index.md` を決定的に再生成する。
   - 変更後のツリー全体に対し、既存の `validate.validate()` を実行し、1件でも問題（レイアウト、frontmatter、`summary`、`citations` の実在/アンカー厳密照合、相対リンク symlink 境界、行数/バイト上限、機密/ローカルURL/認証値、`ops-report` 隔離、自動生成バイト一致）があれば push も PR作成も行わず非ゼロ終了で詳細を返す。
   - コミット作成後に `git diff --raw --no-renames -z <base> <head>` を検査：
     - 差分が空（no-op）なら拒否（`batch` 等で変更なしの場合の扱いを明確化）。
     - すべての変更ファイルが regular file (`100644`) かつ許可パス（`.agents/memory/SUMMARY.md` または `.agents/memory/{workflow,decision,environment,gotcha}[local path]` または4層の `index.md`）のみであること。`ops-report/`・コード・workflow・symlink・実行ビット・rename を全面拒否。
     - コミットの親が厳密に1つ（`<base>`）であること。
3. **`--dry-run`**:
   - 上記の全適用・再生成・機械検査・diff検査を通したうえで、remote への push / PR作成 / merge を一切行わず、変更ファイル一覧・操作概要・検証結果を JSON で stdout に出力して終了コード 0 を返す。
4. **PR作成・PR差分再検証・HEAD固定即自動マージ・競合リトライ**:
   - 専用ブランチ `shared-memory/<YYYYMMDDTHHMMSSZ>-<short-id>` へ push。
   - `gh api` で非draft PRを作成（`base: "main"`, `head: branch`, 日本語タイトル・日本語PR本文に変更一覧・各操作の理由・機械検査通過結果を明記）。
   - マージ直前に `GET repos/{repo}[local path]` と `GET repos/{repo}[local path]` を取得し、`base.ref == "main"`、`head.ref == branch`、`head.sha == committed_head`、`draft is False`、変更ファイル一覧と status（added/modified/removed）がローカルの実diffと完全一致し、全パスが許可範囲内であることを再検証する（逸脱があればマージせず明示エラー）。
   - `PUT repos/{repo}[local path]`（`merge_method: "merge"`, `sha: committed_head`）で即自動マージする。
   - **競合時の自動載せ直しリトライ（最大3回）**: マージが HTTP 405/409 で拒否され `origin/main` が先行更新されていた場合は、最新 `origin/main` を再 fetch → 同じ操作を新 base へ載せ直し → `SUMMARY.md`[local path]`index.md` 再生成 → 全機械検査再通過 → 同一ブランチへ `--force-with-lease` push → PR 状態・差分再検証 → 新 `sha` で merge をリトライする（最大3回）。上限超過・検査失敗・5xx曖昧応答時は明示エラーで終了する。
   - マージ成功後、作成した一時リモートブランチの削除を試み（削除失敗でもマージ成功自体は明確に記録）、PR番号・URL・head SHA・merge SHA・変更ファイル一覧・リトライ回数を JSON で stdout へ出力する。
5. **対話CLI保護（`checkout_lock` 整合）**:
   - `.agents/ops/lib/README.md#対話cliの保護` と `tests/e2e/agents_ops/test_interactive_checkout.py` を確認し、長時間 GitHub 通信・push・merge を行う `python -m shared_memory` を薄入口（`__main__.py` / `cli.py` → `checkout_lock.run_cli()` → `cli_impl.py`）に揃えるか、既存テストの契約と完全に整合させてください。

### 2. `.github/workflows/ci.yml` の限定公開例外（`paths-ignore`）
- `.github/workflows/ci.yml` の `on.push.paths-ignore` と `on.pull_request.paths-ignore` の両方に以下を追加する：
  ```yaml
      - '.agents/memory/SUMMARY.md'
      - '.agents/memory/workflow/*.md'
      - '.agents/memory/decision/*.md'
      - '.agents/memory/environment/*.md'
      - '.agents/memory/gotcha/*.md'
  ```
- **`.agents/memory/ops-report/**` は含めないこと**（5-A 対象外・既存契約維持）。
- コード・`SKILL.md`・`test_agents_docs.py`・`ci.yml` 等を含むPRでは通常どおりフルCIが起動することを維持する。

### 3. ドキュメント整備（二重管理禁止・新運用への一本化）
- `.agents/ops/shared_memory/README.md` を新規作成し、専用CLIのサブコマンド・spec形式・公開Python API（`batch` / `publish`）・機械検査・限定公開例外・競合載せ直し・終了コード・テスト実行方法の正本にする。
- `.agents/skills/memory/SKILL.md` を新運用へ更新する：
  - **更新経路の一本化**: 作業ブランチへの同梱コミットや手動の memory 単独PR作成を廃止し、共有4層の追加・更新・削除はすべて専用CLI（`python -m shared_memory add/update/delete/batch`）から即自動マージする手順へ統一。
  - **Copilot型 JIT 自己修復手順**: 作業中にエントリを読んだり、`citations` が指すコード・README・規則を変更した際、矛盾や参照切れを見つけたらその場で専用CLI（`update` / `delete`）で自己修復する具体的な手順を明記。
  - **形式・上限・自動生成**: `summary`・構造化 `citations`・`applies_to`（`all | claude | codex | gemini`）・`SUMMARY.md`[local path]`index.md` 自動生成（手編集しない）・上限値の正本（`.agents/ops/shared_memory/schema.py`）を反映。
  - **定期保守（AutoDream）**: 旧 `#1192` の記述を除去し、複合トリガーによる4フェーズ定期統合（#2326 で実装中）と `audit` による点検へ整合。
  - `ops-report` 専用領域の節は維持し、「共有5層」の旧表現を残さない。
- `.agents/README.md` を更新する：
  - 自動運用表に `[ops/shared_memory/](ops/shared_memory/README.md)` を追加。
  - 「このファイルの改訂」（限定公開例外の段落）に、共有memory 4層＋`SUMMARY.md` の限定公開例外（専用CLI・機械検査・`ci.yml` 除外）へのリンクを追加し、専用CLIや検査・workflow自体のコード変更は通常PR条件であることを明記。
  - `ops/lib/README.md` の対話CLI一覧を更新した場合、整合させる。
- `tests/e2e/agents_ops/test_agents_docs.py` の `DOCS` に `.agents/ops/shared_memory/README.md` を追加し、リンク整合が常に検査されるようにする。

### 4. テストの設計と実装
- `tests/e2e/agents_ops/test_shared_memory_publish.py`（または `tests/unit/agents_ops/test_shared_memory_publish.py`）を作成し、実一時git repo（bare origin + working checkout）＋ fake GitHub client を用いて以下を網羅テストすること：
  1. `add` / `update` / `delete` / `batch` の正常自動マージ（`SUMMARY.md`・`index.md` 自動生成、呼び出し元 worktree 未汚染、PR作成→差分再検証→HEAD固定マージ、JSON出力）。
  2. `--dry-run` が全検査を通して差分を報告しつつ remote に一切 push / PR 作成しないこと。
  3. マージ競合（HTTP 409/405）＋ `origin/main` の先行更新時に、自動再fetch→載せ直し→再生成→再検査→`--force-with-lease` push→再検証→マージ成功すること、および上限超過（3回超）で明示拒否すること。
  4. 全拒否条件（push/PR/merge が1回も呼ばれないこと）：
     - 許可外パス・`ops-report` 参照・`review` 層指定
     - frontmatter 不備・`summary` 超過・削除済み Markdown アンカー含む `citations` 破損・symlink 脱出相対リンク
     - 機密・ローカルURL・`file:[local path]`・認証値混入
     - `SUMMARY.md` 行数/バイト数超過・エントリ行数超過
     - 既存エントリへの `add`、不在エントリへの `update`[local path]`delete`、`delete` の空 reason、同一エントリへの重複 batch 操作、no-op
     - PR 作成後に GitHub 側の `files` / `head.sha` / `base.ref` / `draft` が期待とズレた場合に merge を拒否すること
  5. `.github/workflows/ci.yml` の `push.paths-ignore` と `pull_request.paths-ignore` が共有memory 4層＋`SUMMARY.md` を含み、`.agents/memory/ops-report/` を含まないことの契約テスト。

## 検証・PR作成
1. 以下をすべて実行し全通過を確認すること：
   - `PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory validate`
   - `PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory sync-index --check`
   - `PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory audit`
   - `.venv/bin/python -m pytest tests/unit/agents_ops/test_shared_memory_core.py tests/e2e/agents_ops/test_shared_memory_publish.py tests/e2e/agents_ops/test_agents_docs.py tests/e2e/agents_ops/test_interactive_checkout.py tests/e2e/agents_ops/test_ops_report_memory.py tests/unit/scripts/ci_metrics/test_lanes.py -n 2 -q`（追加したテストパスに合わせる）
   - `.venv/bin/python -m ruff check .agents/ops/shared_memory tests/unit/agents_ops tests/e2e/agents_ops/test_agents_docs.py tests/e2e/agents_ops/test_shared_memory_publish.py`
   - `.venv/bin/python -m ruff format --check .agents/ops/shared_memory tests/unit/agents_ops tests/e2e/agents_ops/test_agents_docs.py tests/e2e/agents_ops/test_shared_memory_publish.py`
   - `.venv/bin/python -m mypy --follow-imports=skip .agents/ops/shared_memory tests/e2e/agents_ops/test_agents_docs.py`（追加テストファイルも含む）
   - `.agents/ops/review/` の差分が0件であることを確認。
2. コミット（pre-commit通過）→ push → 日本語PRを作成（本文に `Closes #2325`、受入条件ごとの証拠・テスト実測・主要設計を記載。draftにしない）。
3. 最終報告の末尾に `PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":<PR番号>,"head":"<完全な40桁HEAD>","resume_on_success":false}}` を出力して終了すること。
