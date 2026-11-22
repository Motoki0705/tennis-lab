親Issue #2312 の子Issue #2324「共有memoryの4層化・既存棚卸し・構造化citations整備・SUMMARY自動生成と検査強化」を実装してください。

## 厳守事項・制約
- **`.agents/ops/review/` 本体は一切変更しないこと**（不変）。撤去するのは `.agents/memory/review/` 層のみです。
- `.agents/memory/ops-report/` 専用領域の既存契約および関連テスト（`tests/e2e/agents_ops/test_ops_report_memory.py`、`test_ops_report_retrieval.py`）を壊さないこと（`ops-report` エントリに `summary`[local path]`citations` を強制しない）。
- 静かなフォールバック・後方互換シムを置かないこと。
- テストは `tests/README.md` の「テスト作成方針」に沿って設計してから実装すること。
- 途中終了せず、実装・全検証・pre-commit・commit・push・日本語PR作成（`Closes #2324`、非draft）までこのターンで完了させ、PR番号と完全なHEADを `PARENT_RESULT` の `wait.kind="ci"`（`resume_on_success: false`）で宣言して終了すること（CIの結果待ちはこのターンで行わない）。

## 実装要件

### 1. `.agents/ops/shared_memory/` コア基盤の新設
パッケージ `.agents/ops/shared_memory/`（`PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory ...`）を作成し、役割ごとにモジュールを分割してください：

- **スキーマ・定数（単一の正本、`test_agents_docs.py` からも参照）**:
  - `SHARED_LAYERS = ("workflow", "decision", "environment", "gotcha")`（4層・固定順）
  - `LAYER_DESCRIPTIONS = {"workflow": "作業のコツ", "decision": "ユーザーの判断", "environment": "環境の癖", "gotcha": "落とし穴"}`
  - `APPLIES_TO = frozenset({"all", "claude", "codex", "gemini"})`
  - `SUMMARY_MAX_LINES = 80`
  - `SUMMARY_MAX_BYTES = 8192`
  - `INDEX_MAX_ENTRIES = 30`
  - `ENTRY_MAX_LINES = 40`
  - `SUMMARY_ITEM_MAX_CHARS = 160`
  - `STALE_DAYS = 90`
  - 共有4層エントリの frontmatter 仕様:
    - 必須キー: `title`, `summary`, `applies_to`, `source`, `created`, `last_verified`, `evidence`, `citations`
    - `title`: 非空1行文字列
    - `summary`: 非空1行文字列、`SUMMARY_ITEM_MAX_CHARS`（160文字）以内（`SUMMARY.md` に自動生成で掲載する1行要点）
    - `applies_to`: `APPLIES_TO` のいずれか
    - `source`, `evidence`: 非空文字列
    - `created`, `last_verified`: ISO日付 `YYYY-MM-DD`、`created <= last_verified <= today`
    - `citations`: リスト（repo内ファイルに直接紐付かないハードウェア・OS固有の癖などは空リスト `[]` 可）。各要素は以下のキーのみを許可（未知キー・不正型はエラー）:
      - `path`: 必須・非空文字列、repo相対パス（先頭 `[local path]`・`..` 成分・バックスラッシュ禁止）
      - `symbol`: 任意・非空文字列（対象ファイル内にそのテキスト、または `.md` のGitHub式見出しアンカーが存在すること）
      - `note`: 任意・非空文字列
    - `type` キー禁止（ディレクトリが層を表す）
    - ファイル名は `<kebab-case>.md`

- **決定的レンダラー（`render.py`）**:
  - 共有4層のエントリ群から以下を決定的に生成する純粋関数を実装（後続の専用CLI自動マージ #2325 からそのまま再利用できるよう、in-memory のエントリ集合に対しても動作する設計にすること）:
    - 各層 `<layer>[local path]`: ヘッダ＋ファイル名昇順で `- [<title>](<filename>)`、`applies_to != "all"` のとき行末に ` (<applies_to>)`
    - `.agents/memory/SUMMARY.md`: 冒頭ヘッダ・案内文＋4層それぞれの `## [<layer>](<layer>[local path]) — <説明>` の配下にファイル名昇順で `- <summary>` を1行ずつ列挙

- **バリデータと鮮度監査（`validate.py`）**:
  - repo root（および後続 #2325 で再利用できるツリー/ファイルマップ）に対して以下を検査し、構造化された問題一覧を返す:
    - `.agents/memory/` 直下は `SUMMARY.md`・4層・`ops-report/` のみ（`review/` や野良ファイル禁止）
    - 各層配下は `.md` のみ
    - 全共有エントリの frontmatter・必須キー・行数（`<= 40`）・kebab-case・`summary`長（`<= 160`）・機密/個人情報（メールアドレス・GitHubトークン・APIキー・`[local path]` 等）
    - 全 `citations` の実在検査（`path` が repo root 配下に実在し symlink で repo 外へ脱出しないこと。`symbol` 指定時はテキスト一致または `.md` の見出しアンカー一致すること）
    - 相対リンク整合（ファイル存在＋`.md` アンカー存在）
    - `SUMMARY.md` と4層の `index.md` が `render.py` の自動生成出力と完全一致すること
    - `SUMMARY.md` が `SUMMARY_MAX_LINES` 行以下かつ `SUMMARY_MAX_BYTES` バイト以下であること、`index.md` が `INDEX_MAX_ENTRIES` 件以下であること
    - 鮮度監査（`audit`）: `last_verified` から `STALE_DAYS`（90日）超過エントリ、破損 `citations`、上限使用率・超過を構造化JSONで報告

- **CLI（`cli.py` / `__main__.py`）**:
  - `PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory validate [--repo PATH]`（違反時は詳細を出力し終了コード1、正常なら0）
  - `PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory sync-index [--repo PATH] [--check]`（`SUMMARY.md` と4層の `index.md` を生成して書き込む。`--check` は書き込まず差分があれば終了コード1）
  - `PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory audit [--repo PATH] [--today YYYY-MM-DD]`（JSONで鮮度期限超過・破損citations・上限使用率を出力）

### 2. `.agents/memory/` の4層化と既存エントリ全件棚卸し
実コード・README・skill と照合しながら全件点検してください：
- **`.agents/memory/review/` 層を完全削除**し、既存3件（`focused-pull-requests.md`・`one-off-experiment-code-not-in-src.md`・`semantic-implementation-names.md`）を `decision/` へ移動・整備する（相対リンクパス・`summary`・`citations`・`last_verified` を整備）。
- **矛盾・陳腐化エントリを実コードと照合して修正または削除**:
  1. `workflow/worktree-needs-venv-symlink.md`: 現行 `AGENTS.md` と `.agents/ops/storage_tier/README.md`（`storage_tier prepare-worktree` が `outputs/`・`data/`・`ckpt/`・`.venv` のリンクを一括作成）と矛盾。すでに規則・READMEへ昇格済みなら削除、残すべき固有の観察があれば現行仕様へ修正。
  2. `gotcha/src-imported-from-cwd.md`: `outputs/` が git 追跡済み実ディレクトリという古い記述（`git ls-files outputs/` は空、`storage_tier prepare-worktree` が `outputs/` をリンクする）を現行仕様へ修正し `citations` を整備。
  3. `gotcha/main-tree-main-lags-origin.md`: 現在 `tennis-lab-git-sync.timer`（`.agents/ops/git_sync/`）が30分周期で fast-forward する現行仕様に合わせ、lock/dirty/同期間隔で遅れ得る正確な観察へ修正（または不要箇所を整理）。
  4. `gotcha/agents-ops-local-failures-not-regressions.md`: 現行の `tests/e2e/agents_ops` とコード実態を確かめ、有効なら正確に更新、陳腐化していれば削除。
- **昇格済み二重管理エントリの整理**:
  - `decision/pr-ci-required.md`: `.agents/README.md` へ昇格済み（かつログ例外の記載漏れ）→ 削除。
  - `decision/annotation-worker-effort-max.md` / `decision/keep-test-values-out-of-production-plans.md`: `.agents/skills/annotation-workflow/SKILL.md` と `worker-decision.toml` を実確認し、すでにスキル・設定で強制済みなら削除（残すべき一般判断のみ残す／統合）。
  - `decision/no-rereview-after-major-fix.md`: `.agents/README.md` と `.agents/skills/parent-session/references/review.md` を実確認し、昇格済みなら削除、固有の判断意図が残るなら現行仕様に揃えて更新。
- **その他全残存エントリ（`workflow/`・`decision/`・`environment/`・`gotcha/`）**:
  - 1件ずつ実コード・README と照合し、`summary` と適切な構造化 `citations`（repo内ファイルに紐付くものは実在 `path` と適切な `symbol` を付与、純粋なハードウェア[local path] `citations: []`）を整備し、確認できたものは `last_verified` を今日の日付へ更新する。
  - 既存の不変条件（`queue-worker-reaped-with-agent-exec-child.md` / `workers-do-not-wait-on-queue-jobs.md` が存在しないこと等）を保つこと。
- `shared_memory sync-index` で `.agents/memory/SUMMARY.md` と4層の `index.md` を自動生成結果へ同期する。

### 3. `test_agents_docs.py` 強化・単体テスト追加・最小限の文書リンク整合
- `tests/e2e/agents_ops/test_agents_docs.py`:
  - `LAYERS` を4層化し、`shared_memory` の定数・バリデータと単一正本で連携する。
  - 共有4層に対し、`summary` 必須/上限、`citations` 実在/シンボル整合、`SUMMARY.md` / 4層 `index.md` の自動生成完全一致、`SUMMARY.md` の行数（80）＋バイト数（8192）上限、`last_verified` の90日鮮度検査、`applies_to`（`gemini` 追加）を検証する。
  - `ops-report` 専用領域の既存テスト契約は維持する。
- `.agents/skills/memory/SKILL.md` と `.agents/README.md` について、`review/` 層削除によるリンク切れや「共有5層」の明白な不整合など、このPR単体で `test_agents_docs.py` や整合性を壊す箇所を最小限整える（専用CLI一本化・JIT・AutoDreamの本格記述は #2325 で行う）。
- `tests/unit/agents_ops/test_shared_memory_core.py` を新規作成し、スキーマ正常/異常、`citations`（パス脱出・存在しないファイル・存在しないシンボル・Markdown見出しアンカー一致）、`render` の決定性、`SUMMARY_MAX_BYTES` 超過、`audit` の90日期限超過、CLI（`validate` / `sync-index --check` / `audit`）を網羅的にテストする。

## 検証・PR作成
1. 以下をすべて実行し全通過を確認すること：
   - `PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory validate`
   - `PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory sync-index --check`
   - `PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory audit`
   - `.venv/bin/python -m pytest tests/unit/agents_ops/test_shared_memory_core.py tests/e2e/agents_ops/test_agents_docs.py tests/e2e/agents_ops/test_ops_report_memory.py tests/e2e/agents_ops/test_ops_report_retrieval.py tests/unit/scripts/ci_metrics/test_lanes.py -n 2 -q`
   - `.venv/bin/python -m ruff check .agents/ops/shared_memory tests/unit/agents_ops/test_shared_memory_core.py tests/e2e/agents_ops/test_agents_docs.py`
   - `.venv/bin/python -m ruff format --check .agents/ops/shared_memory tests/unit/agents_ops/test_shared_memory_core.py tests/e2e/agents_ops/test_agents_docs.py`
   - `.venv/bin/python -m mypy --follow-imports=skip .agents/ops/shared_memory tests/unit/agents_ops/test_shared_memory_core.py tests/e2e/agents_ops/test_agents_docs.py`
   - `git diff --name-status origin/main` で `.agents/ops/review/` が未変更であることを確認。
2. コミット（pre-commit通過）→ push → 日本語PRを作成（本文に `Closes #2324`、棚卸し判断一覧［修正/削除/統合した各エントリと根拠］、受入条件ごとの検証証拠とテスト実測を記載。draftにしない）。
3. 最終報告の末尾に `PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":<PR番号>,"head":"<完全な40桁HEAD>","resume_on_success":false}}` を出力して終了すること。
