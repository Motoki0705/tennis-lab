親Issue #2312 の子Issue #2326「複合トリガー型AutoDream（全4層4フェーズ定期統合）とops_control・systemd連携の実装」を実装してください。

## 厳守事項・制約
- **`.agents/ops/review/` 本体は一切変更しないこと**（不変）。
- `.agents/memory/ops-report/` 専用領域は対象外（5-A）。AutoDream が専用領域を読み書きしないこと。
- 実ホストでの `spin ops on memory-autodream` / `install` による systemd unit 有効化は行わないこと（merge 後に人間が実施）。
- 静かなフォールバック・失敗の握りつぶしを禁止する（`.agents/README.md`: 利用枠不足や検証失敗等で自動運用が失敗したときは黙って skip せず状態ファイルへ記録し非ゼロ終了すること）。
- テストは `tests/README.md` の「テスト作成方針」に沿って設計してから実装すること。
- 途中終了せず、実装・全検証・pre-commit・commit・push・日本語PR作成（`Closes #2326`、非draft）までこのターンで完了させ、PR番号と完全なHEADを `PARENT_RESULT` の `wait.kind="ci"`（`resume_on_success: false`）で宣言して終了すること（CI待機はこのターンで行わない）。

## 実装要件

### 1. 複合トリガー型 AutoDream 本体（`.agents/ops/shared_memory/autodream*.py`）
#2324・#2325 で整備済みの `.agents/ops/shared_memory/`（`schema.py` / `validate.py` / `operations.py` / `publish.py` / `git.py`）の上に、Claude Code 型 4フェーズ定期統合（**全4層 `workflow`・`decision`・`environment`・`gotcha` 同条件の自動修正・自動削除［2-B］**）を実装してください。

#### 状態ファイルと排他ロック
- 状態ディレクトリ: `${XDG_STATE_HOME:-[local path]}[local path]`（テスト用に `--state-dir` で差し替え可能）。
- 状態ファイル: `autodream_state.json`（`ops_control` の `Target.status_file = "shared-memory/autodream_state.json"` としてそのまま読める形式。原子書込 `NamedTemporaryFile` + `fsync` + `replace`、バージョン明記、破損 JSON や未知 schema は黙って初期化せず明示エラー）。
  - 保存内容: `schema_version`, `last_run_at`, `last_completed_at`（実際に統合完了または no-op 確認した時刻）, `last_consolidated_sha`（前回統合基準の `main` commit SHA）, `last_outcome`（`merged` / `noop` / `skipped_trigger` / `dry_run` / `failed`）, `last_trigger`（発火理由・計測値）, `last_pr` / `last_merge_sha`, `last_error`（失敗時のみ詳細、成功時は `null`）。
- 排他ロック: `autodream.lock`（`fcntl.flock` 非ブロッキング排他）。同時起動時は黙って無視せず明示的に排他競合を記録/通知して終了する。

#### 複合トリガー判定（決定的・境界テスト必須）
最新 `origin/main` を fetch（呼び出し元 checkout を汚さない）したうえで、以下の基準で実行可否を判定する（初期値は定数として1か所に定義し CLI / 引数で上書きテスト可能にすること）：
1. **強制トリガー**: `--force` 指定時は無条件で実行。
2. **即時異常トリガー**（経過日数や更新数にかかわらず即実行）:
   - `audit` で以下を1件以上検知した場合:
     - 破損 `citations`（参照先ファイル/シンボル/アンカー欠損）
     - 上限超過（`SUMMARY.md` 80行[local path]、`index.md` 30件、エントリ40行）
     - 90日期限超過エントリ（`STALE_DAYS` 超過）
     - 生成文書（`SUMMARY.md` / 各層 `index.md`）のずれ（`plan_sync` に実差分あり）
3. **通常定期の複合トリガー**:
   - 初回実行（`last_completed_at` / `last_consolidated_sha` 未記録）の場合は、対象履歴が基準以上または初回統合対象として明確に定義（例: 前回記録なしの場合は7日条件を満たしたものとみなすか、初回から通常定期条件の閾値を明確に仕様化・文書化・テストすること）。
   - 前回統合完了（`last_completed_at`）から **7日（168時間）以上経過** **かつ** 以下のいずれかを満たす場合：
     - 前回統合基準 `last_consolidated_sha..base` の **`main` のマージ/コミット数（first-parent 等で明確定義）が 10 件以上**
     - 同区間における **共有memory 4層＋`SUMMARY.md` の更新コミット数が 3 件以上**
4. **未達時**: LLM ランナーも公開も呼ばず、`last_outcome: "skipped_trigger"` と計測値（経過時間・`main` コミット数・共有memory更新数・audit件数）を状態へ記録し、JSON を出力して終了コード 0 で終了する（このとき `last_completed_at` と `last_consolidated_sha` は進めない）。

#### 4フェーズ統合パイプライン
1. **Phase 1: Orient（現状把握）**:
   - 最新 `origin/main` の `base` SHA を固定し、全4層のエントリ一覧・生バイト・各エントリの **生バイト SHA-256**・`SUMMARY.md`・各層 `index.md` を読み込む（`ops-report/` には一切触れない）。
2. **Phase 2: Gather Signal（シグナル収集）**:
   - `validate.audit(tree, today=today)` で期限超過・破損 `citations`・上限使用率・問題一覧を取得。
   - `validate.plan_sync(tree, today=today)` で生成文書ずれを検知。
   - `last_consolidated_sha..base` の git 履歴から、共有memoryエントリの `citations[].path` が変更・削除されたエントリ一覧、および新規昇格候補の手がかり（`AGENTS.md`・`.agents/README.md`・`.agents/skills/**` の変更パス一覧）を構造化シグナルとしてまとめる。
3. **Phase 3: Consolidate（統合・修正・削除 — 全4層 2-B 同条件）**:
   - 全4層（`workflow`・`decision`・`environment`・`gotcha`）を同じ基準で自動修正・自動削除・統合・相対日時正規化の対象とする（`decision/` を特別に除外しない）。
   - 統合判断エンジンはプラガブルな抽象（DI可能な runner）とし、既定の本番実装は本リポジトリの自動運用契約に従う：
     - `.agents/ops/lib/agents.conf` に専用プロファイル `MEMORY_AUTODREAM_CODEX_MODEL="gpt-6.2-sol"`, `MEMORY_AUTODREAM_CODEX_EFFORT="high"`, `MEMORY_AUTODREAM_CODEX_SERVICE_TIER="default"` を追加（既存プロファイルや関連テスト `test_agent_exec.py` 等と完全整合すること）。
     - 固定した `base` commit を読み取り専用の使い捨て一時チェックアウト（メインcheckoutを汚さず `finally` で確実に削除、`git worktree` 登録を残さない `git archive` 展開等）に展開し、`.agents/ops/lib/agent_exec.sh --profile memory_autodream --mode readonly --json-schema <schema>` で構造化統合計画（JSON Schema 厳格固定）を生成する。
   - 統合計画の JSON Schema は厳格に検証（未知キー拒否）：
     - `summary`: 統合判断の短い要約
     - `operations`: `shared_memory` の `batch` と同じ `add` / `update` / `delete` 操作配列（0件可）。
     - `repair_index`: `bool`（操作0件時に生成文書ずれのみ修復するか）
     - **安全性（楽観ロック強制）**: Phase 3 が生成したすべての `update` / `delete` 操作に対し、Phase 1 で観測した対象エントリの生バイト SHA-256 を `expected_sha256` として付与（すでに指定されている場合は Phase 1 の実ハッシュと一致することを検証）する。これにより実行中に別セッションが更新・再検証したエントリを誤削除・上書きすることを構造的に防ぐ！
     - 計画は公開前に `apply_operations` / `validate` で事前検証し、schema違反・`ops-report` 干渉・不正パス・`citations` 破損・機密混入等があれば一切公開せず `failed` を状態ファイルに記録して非ゼロ終了する。
4. **Phase 4: Prune & Index（圧縮・索引再構築・専用CLI公開）**:
   - `operations` が非空なら `shared_memory.publish.publish_operations(repo, operations, remote=remote, dry_run=dry_run, today=today)` で公開・即自動マージする。
   - `operations` が空で、生成文書に実差分があり `repair_index`（または異常検知修復）対象なら `shared_memory.publish.publish_index_repair(repo, reason=..., remote=remote, dry_run=dry_run, today=today)` で復旧公開する。
   - どちらも変更不要（かつ生成文書ずれなし・残存異常なし）なら PR を作らず `last_outcome: "noop"` として `last_completed_at` / `last_consolidated_sha` を更新して正常終了する。
   - `--dry-run` 時は公開・マージ・状態ファイルの `last_completed_at` 更新を行わず、トリガー判定・シグナル・計画・dry-run 検証結果を JSON で出力する。
   - いずれのフェーズでも例外・ランナー失敗・検証拒否・公開失敗が起きたら、黙って skip せず `autodream_state.json` に `last_outcome: "failed"` と `last_error` を保存（`last_completed_at` / `last_consolidated_sha` は進めない）し、非ゼロ終了コードで返す。

### 2. 有限運用ジョブ入口・`checkout_job.py` 契約・systemd・`ops_control` 連携
- `.agents/ops/lib/README.md#既存cliと新規jobの接続` および `.agents/ops/ops_control/README.md#対象の追加` の契約に完全準拠すること：
  - 有限運用ジョブの公開入口（例: `.agents/ops/shared_memory/autodream_job.py`、先頭で project import 前に `require_checkout_job()` を呼ぶ）を作成し、`checkout_job.py --repo "$PWD" -- .venv/bin/python .agents/ops/shared_memory/autodream_job.py ...` 経由でのみ起動を許可する（直接起動は終了コード 75 で拒否）。
  - 既存の対話CLI `python -m shared_memory` 側にも状態確認用の読み取りコマンド（例: `autodream-status` / `autodream-check`）があると便利なら追加可（ただし有限job本体は `checkout_job.py` 経由の契約を厳守）。
- `.agents/ops/shared_memory/systemd/tennis-lab-memory-autodream.service` と `.timer` を作成する：
  - `service`: `Type=oneshot`, `WorkingDirectory=%h/projects/tennis-lab`, `Environment=PYTHONPATH=%h/projects/tennis-lab/.agents/ops`, `ExecStart=%h/projects/tennis-lab/.venv/bin/python %h/projects/tennis-lab/.agents/ops/checkout_job.py --repo %h/projects/tennis-lab -- %h/projects/tennis-lab/.venv/bin/python %h/projects/tennis-lab/.agents/ops/shared_memory/autodream_job.py run`, `KillMode=control-group`, `TimeoutStopSec=60`。
  - `timer`: 日次実行（例: `OnCalendar=*-*-* 05:00:00 Asia/Tokyo`, `Persistent=true`, `AccuracySec=1s`。毎日起動しても複合トリガー未達なら軽量な `skipped_trigger` で即終了し、異常検知時は当日中に自動修復できる！）。
- `.agents/ops/ops_control/targets.py` に保守ターゲットを追加する：
  - `"memory-autodream": timer_target("memory-autodream", "tennis-lab-memory-autodream", status_file="shared-memory/autodream_state.json")`
  - **`agents` グループには含めないこと**（`ops_control/README.md`: 保守ジョブを `agents` へ追加しない）。

### 3. ドキュメント整備（二重管理禁止・リンク整合）
- `.agents/ops/shared_memory/README.md`: AutoDream の複合トリガー条件・4フェーズ・全4層 2-B 方針・状態ファイル・`checkout_job.py` 実行方法・systemd / `spin ops` 導入手順（merge 後に人間が `spin ops install memory-autodream` / `spin ops on memory-autodream`）を正本として追記。
- `.agents/skills/memory/SKILL.md`: `#2326で実装中` の暫定表記を外し、`.agents/ops/shared_memory/README.md` の AutoDream 節への正確なリンクへ更新。
- `.agents/ops/ops_control/README.md`: 対象表に `memory-autodream` を1行追加。
- `.agents/ops/lib/README.md`: 有限運用ジョブの入口表に `shared_memory/autodream_job.py` を1行追加。
- `.agents/README.md`: 必要に応じて `ops/shared_memory/` の説明を最新化。

### 4. テストの設計と実装
- `tests/unit/agents_ops/test_shared_memory_autodream.py` および/または `tests/e2e/agents_ops/test_shared_memory_autodream.py` を新規作成し、以下を網羅検証すること：
  1. **複合トリガーの全境界条件**:
     - 経過日数不足（6日23時間）＋ `main` 15件 → `skipped_trigger`
     - 7日以上経過＋ `main` 9件＋共有memory更新 2件 → `skipped_trigger`
     - 7日以上経過＋ `main` 10件＋共有memory更新 0件 → 発火
     - 7日以上経過＋ `main` 3件＋共有memory更新 3件 → 発火
     - 経過日数不足でも、破損 `citations` / 90日期限超過 / 上限超過 / 生成文書ずれ のいずれかがあれば即時発火
     - `--force` 指定で条件未達でも発火
  2. **4フェーズ実行と全4層 2-B 統合（fake runner + 実一時 git repo + fake GitHub）**:
     - `workflow`・`decision`・`environment`・`gotcha` 全4層に対する自動 `update` / `delete` / `add` が Phase 1 の生バイト SHA-256（`expected_sha256`）付きで Phase 4（`publish_operations`）へ渡され、自動マージされ、`autodream_state.json` が更新されること。
     - 統合実行中に別セッションが対象エントリを更新した場合、`expected_sha256` 不一致で削除が拒否され、`last_outcome: "failed"` が記録されて非ゼロ終了すること。
     - 生成文書ずれのみのケースで `publish_index_repair` 経由で復旧公開されること。
     - 変更不要ケースで PR を作らず `last_outcome: "noop"` となり `last_completed_at` / `last_consolidated_sha` が更新されること。
     - ランナー失敗・不正 JSON Schema・`ops-report` 干渉・検証違反時は一切公開せず、`last_outcome: "failed"` と `last_error` を記録し、`last_completed_at` を進めず非ゼロ終了すること。
     - `--dry-run` で公開も `last_completed_at` 更新も行われないこと。
     - 排他ロック競合時に二重実行されないこと。
  3. **有限ジョブ入口と `ops_control` 契約**:
     - `autodream_job.py` の直接実行が終了コード 75 で拒否され、`checkout_job.py` 経由でのみ動作すること。
     - `tests/e2e/agents_ops/test_ops_control.py`、`tests/e2e/agents_ops/test_agent_exec.py`、`tests/e2e/agents_ops/test_agents_docs.py` が全通過すること（`memory-autodream` が `agents` グループに含まれないことも確認）。

## 検証・PR作成
1. 以下をすべて実行し全通過を確認すること：
   - `PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory validate`
   - `PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory sync-index --check`
   - `PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory audit`
   - `.venv/bin/python -m pytest tests/unit/agents_ops/test_shared_memory_core.py tests/unit/agents_ops/test_shared_memory_autodream.py tests/e2e/agents_ops/test_shared_memory_publish.py tests/e2e/agents_ops/test_shared_memory_autodream.py tests/e2e/agents_ops/test_ops_control.py tests/e2e/agents_ops/test_agent_exec.py tests/e2e/agents_ops/test_agents_docs.py tests/e2e/agents_ops/test_interactive_checkout.py tests/e2e/agents_ops/test_ops_report_memory.py -n 2 -q`（作成したテストファイル名に合わせること）
   - `.venv/bin/python -m ruff check .agents/ops/shared_memory .agents/ops/ops_control tests/unit/agents_ops tests/e2e/agents_ops`
   - `.venv/bin/python -m ruff format --check .agents/ops/shared_memory .agents/ops/ops_control tests/unit/agents_ops tests/e2e/agents_ops`
   - `.venv/bin/python -m mypy --follow-imports=skip .agents/ops/shared_memory .agents/ops/ops_control`（追加テストファイルも含む）
   - `.agents/ops/review/` の差分が0件であることを確認。
2. コミット（pre-commit通過）→ push → 日本語PRを作成（本文に `Closes #2326`、受入条件ごとの証拠・テスト実測・merge 後に人間が `spin ops install memory-autodream` / `spin ops on memory-autodream` で有効化する旨を記載。draftにしない）。
3. 最終報告の末尾に `PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":<PR番号>,"head":"<完全な40桁HEAD>","resume_on_success":false}}` を出力して終了すること。
