PR #2346（HEAD `f4f2231722ad5e391f358e64771c8d3b8420175e`）の正式レビュー（`parent-2312-4029e9c30022406dba5781d58aa7b0d8`）で、以下の major 指摘が6件出ました。親は**6件すべて採用して直す**と判断しました。再レビューは行いません。

同じブランチ・同じPR（#2346）で6件すべてを修正し、それぞれに対する回帰テストを追加し、全検証・pre-commitを通して commit・push したうえで、`PARENT_RESULT` の `wait.kind="ci"`（`resume_on_success: false`）で新HEADを宣言して終了してください（CI待機はこのターンで行わない）。

## 対応する major 指摘（全6件）

### 1. 並行更新で重複統合の根拠が失われても削除をマージする
- **指摘**: AとBが同じ観察を持つ版で「Bに残るのでAを削除」と計画し、別更新でBだけが削除された版に適用すると、Aの `expected_sha256` は一致し、両方が消えた候補が検査を通る。`autodream.py:264-304` は操作対象の版だけを渡し、`publish.py:60-105` は観測時の `base` を照合せず最新 `main` に載せ直すため。
- **対応**: AutoDream からの公開では、Phase 1 で観測した**根拠となる共有memory 4層全体の版**（観測時の `base` SHA、または共有4層全エントリの `{path: sha256}` スナップショット集合の完全一致条件）を `publish` 境界へ渡し、判断の根拠に使った共有memory集合（残す側を含む他のエントリや観測 `base`）が変化していたら自動載せ直しせず拒否して再計画を要求すること。
- **回帰テスト**: AとBが重複する版で「Bに残るのでAを削除」を計画し、適用前に別更新でBだけが削除された場合、Aの削除公開が拒否されて両方消失しないことを検証すること。

### 2. 一時展開を除外しても runner がホスト上の専用 memory を読める
- **指摘**: 固定HEADのアーカイブには `.agents/memory/ops-report/` が含まれない一方、`autodream_plan.py:149-170` は元checkoutのパスを含む環境を渡し、`agent_settings.py:303-305,360-367` はホスト全体を読める既定の `read-only` sandbox を使う（`lib/README.md:684-687` 記載）。runner が専用領域や固定版外の内容を根拠にでき、本文への転記は引用先検査で止まらない。
- **対応**:
  - runner へ渡す環境から元 checkout のパスや不要な環境参照を除去すること。
  - 実行時の読取範囲を制限すること（Codex CLI の `readOnly` sandbox 制限設定［`restricted` / `readable_roots` 等、実機の `codex` CLI 設定契約を確認のうえ `agent_settings.py` / `agent_exec.sh` または `memory_autodream` プロファイルで明示設定する］や、Linux の隔離境界［例: `ops-report` の `repro_isolation.py` / `repro_linux.py` と同等の mount/user namespace や restricted read-only roots］を用いて、スナップショット外・元 checkout の `.agents/memory/ops-report/` を読めないようにする）。
  - **回帰テスト**: 固定スナップショット内の不在確認だけでなく、実行時の読取範囲制限設定および「元 checkout に置いた `.agents/memory/ops-report/` や固定版外ファイルを runner が読めないこと」を検証すること。

### 3. 公開確定後の停止や片付け失敗で統合状態が `main` と食い違う
- **指摘**:
  - 確定済みの PR とマージ SHA にブランチ削除失敗を添えて返すと、`autodream.py:396-421` は `failed` とマージ SHA を保存し、統合時刻と基準 SHA を更新せず非ゼロ終了する。
  - `publish.py:254-292` の ref 更新が GitHub 側で成立して応答受信前に SIGTERM を受けた場合、次回に結果を照合するための確定前の候補 SHA は状態に保存されない。
- **対応**:
  - 公開の試行開始前（および PR 作成・候補 `head` 確定直後）に、公開インテント（`base`・候補 `head`・専用ブランチ・PR番号・操作種別）を状態ファイルへ原子的に永続化すること。
  - 以下の3状態を明確に区別し、状態ファイルと次回起動時の復旧照合を整合させること：
    1. **公開前の拒否（未マージ）**: `last_outcome: "failed"`、`last_completed_at` / `last_consolidated_sha` は進めない、pending 公開インテントはクリア。
    2. **マージ確定済み＋ブランチ片付け失敗**: `main` への統合は確定済みのため、`last_completed_at` と `last_consolidated_sha`（マージ確定 SHA）を正しく更新したうえで、`branch_cleanup_error`（後処理失敗）を明示記録して運用者が観測・回収できるようにする。
    3. **公開応答喪失・SIGTERM 中断（結果不明）**: 保存済みの pending 公開インテントを残し、次回起動時に remote `main`・PR 状態と候補 `head` を照合して、確定済みなら統合完了状態を回復（必要なら残ブランチ削除を試行）、未反映なら未確定失敗として明確に扱う（重複公開しない）。
  - **回帰テスト**: 既存の fake GitHub を使い、(a) 公開拒否、(b) ref 更新後の応答喪失・SIGTERM 中断からの次回起動照合復旧（反映済み/未反映の両方）、(c) マージ確定＋ブランチ片付け失敗、の各状態を検証すること。

### 4. runner のハングで日次ジョブと `git-sync` が止まり続ける
- **指摘**: `autodream_plan.py:149-170` は共通 runner の `--timeout` を指定せず、外側の `run_process_group` にも `timeout=None` を渡す。oneshot service には起動タイムアウトがなく、`TimeoutStartSec=60` ではなく `TimeoutStopSec=60` のみ。Codex が応答を返さなくなると日次実行と共有 checkout lock を使う `git-sync` が止まり続ける。
- **対応**:
  - `autodream_plan.py` / `autodream.py` に明示的な有限タイムアウト（例: 既定の runner タイムアウト定数＋CLIオプション `--runner-timeout` 等、共通 runner の `--timeout` と外側 `run_process_group` の上限の両方に反映）を設けること。
  - `tennis-lab-memory-autodream.service` にも明示的な有限の `TimeoutStartSec=` を設定すること（`TimeoutStopSec=60` と併用）。
  - **回帰テスト**: ハングする fake runner に対し、手動起動・ジョブ実行の両経路でタイムアウトが発火し、`last_outcome: "failed"` と明確なタイムアウト理由が記録され、非ゼロ終了し、排他ロック・checkout lock・一時展開ディレクトリ・子プロセス群がすべて回収されることを検証すること。

### 5. `citations` フィールドの欠落を即時トリガーが見逃す
- **指摘**: `citations` だけが欠けたエントリを追加し、生成文書が直前の有効エントリ集合と一致する状態で `audit`・`plan_sync`・`decide_trigger` を実行すると、`schema.py:231-233` は欠落を一般のスキーマ違反に分類し、生成差分も出ないため、前回から1時間・更新1件では `reasons=()`（`skipped_trigger`）になる。
- **対応**: `citations` フィールドの欠落（および `citations` の型不正・要素スキーマ違反を含む `citations` 由来の全違反）を `audit` の `broken_citations`（および即時トリガー条件）へ確実に含めること。
- **回帰テスト**: `citations` フィールドが欠落し、かつ生成文書ずれ・経過時間条件・件数条件が一切併発しない境界ケースで、`audit` の `broken_citations` に分類され、即時トリガーが発火することを検証すること。

### 6. 破損した状態ファイルで起動が失敗しても `failed` を記録しない
- **指摘**: 状態ファイルの読取りへ壊れた JSON を注入すると、`autodream.py:337-352` の `read_state` が失敗し、Orient も失敗状態の書込みも呼ばれず、`spin ops status` の `last_run` に今回の失敗とエラーが残らない。
- **対応**: 排他ロック取得後に状態ファイルの読込・schema 検証が失敗した場合、**破損した原本を別ファイル（例: 同ディレクトリ内の退避ファイル）へ保全したうえで**、公開操作を一切開始せず、運用者が `spin ops status`（`status_file`）で今回の失敗・エラー・退避した原本の所在を確認できる形で `failed` 状態を永続化し非ゼロ終了すること（破損前の `last_completed_at` や `last_consolidated_sha` を捏造せず、手動確認・復旧が必要であることが明確に分かる状態にする）。
- **回帰テスト**: 壊れた JSON の状態ファイルを注入した起動で、(a) 破損原本が保全されていること、(b) `spin ops status` の `last_run` から今回の `failed` とエラー・退避先を取得できること、(c) runner・公開・GitHub 書込が一切始まらず非ゼロ終了すること、を検証すること。

## 補足・検証手順
- ドキュメント（`.agents/ops/shared_memory/README.md`・`.agents/ops/lib/README.md` 等）も上記6点の仕様（観測版の整合、runner の読取隔離・タイムアウト、公開インテントと復旧状態、破損状態の保全記録等）に合わせて更新すること。
- 修正後、以下すべてを実行し全通過を確認すること：
  - `PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory validate`
  - `PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory sync-index --check`
  - `PYTHONPATH=.agents/ops .venv/bin/python -m shared_memory audit`
  - `.venv/bin/python -m pytest tests/unit/agents_ops/test_shared_memory_core.py tests/unit/agents_ops/test_shared_memory_autodream.py tests/e2e/agents_ops/test_shared_memory_publish.py tests/e2e/agents_ops/test_shared_memory_autodream.py tests/e2e/agents_ops/test_ops_control.py tests/e2e/agents_ops/test_agent_exec.py tests/e2e/agents_ops/test_agents_docs.py tests/e2e/agents_ops/test_interactive_checkout.py tests/e2e/agents_ops/test_ops_report_memory.py tests/e2e/development/test_ci_test_lanes.py -n 2 -q`
  - `.venv/bin/python -m ruff check .agents/ops/shared_memory .agents/ops/ops_control .agents/ops/lib/agent_settings.py tests/unit/agents_ops/test_shared_memory_autodream.py tests/e2e/agents_ops/test_shared_memory_autodream.py tests/e2e/agents_ops/test_shared_memory_publish.py tests/e2e/agents_ops/test_ops_control.py tests/e2e/agents_ops/test_agent_exec.py`
  - `.venv/bin/python -m ruff format --check .agents/ops/shared_memory .agents/ops/ops_control .agents/ops/lib/agent_settings.py tests/unit/agents_ops/test_shared_memory_autodream.py tests/e2e/agents_ops/test_shared_memory_autodream.py tests/e2e/agents_ops/test_shared_memory_publish.py tests/e2e/agents_ops/test_ops_control.py tests/e2e/agents_ops/test_agent_exec.py`
  - `.venv/bin/python -m mypy --follow-imports=skip .agents/ops/shared_memory .agents/ops/ops_control .agents/ops/lib/agent_settings.py tests/unit/agents_ops/test_shared_memory_autodream.py tests/e2e/agents_ops/test_shared_memory_autodream.py tests/e2e/agents_ops/test_shared_memory_publish.py`
  - `.agents/ops/review/` と `.agents/memory/` の差分が0件であることを確認。
- コミット（pre-commit通過）して同じブランチへ push し、PR #2346 に6件それぞれの修正内容と回帰テスト結果をコメントすること。
- 最終報告で6件それぞれの修正コミット・回帰テスト結果・新HEADを報告し、末尾に `PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2346,"head":"<新40桁HEAD>","resume_on_success":false}}` を出力して終了すること。
