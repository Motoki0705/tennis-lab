# Issue #2690 実装・オフライン比較・テスト整備（Turn 1）

Issue #2690「[CI高速化] CPUテストの分割用実測プロファイルを最新化し、重いshardへの偏りを解消する」の実装・オフライン検証・テスト・文書整備を担当してください。

## 目的と厳守事項（スコープ境界）
- 既存の CPU CI のテスト集合・検証水準を維持したまま、最新の適格な成功 `main` push CI の実測データから `scripts/ci_metrics/cpu-file-times.json` を再現可能に再生成し、重い shard への偏りを解消する。
- 以後も再現可能に profile を更新でき、鮮度・負荷偏りの劣化を観測できる仕組みを整備する。
- **対象外（絶対に手を付けない）**:
  - テスト削除・重複統合・PR対象の選別規則変更（`lanes.json` / `lanes.py` の選別方針）
  - テスト本体や fixture の機能契約・scope の変更
  - runner 数・worker 数の増加、GPU テスト、記録専用 PR のマージ後 CI 省略、`#2683` のマージ管理フロー変更
  - 毎回の `main` CI への自動コミットや自動マージの導入
- 既存の `Python` 必須チェック、`system_deps` 隔離、fail-closed の完全性監査、attempt 整合規則、分割 guard を一切弱めないこと。
- 静かなフォールバック禁止。失敗時は明示エラーで止め、既存 `cpu-file-times.json` を書き換えないこと。

## 実施内容

### 1. 最新適格 full main CI の確認・取得と再現可能な更新入口の整備
1. 着手時点の最新成功 `main` push CI run（例: `45955929416`、次点 `45931829509` や `45693968766`）を確認し、適格な最新 run を特定する。
2. `scripts/ci_metrics/` に、再現可能な profile 更新入口（例: `python -m scripts.ci_metrics update-profile` サブコマンドと対応モジュール）を実装する。
   - **指定 run（`--run-id`）** と **最新適格 main run（`--latest-main` 等）** の両方を選べるようにする。
   - Actions の run ID・HEAD・attempt・lane plan・全 artifact（`cpu-lane-plan-RUN`、`cpu-test-coverage-RUN-ATTEMPT`、通常 `0..count-1` と `system-deps` `count` の全 `cpu-test-measurements-RUN-shard-N`）を固定・照合する。
   - 単なる workflow の `success` だけでなく、`coverage.json` の `success is True`・`full_cpu_success is True`・`observations_complete is True`・`deferred_count == 0`・`event == "push"`・`ref == "refs/heads/main"`・`source_sha == head_sha`・run/attempt 整合・全 shard/worker 観測完全性・正当な既存 skip のみ（テスト失敗 0 件）を厳格に照合する。
   - 記録専用軽量 run、部分 PR、欠測、失敗、取消、期限切れ、異なる run/HEAD/plan の混在はすべて明示エラーで拒否する。既存の正当な部分 rerun の attempt 整合規則（`plan_attempt <= shard_attempt <= aggregate_attempt`）は維持する。
   - 既存の `build_profile` / `inspect_shards` / `inspect_artifacts` 等を適切に再利用する。低レベル `profile` コマンドの既存契約は壊さない。
   - 失敗時は既存 `--output`（`cpu-file-times.json`）を一切変更しない（一時ファイル経由の原子置換）。
   - `source` metadata（run_id, commit, attempt, plan_attempt, shard_attempts, shard_count, selected_count, inventory_sha256, counts 等）、生成コマンド、旧重みとの差分、未計測 file/case、shard ごとの推定負荷を含むレポート（JSON/Markdown や標準出力サマリ）を出力できるようにする。

### 2. fixture 独立性の検証と A/B/C オフライン割当比較
1. `tests/integration/agents_ops/test_storage_tier_crash_boundaries.py` と `tests/integration/agents_ops/test_storage_tier_recovery_boundaries.py`（および参照する `tests/support/agents_ops/storage_crashes.py`, `storage_tier.py`, 関連 `conftest.py`）の fixture graph・`tmp_path`・ファイル/プロセス/ロック・`sys.path`[local path]。
2. 実際に `scripts/ci_metrics/shard_plugin.py` の収集時 guard に両 file を `--split-file` として通し、function fixture のみで安全に case 分割できることを検証する。万一分割不可能な根拠があれば無理に分割せず、根拠と安全な代替案を示す。
3. 最新の全件 inventory と最新適格 main CI の実測値に対し、通常 10 shard について以下 3 条件をオフライン比較する:
   - **A**: 現行 `scripts/ci_metrics/cpu-file-times.json`・現行 `split_files`（3 file）
   - **B**: 最新実測 profile・現行 `split_files`（3 file、重み更新のみ）
   - **C**: 最新実測 profile・既存 3 file + 停止/復旧 2 file（安全に分割可能な場合。他に支配的 file がある場合はそれも分析して報告する）
4. 各条件について以下を記録する（推定 phase 秒を wall time と混同しないこと）:
   - 各 shard（0〜9）の推定負荷（phase 秒）、最小・最大・平均・最大/平均比
   - 現行割当に対する最新実測 phase 秒の shard ごとの分布（旧割当で実際にどれだけ偏っていたか）
   - 未計測 file 数、未計測 split case 数
   - 支配的な上位 file の実測 phase 秒と割当結果
5. 整備した更新入口コマンドを実際に実行し、採用候補（原則 C、不可なら B）で `scripts/ci_metrics/cpu-file-times.json` を再生成して反映する。

### 3. 鮮度・偏りの可視化（既存 growth report の拡張）
- `scripts/ci_metrics/growth.py` と `scripts/ci_metrics/shards.py`（および `aggregate_snapshot` / `analyze` / `markdown`）を確認し、既存の `growth-summary.json` / `growth-summary.md` に profile の鮮度・負荷偏り指標を追加する:
  - profile の出典（`run_id`, `commit`, `selected_count`, `split_files` 数 等）
  - 現行観測 inventory との差（未計測 file 数・未計測 file 率、未計測 split case 数等）
  - 通常 shard ごとの推定負荷 vs 実測 phase 秒、実測負荷の min/max/mean および max/mean 比（偏り指標）
- 既存契約（growth report は情報提供のみで coverage gate の成否を変えない、既存テスト互換）を維持する。

### 4. 回帰テスト・文書整備
1. **回帰テスト**（`tests/e2e/development/test_ci_*.py` 等へ追加・更新）:
   - 不正・部分観測（`full_cpu_success=False`、`deferred_count > 0`、テスト失敗含む run、PR event / 非 main branch、run/HEAD/plan 不一致、期限切れ/欠落/重複 artifact 等）で profile 更新が拒否され、既存の `cpu-file-times.json` が変更されないこと
   - 更新途中の失敗で既存ファイルが壊れない・書き換わらないこと
   - 未計測 file・split file 内の新規 case が省略されず実行対象に入り manifest に記録されること
   - 更新後 profile で同一入力に対し決定的な割当が行われ、full CPU・代表的 scoped PR の両方で requested nodeid の欠落・重複がなく、`system_deps` 隔離・正当な skip・固定 profile/plan・最終 `Python` 完全性監査が成立すること
   - 新 `scripts/ci_metrics/cpu-file-times.json` に対する `read_timing_profile`・実コレクションの fixture guard 整合
   - growth report の新鮮度・偏り指標の出力と欠損時の安全な扱い
2. **文書整備**（公開文書は英語を正本とし、日本語・簡体字中国語も 3 言語で同期更新する）:
   - `scripts/ci_metrics/sharding.md`, `scripts/ci_metrics/sharding.ja.md`, `scripts/ci_metrics/sharding.zh-CN.md`
   - `scripts/ci_metrics/README.md`, `scripts/ci_metrics/README.ja.md`, `scripts/ci_metrics/README.zh-CN.md`
   - 記載内容: 新 profile の出典・更新差分・A/B/C オフライン比較結果・fixture 境界検証結果・更新コマンドの使用方法と更新判断基準（自動コミット/自動マージしない方針含む）・鮮度/偏り指標の説明。
   - 実 CI 比較（AC-4）はこの Turn 1 の後、親の確認を経て Turn 2 で同一条件の比較 CI を実行・追記する予定。

### 5. ローカル検証と commit / push（PR はまだ作らず、検証済み branch を報告）
- 以下のローカル検証を worktree 内で実行してすべて通すこと:
  - `.venv/bin/python -m pytest tests/e2e/development/test_ci_artifacts.py tests/e2e/development/test_ci_growth.py tests/e2e/development/test_ci_metrics.py tests/e2e/development/test_ci_shards.py tests/e2e/development/test_ci_test_lanes.py tests/e2e/development/test_readme_layout.py`（および新規追加テスト）
  - `.venv/bin/python -m ruff check scripts/ci_metrics tests/e2e/development`
  - `.venv/bin/python -m mypy --follow-imports=skip scripts/ci_metrics`
  - `git status` / `git check-ignore` で意図しないファイルや抜け漏れがないことの確認、pre-commit 相当の確認
- 変更をきれいに commit してリモート branch へ push する（**まだ PR 作成や実 CI dispatch は行わず**、次のターンで親と方針・比較手順を確認したうえで実 CI 比較へ進む）。
- 既存の `cpu-file-times.json` 比較用に、今回の変更で追加したコードと旧 `cpu-file-times.json` のみの差分が最小・明確であることも確認する。

## 最終報告に含めるもの
1. 採用した最新適格 main CI run の詳細（run ID, commit SHA, attempt, plan_attempt, shard_attempts, artifact IDs, `coverage.json` 検証結果）と実際に実行した更新コマンド
2. fixture 独立性検証の結果（`test_storage_tier_crash_boundaries.py`, `test_storage_tier_recovery_boundaries.py` + `shard_plugin` guard 実行結果）
3. A / B / C オフライン比較の詳細数値表（各 shard 0〜9 の推定負荷、min/max/mean、max/mean 比、未計測件数、旧割当での実測 shard 負荷分布、上位の重い file）
4. 実装した更新入口 CLI・growth report 鮮度指標・文書・追加テストの概要
5. ローカル検証コマンドと結果、commit SHA、branch 名
6. 末尾に規定の `PARENT_RESULT`（`state: "done"`）を出力すること。
