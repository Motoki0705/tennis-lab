# Issue #2690 Turn 2: 実 CI 比較結果の反映・最終検証・ready PR 作成

親側で、あなたの Turn 1 コミット（`3e9e0b7380b050f205f20372dd31f298dbe4a483`）を基準に、`scripts/ci_metrics/cpu-file-times.json` だけを旧版へ戻した同一ソース条件 A（`299b5c07a4a6153234cf4cbddc66f1c7a7b74e70`）と新 profile 条件 C（`3e9e0b7380b050f205f20372dd31f298dbe4a483`）の間で、同一テスト集合・同一 runner 構成（通常 10 shard × 2 worker + `system-deps` 1）の実 GitHub Actions CI 比較を全 6 run 完了し、一時比較ブランチおよび一時 Draft PR（#2715, #2716）はすべて削除・クローズ済みです。

## 回収済みの実 CI 比較証拠（読み取り元）
以下のディレクトリに全 6 run の `actions-summary.json`、`actions-raw.json`、`coverage/{coverage.json,growth-summary.json,growth-summary.md,artifacts.json}`、および全 11 shard の計測 artifact（`shards/shard-0..10/tests/`）を保存しています：

- ルート: `[local path]`
- マニフェスト: `[local path]`
- 速報集計: `[local path]`

対象 6 run（すべて `conclusion=success`, `attempt=1`, `UV_CACHE_HIT=true` 10/10, `issues=[]`, `system_deps={'passed': 17}`）:
1. `full-a-1`: run `46045996862` (`workflow_dispatch`, A, U=26208, R=26208, D=0)
2. `full-c-1`: run `46050353999` (`workflow_dispatch`, C, U=26208, R=26208, D=0)
3. `full-a-2`: run `46052765632` (`workflow_dispatch`, A, U=26208, R=26208, D=0)
4. `full-c-2`: run `46056231956` (`workflow_dispatch`, C, U=26208, R=26208, D=0)
5. `scoped-a-1`: run `46058668606` (`pull_request`, A + 無害1行コメント `src/tasks/ball_refiner_3d/__init__.py`, U=26208, R=23334, D=2874)
6. `scoped-c-1`: run `46061213080` (`pull_request`, C + 同一無害1行コメント `src/tasks/ball_refiner_3d/__init__.py`, U=26208, R=23334, D=2874)

## 実施内容

### 1. 実 CI 証拠の厳密検証・集計（AC-3, AC-4）
- 上記の回収済み raw データ（`actions-raw.json` / `actions-summary.json` / `coverage.json` / `growth-summary.json` / 各 shard の `environment.json`, `summary.json`, `shard-gw*.json`, `events.jsonl` 等）を直接検証し、以下を正確に集計する：
  - full CPU（A×2 vs C×2）と代表的 scoped PR（A×1 vs C×1）それぞれの：
    - 各 shard の推定累積 phase 秒・実測 phase 秒・Test ステップ wall time・job wall time
    - 最長 shard（Test ステップ wall time・job wall time）、準備時間、収集時間（計測ログから取得可能なもの）、キュー待ち（job queue / dispatch_to_first_job）
    - `Python` 必須 check 完了時間、全 runner 時間（`all_jobs_runner_s`）、成功/失敗/取消
    - cache 条件（`UV_CACHE_HIT` 等）、試行数
    - 実行 nodeid の欠落/重複 0、`system_deps` 17 件全 passed、正当な skip、固定 profile/plan、完全性監査（`full_cpu_success` / `scoped_cpu_success`）の成立
    - 短縮効果と runner 費用の評価（推定 phase 秒を wall time と混同しないこと）
- ※ `full-a-1..2` / `full-c-1..2` は `3e9e0b738` に対して同一テスト集合、`scoped-a-1` / `scoped-c-1` も同一テスト集合であることを確認済み。各 run の Actions URL も記載する。

### 2. 3 言語文書への反映（英語正本 + 日本語 + 簡体字中国語）
- `scripts/ci_metrics/sharding.md`、`scripts/ci_metrics/sharding.ja.md`、`scripts/ci_metrics/sharding.zh-CN.md` の実 CI 比較節（現在「同一条件のActionsで比較します」となっている箇所等）を更新し、上記 6 run の実測比較（full CPU A×2 vs C×2、代表的 scoped PR A×1 vs C×1 の速度・runner 消費・cache 条件・完全性監査・短縮効果）を明確・簡潔に記録する。
- 必要であれば、既存 `scripts/ci_metrics/` の他ファイル（例: 根拠 JSON や README）との整合を確認し、既存のテスト（`test_readme_layout.py`, `test_ci_*.py`）や文書配置規則に違反しないようにする。
- 公開文書にローカル絶対パス（`[local path]` や `[local path]`）を絶対に書かないこと。

### 3. ローカル検証・commit・push・ready PR 作成
- 必要なローカル検証を worktree 内で実行してすべて通すこと：
  - `.venv/bin/python -m pytest tests/e2e/development/test_ci_artifacts.py tests/e2e/development/test_ci_growth.py tests/e2e/development/test_ci_metrics.py tests/e2e/development/test_ci_shards.py tests/e2e/development/test_ci_test_lanes.py tests/e2e/development/test_readme_layout.py`
  - `.venv/bin/python -m ruff check scripts/ci_metrics tests/e2e/development`
  - `.venv/bin/python -m mypy --follow-imports=skip scripts/ci_metrics`
  - pre-commit 相当の確認
- 変更を commit してリモート branch `codex/parent-2690-issue-2690-primary` に push する。
- **日本語の PR 本文** で **ready PR**（ドラフトではない）を作成し、本文に `Closes #2690` を明記する。
  - PR 本文には Issue #2690 の Acceptance checklist 全 5 項目に対応する証拠（計測元 run・source metadata・再現コマンド・更新差分、fixture 独立性検証と A/B/C オフライン比較表、全件[local path] PR の安全契約・回帰テスト、実 CI 6 run の詳細比較表・短縮効果・runner 費用評価・Actions URL、次回更新手順・判断基準・鮮度/偏り可視化）を漏れなく簡潔にまとめること。
  - ローカルの絶対パスやセッション識別子は PR 本文に載せないこと。
- **PR CI の完了は待たずに**、ready PR の番号・URL・完全な HEAD SHA・ローカル検証結果を最終報告に記載し、末尾に `PARENT_RESULT {"schema_version":1,"state":"done","wait":null}` を返してください（正式レビューと PR CI 監視は親が `review-request` で自動進行へ渡します）。
