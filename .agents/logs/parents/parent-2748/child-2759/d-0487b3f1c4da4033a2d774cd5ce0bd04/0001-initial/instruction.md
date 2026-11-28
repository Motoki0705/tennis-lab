# 実装指示: #2759 S2 — ルールベース打者推定と val/test 22クリップの独立確認 GT 検証

親 Issue #2748 の Phase 1 子 Issue #2759 を実装し、日本語本文で `Closes #2759` を含む ready PR を作成してください。

## 背景・前提契約（#2757 基盤・親合意）
- 先行 PR #2762（#2757）で `src/tennis_scene/analytics/` 配下に PostgreSQL 17 + SQLAlchemy 2.0 + Alembic スキーマ（`shot_hitter_estimates` 含む）が構築済みです。まず `src/tennis_scene/analytics/README.md`, `db/models.py`, `export.py`, `__main__.py` を必ず確認してください。
- 対象データはローカルの frozen `data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v4`（69クリップ、47 train / 11 val / 11 test の固定 split）。既存 V4 ファイルは不変とし書き換えないこと。
- **最重要の科学的厳密性（リーク禁止）**:
  - val 11クリップ＋test 11クリップ（計22クリップ）の全 shot に対する独立確認 GT（各 shot の実際の打者の `local_player_id`）は、**ルールベース打者推定の実行結果を見る前に**多視点フレーム画像（cam0〜cam3 へ各選手の `local_player_id` / track ID・骨格・ボール位置を投影描画した画像）から独立に視覚確認して確定すること。
  - ルールの閾値・重み・棄却条件は **train / val のみ** で調整・固定し、**test 11クリップは一度も調整に使わず最終評価のみ** を行うこと。

## 具体的な実装要件

### 1. ルールベース打者推定 (`src/tennis_scene/analytics/hitter.py`)
- 既存 `src/tennis_scene/ball_events/rules.py::_player_support` を調査し、共通化できる部品はクリーンに再利用・抽出する（既存 `rules.py` の挙動を壊さないこと）。
- V4 の各 `shot` イベントに対して以下を入力に候補をスコアリングする：
  - shot 近傍フレーム窓（`ball_event_frame_uncertainty` を考慮し、中心フレームだけでなく近傍の有効フレームを明示的に評価）
  - 3D ボール位置（`ball_3d`, `ball_3d_valid`）と各選手の左右手首（COCO17 の wrist 関節 `player_kp_3d`, `player_kp_3d_vis`）および root（`player_position`, `player_valid`）の 3D 距離
  - V4 の `ball_event_side`（`+1` / `-1` / `0`）と各選手のコート側 Y 符号の整合
  - 接触前後の移動・有効性 mask 品質
- **local_player_index と local_player_id の厳格な区別**:
  - 配列添字 `local_player_index`（`0..P-1`）とクリップ内永続 ID `local_player_id`（`player_track_ids[index]`）を別フィールドで保存する。
- **曖昧・低品質ケースの誠実な保留（無理に確定しない）**:
  - `resolution_status`: `resolved` / `ambiguous` / `unresolved`
  - 候補ごとのスコア一覧 (`candidates_json`)、次点との差 (`margin_to_second`)、採用した landmark (`left_wrist` / `right_wrist` / `root`)、利用フレーム・距離・side 整合の根拠 (`evidence_json`)、保留/棄却理由 (`rejection_reason`: 例 `ball_3d_invalid`, `no_valid_player`, `side_mismatch`, `too_far`, `root_only_low_confidence`, `same_side_competing_players`, `net_area_ambiguous`, `high_time_uncertainty` 等) を明確に保存する。
  - `rule_version` を版付きで定義し、同じ入力・同じ版で完全に決定論的に再現できるようにする。
- PostgreSQL の `shot_hitter_estimates` への保存ヘルパーと、`src/tennis_scene/analytics/__main__.py` への最小限の CLI 入口（並列開発中の #2758/#2760 と競合を最小化するため、実装本体は `hitter.py` / `hitter_eval.py` に分離）を接続する。

### 2. val 11 + test 11（計22クリップ）の独立映像確認 GT 作成と評価 (`hitter_eval.py`)
- `manifest.json` から val 11クリップ・test 11クリップの全 `shot` イベント一覧を抽出する（まず件数と内訳を確認する）。
- 多視点確認画像（各 shot の中心フレームおよび前後フレームで cam0〜cam3 に各選手の `local_player_index` / `local_player_id` と 3D 骨格・ボール位置をラベル描画した合成パネル画像）を生成するスクリプトを作成し、**ルール推定を実行する前に**、全22クリップの全 shot について画像を目視確認して独立 GT（`clip_id`, `source_event_id`, `frame`, `subtype`, `split`, `gt_local_player_id`, `gt_local_player_index`, `visibility_status`, `audit_note` 等）を作成・保存する。
  - 独立 GT と監査メタデータ・評価レポートはリポジトリ内の適切な場所（例：`knowledge/runs/run-i2759-v4-hitter-rule-eval/` 等、または再現可能な JSON/CSV + 評価スクリプト）に保存し、元映像・V4 のハッシュと紐づける（巨大な中間画像はコミットしないこと）。
- train/val だけでルール閾値・棄却条件を決定・固定した後、val 11クリップと test 11クリップそれぞれについて以下を計測・報告する：
  - `total_shots`, `evaluable_shots`, `resolved_count`, `ambiguous_count`, `unresolved_count`, `coverage`（確定率）
  - `resolved_accuracy`（確定した中の正解率）、`false_resolution_rate`（誤確定率 = 誤確定数 / 全対象または確定数、定義を明示）、`overall_accuracy`
  - 保留理由別内訳、誤確定ケースの全件一覧（clip / event / 原因分析）
  - 47/11/11 split の重複ポイント・同一映像跨ぎ有無の検査結果

### 3. テスト
- **DB 非依存ユニットテスト** (`tests/unit/tennis_scene/analytics/test_hitter.py`):
  - 明確な手首ヒット（`resolved`）、同側2人競合（`ambiguous`）、ネット際曖昧、root のみ、ball/player mask 欠測、side 不一致、時刻±幅考慮、`local_player_index` vs `local_player_id` 非一致ケース、決定論性を検証。
- **PostgreSQL 結合テスト** (`tests/integration/tennis_scene/analytics/test_hitter_postgres.py`, `@pytest.mark.postgres`):
  - `shot_hitter_estimates` への保存・制約・CLI 照会・JSON/CSV エクスポート往復を実 PostgreSQL で検証。

## 検証・引き渡し条件
1. 独立 GT 作成→val 閾値固定→test 未調整評価の全手順を実行し、val/test 別の実測指標・誤確定分析・保留内訳を PR 本文と評価レポートへ明記する。
2. ローカル PostgreSQL 17 結合テスト（`CI_REQUIRE_POSTGRES=1 .venv/bin/python -m pytest -n 0 -m postgres tests/integration/tennis_scene/analytics`）、新規・既存ユニットテスト（`rules.py` 回帰含む）、pre-commit（Ruff・mypy `--follow-imports=skip`）を通す（検証後の Compose は停止）。
3. 日本語本文で `Closes #2759` を含む ready PR を作成し、PR 番号・完全な HEAD SHA・実測結果・検証コマンドを報告して、PR CI 完了前に `PARENT_RESULT {"schema_version":1,"state":"done","wait":null}` を返す。

## 並列実行・共有環境に関する重要注意（必読）
- 共有 `.venv` には `sqlalchemy` / `psycopg` / `alembic` が未導入のため、**共有 `.venv` 自体は `uv sync` 等で書き換えず**、`uv.lock` に固定されたバージョンの追加パッケージだけを一時ディレクトリ（例: `[local path]`）へ展開して `PYTHONPATH` に追加する形でローカル検証を行ってください（`pyproject.toml` / `uv.lock` は変更不要）。
- 他子 Issue（#2758・#2760）と並列実行するため、ローカル Docker Compose 検証では必ず固有ポート・固有プロジェクト名を使用してください：
  `POSTGRES_PORT=15459 docker compose -p tennis-analytics-2759 up -d --wait`
  接続 URL は `postgresql+psycopg:[local path]` とし、検証完了後は `docker compose -p tennis-analytics-2759 down` で停止してください。
