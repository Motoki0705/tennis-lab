# 実装指示: #2758 S1 — V4詳細注釈・projects.json・手動補足アダプターと69クリップ利用可能率監査

親 Issue #2748 の Phase 1 子 Issue #2758 を実装し、日本語本文で `Closes #2758` を含む ready PR を作成してください。

## 背景・前提契約（#2757 基盤）
- 先行 PR #2762（#2757）で `src/tennis_scene/analytics/` 配下に PostgreSQL 17 + SQLAlchemy 2.0 + Alembic スキーマ（18表）・指標辞書・CLI/エクスポート基盤が構築済みです。
- まず `src/tennis_scene/analytics/README.md`, `db/models.py`, `db/connection.py`, `export.py`, `metrics.py`, `__main__.py` を必ず読み、既存の複合外部キー・制約・`adopted_in_v4`[local path]`in_point`[local path]`excluded_reason`・`terminal_cause`×`terminal_cause_source`・`(match_context_id, match_point_key)` の契約に完全準拠してください。
- 既存の frozen V4 ファイル（`data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v4`）および `data/tennis_multiview/processed/meiji_4cam/projects.json` は不変とし、一切書き換えないこと。

## 具体的な実装要件

### 1. frozen multiview-v4 リーダー (`src/tennis_scene/analytics/ingest/v4_reader.py` 等)
- V4 の `manifest.json`・`SHA256SUMS`・各クリップの `clip.json`・`scene.metadata.json`・`scene.npz`・`event_detection` コンポーネント（`event_evidence` を含む）を読み込み、ハッシュ整合を検証する。
- V4 配列の enum 順（`metadata.ball_events`）に従って13 subtype・kind・status・frame_uncertainty・side を読み込む。
- **永続 ID 契約**: NPZ 行番号を永続 ID にせず、component の `event_evidence` にある元 `event_id`（`source_event_id`）を保持する。
- **V4 採否とポイント内適格性の分離**:
  - V4 採用イベント（620件）は `adopted_in_v4=True`、V4 除外イベント（293件）は `adopted_in_v4=False` かつ `excluded_reason` 必須（旧ラベル等は `provenance_json` へ保全）。
  - `clip_009/e003` のような「V4 採用だが元ラベルはポイント外（レット直後の初回バウンド）」を正しく `adopted_in_v4=True, in_point=False` として保持し、通常ラリーに混ぜないこと。
- 47/11/11 の `train`[local path]`val`[local path]`test` split を `manifest.json` から正確に引き継ぐ。

### 2. `projects.json` スコア結合とハイブリッド出典分離 (`score_adapter.py`, `supplement.py`)
- 既存の `src/tennis_scene/clip_studio/project.py` および `src/utils/tennis_scoring.py` を再利用して `projects.json` を読み込み、V4 の clip ID・元動画時刻（`global_start_sec` / `global_end_sec`）・入力ハッシュ・スコア連鎖（`before` → `advance_point` → `after`、試合完了判定）を厳密に照合する。時刻不一致や未対応状態（`not_point`[local path]`multiple_points`[local path]`unknown`）を黙って丸めない。
- **ハイブリッド出典分離方針（親合意の厳守）**:
  - `projects.json` の確認済み `status=='confirmed'` のみ勝者・スコア確定として扱う。
  - `projects.json`（勝者/サーバー/スコア）＋V4 確定イベント（`net_fault`・`service_let`・サーブ列）から決定論的に確定できる範囲のみ `terminal_cause_source='derived_v4_score'` とする（#2757 の DB 制約・`metrics.py` に従い `net_error` および決定論的に確定する `double_fault` に限定。未確認の `out_long`[local path]`out_wide`[local path]`out_both`[local path]`unreturned`[local path]`other` を幾何から自動確定せず `terminal_cause='unknown'`, `terminal_cause_source='unknown'` とする）。
  - 任意の手動補足テーブル（JSON または CSV：clip/point ごとの `terminal_cause`, `error_assessment` (UE/FE), サーブ試行補足、確認者/メモ等のスキーマを定義・検証）を取り込むアダプターを実装する。手動補足がある項目は `manual_confirmed` 出典として上書き/適用し、その入力 SHA-256 を `ingest_runs.manual_supplement_sha256` へ保存する。
  - 試合内の正規ポイント識別子 `(match_context_id, match_point_key)` を決定論的に付与する（1 clip 1 point の連続スコア連鎖や元動画時刻から一意化し、重複クリップが同一実ポイントを指す場合も整合する設計）。

### 3. PostgreSQL 冪等投入・利用可能率監査レポート・CLI 接続
- `ingest_runs`, `match_contexts`, `clips`, `points`, `ball_events` への冪等投入（同一チーム・同一入力ハッシュ・同一定義版での再実行検証・再現性保証）を実装する。
- 全69クリップの監査レポート生成機能を実装する：
  - 入力ハッシュ（V4 manifest / SHA256SUMS / projects.json / 任意手動補足）、クリップ数・時刻照合結果（最大誤差）・スコア連鎖整合
  - 列ごとの利用可能率（`score_before`, `score_after`, `server_team`, `winner_team`, `terminal_cause`, `error_assessment`, V4 採用/除外イベント件数、各 subtype 件数、欠測・未確定理由の内訳）
  - JSON / CSV の機械可読出力、2回実行での完全一致（決定論性）。
- `src/tennis_scene/analytics/__main__.py` へ `ingest` および `audit` サブコマンドを追加（並列開発中の #2759/#2760 と競合を最小化するため、実装本体は `src/tennis_scene/analytics/ingest/` に分離し `__main__.py` の変更は最小限にする）。

### 4. テストと実データ監査
- **DB 非依存ユニットテスト** (`tests/unit/tennis_scene/analytics/test_ingest.py`):
  - V4 採用/除外イベント・`clip_009/e003` パターン・時刻不一致検出・スコア連鎖検証・ハイブリッド出典分離（未確認アウトが `unknown` に留まること）・手動補足テーブル適用・監査レポート生成を検証。
- **PostgreSQL 結合テスト** (`tests/integration/tennis_scene/analytics/test_ingest_postgres.py`, `@pytest.mark.postgres`):
  - 実 PostgreSQL 上で投入→CLI 照会→`export`→再投入冪等性を検証。
- **実データ監査（ローカルの meiji_4cam/multiview-v4 全69クリップ＋`projects.json`）**:
  - 実際に全69クリップに対して監査・PostgreSQL 投入（Compose の PostgreSQL 17 を利用）を実行し、69クリップ・620採用[local path] PR 本文に記録する。2回実行して出力が完全一致することを確認する（検証後の Compose は停止）。

## 検証・引き渡し条件
1. ローカルの PostgreSQL 17 上での結合テスト（`CI_REQUIRE_POSTGRES=1 .venv/bin/python -m pytest -n 0 -m postgres tests/integration/tennis_scene/analytics`）と実データ69クリップ監査を実行する。
2. 新規ユニットテスト・既存 analytics ユニットテスト・pre-commit（Ruff・mypy `--follow-imports=skip`）を通す。
3. 日本語本文で `Closes #2758` を含む ready PR を作成し、PR 番号・完全な HEAD SHA・69クリップ実測結果・検証コマンドを報告して、PR CI 完了前に `PARENT_RESULT {"schema_version":1,"state":"done","wait":null}` を返す。

## 並列実行・共有環境に関する重要注意（必読）
- 共有 `.venv` には `sqlalchemy` / `psycopg` / `alembic` が未導入のため、**共有 `.venv` 自体は `uv sync` 等で書き換えず**、`uv.lock` に固定されたバージョンの追加パッケージだけを一時ディレクトリ（例: `[local path]`）へ展開して `PYTHONPATH` に追加する形でローカル検証を行ってください（`pyproject.toml` / `uv.lock` は変更不要）。
- 他子 Issue（#2759・#2760）と並列実行するため、ローカル Docker Compose 検証では必ず固有ポート・固有プロジェクト名を使用してください：
  `POSTGRES_PORT=15458 docker compose -p tennis-analytics-2758 up -d --wait`
  接続 URL は `postgresql+psycopg:[local path]` とし、検証完了後は `docker compose -p tennis-analytics-2758 down` で停止してください。
