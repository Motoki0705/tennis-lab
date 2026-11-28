# 実装指示: #2757 PostgreSQL正規化DB基盤・Docker Compose・指標辞書・CLI/エクスポート骨格

親 Issue #2748 の Phase 1 先行基盤として、子 Issue #2757 を実装し、日本語本文で `Closes #2757` を含む ready PR を作成してください。

## 背景・合意事項（親 #2748）
- ベースは PR #2745 マージ後の `main`（`755941017c00360bb79f4e8e047faa0c085110b1`）。
- 初期段階から **PostgreSQL 17 + Docker Compose** を採用し、**`psycopg[binary]` + SQLAlchemy 2.0 + Alembic** でスキーマとマイグレーションを管理する。
- 将来の Web API・認証・チーム別アクセス制御を見据えたスキーマ境界を初期マイグレーションに組み込む（Web UI・認証ログイン実体・PostgreSQL RLS 有効化は別 Issue に分離）。
- 指標辞書契約・JSON/CSV エクスポート・CLI 照会骨格・CI の PostgreSQL 結合テストジョブまでを本 PR の範囲とする。
- 下流の #2758 (S1)・#2759 (S2)・#2760 (S4) がこの基盤の上で並列実装されるため、契約を明確かつ再利用しやすく設計すること。

## 具体的な実装要件

### 1. 依存関係と Docker Compose
- `pyproject.toml` に `psycopg[binary]`, `sqlalchemy>=2.0`, `alembic` を追加し、`uv.lock` を更新する（`uv lock` が長時間になりそうなら `parent.local_jobs start` を利用）。
- worktree の共有 `.venv` を破壊しないよう配慮し、ローカル検証で新依存が使えることを確認すること。
- リポジトリルートに `docker-compose.yml`（`postgres:17`、ヘルスチェック、環境変数による DB 名/ユーザー/パスワード/ポート切替、ローカル開発用既定値、データ永続化ボリューム）を配置する。

### 2. PostgreSQL 正規化スキーマと Alembic (`src/tennis_scene/analytics/db/`)
- SQLAlchemy 2.0 (`DeclarativeBase` / `Mapped`) + `psycopg` v3 (`postgresql+psycopg:[local path]`) + Alembic 初期マイグレーションを実装する。
- `alembic upgrade head` → `alembic downgrade base` → `alembic upgrade head` の往復が通ること。
- **アクセス境界層（将来の Web API / 認証 / チーム別アクセス制御）**:
  - `organizations` (`id`, `slug` unique, `name`, `created_at`, `updated_at`)
  - `teams` (`id`, `organization_id` FK, `slug`, `name`, `created_at`, `updated_at`, unique `(organization_id, slug)`)
  - `users` (`id`, `external_subject` unique, `email`, `display_name`, `is_active`, `created_at`, `updated_at`) — 認証実体（パスワード・セッション）を持たない最小参照表
  - `team_memberships` (`id`, `team_id` FK, `user_id` FK, `role`, `created_at`, `updated_at`, unique `(team_id, user_id)`)
- **全ドメイン表の共通列・制約・インデックス**:
  - `owner_team_id` (NOT NULL FK -> `teams.id`)
  - `visibility` (`private` / `team` / `org` / `public`, CHECK 制約、default `'team'`)
  - `created_at`, `updated_at` (TIMESTAMPTZ)
  - 将来の PostgreSQL RLS 導入を見据えた `(owner_team_id, ...)` 複合インデックス
- **Phase 1 ドメイン表**:
  - `court_net_profiles`: `profile_key`, `sport_type` (`soft_tennis`[local path]`tennis`), `net_height_center_m`, `net_height_post_m`, `net_profile_kind` (`flat`[local path]`itf_parabolic`[local path]`measured_curve`), `net_post_offset_x_m`, `ball_radius_m`, `source_kind` (`rule_default`[local path]`measured`[local path]`custom`), `parameters_json`
  - `ingest_runs`: `dataset_id`, `video_id`, `v4_manifest_sha256`, `projects_json_sha256`, `manual_supplement_sha256` (nullable), `schema_version`, `definition_version`, `status`, `summary_json`
  - `match_contexts`: `ingest_run_id` FK, `dataset_id`, `video_id`, `match_key`, `sport_type`, `rules_json`, `side_a_name`, `side_b_name`, `first_server`, `court_net_profile_id` FK, `time_conversion_json`
  - `clips`: `match_context_id` FK, `clip_id`, `clip_name`, `split` (`train`[local path]`val`[local path]`test`[local path]`unassigned`), `global_start_sec`, `global_end_sec`, `fps`, `num_frames`, `scene_schema_version`, `scene_npz_sha256`, `annotation_sha256`, `metadata_json`
  - `points`: `clip_id` FK, `point_key`, `start_frame`, `end_frame`, `completeness` (`complete`[local path]`truncated_start`[local path]`truncated_end`[local path]`truncated_both`[local path]`unknown`), `score_status` (`confirmed`[local path]`auto`[local path]`unknown`[local path]`not_point`[local path]`multiple_points`), `score_before_json`, `score_after_json`, `server_team` (`A`[local path]`B`[local path]`winner_team` (`A`[local path]`B`[local path]`score_source`, `terminal_cause` (`net_error`[local path]`out_long`[local path]`out_wide`[local path]`out_both`[local path]`double_fault`[local path]`unreturned`[local path]`other`[local path]`unknown`), `terminal_cause_source` (`manual_confirmed`[local path]`derived_v4_score`[local path]`unknown`), `error_assessment` (`forced`[local path]`unforced`[local path]`undetermined`), `error_assessment_source`, `note`
  - `ball_events`: `clip_id` FK, `point_id` FK (nullable), `source_event_id` (V4 元 event ID、NPZ 行番号を永続 ID にしない), `frame`, `time_s`, `frame_uncertainty`, `kind` (`shot`[local path]`bounce`[local path]`net`[local path]`unknown`), `subtype` (V4 13 subtype 対応), `status` (`certain`[local path]`probable`[local path]`uncertain`), `court_side` (`-1`[local path]`0`[local path]`1`), `event_source` (`annotation`[local path]`rule`[local path]`refiner`), `in_point` bool, `excluded_reason` (nullable), `provenance_json`
  - `shot_hitter_estimates`: `ball_event_id` FK, `rule_version`, `resolution_status` (`resolved`[local path]`ambiguous`[local path]`unresolved`), `local_player_index` (nullable), `local_player_id` (nullable), `selected_landmark` (`left_wrist`[local path]`right_wrist`[local path]`root`[local path]`best_score` (nullable), `margin_to_second` (nullable), `rejection_reason` (nullable), `candidates_json`, `evidence_json`
  - `flight_arcs`: `clip_id` FK, `point_id` FK (nullable), `shot_event_id` FK (nullable), `arc_index`, `start_event_id` FK (nullable), `end_event_id` FK (nullable), `start_frame`, `end_frame`, `valid_frame_count`, `total_frame_count`, `source_kind` (`observed`[local path]`estimated`), `quality_json`
  - `shot_geometry_features`: `ball_event_id` FK, `flight_arc_id` FK (nullable), `profile_id` FK, `definition_version`, `contact_frame`, `contact_time_s`, `contact_frame_uncertainty`, `contact_x_m`, `contact_y_m`, `contact_z_m`, `contact_height_m`, `outgoing_speed_mps`, `incoming_speed_mps`, `speed_window_frames`, `speed_method`, `outgoing_heading_rad`, `first_bounce_frame`, `first_bounce_time_s`, `first_bounce_x_m`, `first_bounce_y_m`, `first_bounce_z_m`, `landing_depth_m`, `landing_lane`, `landing_course`, `line_margin_m`, `landing_result` (`in`[local path]`out_long`[local path]`out_wide`[local path]`out_both`[local path]`uncertain`[local path]`not_applicable`), `landing_missing_reason`, `bin_version`, `bins_json`, `quality_json`
  - `net_plane_crossings`: `flight_arc_id` FK, `ball_event_id` FK (nullable), `crossing_index`, `is_primary` bool, `status` (`crossed`[local path]`no_crossing_confirmed`[local path]`unknown_gap`[local path]`not_applicable`), `crossing_in_net_span` bool (nullable), `fractional_frame`, `time_s`, `time_to_net_s`, `x_m`, `y_m`, `z_center_m`, `direction_y` (`-1`[local path]`1`[local path]`net_height_m`, `net_clearance_center_m`, `net_clearance_bottom_m`, `profile_source_kind`, `source_kind` (`observed`[local path]`estimated`), `used_frames_json`, `fit_error_m`
- **Phase 2 予約表（#2746 完了後に接続）**:
  - `player_identity_maps`, `team_pair_assignments`, `serve_attempts`, `metric_snapshots`（`#2748` の出力契約カラムを満たす）

### 3. 指標辞書契約 (`src/tennis_scene/analytics/metrics.py`)
- Issue #2748 の「構想1：2. 基本スタッツの最小セット」、スコア結合追加項目、条件別得点率／ミス分析の全 `metric_id` を定義版付きで機械可読に定義する。
- 出力レコード契約：`metric_id, definition_version, subject_type, subject_id, filters, bin_version, numerator, denominator, value, candidate_count, resolved_count, unknown_count, excluded_by_reason, status, source_ids`。
- 不変条件をコードで強制・検証する：
  - `denominator == 0` のとき `value is None`、`status` は `complete`[local path]`partial`[local path]`unavailable` を厳格に区別。
  - `rally.contact_count`（最終ミス接触含む）と `rally.completed_shot_count`（最終ミス接触除く）を明確に区別。
  - 条件別 point 得点率の `(point_id, subject_id, bin_id)` 一意化ヘルパー、ミス率（打球試行単位）との分母分離。
  - 実着地深さ・コース別は「観測着地を持つ打球」対象であることを明示し、全ミス率を捏造しない契約。
  - `double_fault` と `net_error` の排他的計上、UE/FE 未評価を 0 件にしない契約。

### 4. JSON/CSV エクスポート & CLI 照会骨格 (`src/tennis_scene/analytics/export.py`, `__main__.py`)
- `python -m src.tennis_scene.analytics` で以下を実行可能にする：
  - `db upgrade` / `db downgrade` / `db check`
  - `metrics list` / `metrics export --format json|csv`
  - `query`（対象テーブル・`--owner-team` 等でフィルタし JSON/CSV 出力）
  - `export`（指定 `ingest_run` / `match_context` 配下のレコードを再現可能な JSON/CSV へ書き出し）

### 5. テストと CI（`.github/workflows/ci.yml`）
- **DB 非依存ユニットテスト**（`tests/unit/tennis_scene/analytics/`）：指標辞書・契約検証・スキーマメタデータ検査（全ドメイン表の `owner_team_id`[local path]`visibility`[local path]）・エクスポート整形。通常 CPU テストで常時実行。
- **PostgreSQL 結合テスト**（`tests/integration/tennis_scene/analytics/` 等）：実 PostgreSQL に対し Alembic `upgrade head -> downgrade base -> upgrade head`、マルチテナント境界テーブル・全ドメイン表の CRUD/制約違反検出、fixture 投入→CLI 照会→JSON/CSV エクスポートの一致を検証。`DATABASE_URL` 未設定時の扱いも明示（静かなフォールバックで成功扱いにしない。CI の postgres ジョブでは必須実行、通常の CPU shard とは明確に分離）。
- `.github/workflows/ci.yml`（および必要なら `scripts/ci_metrics/` 等のレーン設計・テスト）を確認・整合させた上で、`postgres:17` service container を用いた PostgreSQL 結合テストジョブを追加し、集約 `Python` チェックに含める。
- `src/tennis_scene/analytics/README.md` を作成し、構成・Docker Compose 起動・Alembic・CLI・スキーマ契約・CI テスト方針を簡潔に記載する。

## 検証・引き渡し条件
1. ローカルで `docker compose up -d` により PostgreSQL 17 を起動し、Alembic 往復・PostgreSQL 結合テスト・CLI 照会・JSON/CSV エクスポートを実測検証する（検証後、必要に応じクリーンアップまたは停止状態を明記）。
2. 新規ユニットテスト・関連既存テスト（CI workflow 整合テスト等があれば含む）・pre-commit（ruff + mypy `--follow-imports=skip`）をすべて通す。
3. `git check-ignore` で新規ファイル（`docker-compose.yml`, `alembic.ini`, migrations 等）が stage 漏れしていないことを確認する。
4. 日本語本文で `Closes #2757` を含む ready PR を作成し、PR 番号・完全な HEAD SHA・検証コマンドと実測結果・未確認事項を報告して、PR CI 完了前に `state: "done"` で終了する。
