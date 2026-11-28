# 実装指示: #2760 S4 — 競技プロファイルと FlightArc・打点高・片側球速・ネット通過・実着地特徴抽出

親 Issue #2748 の Phase 1 子 Issue #2760 を実装し、日本語本文で `Closes #2760` を含む ready PR を作成してください。

## 背景・前提契約（#2757 基盤・親合意）
- 先行 PR #2762（#2757）で `src/tennis_scene/analytics/` 配下に PostgreSQL 17 + SQLAlchemy 2.0 + Alembic スキーマ（`court_net_profiles`, `flight_arcs`, `shot_geometry_features`, `net_plane_crossings` を含む）が構築済みです。まず `src/tennis_scene/analytics/README.md`, `db/models.py`, `export.py`, `__main__.py` を必ず確認し、全 DB 制約（複合外部キー・`landing_source_kind`×`landing_missing_reason` 整合・`net_plane_crossings.court_net_profile_id` 参照等）に完全準拠してください。
- 既存の `src/utils/schema/court.py`（硬式 ITF 寸法）や `src/utils/rendering/trajectory_analysis.py`、`src/tasks/blcs/generate_dataset/api_server/metrics.py` を調査し、共通化すべき幾何部品は再利用・整理する（既存コードの挙動は壊さないこと）。
- 既存の frozen V4（`data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v4`）は不変とし書き換えないこと。

## 具体的な実装要件

### 1. 競技・コート・ネット・球半径プロファイル (`src/tennis_scene/analytics/profiles.py`)
- 以下の標準プロファイルを定義し、`source_kind`（`rule_default` / `measured` / `custom`）とパラメータを明示保存・切替・上書き可能にする：
  - `soft_tennis_standard`: ネット全面フラット `1.07m`（`net_height_center_m=1.07`, `net_height_post_m=1.07`, `net_profile_kind='flat'`）、球半径 `0.033m`（直径 66mm）、ダブルスコート標準寸法（`23.77m × 10.97m`, サービスライン `6.40m`）。**`meiji_4cam` の既定プロファイル**（`source_kind='rule_default'`、実測値があれば `measured`[local path]`custom` で上書き可）。
  - `tennis_itf_standard`: ネット中央 `0.914m`・ポスト `1.07m`（`net_profile_kind='itf_parabolic'`、既存 `court.py::net_height_at_x` と整合）、球半径 `0.0335m`。
- 任意の `x` に対する `net_height_at_x(x)` と、ネットスパン判定（`|x| <= half_doubles_width + net_post_offset_x`）をプロファイルごとに厳密に計算する。

### 2. FlightArc 分割と一球ごとの幾何特徴抽出 (`src/tennis_scene/analytics/geometry.py`)
- **`FlightArc` 分割**:
  - V4 の `shot` / `bounce` / `net` / `unknown` イベントで連続飛行区間を分割する。接触や不明区間を跨ぐ1本のスプラインや中心差分で処理しない。
  - 各 arc の `start_event` / `end_event`、`start_frame` / `end_frame`、`valid_frame_count` / `total_frame_count`、`source_kind` (`observed` / `estimated`)・品質メタを保持する。
- **解析座標系の規約**:
  - 位置の正本は元の court 座標。打球者が相手コートへ向かう方向を `+Y` に正規化した解析座標も算出する際は **XY を 180° 回転 (`(-x, -y, z)`)** し、Y だけ反転して左右を壊さないこと。
- **打点特徴 (`shot_geometry_features`)**:
  - `contact_frame`, `contact_time_s`, `contact_frame_uncertainty`, `contact_x/y/z_m`, `contact_height_m`（コート面からのボール中心 Z。手首高や root 高を打点高にしない）。イベント時刻の曖昧さが高さ帯を跨ぐ場合は品質メタへ明示記録する。
  - `outgoing_speed_mps` / `incoming_speed_mps`: 接触後／前の別々の連続飛行区間内で片側推定（欠測や接触境界を絶対に跨がない）。算出窓 (`speed_window_frames` に incoming/outgoing 別フレーム列) と推定種別 (`speed_method`) を保存。m/s 正本。
  - `outgoing_heading_rad`: 打球者基準解析座標での初期飛行方向（着地コースと別列）。
- **ネット平面通過 (`net_plane_crossings`)**:
  - `net_crossing` と `net_contact`（V4 の `service_let` / `net_in` / `net_fault`）を明確に別概念・別レコードとする。
  - 有効な連続サンプルが `y=0` 平面を跨ぐ全区間を検出し、線形補間（最小基準）でサブフレーム時刻・`fractional_frame`・`x_m, y_m=0, z_center_m`・`direction_y`・`used_frames_json` を算出する。複数交差・戻り球を失わず、相手側へ向かう最初の有効交差を `is_primary=True` とする。
  - ネットスパン外側を通過した軌道には `crossing_in_net_span=False` を記録し、「ネット上を通過」と区別する。
  - プロファイルから `net_height_m`, `net_clearance_center_m = z_center - z_net(x)`, `net_clearance_bottom_m = net_clearance_center_m - ball_radius_m`, `time_to_net_s = crossing_time_s - contact_time_s` を算出（ネット接触時刻を `time_to_net_s` に代用しない）。`court_net_profile_id` と `profile_source_kind` を保持する。
  - `status`: `crossed` / `no_crossing_confirmed` / `unknown_gap` / `not_applicable` を厳格に区別し、非通過・不明に時刻 0 や高さ 0 を入れない（NULL）。余裕が負という数値だけで失点と決めない。
- **実着地・深さ・コース・ライン余裕**:
  - 次の打者に触れられる前の最初の実バウンド（自陣バウンドやレット後バウンド等は合法着地と区別）から `first_bounce_*`、`landing_depth_m`（相手側 baseline から実着地点までの符号付き距離：深いほど小、baseline 越えは負、コート内へ clamp しない）、`landing_lane`（左右中央3分割）、`landing_course`（打点と実着地からのクロス／ストレート等、境界曖昧なら保留）、`line_margin_m`、`landing_result`（`in`[local path]`out_long`[local path]`out_wide`[local path]`out_both`[local path]`uncertain`[local path]`not_applicable`、service box とラリー領域、singles/doubles、球半径/ライン接地規約を明示）を算出する。
  - **重要（欠測理由と捏造禁止）**: ネットミス・相手ボレーで遮られた球・追跡欠損・途中終了に実着地点を捏造しない。`landing_depth_m=None`、`landing_source_kind=None`、`landing_missing_reason`（`net_stopped`, `intercepted_before_bounce`, `tracking_gap`, `point_truncated`, `self_court_bounce`, `let_after_bounce` 等）を保存する。観測着地と推定着地は `landing_source_kind` (`observed`[local path]`estimated`) で厳密に分離する。
- **版付きビン設定 (`bin_version`, `bins_json`)**:
  - 初期候補（打点高 0.5m 刻み、着地左右3分割、相手 baseline から `0-2m` / `2-4m` / `4m+` / `out_long`）を版付き設定として定義し、V4 の train/val での分布・疎な区分を監査レポートに含める（test を見てビンを調整しない）。

### 3. DB 保存・CLI・検証
- `court_net_profiles`, `flight_arcs`, `shot_geometry_features`, `net_plane_crossings` への PostgreSQL 保存ヘルパーと、`src/tennis_scene/analytics/__main__.py` への最小限の CLI 入口（並列開発中の #2758/#2759 と競合を最小化するため本体は `profiles.py` / `geometry.py` に分離）を接続する。
- **合成真値検証**: 既存合成データ生成コード（`ShotEventInfo` / `PhysicsEvent` 等、または厳密解析弾道モデル）の真値を用いて、ネット通過時刻/位置/余裕・打点高・片側球速・実着地/深さ/ライン余裕の数値誤差を定量検証するユニットテストを追加する。
- **境界 fixture ユニットテスト** (`tests/unit/tennis_scene/analytics/test_profiles.py`, `test_geometry.py`):
  - `soft_tennis_standard` vs `tennis_itf_standard` 切替・実測上書き
  - `net_stopped`・`intercepted_before_bounce`・`tracking_gap`・`point_truncated`（架空着地・通過時刻が生成されないこと）
  - ネット外回り (`crossing_in_net_span=False`)・複数交差・`net_in` 後の継続・接触付近の片側速度（接触跨ぎ禁止）・side 交代（XY 180°回転で左右不変）
- **PostgreSQL 結合テスト** (`tests/integration/tennis_scene/analytics/test_geometry_postgres.py`, `@pytest.mark.postgres`):
  - 全4表の保存・制約・CLI 照会・JSON/CSV エクスポート往復を実 PostgreSQL で検証。
- **V4 全69クリップ実データ監査**:
  - frozen `meiji_4cam/multiview-v4` 全69クリップに対して S4 抽出を実行し、各幾何特徴・ネット通過・実着地の抽出可能率、欠測理由別内訳、train/val のビン分布を実測して PR 本文へ報告する（V4 の ball_3d 再構成値自身を独立 3D 真値と誤認しないこと）。

## 検証・引き渡し条件
1. 合成真値検証テスト・境界 fixture ユニットテスト・ローカル PostgreSQL 17 結合テスト（`CI_REQUIRE_POSTGRES=1 .venv/bin/python -m pytest -n 0 -m postgres tests/integration/tennis_scene/analytics`）・pre-commit（Ruff・mypy `--follow-imports=skip`）を通す（検証後の Compose は停止）。
2. frozen V4 全69クリップの抽出可能率・欠測理由内訳・train/val ビン分布の実測結果を PR 本文に記録する。
3. 日本語本文で `Closes #2760` を含む ready PR を作成し、PR 番号・完全な HEAD SHA・実測結果・検証コマンドを報告して、PR CI 完了前に `PARENT_RESULT {"schema_version":1,"state":"done","wait":null}` を返す。

## 並列実行・共有環境に関する重要注意（必読）
- 共有 `.venv` には `sqlalchemy` / `psycopg` / `alembic` が未導入のため、**共有 `.venv` 自体は `uv sync` 等で書き換えず**、`uv.lock` に固定されたバージョンの追加パッケージだけを一時ディレクトリ（例: `[local path]`）へ展開して `PYTHONPATH` に追加する形でローカル検証を行ってください（`pyproject.toml` / `uv.lock` は変更不要）。
- 他子 Issue（#2758・#2759）と並列実行するため、ローカル Docker Compose 検証では必ず固有ポート・固有プロジェクト名を使用してください：
  `POSTGRES_PORT=15460 docker compose -p tennis-analytics-2760 up -d --wait`
  接続 URL は `postgresql+psycopg:[local path]` とし、検証完了後は `docker compose -p tennis-analytics-2760 down` で停止してください。
