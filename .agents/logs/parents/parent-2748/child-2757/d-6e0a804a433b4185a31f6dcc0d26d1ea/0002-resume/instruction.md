# PR #2762 正式レビュー major 指摘（6件すべて採用）の修正指示

正式レビュー（run `parent-2748-7cb0f4ebdbfa455b9e5cef246b501d24`、HEAD `a175d094f6a7ff0e478601d323244a76dd64a6aa`）で挙がった以下6件の major 指摘を**すべて採用**します。同一ブランチ・同一 PR (#2762) に対して修正し、テスト・Alembic 往復・PostgreSQL 結合テスト・pre-commit を通して push してください。

---

## 修正対象（全6件）

### 1. 同一チーム内の別試合・別クリップ誤参照の防止（複合外部キー）
- 指摘: `ShotGeometryFeature` と `NetPlaneCrossing` の打球 (`ball_event_id`)・飛行区間 (`flight_arc_id`)、および `ServeAttempt` のポイント (`point_id`)・打球 (`shot_event_id`) が、`owner_team_id` と行 ID だけの参照になっており、同一チーム内の別試合・別クリップを結び付けられてしまい、試合エクスポートで根拠が欠ける。
- 修正:
  - 同一クリップ（および必要に応じて同一ポイント）を共有すべき参照を `(owner_team_id, clip_id, ...)` 等の複合外部キーで DB レベルで厳密に拘束する（初期マイグレーション `0001_create_analytics_schema.py` と `models.py` の双方を同期更新。まだ未マージの新規初期リビジョンなので初期リビジョン自体を更新し `alembic check` と往復を通す）。
  - 既存の PostgreSQL 結合テストに「同一チーム・別試合（別クリップ）」の誤参照が外部キー制約で拒否されるテストを追加する。

### 2. V4 の採否とラリー集計適格性の分離＋終了原因×出典の許容組み合わせ固定
- 指摘:
  1. `ball_events` が `in_point=false` かつ `excluded_reason=NULL` を許容する。固定 V4 の `clip_009/e003` は元ラベルではポイント外だが、レット直後の初回バウンドとして V4 に採用されている。
  2. `counted_terminal_cause('out_long', 'derived_v4_score')` が `out_long` を返し、DB 制約も許容している。人手未確認のアウトを不明 (`unknown`) で保持する合意に反する。
- 修正:
  - `ball_events` で「V4 採用の有無（例: `adopted_in_v4` bool）」と「ラリー集計への適格性（`in_point` / `rally_eligible` 等）」を明確に分離し、V4 で除外されたイベント（`adopted_in_v4=false`）には `excluded_reason IS NOT NULL` を必須とする DB CHECK 制約を課す。`clip_009/e003` のように「V4 採用かつポイント外／ラリー非対象（レット後初回バウンド）」を表現できる契約にし、テストで検証する。
  - 終了原因 (`terminal_cause`) と出典 (`terminal_cause_source`) の許容組み合わせを DB CHECK 制約と `metrics.py` の両方で一致・固定する：
    - `derived_v4_score` で許容するのは、projects.json（勝者/サーバー/スコア）＋V4 確定イベント（`net_fault`・`service_let`・サーブ列）から決定論的に確定できる範囲（`net_error`・`double_fault` 等、合意に沿う確定可能集合）のみ。人手未確認の `out_long`[local path]`out_wide`[local path]`out_both`[local path]`unreturned`[local path]`other` を `derived_v4_score` で計上・保存できないように拒否する（未確認時は `terminal_cause='unknown'` / `terminal_cause_source='unknown'`）。
    - `manual_confirmed` は人手確認済みの各終了原因を許容。
    - `counted_terminal_cause('out_long', 'derived_v4_score')` が計上されない（または明示的に拒否される）ことをユニットテスト・DB 結合テストで確認する。

### 3. 観測着地とネット余裕の出典を DB から確定可能にする
- 指摘:
  1. `ShotGeometryFeature` は `first_bounce_frame` があるだけで着地深さや in/out 判定を受理し、観測 (`observed`) と推定 (`estimated`) を区別する必須項目がない。また `landing_missing_reason='tracking_gap'` と着地深さの併存も通る。
  2. `NetPlaneCrossing` は `profile_source_kind` のみで、算出に使った具体的な `court_net_profiles` 行を特定できない。
- 修正:
  - `ShotGeometryFeature` に着地の出典区分（例: `landing_source_kind`: `observed` / `estimated` / `NULL`）を追加し、「着地あり（`first_bounce_*` / `landing_depth_m` / `landing_result != 'not_applicable'` 等）⇔ `landing_source_kind IS NOT NULL` かつ `landing_missing_reason IS NULL`」「着地欠測 ⇔ `landing_missing_reason IS NOT NULL` かつ着地座標・深さ等はすべて `NULL`」という整合 CHECK 制約を DB とモデルに固定する。観測着地限定集計では `landing_source_kind == 'observed'` で明確に識別可能にする。
  - `NetPlaneCrossing` に `court_net_profile_id`（`owner_team_id` と組の複合 FK → `court_net_profiles`）を追加し、ネット高・余裕の算出元プロファイル行を DB から一意に特定できるようにする（`status='crossed'` のとき必須等、整合制約も明示）。
  - PostgreSQL 結合テストに、観測着地・推定着地・着測欠測（`tracking_gap` と深さ併存の拒否）・ネット通過プロファイル参照の境界例を追加する。

### 4. 複数クリップにまたがる同一ポイントの識別規則
- 指摘: `points.id` はクリップごとの行、`point_key` の一意性もクリップ内に限られており、複数クリップが同一実ポイントをカバーするときに `unique_point_bins` が重複排除できない。
- 修正:
  - `points` に試合内 (`match_context_id`) で同一実ポイントを識別する正規ポイントキー（例: `canonical_point_id` / `match_point_key`、同一試合内で重複クリップの同一点が同じ値を持つ設計。必要なら `match_context_id` を含めた複合表現または明示列）を導入する。
  - `metrics.py` の `unique_point_bins`（および関連ヘルパー）が DB 行 ID ではなく試合内の正規ポイント識別子で一意化されるように契約を固定し、「重複クリップの同一ポイントは 1 件、同じビンの別ポイントは別件になる」ユニットテストおよび DB 結合テストを追加する。

### 5. `conditional.shot_error_rate` のサーブ／ラリー混在時の分母一意化
- 指摘: 打点高だけを指定した `conditional.shot_error_rate` が有効として受理されるが、同じ高さ帯にサーブ fault とラリー打球が混在した場合に分子・分母の規則が定まらない。
- 修正:
  - `conditional.shot_error_rate` において `shot_role` フィルタ（例: `serve` / `return` / `rally` / `non_serve` のいずれか）を必須とする（サーブ fault は `shot_role='serve'` でのみ、非サーブ打球ミスは `shot_role in ('return', 'rally', 'non_serve')` で集計し、混在を契約レベルで禁止する）。
  - 打点高のみ（`shot_role` なし）の `conditional.shot_error_rate` を明確に拒否し、サーブとラリーが手計算期待値通りに分離されるテストを追加する。

### 6. 件数指標の成立条件・根拠 `source_ids` 必須化と保存／出力経路での `MetricRecord` 検証
- 指摘:
  1. `point.played_count` で分子3・分母1・値3・`status='complete'`・`source_ids={}` が受理されてしまう。
  2. `metric_snapshots` への書き込みと CLI 出力・エクスポート経路が `MetricRecord` の検証を通らない。
- 修正:
  - 各 `MetricDefinition` に計数関係（例: 件数系で対象数と一致する `exact_count` / 割合系 `ratio` (`0 <= numerator <= denominator`) / ポイント当たり複数接触を許す `multi_per_unit` 等）と、`denominator > 0` 時に必要な根拠キー（`source_ids` に分子・分母の対象 ID へ遡れる非空の根拠を必須とする規則、および件数・一意性の整合検証）を定義し、`MetricRecord` の検証で強制する。
  - `metric_snapshots` への書き込みヘルパー（および ORM バリデーション／リポジトリ関数）と CLI 照会／エクスポート経路で `MetricRecord` 検証を通るように接続し、件数不整合や空の `source_ids` を持つ `complete` 指標が保存・出力されないことをユニットテストおよび PostgreSQL 結合テストで確認する。

---

## 検証・引き渡し条件
1. ローカルの PostgreSQL 17（`docker compose`）上で `alembic upgrade head` → `downgrade base` → `upgrade head` と `alembic check`、および `CI_REQUIRE_POSTGRES=1 .venv/bin/python -m pytest -n 0 -m postgres tests/integration/tennis_scene/analytics` を実行し、全件合格を確認する。
2. DB 非依存ユニットテスト・CI 構成テスト・pre-commit（Ruff・mypy `--follow-imports=skip`）をすべて通す。
3. 同一ブランチ `codex/parent-2748-issue-2757-primary` に commit・push し、PR #2762 の説明も更新した上で、新 HEAD SHA・各指摘（1〜6）の修正内容と検証コマンド・実測結果を報告し、PR CI 完了前に `PARENT_RESULT {"schema_version":1,"state":"done","wait":null}` を返す。
