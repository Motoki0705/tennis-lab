---
id: run-i2733-meiji-event-labels-v4
type: run
task: tennis_scene
sequence: 108
recorded_at: '2026-11-28'
title: 明治4カメラ全69クリップのボールイベント詳細再注釈とmultiview-v4生成
issue: 2733
provider: gemini
date: '2026-11-28'
status: done
summary_impact: none
config:
  base_benchmark: meiji_4cam/multiview-v3
  output_benchmark: meiji_4cam/multiview-v4
  scene_schema_version: 4
  annotation_schema: meiji_event_labels.v2
  codex_model: gpt-6.2-sol
  codex_reasoning_effort: xhigh
  parallel_chunks: 8
  split_seed: 42
metrics:
  clips: 69
  v1_total_events: 913
  v1_in_point_events: 640
  primary_review_targets: 355
  v2_adopted_events: 620
  v2_excluded_events: 293
  shot_events: 330
  bounce_events: 259
  net_events: 31
  unknown_events: 0
  service_let_events: 4
  service_let_first_bounces: 4
  net_in_events: 1
  net_fault_events: 26
  non_event_byte_identical_clips: 69
  sha256sums_hardlinked_v3_files: 18786
  sha256sums_new_v4_files: 2555
  sha256sums_mismatches: 0
repro:
  commit: 744504dbafe6296f99bfea26cb569454fa72f7d5
  branch: feat/meiji-event-labels-v4
  command: bash knowledge/runs/run-i2733-meiji-event-labels-v4/repro.sh
artifacts:
  run_dir: knowledge/runs/run-i2733-meiji-event-labels-v4
  report: knowledge/runs/run-i2733-meiji-event-labels-v4/README.md
  inputs: knowledge/runs/run-i2733-meiji-event-labels-v4/inputs.json
  ledger_reconciliation: knowledge/runs/run-i2733-meiji-event-labels-v4/ledger-reconciliation.json
  clip_diffs: knowledge/runs/run-i2733-meiji-event-labels-v4/clip-diffs.csv
  adopted_events: knowledge/runs/run-i2733-meiji-event-labels-v4/adopted-events.csv
  excluded_events: knowledge/runs/run-i2733-meiji-event-labels-v4/excluded-events.csv
  verification: knowledge/runs/run-i2733-meiji-event-labels-v4/verification.json
parents:
- run-i2542-meiji-ball-events
relations: []
papers: []
tags:
- meiji_4cam
- ball_events
- scene_result_v4
- multiview_v4
---

## 考察 / Findings

### 要約

明治4カメラ `multiview-v3` 全69クリップ（元注釈 `meiji_event_labels.v1` 全913イベント）に対し、同期4視点映像を Codex（`gpt-6.2-sol`, `xhigh`）で8チャンク並列確認および境界ケースの独立監査（Pass 2）を実施し、詳細イベント注釈 `meiji_event_labels.v2` と `SceneResult v4` 形式のベンチマーク `multiview-v4` を生成・検証した。

- **1対1照合の成立**: 初期確認対象355件（サーブ94、グラウンドストローク183、ボレー44、ネット接触32、ポイント中 `other` 2）を起点に、全913件が採用620件・除外293件へ重複0・未処理0で1対1分割された。
- **スキーマとラベルの解消**: `SceneResult v4` では採用イベントのみを格納して `ball_event_in_point` を廃止し、`ball_event_kind`（4値）と `ball_event_subtype`（13値）へ移行した。旧 `other` と未設定（`None`）コードは0件となり、`unknown` も今回の全69クリップでは0件だった。
- **実データ整合性の確認**: `multiview-v4`（`manifest.json` SHA-256: `8e012944b1407d696a1ab1d325ae3c6b60cf26f071752510ea2fc688b031893c`、`SHA256SUMS` SHA-256: `06a1ba201881cfe8cf873e721adc07d5367c0ae9e3893aaaf0570d098648eaae`）の全21,341ファイルで重複0・不一致0を確認し、全69クリップの `scene.npz` 非イベント配列が不変の v2 `scene_assembly` と byte-for-byte 一致すること、および `MeijiBenchmark.load`・`Annotations.from_scene` による読み戻し通過を確認した。

### 主要な観察と境界調整

- **打球細分類の分布**: サーブ94件は `serve_overhead` 78・`serve_underhand` 16（16件すべて同一の左利き選手で一貫）、グラウンドストロークは `groundstroke_forehand` 156・`groundstroke_backhand` 24、ボレーはフォア／バック優先規則により `volley_forehand` 25・`volley_backhand` 19・`volley_center` 1（`clip_000` `e004` f=225）、スマッシュは既存引継ぎ11件となった。
- **ネット接触の分離とサービスレット初回バウンドの統一**: 旧 `let_cord` 5件を `service_let` 4件（`clip_009@106`, `clip_012@135`, `clip_023@428`, `clip_035@199`）とラリー中の `net_in` 1件（`clip_026@281`）へ分離した。確定した4件の `service_let` すべてで直後の初回バウンド（`clip_009@132`, `clip_012@152`, `clip_023@460`, `clip_035@279`）を採用に統一した（旧 `in_point=false` の2件を追加採用）。
- **ポイント終了境界の是正**: `net_fault` 直後の打者側落下バウンドで旧 `in_point=true` となっていた14件を除外へ統一した。さらに Pass 2 独立監査により、`clip_016`（第1サーブ・ロングフォルト後の返送列5件、旧 `net_fault` f=282 を含む）および `clip_056`（第1サーブ・フォルト後の返送・前衛止球列3件、旧ポイント中 `other` `e004@262` を含む）を除外し、一方で `clip_035` `e041@2045`（旧ポイント中 `other`）は同側後衛の連続接触によるポイント終了原因として `shot/volley_forehand` で採用した。

### 限界と `summary.md` への影響（`summary_impact: none`）

本記録は学習モデルの新規評価や標準パイプライン採用構成の変更ではなく、ベンチマーク注釈（`multiview-v4`）および `SceneResult v4` スキーマの構築・検証までを対象とする。そのため `knowledge/summary.md` に記載された 3D ボールイベント検出モデル（Refiner／規則）の現在の判断は変更せず `summary_impact: none` とした。TensorBoard 曲線は学習ランではないため該当しない。詳細定義・実クリップ判定例・全913件遷移表・全69クリップ差分は [検証レポート](../../runs/run-i2733-meiji-event-labels-v4/README.md) と同ディレクトリの CSV/JSON 成果物を参照。

### 次に有効な実験

- `multiview-v4` の固定 split（`train` 47 / `val` 11 / `test` 11）を用いて、3D ボールイベント検出モデル（`ball_refiner_3d/post_training` 等）の細分類（13クラス）および統一境界下での再評価・追加学習を行う。
