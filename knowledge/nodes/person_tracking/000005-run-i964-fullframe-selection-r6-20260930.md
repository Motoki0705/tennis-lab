---
id: run-i964-fullframe-selection-r6-20260930
type: run
task: person_tracking
sequence: 5
recorded_at: '2026-09-30'
title: ROI前の人物7条件と観測を切らないコート選別の開発比較
issue: 964
provider: codex
status: running
config:
  device: cpu
  core_half_width_m: 4.115
  baseline_limit_m: 16.885
  min_dwell_fraction: 0.25
  max_candidates: 6
  observation_spatial_gate: false
metrics:
  source_variants: 7
  camera_clips_per_source: 12
  reference_player_units: 20558
  wide_reference_units: 143
artifacts:
  run_dir: knowledge/runs/run-i964-fullframe-selection-r6-20260930
  output_dir: /home/kamimura/projects/tennis-lab/outputs/person_tracking/evaluate/court_selection/i964-fullframe-r6-20260929
parents:
- run-i964-coco-fullframe-r5-20260929
- run-i964-court-selection-r5-20260929
relations: []
papers: []
tags:
- dev-only
- cpu
- pre-roi
- selection
date: '2026-09-30'
repro:
  commit: 80603287169034ac864991888ec7537ce2971d03
  branch: campaign930/i964-2-tracking
---

## 実施中の範囲

4本の固定dev clipに対し、FT .01/.02/.05、全画面COCO .05/.10/.30、両者 .30 unionを同じ800/1333・ROI前で比較する。全24 raw archiveのhashを確認し、全84 source camera-clipの共通BoT-SORT再生が完了した。CLIP/選別とcamera間対応をCPU実行中。最終比較・推薦は全条件完了後に確定する。

前回データ固定の補助auditでは、|x|>5.485mの選手参照143 unitについてFT保持46→141、旧union 5→123、旧経路0→123へ回復した。いずれも隣コート395/395除外を維持した。領域で選択済み断片の観測を切らない修正の効果で、ROI差が残るためこの補助表を公平なsource比較にはしない。

参照はCOCO由来でCOCOに有利。旧box一致は検出recallではなく、未ラベル予測をFPと扱わない。未見・最終tracking/encoder比較・既定変更・GPU追加は実施していない。学習を含まず、TensorBoard曲線は無い。
