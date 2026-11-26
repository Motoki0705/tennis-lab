---
id: run-i2445-reserved-tracking
type: run
task: tennis_scene
sequence: 103
recorded_at: '2026-11-24'
title: 固定した3D追跡と選手選別のMeiji 3cam予約採点
issue: 2445
provider: codex
date: '2026-11-24'
status: done
summary_impact: none
config:
  device: cpu
  players_per_side: 1
  split: meiji_3cam_reserved
  max_fragment_gap_seconds: 1.5
metrics:
  reserved_scoring_attempts: 1
  reserved_all_clips: 3
  reserved_all_projected_idf1: 0.8041545447130156
  reserved_all_projected_switches: 0
  reserved_all_projected_fragments: 546
  reserved_all_selection_precision: 0.9998647734956051
  reserved_all_selection_recall: 0.6725180772204284
  reserved_without_repaired_clips: 2
  reserved_without_repaired_projected_idf1: 0.8501223961941712
  reserved_without_repaired_projected_switches: 0
  reserved_without_repaired_projected_fragments: 374
  reserved_without_repaired_selection_precision: 1.0
  reserved_without_repaired_selection_recall: 0.7393155526992288
repro:
  commit: 33899a816aa06b9562b2a663fe4a579d9b6eba10
  command: bash knowledge/runs/run-i2445-multiview-tracking/reproduce.sh holdout
artifacts:
  run_dir: knowledge/runs/run-i2445-reserved-tracking
  meiji3: knowledge/runs/run-i2445-reserved-tracking/meiji3.json
parents:
- run-i2445-player-slot-integration
- run-multiview-pose-link-recall-20261124
tags: [meiji, multiview, cpu, tracking, holdout]
---

固定設定で予約未見の採点を一度実行した。全3本の選別後の射影IDF1は80.42%、recallは67.25%。校正補完の001/003を除く2本では85.01%、73.93%だった。全3本で各側1人を選別した。開発用4本との差は全3本でIDF1 −4.08ポイント、recall −5.93ポイント。標準pipelineへの採用判断は据え置く。 [指標と照合記録](../../runs/run-i2445-reserved-tracking/README.md)
