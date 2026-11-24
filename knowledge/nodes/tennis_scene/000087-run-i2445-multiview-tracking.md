---
id: run-i2445-multiview-tracking
type: run
task: tennis_scene
sequence: 87
recorded_at: '2026-11-24'
title: Meijiの時間窓による3D人物追跡と選手選別
issue: 2445
provider: codex
date: '2026-11-24'
status: done
summary_impact: none
config:
  device: cpu
  window_seconds: 1.2
  commit_seconds: 0.5
  max_occlusion_seconds: 0.4
  meiji3_players_per_side: 1
  meiji4_players_per_side: 2
metrics:
  meiji3_dev_projected_idf1: 0.7609206483099356
  meiji3_dev_projected_id_switches: 0
  meiji3_dev_projected_fragments: 190
  meiji3_dev_selection_precision: 0.999446027223805
  meiji3_dev_selection_recall: 0.6143107306158186
  meiji3_reserved_scoring_attempts: 0
  meiji4_selected_four_clips: 69
  meiji4_source_frames: 46228
  meiji4_full_lineup_measured_frames: 43726
  meiji4_full_lineup_available_frames: 43966
  meiji4_selected_measured_frames: 182357
  meiji4_selected_predicted_frames: 291
repro:
  commit: 6aecd7bb11a32fd5eb5c8a1491d0092276c133d1
  command: bash knowledge/runs/run-i2445-multiview-tracking/reproduce.sh all
artifacts:
  run_dir: knowledge/runs/run-i2445-multiview-tracking
  meiji3: knowledge/runs/run-i2445-multiview-tracking/meiji3.json
  meiji4: knowledge/runs/run-i2445-multiview-tracking/meiji4.json
parents:
- run-multiview-pose-link-recall-20261124
tags: [meiji, multiview, cpu, tracking]
---

保存済み3D候補を1.2秒の窓で連結し、ダブルス幅での観測滞在と運動から選手を選んだ。meiji_4camは全69本で両側2 ID、020・058では全元フレームに各4人の実測が揃った。開発用Meiji 3camの人手ラベルによるIDF1は76.09%、選別のprecision 99.94%、recall 61.43%。000/000の正側に選別不足が残る。予約未見の採点は、ViTPose不足と共有GPUロックの権限エラーにより0回。標準pipelineの採用判断は据え置く。 [設定根拠・入力の版・計測結果](../../runs/run-i2445-multiview-tracking/README.md)
