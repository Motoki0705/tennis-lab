---
id: run-i2445-player-slot-integration
type: run
task: tennis_scene
sequence: 102
recorded_at: '2026-11-24'
title: 選手枠への3D断片統合をMeijiで評価
issue: 2445
provider: codex
date: '2026-11-24'
status: done
summary_impact: none
config:
  device: cpu
  assignment: exclusive player slot paths
  window_seconds: 1.2
  max_fragment_gap_seconds: 1.5
metrics:
  meiji3_dev_projected_idf1: 0.844954927125165
  meiji3_dev_projected_id_switches: 0
  meiji3_dev_projected_fragments: 387
  meiji3_dev_projected_all_id_switches: 737
  meiji3_dev_projected_switches_added: 0
  meiji3_dev_selection_precision: 0.9995349146236131
  meiji3_dev_selection_recall: 0.7317832473976068
  meiji3_reserved_scoring_attempts: 0
  meiji4_full_lineup_clips: 69
  meiji4_full_lineup_measured_frames: 43726
  meiji4_full_lineup_available_frames: 43966
repro:
  commit: cea2690128406c0045090b57c26a6bca68db8818
  command: bash knowledge/runs/run-i2445-multiview-tracking/reproduce.sh all
artifacts:
  run_dir: knowledge/runs/run-i2445-player-slot-integration
  meiji3: knowledge/runs/run-i2445-player-slot-integration/meiji3.json
  meiji4: knowledge/runs/run-i2445-player-slot-integration/meiji4.json
parents:
- run-i2445-multiview-tracking
tags: [meiji, multiview, cpu, tracking]
---

選手枠に3D断片を割り当て、000/000の正側で領域内実測を229から593フレームに増やした。開発4本の選別recallは61.43%から73.18%、選別後IDF1は76.09%から84.50%。新規switchは0件、採用後のfragmentは190から387件。4camの69本では各側2 ID、4人の実測43,726フレーム、020・058はいずれも全フレームで4人の実測を保持した。予約未見の採点は0回。標準pipelineの採用判断は据え置く。 [測定・閾値・出力](../../runs/run-i2445-player-slot-integration/README.md)

2026-11-26のレビュー修正で、連結可能な同一選手の断片がダブルスの2枠を占める問題に対処した。40フレームの再現で2選手を各1枠に割り当て、1選手だけなら欠員1を記録した。修正commitでの再実行では開発4本の各視点の採点と、4cam 69本の選別・配列が上記の結果と一致した。 [修正の実行と照合](../../runs/run-i2445-player-slot-integration/README.md#ダブルス枠のレビュー修正)
