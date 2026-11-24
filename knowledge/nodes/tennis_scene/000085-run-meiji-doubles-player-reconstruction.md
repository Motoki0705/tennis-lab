---
id: run-meiji-doubles-player-reconstruction
type: run
task: tennis_scene
sequence: 85
recorded_at: '2026-11-24'
title: Meiji 4camのダブルス幅による選手再構成
issue: 2460
provider: codex
date: '2026-11-24'
status: running
summary_impact: none
config:
  dataset: meiji_4cam
  video: video_000
  input_run: i2363-all-v1
  selection_core_half_width_m: 5.485
  selection_min_presence_fraction: 0.25
  cpu_workers: 2
metrics:
  cpu_prepared_clips: 69
  verified_selection_masks: 276
  changed_selection_masks: 72
  reused_upstream_references: 1104
  joint_player_frames_before: 133887
  joint_player_frames_after_cpu: 147105
  stopped_clips_before: 14
  stopped_clips_after_cpu: 11
  cpu_clip_seconds_sum: 3812.0344
  pilot_clip_000_joint_player_frames_before: 1416
  pilot_clip_000_joint_player_frames_after_cpu: 1416
  pilot_clip_020_joint_player_frames_before: 0
  pilot_clip_020_joint_player_frames_after_cpu: 2579
  pilot_cpu_wall_seconds: 102.65
repro:
  commit: 27a097321dad3b6f924fa613b5a5887be053e16d
  command: >-
    PYTHONPATH=. CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    .venv/bin/python -B knowledge/runs/run-meiji-doubles-player-reconstruction/reconstruct.py cpu
    --dataset "$MEIJI_FIXED_DATASET" --comparison "$I2440_REPLAY_ROOT"
    --output outputs/meiji-doubles-player-reconstruction-final --workers 2
artifacts:
  run_dir: knowledge/runs/run-meiji-doubles-player-reconstruction
  cpu_clips: knowledge/runs/run-meiji-doubles-player-reconstruction/cpu-clips.csv
  cpu_upstream: knowledge/runs/run-meiji-doubles-player-reconstruction/cpu-upstream.csv
  cpu_summary: knowledge/runs/run-meiji-doubles-player-reconstruction/cpu-summary.json
parents:
- run-meiji-external-ball-20261124
- run-i2440-player-selection
relations: []
papers: []
tags:
- fixed-geometry
- player-selection
---

## 考察

#2458のボール入りsceneを起点に、#2440の既定で全69 clipの選手を選別した。有効な3D関節を持つ
選手フレームは133,887件から147,105件へ増え、clip_012・020・058の3件が各4人で回復した。
276視点の選別maskは#2440のダブルス幅と一致した。
[clip別の計測と再現手順](../../runs/run-meiji-doubles-player-reconstruction/README.md)

身体配置のGPU実行とdatasetの参照切替は後続工程。共有GPUロックとqueueの稼働を確認した。
このrunは保存済みdatasetへの適用結果を記録する。標準の選別設定は #2440 の判断に従う。
