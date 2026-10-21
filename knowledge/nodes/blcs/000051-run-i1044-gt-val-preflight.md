---
id: run-i1044-gt-val-preflight
type: run
task: blcs
sequence: 51
recorded_at: '2026-10-13'
title: 保存物理GTの全validation自己評価
issue: 1044
provider: codex
session: 01a1327c-d543-7673-8143-2764d9f62447
date: '2026-10-13'
status: done
config:
  diagnostic: true
  completed_comparison_run: false
  split: val
  scene_count: 100
  full_clips: true
metrics:
  val_trajectory_rmse_m: 0.0
  val_trajectory_count: 37287.0
  val_velocity_head_rmse_mps: 0.0
  val_position_difference_velocity_rmse_mps: 0.0
  val_rollout_position_rmse_m: 3.8394792721829886e-07
  val_rollout_velocity_rmse_mps: 4.1657735546203305e-07
  val_reprojection_rmse_px: 2.7926874707956362e-05
  val_reprojection_evaluable_rate: 1.0
repro:
  commit: 90e52a25126cb4d5a790e2a36ebfd09c9f1480da
  branch: codex/issue-1044-blcs-physics
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/i1044-mpl .venv/bin/python
    -m src.tasks.blcs.experiments.evaluation --baseline gt --data-root /home/kamimura/projects/tennis-lab/data
    --output outputs/blcs/i1044/gt-val-preflight --split val --fitting-clips 2
artifacts:
  run_dir: knowledge/runs/run-i1044-gt-val-preflight
  log: knowledge/runs/run-i1044-gt-val-preflight/queue.log
  output_dir: outputs/blcs/i1044/gt-val-preflight
  evaluation_json: {status: deleted, record: knowledge/runs/run-i1044-gt-val-preflight/artifact-manifest.json#/files/0}
  pred_val_npz: {status: deleted, record: knowledge/runs/run-i1044-gt-val-preflight/artifact-manifest.json#/files/1}
parents: []
relations: []
papers: []
tags:
- issue1044
- qualification
---

固定validation全100 scene・全37,287 frameのGT自己評価を行った。位置・速度head・同一差分の誤差は0、共有積分器のrollout残差は浮動小数点の丸め水準だった。clean再投影も丸め水準で、clockと尺度の接続を確認した。先頭2 clipの有限反復ODE fitには非ゼロ残差が残り、最適化の有限予算と物理方程式の整合を区別する。モデル学習やtestではない。

正確な値・分母・支持数とclip別診断はbundleのevaluation.json、再計算可能な予測はpred_val.npzに保存した。

学習scalarは存在しないか、学習を伴わない評価のため学習曲線は対象外。

再現資料のcommit・patch・commandはqueue捕捉原本を維持した。巨大checkpointはgitへ追加していない。
