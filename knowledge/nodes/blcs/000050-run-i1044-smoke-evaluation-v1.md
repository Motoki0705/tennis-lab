---
id: run-i1044-smoke-evaluation-v1
type: run
task: blcs
sequence: 50
recorded_at: '2026-10-13'
title: 物理headスモーク重みの全長評価と保存確認
issue: 1044
provider: codex
session: 01a1327c-d543-7673-8143-2764d9f62447
date: '2026-10-13'
status: done
config:
  diagnostic: true
  completed_comparison_run: false
  split: val
  scene_count: 2
  full_clips: true
metrics:
  val_trajectory_rmse_m: 11.164987645050063
  val_trajectory_count: 689.0
  val_velocity_head_rmse_mps: 12.085190756143152
  val_position_difference_velocity_rmse_mps: 15.527036060913368
  val_rollout_position_rmse_m: 0.5802822670778126
  val_rollout_velocity_rmse_mps: 0.8170043079873502
  val_reprojection_rmse_px: 299.8527125838946
  val_reprojection_evaluable_rate: 1.0
repro:
  commit: 90e52a25126cb4d5a790e2a36ebfd09c9f1480da
  branch: codex/issue-1044-blcs-physics
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/i1044-mpl .venv/bin/python
    -m src.tasks.blcs.experiments.evaluation --checkpoint outputs/blcs/i1044/smoke-parameters-v3/checkpoints/final-selected.ckpt
    --data-root /home/kamimura/projects/tennis-lab/data --output outputs/blcs/i1044/smoke-evaluation-v1
    --split val --limit 2 --fitting-clips 0
artifacts:
  run_dir: knowledge/runs/run-i1044-smoke-evaluation-v1
  log: knowledge/runs/run-i1044-smoke-evaluation-v1/queue.log
  output_dir: outputs/blcs/i1044/smoke-evaluation-v1
  evaluation_json: knowledge/runs/run-i1044-smoke-evaluation-v1/evaluation.json
  pred_val_npz: {status: deleted, record: knowledge/runs/run-i1044-smoke-evaluation-v1/artifact-manifest.json#/files/0}
parents: []
relations: []
papers: []
tags:
- issue1044
- qualification
---

4更新の物理headスモーク重みをvalidationの先頭2 clipで全長推論した。直接出力、GT区間での積分、予測イベント区間での積分を区別して保存した。保存NPZだけから通常指標を再計算できることも確認した。品質比較やtest評価ではない。

正確な値・分母・支持数とclip別診断はbundleのevaluation.json、再計算可能な予測はpred_val.npzに保存した。

学習scalarは存在しないか、学習を伴わない評価のため学習曲線は対象外。

再現資料のcommit・patch・commandはqueue捕捉原本を維持した。巨大checkpointはgitへ追加していない。
