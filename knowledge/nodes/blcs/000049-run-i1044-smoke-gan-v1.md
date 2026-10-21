---
id: run-i1044-smoke-gan-v1
type: run
task: blcs
sequence: 49
recorded_at: '2026-10-13'
title: 任意GAN経路と物理損失の接続確認
issue: 1044
provider: codex
session: 01a1327c-d543-7673-8143-2764d9f62447
date: '2026-10-13'
status: done
config:
  diagnostic: true
  completed_comparison_run: false
  arm: rollout_velocity
  seed: 42
  optimizer_updates: 4
  smoke: true
  chunked: false
  gan: true
metrics:
  val_trajectory_rmse_m: 17.779537200927734
repro:
  commit: 90e52a25126cb4d5a790e2a36ebfd09c9f1480da
  branch: codex/issue-1044-blcs-physics
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/i1044-mpl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m src.tasks.blcs.experiments.issue1044
    --arm rollout_velocity --steps 4 --smoke --gan --data-root /home/kamimura/projects/tennis-lab/data
    --output blcs/i1044/smoke-gan-v1
artifacts:
  run_dir: knowledge/runs/run-i1044-smoke-gan-v1
  log: knowledge/runs/run-i1044-smoke-gan-v1/queue.log
  output_dir: outputs/blcs/i1044/smoke-gan-v1
  experiment_config_yaml: knowledge/runs/run-i1044-smoke-gan-v1/experiment-config.yaml
  config_yaml: knowledge/runs/run-i1044-smoke-gan-v1/config.yaml
  selection_json: knowledge/runs/run-i1044-smoke-gan-v1/selection.json
  tb_logdir: outputs/blcs/i1044/smoke-gan-v1/logs/version_0
  scalars: {status: deleted, record: knowledge/runs/run-i1044-smoke-gan-v1/artifact-manifest.json#/files/0}
parents: []
relations: []
papers: []
tags:
- issue1044
- qualification
---

GAN有効のmanual optimizationでglobal optimizer stepが4に達し、validationとcheckpoint保存を完了した。この更新数はGとDのoptimizerを含み、GAN無効の4更新と教師あり更新数が同じではない。本比較10 runのGANは無効に固定する。

選定validationの軌道RMSEは 17.779537 m、処理時間は 33.15秒。選定にtestは使用していない。

再現資料のcommit・patch・commandはqueue捕捉原本を維持した。巨大checkpointはgitへ追加していない。

kg_curvesで確認したが、4更新以下のスモークを収束曲線として解釈できないため、図は掲載せず全scalarを保存した。
