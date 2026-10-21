---
id: run-i1044-smoke-chunked-v2
type: run
task: blcs
sequence: 48
recorded_at: '2026-10-13'
title: chunk切替を含む物理損失の4更新スモーク
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
  chunked: true
  gan: false
metrics:
  val_trajectory_rmse_m: 8.853866577148438
repro:
  commit: 90e52a25126cb4d5a790e2a36ebfd09c9f1480da
  branch: codex/issue-1044-blcs-physics
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/i1044-mpl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m src.tasks.blcs.experiments.issue1044
    --arm rollout_velocity --steps 4 --smoke --chunked --data-root /home/kamimura/projects/tennis-lab/data
    --output blcs/i1044/smoke-chunked-v2
artifacts:
  run_dir: knowledge/runs/run-i1044-smoke-chunked-v2
  log: knowledge/runs/run-i1044-smoke-chunked-v2/queue.log
  output_dir: outputs/blcs/i1044/smoke-chunked-v2
  experiment_config_yaml: knowledge/runs/run-i1044-smoke-chunked-v2/experiment-config.yaml
  config_yaml: knowledge/runs/run-i1044-smoke-chunked-v2/config.yaml
  selection_json: knowledge/runs/run-i1044-smoke-chunked-v2/selection.json
  tb_logdir: outputs/blcs/i1044/smoke-chunked-v2/logs/version_0
  scalars: {status: deleted, record: knowledge/runs/run-i1044-smoke-chunked-v2/artifact-manifest.json#/files/0}
parents: []
relations: []
papers: []
tags:
- issue1044
- qualification
---

再生成データのchunk切替を含め、候補Aの損失による4更新とvalidationが完了した。所要時間は生成・切替を含む。短い処理確認であり、固定snapshotの正式比較runとは母集団も予算も異なる。

選定validationの軌道RMSEは 8.853867 m、処理時間は 113.82秒。選定にtestは使用していない。

再現資料のcommit・patch・commandはqueue捕捉原本を維持した。巨大checkpointはgitへ追加していない。

kg_curvesで確認したが、4更新以下のスモークを収束曲線として解釈できないため、図は掲載せず全scalarを保存した。
