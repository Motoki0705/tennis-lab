---
id: run-i1044-smoke-chunked-v1
type: run
task: blcs
sequence: 47
recorded_at: '2026-10-13'
title: chunk切替時に古いDataLoaderが残る不具合の検出
issue: 1044
provider: codex
session: 01a1327c-d543-7673-8143-2764d9f62447
date: '2026-10-13'
status: failed
config:
  diagnostic: true
  completed_comparison_run: false
metrics: {}
repro:
  commit: e25ce1c8702a37febafea375b10fd3f162f16e6c
  branch: codex/issue-1044-blcs-physics
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/i1044-mpl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m src.tasks.blcs.experiments.issue1044
    --arm rollout_velocity --steps 4 --smoke --chunked --data-root /home/kamimura/projects/tennis-lab/data
    --output blcs/i1044/smoke-chunked-v1
artifacts:
  run_dir: knowledge/runs/run-i1044-smoke-chunked-v1
  log: knowledge/runs/run-i1044-smoke-chunked-v1/queue.log
  output_dir: outputs/blcs/i1044/smoke-chunked-v1
  experiment_config_yaml: knowledge/runs/run-i1044-smoke-chunked-v1/experiment-config.yaml
  config_yaml: knowledge/runs/run-i1044-smoke-chunked-v1/config.yaml
  tb_logdir: outputs/blcs/i1044/smoke-chunked-v1/logs/version_0
  scalars: {status: deleted, record: knowledge/runs/run-i1044-smoke-chunked-v1/artifact-manifest.json#/files/0}
parents: []
relations: []
papers: []
tags:
- issue1044
- qualification
---

次chunkへの切替後、古いDataLoaderが破棄済みsceneのcam_5_ball_uvを参照してKeyErrorとなった。BLCSのchunked設定へreload_dataloaders_every_n_epochs=1を指定し、loaderとchunkの寿命を一致させた。比較学習にはこの未完了runを使わない。

再現資料のcommit・patch・commandはqueue捕捉原本を維持した。巨大checkpointはgitへ追加していない。

kg_curvesで確認したが、4更新以下のスモークを収束曲線として解釈できないため、図は掲載せず全scalarを保存した。
