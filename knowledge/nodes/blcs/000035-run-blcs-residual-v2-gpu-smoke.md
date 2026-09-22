---
id: run-blcs-residual-v2-gpu-smoke
type: run
task: blcs
sequence: 35
recorded_at: '2026-09-21'
title: BLCS Court14残差v2 GPU事前確認（warmup設定拒否）
provider: codex
session: 01a0bf91-5494-74c0-8793-3424866618ed
date: '2026-09-21'
status: failed
config:
  recipe: train_triangulation_residual_v2
  model: blcs_triangulation_residual_v2
  loss: balanced_regret
  train_scenes: 64
  val_scenes: 32
  test_scenes: 32
  views: 6
  sequence_length: 64
  batch_size: 32
  precision: bf16-mixed
  max_epochs: 1
  warmup_epochs: 1
  seed: 42
metrics: {}
repro:
  commit: 4671f9bad0873a23054a3311009b9ecf4a6bbc88
  branch: codex/plcs-triangulation-residual
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
    .venv/bin/python -c 'import runpy,torch; runpy.run_module("src.tasks.blcs.scripts.train_triangulation_residual",run_name="__main__");
    print("CUDA_PEAK",torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved(),flush=True)'
    --config-name train_triangulation_residual_v2 paths.data_root=/home/kamimura/projects/tennis-lab/data
    paths.output_root=/home/kamimura/projects/tennis-lab/outputs run.output_dir=blcs/triangulation_residual_v2_gpu_smoke_20260921
    run.seed=42 data.train_limit=64 data.val_limit=32 data.test_limit=32 data.num_workers=2
    data.min_views=6 data.max_views=6 v2.evaluation_views=6 training.trainer.max_epochs=1
    training.trainer.enable_progress_bar=false
artifacts:
  run_dir: knowledge/runs/run-blcs-residual-v2-gpu-smoke
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1789944199254774101_314430_blcs-residual-v2-gpu-smoke.log
parents:
- run-blcs-triangulation-residual-physics-v1
relations: []
papers: []
tags:
- triangulation-residual-v2
- gpu-smoke
- diagnostic
---

## 考察 / Findings

### 要約
1 epochへの短縮に対してwarmup_epochs=1を残したため、設定検証が「warmupはmax_epochs未満」として拒否した。学習・推論は開始しておらず、性能metricや曲線は存在しない。

### 対応
事前確認だけwarmup_epochs=0へ変更し、別run（r2）で実行した。元コマンド・失敗logは証拠として保存する。本学習の30 epoch/warmup 1 epoch設定とは無関係。元コマンドとr2はoutput_dir文字列を共有するが、この失敗runはそこへ学習成果物を作成していない。
