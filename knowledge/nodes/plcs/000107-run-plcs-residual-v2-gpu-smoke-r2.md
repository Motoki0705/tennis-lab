---
id: run-plcs-residual-v2-gpu-smoke-r2
type: run
task: plcs
sequence: 107
recorded_at: '2026-09-21'
title: PLCS Court14残差v2 GPU事前確認（完走）
provider: codex
session: 01a0bf91-5494-74c0-8793-3424866618ed
date: '2026-09-21'
status: done
config:
  recipe: train_triangulation_residual_v2
  model: plcs_triangulation_residual_v2
  loss: balanced_regret
  train_scenes: 64
  val_scenes: 32
  test_scenes: 32
  views: 6
  sequence_length: 64
  batch_size: 32
  precision: bf16-mixed
  max_epochs: 1
  warmup_epochs: 0
  seed: 42
metrics:
  world_mpjpe_m: 0.09175436943769455
  initial_world_mpjpe_m: 0.09043184667825699
  root_error_m: 0.07052482664585114
  initial_root_error_m: 0.06971637904644012
  relative_mpjpe_m: 0.0916418731212616
  initial_relative_mpjpe_m: 0.09096907824277878
  peak_cuda_allocated_bytes: 1706594816
  peak_cuda_reserved_bytes: 2134900736
repro:
  commit: 4671f9bad0873a23054a3311009b9ecf4a6bbc88
  branch: codex/plcs-triangulation-residual
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
    .venv/bin/python -c 'import runpy,torch; runpy.run_module("src.tasks.plcs.scripts.train_triangulation_residual",run_name="__main__");
    print("CUDA_PEAK",torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved(),flush=True)'
    --config-name train_triangulation_residual_v2 paths.data_root=/home/kamimura/projects/tennis-lab/data
    paths.output_root=/home/kamimura/projects/tennis-lab/outputs run.output_dir=plcs/triangulation_residual_v2_gpu_smoke_20260921
    run.seed=42 data.train_limit=64 data.val_limit=32 data.test_limit=32 data.num_workers=2
    data.min_views=6 data.max_views=6 v2.evaluation_views=6 training.trainer.max_epochs=1
    training.trainer.enable_progress_bar=false training.warmup_epochs=0
artifacts:
  run_dir: knowledge/runs/run-plcs-residual-v2-gpu-smoke-r2
  predictions: knowledge/runs/run-plcs-residual-v2-gpu-smoke-r2/pred_test.npz
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1789944265760495498_317621_plcs-residual-v2-gpu-smoke-r2.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/plcs/triangulation_residual_v2_gpu_smoke_20260921/logs/version_0
  curves: knowledge/runs/run-plcs-residual-v2-gpu-smoke-r2/curves.png
  tb_logdir: outputs/plcs/triangulation_residual_v2_gpu_smoke_20260921/logs/version_0
parents:
- run-plcs-residual-v2-gpu-smoke
relations: []
papers: []
tags:
- triangulation-residual-v2
- gpu-smoke
- diagnostic
---

## 考察 / Findings

### 要約
6 camera・batch 32・BF16でforward/backward、validation、best checkpointの保存・再読込、test予測保存が完走した。最大予約メモリは約2.1 GBで、16 GB GPUにhalf予約2本を配置できる。

### 実験条件と解釈
Court14校正・持続誤検出・balanced regret objectiveを使う。train 64 scene/2 update、val/test各32 scene、1 epochの動作確認であり、精度比較には使わない。表示metricはこの限定testのみ。曲線も1 epochだけで、収束を示さない。

### 次の比較
warmupを0にして設定拒否を修正した。同一v2データ・seed・モデルで従来lossとbalanced lossを各taskで比較し、meanだけでなく中央値、改善率、誤差成分別の指標を確認する。実clipの独立3D正解はない。
