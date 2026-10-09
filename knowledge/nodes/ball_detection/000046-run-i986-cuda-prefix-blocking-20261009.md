---
id: run-i986-cuda-prefix-blocking-20261009
type: run
task: ball_detection
sequence: 46
recorded_at: '2026-10-09'
title: 本学習の先頭窓をCUDA同期で確認
issue: 986
provider: codex
session: 01a0fb0c-97b5-7851-b805-25dae445a6a2
date: '2026-10-09'
status: done
config:
  model: conv2d-query_only
  precision: bf16
  compile_mode: default
  jpeg_decoder: nvjpeg
  batch_size: 1
  workers: 8
  prefetch_factor: 4
  seed: 42
metrics:
  windows_per_second: 9.057490712212585
  optimizer_updates: 18
repro:
  commit: a7c6419d9dd0de01a0612b4bacfee521f23a48b9
  branch: codex/ball-query-only-training
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONPATH=. CUDA_LAUNCH_BLOCKING=1 CUDA_LOG_FILE=stderr OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
    TORCH_LOGS=graph_breaks,recompiles PYTHONUNBUFFERED=1 timeout 900 .venv/bin/python tests/benchmarks/ball_mdd_query_gpu.py
    --manifest /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/mdd_only_windows.json
    --model-config /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/models/conv2d-query_only.yaml
    --output /home/kamimura/projects/tennis-lab/outputs/ball_detection/analyze/native-rgb-reader/20261009-production-prefix-blocking-v1/report.json
    --mode pipeline --workers 8 --prefetch-factor 4 --pin-memory --precision bf16 --compile-mode default --jpeg-decoder
    nvjpeg --image-prefetch --preverify --batch-size 1 --sampler-windows 6000 --warmup 2 --steps 16
artifacts:
  run_dir: knowledge/runs/run-i986-cuda-prefix-blocking-20261009
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791512129661718877_1654596_i986-production-prefix-blocking-v1.log
parents:
- run-i986-main-nvjpeg-cuda-failure-20261009
relations: []
papers: []
tags:
- cuda-diagnostic
- nvjpeg
- input-performance
---

CUDA_LAUNCH_BLOCKING=1で本学習6000窓samplerの先頭18更新を処理できた。検証用の全CUDA同期とallocator上限90%を含み、速度を本学習へ外挿しない。元エラーの再現にはならず、原因は未確定のまま。

[全stepと実行設定](../../runs/run-i986-cuda-prefix-blocking-20261009/report.json)を保存。速度の優劣や精度の確認が目的ではない。TensorBoardなし。元scriptと同一YAMLの参照修復を保持した。
