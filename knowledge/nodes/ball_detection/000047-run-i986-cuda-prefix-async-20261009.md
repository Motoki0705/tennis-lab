---
id: run-i986-cuda-prefix-async-20261009
type: run
task: ball_detection
sequence: 47
recorded_at: '2026-10-09'
title: 全GPU同期なしの本学習prefix確認
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
  windows_per_second: 14.167566824694367
  optimizer_updates: 128
repro:
  commit: a7c6419d9dd0de01a0612b4bacfee521f23a48b9
  branch: codex/ball-query-only-training
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONPATH=. CUDA_LOG_FILE=stderr OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
    TORCH_LOGS=graph_breaks,recompiles PYTHONUNBUFFERED=1 timeout 1000 .venv/bin/python tests/benchmarks/ball_mdd_query_gpu.py
    --manifest /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/mdd_only_windows.json
    --model-config /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/models/conv2d-query_only.yaml
    --mode pipeline --workers 8 --prefetch-factor 4 --pin-memory --precision bf16 --compile-mode default --jpeg-decoder
    nvjpeg --preverify --batch-size 1 --sampler-windows 6000 --warmup 4 --steps 124 --no-synchronize-steps --image-prefetch
    --output /home/kamimura/projects/tennis-lab/outputs/ball_detection/analyze/native-rgb-reader/20261009-production-prefix-v2/async.json
artifacts:
  run_dir: knowledge/runs/run-i986-cuda-prefix-async-20261009
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791512396754978286_1666698_i986-production-prefix-async-v2.log
parents:
- run-i986-cuda-prefix-blocking-20261009
relations: []
papers: []
tags:
- cuda-diagnostic
- nvjpeg
- input-performance
---

本学習6000窓samplerの先頭128更新を、画像先読みあり・step後の全GPU同期なし・allocator上限変更なしで完了した。元エラーは再現せず、非同期経路だけで必ず失敗するとは言えない。

[全stepと実行設定](../../runs/run-i986-cuda-prefix-async-20261009/report.json)を保存。速度の優劣や精度の確認が目的ではない。TensorBoardなし。元scriptと同一YAMLの参照修復を保持した。
