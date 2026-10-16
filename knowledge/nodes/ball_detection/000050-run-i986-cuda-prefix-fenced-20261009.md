---
id: run-i986-cuda-prefix-fenced-20261009
type: run
task: ball_detection
sequence: 50
recorded_at: '2026-10-09'
title: 画像側fence付きで本学習prefix256更新を確認
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
  windows_per_second: 13.308560433316561
  mean_reader_wait_seconds: 0.002752960234555224
  optimizer_updates: 256
repro:
  commit: a7c6419d9dd0de01a0612b4bacfee521f23a48b9
  branch: codex/ball-query-only-training
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2 TORCH_LOGS=graph_breaks,recompiles
    PYTHONUNBUFFERED=1 timeout 1200 .venv/bin/python tests/benchmarks/ball_mdd_query_gpu.py --manifest /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/mdd_only_windows.json
    --model-config /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/models/conv2d-query_only.yaml
    --output /home/kamimura/projects/tennis-lab/outputs/ball_detection/analyze/native-rgb-reader/20261009-production-prefix-fenced-v3/report.json
    --mode pipeline --workers 8 --prefetch-factor 4 --pin-memory --precision bf16 --compile-mode default --jpeg-decoder
    nvjpeg --image-prefetch --preverify --batch-size 1 --sampler-windows 6000 --warmup 4 --steps 252 --no-synchronize-steps
artifacts:
  run_dir: knowledge/runs/run-i986-cuda-prefix-fenced-20261009
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791512639776318187_1692229_i986-production-prefix-fenced-v3.log
parents:
- run-i986-cuda-producer-fence-20261009
relations: []
papers: []
tags:
- cuda-diagnostic
- nvjpeg
- input-performance
---

本学習と同じ6000窓samplerの先頭256更新を、画像側fence付き・BF16・compile default・8 workers/prefetch4で完了した。stepごとの全GPU同期とallocator上限変更は行わない。13.309窓/秒、CPU reader待ちは平均2.75msだった。

元のCUDA unknown errorは再現せず、今回のfenceがその原因を解消したとまでは断定しない。CPU準備が完了してからCUDA stateを作る起動順にも整理し、実CLIの保存・再開確認を経て新しい本学習runを試す。

[全step](../../runs/run-i986-cuda-prefix-fenced-20261009/report.json)にloss・勾配・窓列hashを保存。TensorBoardなし。元scriptと同一YAML参照修復を保持した。
