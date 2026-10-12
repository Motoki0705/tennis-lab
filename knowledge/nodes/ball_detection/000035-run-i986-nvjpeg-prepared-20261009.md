---
id: run-i986-nvjpeg-prepared-20261009
type: run
task: ball_detection
sequence: 35
recorded_at: '2026-10-09'
title: RAM内JPEGの基準：復号込みの同期モデル速度
issue: 986
provider: codex
session: 01a0fb0c-97b5-7851-b805-25dae445a6a2
date: '2026-10-09'
status: done
config:
  precision: bf16
  compile_mode: default
  batch_size: 1
  warmup_windows: 12
  measured_windows: 192
metrics:
  windows_per_second: 12.35261415522289
  preverification_seconds: 346.82961601500574
  input_setup_seconds: 67.72936660501
repro:
  commit: 2da3351a28cf6c685767338f0f768d3c357c402e
  branch: codex/ball-query-only-training
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2 TORCH_LOGS=graph_breaks,recompiles
    PYTHONUNBUFFERED=1 timeout 1200 .venv/bin/python tests/benchmarks/ball_mdd_query_gpu.py --manifest /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/mdd_only_windows.json
    --model-config /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/models/conv2d-query_only.yaml
    --precision bf16 --compile-mode default --batch-size 1 --pin-memory --warmup 12 --steps 192 --jpeg-decoder nvjpeg
    --mode prepared --workers 8 --preverify --output /home/kamimura/projects/tennis-lab/outputs/ball_detection/analyze/native-rgb-reader/20261009-nvjpeg-v1/prepared.json
artifacts:
  run_dir: knowledge/runs/run-i986-nvjpeg-prepared-20261009
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791503742192204096_1110970_i986-nvjpeg-prepared-v1.log
parents:
- run-i986-nvjpeg-synchronous-20261009
relations: []
papers: []
tags:
- input-performance
- nvjpeg
- diagnostic
---

同じ204窓のJPEGをRAM/pinned memoryへ用意すると、復号込みの同期更新は12.353窓/秒だった。先のreader込みより速く、decode以外の供給待ち・CPU競合も減らす余地があった。

事前検証は8 processesで346.83秒、JPEGの準備は67.73秒。これらを除いた192更新の速度であり、RAM準備を無料とは扱わない。初回試行とは検証の並列度・実行順が違うため、準備時間の差を厳密な因果効果とはしない。

[全step](../../runs/run-i986-nvjpeg-prepared-20261009/prepared.json)。次は同一run・同一窓で同期とCUDA先読みを比較する。TensorBoardなし。原scriptと同一YAMLの参照修復を保持した。
