---
id: run-i986-nvjpeg-synchronous-20261009
type: run
task: ball_detection
sequence: 34
recorded_at: '2026-10-09'
title: 同期nvJPEG入力：reader込み192更新
issue: 986
provider: codex
session: 01a0fb0c-97b5-7851-b805-25dae445a6a2
date: '2026-10-09'
status: done
config:
  precision: bf16
  compile_mode: default
  workers: 8
  batch_size: 1
  warmup_windows: 12
  measured_windows: 192
metrics:
  windows_per_second: 9.808679471315925
  mean_reader_wait_seconds: 0.0060020832556044
  preverification_seconds: 493.3572533409897
repro:
  commit: 2da3351a28cf6c685767338f0f768d3c357c402e
  branch: codex/ball-query-only-training
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2 TORCH_LOGS=graph_breaks,recompiles
    PYTHONUNBUFFERED=1 timeout 1200 .venv/bin/python tests/benchmarks/ball_mdd_query_gpu.py --manifest /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/mdd_only_windows.json
    --model-config /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/models/conv2d-query_only.yaml
    --precision bf16 --compile-mode default --batch-size 1 --pin-memory --warmup 12 --steps 192 --jpeg-decoder nvjpeg
    --mode pipeline --workers 8 --preverify --output /home/kamimura/projects/tennis-lab/outputs/ball_detection/analyze/native-rgb-reader/20261009-nvjpeg-v1/pipeline.json
artifacts:
  run_dir: knowledge/runs/run-i986-nvjpeg-synchronous-20261009
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791503742151889422_1110953_i986-nvjpeg-pipeline-v1.log
parents:
- run-i986-nvjpeg-pixels-20261009
relations: []
papers: []
tags:
- input-performance
- nvjpeg
- diagnostic
---

clipを事前検証した後のreader込みは9.809窓/秒、平均step 101.95msのうちreader待ちは6.00ms（約5.9%）。GPUにRGBが常駐した旧基準とは異なり、JPEG復号も各stepへ含める。

ただしこの試行は事前hashを逐次実行し493.36秒を要した。準備費用を隠して総時間の改善率を述べない。decodeとモデルを順番に実行するため、入力をRAMに置いた対照とCUDA先読みを次に比較する。

[全step・入力identity](../../runs/run-i986-nvjpeg-synchronous-20261009/pipeline.json)。単発の資源診断で精度評価ではない。TensorBoardなし。原scriptを残し、同じhashのモデルYAMLをbundleへ移した。
