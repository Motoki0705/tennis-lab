---
id: run-i986-jpeg-prefetch-integrity-20261009
type: run
task: ball_detection
sequence: 37
recorded_at: '2026-10-09'
title: JPEG先読みの画素・順序・終了処理を検証
issue: 986
provider: codex
session: 01a0fb0c-97b5-7851-b805-25dae445a6a2
date: '2026-10-09'
status: done
config:
  sources: 3
  frame_steps:
  - 1
  - 2
  - 4
  batch_sizes:
  - 2
  - 2
  - 2
  - 2
  - 1
metrics:
  windows: 9
  rgb_bit_identical: true
  teachers_and_order_unchanged: true
repro:
  commit: 2da3351a28cf6c685767338f0f768d3c357c402e
  branch: codex/ball-query-only-training
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2 TORCH_LOGS=graph_breaks,recompiles
    PYTHONUNBUFFERED=1 timeout 900 .venv/bin/python tests/benchmarks/ball_jpeg_prefetch_check.py --manifest /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/mdd_only_windows.json
    --output /home/kamimura/projects/tennis-lab/outputs/ball_detection/analyze/native-rgb-reader/20261009-prefetch-check-v1/report.json
artifacts:
  run_dir: knowledge/runs/run-i986-jpeg-prefetch-integrity-20261009
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791505863279715372_1277814_i986-jpeg-prefetch-check-v1.log
parents:
- run-i986-jpeg-prefetch-sweep-20261009
relations: []
papers: []
tags:
- input-performance
- nvjpeg
- diagnostic
---

実9窓をBS2（最終batchだけ1）で先読みし、同期nvJPEGとRGBをbit一致で照合した。clip・frame順、教師、mask、実PTSも一致。readerの例外が呼出元へ届き、途中終了でもproducer threadをjoinすることを確認した。

これは同じnvJPEG方式の同期/非同期比較であり、OpenCVとの画素一致ではない。CUDA event、record_stream、CPU byte bufferの保持による寿命管理の診断で、全datasetや長期学習は別に確認する。

[結果](../../runs/run-i986-jpeg-prefetch-integrity-20261009/report.json)。TensorBoardなし。保存commit＋patchを使い、GPU再実行は共有queueへ投入する。
