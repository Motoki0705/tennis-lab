---
id: run-i986-reader-pagefaults-20261009
type: run
task: ball_detection
sequence: 40
recorded_at: '2026-10-09'
title: 遅い窓の追跡：mmapの初回ページフォルト
issue: 986
provider: codex
session: 01a0fb0c-97b5-7851-b805-25dae445a6a2
date: '2026-10-09'
status: done
config:
  model: conv2d-query_only
  precision: bf16
  jpeg_decoder: nvjpeg
  image_prefetch: true
  batch_size: 1
  workers: 2
metrics:
  first_images_seconds: 0.17470790037441475
  first_images_cpu_seconds: 0.14335499237500002
  first_major_faults: 121.375
  first_block_reads: 29946
repro:
  commit: 2da3351a28cf6c685767338f0f768d3c357c402e
  branch: codex/ball-query-only-training
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2 PYTHONUNBUFFERED=1
    timeout 600 .venv/bin/python tests/benchmarks/ball_reader_trace.py --sustained-report /home/kamimura/projects/tennis-lab/outputs/ball_detection/analyze/native-rgb-reader/20261009-sustained-v1/report.json
    --output /home/kamimura/projects/tennis-lab/outputs/ball_detection/analyze/native-rgb-reader/20261009-reader-trace-v1/report.json
artifacts:
  run_dir: knowledge/runs/run-i986-reader-pagefaults-20261009
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791508049426352171_1393480_i986-reader-slow-trace-v1.log
parents:
- run-i986-nvjpeg-long-reader-20261009
relations: []
papers: []
tags:
- input-performance
- nvjpeg
- reader-diagnostic
---

直前の長い試行でreader待ちが最大だった8窓を選び、同じ順で4回読み直した。事前に同じdual hashを通し、対象fileのclean page破棄をOSへ一度だけ要求した（完全なcold状態を保証する操作ではない）。

初回の画像読込は平均174.7ms、うちCPU143.4ms、major page fault 121.4回/窓、block readは約15.33MB/窓だった。要求JPEGは平均6.73MB。3回目以降は画像約1.1〜1.6ms、page fault・block readが0となった。hash再計算は0で、pin memoryは初回約13.9ms、以降約1〜2msだった。

clip検証やモデルの演算より、mmapから初めてJPEGをコピーする処理が大きい。特に遅い窓を選んだ診断なので全窓平均へ外挿せず、必要範囲だけを直接読む方式を次に試す。

[worker CPU/壁時間・fault・pin・collate](../../runs/run-i986-reader-pagefaults-20261009/report.json)と、選択に使った親reportを保存。TensorBoardなし。元scriptと親report参照修復を併記した。
