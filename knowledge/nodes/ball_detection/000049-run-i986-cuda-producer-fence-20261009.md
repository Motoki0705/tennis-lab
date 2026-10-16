---
id: run-i986-cuda-producer-fence-20261009
type: run
task: ball_detection
sequence: 49
recorded_at: '2026-10-09'
title: 画像側streamの再利用順序を保証して遅延試験
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
  windows: 9
  rgb_bit_identical: true
  delayed_producer: true
repro:
  commit: a7c6419d9dd0de01a0612b4bacfee521f23a48b9
  branch: codex/ball-query-only-training
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2 TORCH_LOGS=graph_breaks,recompiles
    PYTHONUNBUFFERED=1 timeout 1200 .venv/bin/python tests/benchmarks/ball_jpeg_prefetch_check.py --manifest /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/mdd_only_windows.json
    --output /home/kamimura/projects/tennis-lab/outputs/ball_detection/analyze/native-rgb-reader/20261009-producer-fence-integrity-v1/report.json
    --delay-producer
artifacts:
  run_dir: knowledge/runs/run-i986-cuda-producer-fence-20261009
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791512639733753209_1692206_i986-producer-fence-integrity-v1.log
parents:
- run-i986-main-nvjpeg-cuda-failure-20261009
relations: []
papers: []
tags:
- cuda-diagnostic
- nvjpeg
- input-performance
---

torchvision v0.28の[CUDA実装](https://github.com/pytorch/vision/blob/v0.28.0/torchvision/csrc/io/image/cuda/decode_jpegs_cuda.cpp)は、出力tensorを呼出元streamのallocatorから確保し、nvJPEGは内部streamへ投入する。decode後の内部stream同期・callerへの待ちはあるが、出力storageを再利用する前に以前のcaller stream上のstackを待つ関係が見当たらない。

画像側streamをdecode直前に同期して、前回stack等の完了を保証する。モデル用streamまで全GPU同期する処理にはしない。この順序の問題はsource上のリスクであり、元のunknown errorの原因と実証したものではない。

decode後・stack前にGPU待ちを意図的に挿入した9窓試験でも、同期nvJPEGとのRGB bit一致・教師/順序・例外伝播・早期終了を確認した。[結果](../../runs/run-i986-cuda-producer-fence-20261009/report.json)。OpenCVとの画素一致とは別の検証。TensorBoardなし。
