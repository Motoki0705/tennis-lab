---
id: run-i986-main-nvjpeg-cuda-failure-20261009
type: run
task: ball_detection
sequence: 45
recorded_at: '2026-10-09'
title: 本学習起動でCUDA unknown error：原因は未確定
issue: 986
provider: codex
session: 01a0fb0c-97b5-7851-b805-25dae445a6a2
date: '2026-10-09'
status: failed
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
  exit_code: 134
  checkpoint_count: 0
  train_verification_seconds: 258.0657058190118
  val_verification_seconds: 104.74419478198979
repro:
  commit: a7c6419d9dd0de01a0612b4bacfee521f23a48b9
  branch: HEAD
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2 TORCH_LOGS=graph_breaks,recompiles
    PYTHONUNBUFFERED=1 .venv/bin/python -m src.tasks.ball_detection.scripts.train_mdd_pose --manifest /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/mdd_only_windows.json
    --model-config /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/models/conv2d-query_only.yaml
    --output /home/kamimura/projects/tennis-lab/outputs/ball_detection/train/conv2d-query-only-bf16/s42-u60000-nvjpeg-v1
    --device cuda --precision bf16 --compile-mode default --batch-size 1 --num-workers 8 --pin-memory --prefetch-factor
    4 --cpu-threads 2 --jpeg-decoder nvjpeg --input-verification upfront --image-prefetch --epochs 10 --windows-per-epoch
    6000 --learning-rate 0.0001 --seed 42 --mdd-a 0.2 --mdd-b 0.15 --selection-scope common --log-every 50
artifacts:
  run_dir: knowledge/runs/run-i986-main-nvjpeg-cuda-failure-20261009
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791511474412814944_1611763_i986-query-bf16-s42-u60000-nvjpeg-v1.log
parents:
- run-i986-final-pread-cli-20261009
relations: []
papers: []
tags:
- cuda-diagnostic
- nvjpeg
- input-performance
---

固定commit a7c6419d9の本学習を開始したが、全train/valの検証後、最初の50更新ログより前にCUDA unknown errorで終了した。epoch checkpointはなく、完了更新数は不明。画像入力待ちの改善を、この起動の安定性達成とは扱わない。

主例外はモデル出力のfinite検査で観測され、generatorのstream同期・CUDA allocatorの終了処理にも同じエラーが出た。非同期の先行エラーを後で観測した可能性があり、finite検査自体が原因とは断定できない。同時期のWindows Display/nvlddmkmイベント照会は一致なしで、driver resetの証拠も得られていない。

[失敗log](../../runs/run-i986-main-nvjpeg-cuda-failure-20261009/failure.log)、[設定](../../runs/run-i986-main-nvjpeg-cuda-failure-20261009/config.json)、[receipt](../../runs/run-i986-main-nvjpeg-cuda-failure-20261009/failure_receipt.json)を保持した。次は同じ6000窓samplerの先頭を、同期・非同期条件で切り分ける。モデル・BF16・学習予算を緩めない。
