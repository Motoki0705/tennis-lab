---
id: run-i986-fenced-startup-cli-20261009
type: run
task: ball_detection
sequence: 51
recorded_at: '2026-10-09'
title: 画像側fence・CPU検証後のCUDA起動でCLI保存再開を確認
issue: 986
provider: codex
session: 01a0fb0c-97b5-7851-b805-25dae445a6a2
date: '2026-10-09'
status: done
config:
  precision: bf16
  compile_mode: default
  batch_size: 1
  workers: 8
  prefetch_factor: 4
  jpeg_decoder: nvjpeg
  image_prefetch: true
metrics:
  optimizer_updates: 24
repro:
  commit: a7c6419d9dd0de01a0612b4bacfee521f23a48b9
  branch: codex/ball-query-only-training
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2
    TORCH_LOGS=graph_breaks,recompiles PYTHONUNBUFFERED=1 timeout 900 .venv/bin/python
    tests/benchmarks/ball_mdd_query_gpu_smoke.py --manifest /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/mdd_only_windows.json
    --model-config /home/kamimura/projects/tennis-lab/outputs/ball_detection/precompute/mdd_coordinates_36/20261008-fps124-v1/models/conv2d-query_only.yaml
    --output /home/kamimura/projects/tennis-lab/outputs/ball_detection/analyze/native-rgb-reader/20261009-fenced-cli-v5
    --workers 8 --prefetch-factor 4 --batch-size 1 --compile-mode default --jpeg-decoder
    nvjpeg --input-verification upfront --image-prefetch
artifacts:
  run_dir: knowledge/runs/run-i986-fenced-startup-cli-20261009
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791513179862052110_1733656_i986-fenced-startup-cli-v5.log
parents:
- run-i986-cuda-prefix-fenced-20261009
relations: []
papers: []
tags:
- cuda-diagnostic
- checkpoint-roundtrip
---

画像側streamのfenceを追加し、CPUの整合性検証をCUDA state生成前へ移した実CLIで、24更新・v4保存・再開・独立process評価を確認した。BF16/compile/decoder/先読みの設定を維持し、精度やモデル構造を緩めていない。

[smoke結果](../../runs/run-i986-fenced-startup-cli-20261009/smoke.json)・[validation](../../runs/run-i986-fenced-startup-cli-20261009/validation.json)を保存。checkpoint hashは`76cb88826ad16bc7f9c1f72193e3b4770b5333adcf8a9b7b1a57f3d66a9aa885`。3 sourceのtrain/val各3 clipの診断で、testは未使用。元の本学習CUDA unknown errorの原因が確定したとは扱わず、固定した新しいcommitで本学習を再試行する。TensorBoardなし。元scriptとYAML参照修復を保持した。
