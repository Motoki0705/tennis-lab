---
id: run-plcs-reid-eager-gpu-smoke-20260924
type: run
task: plcs
sequence: 121
recorded_at: '2026-09-24'
title: eager Re-IDの反復学習・全validation・再読込GPU確認
issue: 915
provider: codex
session: 01a0d0f9-6057-7961-ad6b-adc46744d619
date: '2026-09-24'
status: done
config:
  model: plcs_player_reid
  compile: false
  precision: bf16-mixed
  hidden_dim: 256
  num_stages: 4
  num_heads: 8
  ffn_dim: 768
  num_slots: 4
  seq_len: 512
  batch_size: 4
  epochs: 2
  train_batches_per_epoch: 4
  validation_scenes: 100
  test_batches: 2
  seed: 42
  data: plcs/tracked_person_reid_v1
metrics:
  loss: 1.311019
  pair_precision: 0.35
  pair_recall: 1.0
  pair_f1: 0.518519
  pair_balanced_accuracy: 0.5
  player_accuracy: 1.0
  cosine_threshold: 0.45
  matching_precision: 0.5
  matching_recall: 0.5
  matching_f1: 0.5
  group_accuracy: 0.375
  track_acceptance_recall: 1.0
  peak_reserved_bytes: 3334471680
  duration_s: 13.892964931001188
repro:
  commit: 3ca9b8a4d3792dc613d2f30443c8d9e03c12c83c
  branch: codex/plcs-track-reid
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python tests/benchmarks/person_association.py
    --device cuda --data-root /home/kamimura/projects/tennis-lab/data --scene-dir
    plcs/tracked_person_reid_v1 --output /home/kamimura/projects/tennis-lab/outputs/plcs/diagnostics/tracked_reid_eager_smoke_20260924
artifacts:
  run_dir: knowledge/runs/run-plcs-reid-eager-gpu-smoke-20260924
  predictions: knowledge/runs/run-plcs-reid-eager-gpu-smoke-20260924/pred_test.npz
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790220145931913139_784188_plcs-reid-eager-gpu-smoke-20260924.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/plcs/diagnostics/tracked_reid_eager_smoke_20260924
  tb_logdir: outputs/plcs/diagnostics/tracked_reid_eager_smoke_20260924/logs/version_0
  curves: knowledge/runs/run-plcs-reid-eager-gpu-smoke-20260924/curves.png
parents:
- run-plcs-reid-safe-mask-gpu-smoke-20260924
relations: []
papers: []
tags:
- reid
- gpu_smoke
- eager
- fixed_tracks
---

## 観測

compile=falseを明示し、同じRe-ID構造の2epoch・各4train batch、全100scene validation、2test batch、checkpoint再読込をGPUで実行した。すべて有限値で完了し、最大予約メモリは約3.11 GiB、診断本体は約13.9秒だった。validationによる閾値更新とcheckpoint契約も確認した。

testは8sceneだけで、matching F1=0.5、group正解率=0.375は配線確認値である。8更新しか行っておらず、性能基準や収束を示さない。初回compiled smokeとはbatch/評価件数が異なるので精度差を比較しない。全100sceneでのvalidationを含む反復が正常だったことを根拠に、同configからseed42で新規60epochの本学習へ進む。以前のNaN runをresumeしない。sideの学習・構造確定は行わない。
