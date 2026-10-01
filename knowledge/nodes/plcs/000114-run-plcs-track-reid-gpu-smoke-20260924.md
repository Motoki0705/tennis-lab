---
id: run-plcs-track-reid-gpu-smoke-20260924
type: run
task: plcs
sequence: 114
recorded_at: '2026-09-24'
title: 固定track Re-IDの初回GPU配線確認
issue: 915
provider: codex
session: 01a0d0f9-6057-7961-ad6b-adc46744d619
date: '2026-09-24'
status: done
config:
  model: plcs_player_reid
  hidden_dim: 256
  num_stages: 4
  num_heads: 8
  ffn_dim: 768
  num_slots: 4
  seq_len: 512
  batch_size: 2
  train_batches: 4
  validation_batches: 2
  test_batches: 2
  compile: true
  precision: bf16-mixed
  seed: 42
  dataset: plcs/tracked_person_reid_v1
metrics:
  loss: 1.724595
  pair_precision: 0.320755
  pair_recall: 1.0
  pair_f1: 0.485714
  pair_balanced_accuracy: 0.5
  player_accuracy: 1.0
  cosine_threshold: 0.725
  matching_precision: 0.490196
  matching_recall: 0.490196
  matching_f1: 0.490196
  group_accuracy: 0.5
  track_acceptance_recall: 1.0
  peak_reserved_bytes: 1361051648
  duration_s: 39.65712581400294
repro:
  commit: 53db1868054e8330126d34a458edb1424646d61e
  branch: codex/plcs-track-reid
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python tests/benchmarks/person_association.py
    --device cuda --data-root /home/kamimura/projects/tennis-lab/data --scene-dir
    plcs/tracked_person_reid_v1 --output /home/kamimura/projects/tennis-lab/outputs/plcs/diagnostics/tracked_reid_smoke_20260924
artifacts:
  run_dir: knowledge/runs/run-plcs-track-reid-gpu-smoke-20260924
  predictions: knowledge/runs/run-plcs-track-reid-gpu-smoke-20260924/pred_test.npz
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790218808214964499_733501_plcs-track-reid-gpu-smoke-20260924.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/plcs/diagnostics/tracked_reid_smoke_20260924
  tb_logdir: outputs/plcs/diagnostics/tracked_reid_smoke_20260924/logs/version_0
  curves: knowledge/runs/run-plcs-track-reid-gpu-smoke-20260924/curves.png
parents:
- run-plcs-association-refactor-512-smoke
relations: []
papers: []
tags:
- reid
- fixed_tracks
- gpu_smoke
- synthetic
- partial_evaluation
---

## 観測

2D tracker IDがシーン全体で同一人物を指す前提のPLCS専用Re-IDを、RTX 5060 Tiの共有queueで検証した。D256・4stageの時間/query attention、bf16、compile有効で4更新、validation 2batch、test 2batch、checkpoint再読込が完了した。最大予約メモリは約1.27 GiBだった。入力・教師・embedding・予測groupを保存した。

対象は専用合成データ1000sceneの固定800/100/100 splitから読み出した少数batchで、全test評価ではない。test matching F1=0.490196、group正解率=0.5は4sceneの配線検証値であり、収束・実動画精度・旧モデルよりの改善を主張しない。sideは独立境界へ分離しただけで学習していない。

## 制限と後続

この初回スモークはsanity validationを省略し、batch2だった。その後、batch4の本学習でtrain損失が低下する一方validationがNaNになる異常を検出して停止した。保存重みと全入力は有限値で、GPU probeではcompiledモデルの評価→学習→評価により再現し、eagerでは再現しなかった。したがって本スモークだけを長時間学習の安定性の根拠にはしない。切替を反復する検証と明示的な非有限値エラーを追加して再実行する。

データ生成は全1000scene保存後に旧均等occupancy検査で停止したが、固定人数契約では均等化を要求しない方針を明示し、全入力のshape/有限値・camera別累計人数・full-source区間を再監査してエラー0件だった。別rootの再生成でCRCエラーが発生したが、元167 archiveの全CRC検査では再現しなかった。失敗した部分データは採用していない。scene splitなので未見source motionの評価ではない。
