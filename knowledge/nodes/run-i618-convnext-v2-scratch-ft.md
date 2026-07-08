---
id: run-i618-convnext-v2-scratch-ft
type: run
title: i618_convnext_v2_scratch_ft
issue: 620
provider: claude
date: '2026-07-08'
status: done
config:
  model: conv_next_unet
metrics:
  precision: 0.678532
  recall: 0.75703
  f1: 0.715635
  mean_distance_px: 2.107013
repro:
  commit: 41b318b39f757e09897964a32f28148a777b87f5
  branch: feat/issue-618-ball-subpixel-retrain
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.tasks.ball_detection.scripts.train
    model=conv_next_unet data.heatmap_size='[72,128]' data.batch_size=6 data.num_workers=2
    metrics.nms_kernel=3 training.learning_rate=2e-5 training.trainer.max_epochs=12
    training.checkpoint.monitor=val/f1 training.checkpoint.mode=max training.early_stopping.monitor=val/f1
    training.early_stopping.mode=max training.early_stopping.patience=6 training.early_stopping.min_delta=0.0005
    'run.init_weights="outputs/ball_detection/convnext_v2_scratch/logs/version_0/checkpoints/ball-detection-epoch=25.ckpt"'
    run.output_dir=outputs/ball_detection/convnext_v2_scratch_ft
artifacts:
  run_dir: knowledge/runs/run-i618-convnext-v2-scratch-ft
  predictions: knowledge/runs/run-i618-convnext-v2-scratch-ft/pred_test.npz
  log: .training_queue/logs/1783455473439495639_216909_i618_convnext_v2_scratch_ft.log
  output_dir: outputs/ball_detection/convnext_v2_scratch_ft/logs/version_0
parents:
- run-i618-convnext-v2-scratch
relations:
- to: run-i618-convnext-v2-scratch
  rel: compares
tags:
- ball_detection
- conv_next_unet
- fine-tune
- negative-result
---

## 考察 / Findings

### 要約
scratch-e25 からの低 lr (2e-5) TrackNet-only 継続 fine-tune（issue #620 B1 プローブ）。
val/f1 best 0.7108 (ep2) で、起点の scratch (val 0.7198) にも ft-e13 (0.7126) にも
届かず**負の結果**。「データを変えずに低 lr 継続するだけではベンチも頑健性も
改善しない」ことを安価に確定した。ckpt は --prune-ckpt により削除済み。

### アーキテクチャ詳細
モデル・データは [[run-i618-convnext-v2-scratch]] と同一。差分は
init_weights=scratch best (ep25) / lr 2e-5 / max 12ep / ES patience 6 のみ。

### メトリクスの解釈
frontmatter metrics (f1 0.7156) は last-epoch (ep8, val 0.6925) の test 値。
val/f1 は ep2 の 0.7108 が最高で以降劣化 — 収束済みの重みに同一分布の低 lr を
当てても改善余地がないことを示す。

### アーキテクチャ⇄メトリクスの因果考察
scratch-e25 は既に TrackNet 分布へ深く適合しており、同一データの継続学習は
val の揺らぎの範囲でしか動かない。#618 で観測した「ft が正則化として働く」
効果は旧 ckpt という異なる初期値に由来し、lr の低さ自体ではない（本 run が反証）。

### 既存実験との比較
- [[run-i618-convnext-v2-scratch]] (val 0.7198 / test-best 0.7692): 起点。本 run はこれを下回る。
- 対照アーム [[run-i620-convnext-webmix-ft]]（同一起点・web 混合データ）が本命。

### 次に有効な実験
- B2 (web 混合 ft) の結果評価が対照実験として直接の次ステップ。
- 頑健性は学習でなく trajectory gate（issue #620 B3、実装検証済み）でも担保可能。
