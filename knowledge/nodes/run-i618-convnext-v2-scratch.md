---
id: run-i618-convnext-v2-scratch
type: run
title: i618_convnext_v2_scratch
issue: 618
provider: claude
session: 147b5124-0694-4620-bb75-11cb13e410c0
date: '2026-07-07'
status: done
config:
  model: conv_next_unet
metrics:
  precision: 0.662902
  recall: 0.76739
  f1: 0.71133
  mean_distance_px: 2.058813
repro:
  commit: 640766639f3b16ae5048f31602e29a2f03b41cc5
  branch: feat/issue-618-ball-subpixel-retrain
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.tasks.ball_detection.scripts.train
    model=conv_next_unet data.heatmap_size='[72,128]' data.batch_size=6 data.num_workers=2
    metrics.nms_kernel=3 training.trainer.max_epochs=60 training.checkpoint.monitor=val/f1
    training.checkpoint.mode=max training.early_stopping.monitor=val/f1 training.early_stopping.mode=max
    training.early_stopping.patience=10 training.early_stopping.min_delta=0.0005 run.output_dir=outputs/ball_detection/convnext_v2_scratch
artifacts:
  run_dir: knowledge/runs/run-i618-convnext-v2-scratch
  predictions: knowledge/runs/run-i618-convnext-v2-scratch/pred_test.npz
  log: .training_queue/logs/1783405858454132778_1226767_i618_convnext_v2_scratch.log
  output_dir: outputs/ball_detection/convnext_v2_scratch/logs/version_0
  curves: knowledge/runs/run-i618-convnext-v2-scratch/curves.png
  tb_logdir: outputs/ball_detection/convnext_v2_scratch/logs/version_0
parents:
- run-i618-convnext-v2-ft
relations:
- to: run-i618-convnext-v2-ft
  rel: compares
tags:
- ball_detection
- conv_next_unet
- subpixel
- from-scratch
---

## 考察 / Findings

### 要約
val/f1 監視レシピでのフルスクラッチ再学習。best ckpt (epoch=25) は native
プロトコルで test F1 **0.7692**（[[run-i618-convnext-v2-ft]] の 0.7218 を +4.7pt
上回り、TrackNet ベンチ最良）。ただし実クリップ tennis_clip では 179.9px の
テレポート 1 件と coverage 微減（91.1% vs 92.0%）で ft に僅差で劣り、
**デプロイ ckpt は ft-e13 を維持**した。

### アーキテクチャ詳細
モデル・データ・損失は ft と完全同一（conv_next_unet, mdd 2ch, T=8, 288x512,
heatmap 72x128, focal_bce）。差分は初期化のみ:
`run.init_weights` なし（スクラッチ）、lr 1e-4（ft は 5e-5）、max_epochs 60
（early stopping val/f1, patience 10 → ep35 で停止）。checkpoint/early-stopping
とも val/f1 (max) 監視、metrics は nms_kernel 3 + subpixel 精緻化の native
プロトコル。

### メトリクスの解釈
frontmatter の metrics (f1 0.7113) は学習ジョブが **last ckpt (ep35)** で流した
test 値。best ckpt (ep25, val/f1 0.7198) の manifest 評価は
val 0.7448 / **test 0.7692**（precision 0.7976 / recall 0.7427 / 2.01px）。
ft と同様、last と best の乖離が大きい（0.7113 vs 0.7692）ので、比較には
必ず manifest（outputs/ball_detection/evaluation_native/）を使うこと。
curves.png では val/f1 が ep25 以降 0.63-0.68 に落ちて回復せず、
val/loss は単調減少を続ける — val/loss と val/f1 の乖離が再確認できる。

### アーキテクチャ⇄メトリクスの因果考察
スクラッチ + 60ep 予算は ft（24ep, 低 lr）より TrackNet 分布への適合が深く、
ベンチでは precision (0.80 vs 0.74) を中心に上回った。一方 tennis_clip
（640x360 放送クリップ、ドメイン外）では score 0.58 の偽ピーク（t=640,
画面上部）を 1 件出しており、「TrackNet への深い適合がドメイン外の
頑健性をわずかに削った」という仮説と整合する。旧 ckpt からの ft は
weight 初期化が正則化として働いた可能性がある（仮説）。

### 既存実験との比較
- [[run-i618-convnext-v2-ft]]: TrackNet test F1 0.7218 / tennis_clip coverage
  92.0%・テレポート 0。scratch はベンチ +4.7pt、実クリップ僅差負け。
- 旧 convnext (run なし, manifest `convnext-tracknet`): test F1 0.5764。
  レシピ（native 評価 + val/f1 監視 + subpixel）でどちらも大幅超過。

### 次に有効な実験
- scratch ckpt に低 lr の短い ft（旧 ckpt 由来でなく scratch→ft の 2 段）を
  当て、ベンチ性能を保ったまま実クリップ頑健性が戻るか検証。
- 実クリップ偽ピーク対策として web データ混合（issue #579 phase2 相当）を
  conv_next_unet 系にも適用する。
- score 閾値 0.5→0.6 で v3 テレポートが消えるかの感度分析（deploy 前提の
  閾値調整は fragile なので参考情報として）。
