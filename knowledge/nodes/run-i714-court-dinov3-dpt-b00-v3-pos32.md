---
id: run-i714-court-dinov3-dpt-b00-v3-pos32
type: run
title: DINOv3+DPT B00 KP7 v3（aspect ratio修正・positive weight 32）
issue: 714
provider: codex
date: '2026-08-13'
status: failed
session: 019ff617-dfb3-7182-8c2a-1d0705cf3ff0
config:
  model: dinov3 + dpt + lora
  loss: focal BCE, kp positive_weight 32.0
  data: synthetic_court B00 KP7, train short-side 256, validation short-side 512
  max_epochs: 50
  seed: 714
metrics:
  validation_events: 26
  best_val_kp_mean_dist_px: 186.4901580810547
  final_val_kp_mean_dist_px: 192.9145965576172
  best_val_kp_mean_dist_normalized: 0.17868904769420624
  best_val_loss: 0.03701108694076538
  held_out_test_kp_mean_dist_px: null
repro:
  commit: d3226c0dcd5ea622cdc82d752147d8153f66d802
  branch: feat/issue-714-court-data-composition-v2
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: "PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.tasks.court_detection.scripts.train data/source=synthetic_court data/processing=kp model/encoder=dinov3 model/decoder=dpt training=lora loss.kp.positive_weight=32.0 data.batch_size=4 data.num_workers=4 data.augmentation.train_scales='[256]' data.augmentation.val_short_side=512 training.trainer.max_epochs=50 training.checkpoint.monitor=val/kp_mean_dist training.checkpoint.mode=min training.early_stopping.monitor=val/kp_mean_dist training.early_stopping.mode=min training.early_stopping.patience=10 training.qualitative_logging.enabled=false run.seed=714 run.output_dir=issue714/i714-court-dinov3-dpt-b00-v3-pos32 run.test_after_fit=true"
artifacts:
  log: .training_queue/logs/1786557224699804198_2236285_i714-court-dinov3-dpt-b00-v3-pos32.log
  job: .training_queue/failed/1786557224699804198_2236285_i714-court-dinov3-dpt-b00-v3-pos32.job
parents:
- run-i714-court-dinov3-dpt-b00-v2-aligned256
relations: []
tags:
- court-detection
- synthetic-court
- b00
- dinov3
- dpt
- kp7
- positive-weight
- negative
---

## 考察 / Findings

### 要約

shared geometryでaspect ratioを保存し、KP focal BCEのpositive weightを32へ上げたrun。
26回のvalidation後にearly stoppingし、best `val/kp_mean_dist`は186.4902 pxだった。
32 pxのvalidation合格条件を大幅に超過したため不採用とする。

### アーキテクチャ詳細

full B00、DINOv3 ViT-B/16 + DPT + LoRA、batch size 4、train short-side 256、
validation short-side 512、seed 714を使用した。run開始時のcommit `d3226c0d`はaspect-ratio
geometry修正を含むが、後から修正したtrusted checkpoint loaderとheld-out target bundle拡張は
含まない。best checkpointはepoch 15として保存された。

### メトリクスの解釈

best validation距離186.4902 px、最終192.9146 px、best normalized distance 0.178689、
best validation loss 0.037011だった。旧checkpoint loaderのためtest-after-fitで
`DictConfig`のweights-only unpickling errorが起き、held-out metric/prediction bundleはない。
ただしvalidation閾値だけで不採用は確定しており、test欠落を合格証拠で補完しない。

### アーキテクチャ⇄メトリクスの因果考察

v2のpositive weight 1で得た42.7476 pxに対し、その他の修正を含むv3は186.4902 pxへ
悪化した。32倍のpositive weightingは疎なGaussian targetに対して偽陽性を強く許容し、
peak位置metricを悪化させた過補償が有力である。ただしv2とはgeometryとvalidation解像度も
異なるため、次runではv3からpositive weightだけを戻して検証する。

### 既存実験との比較

v1の134.1354 px、v2の42.7476 pxのいずれよりも悪い。aspect ratio保存だけでは
positive weight 32の悪化を相殺できなかった。v2診断で全channelにpeakが存在したことからも、
強いpositive weightingは必要条件ではなかった。

### 次に有効な実験

v3のdata、geometry、model、seed、validation解像度を固定し、
`loss.kp.positive_weight=1.0`だけへ戻すv4を実行する。これによりweight 32の因果を分離する。
