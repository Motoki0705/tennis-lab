---
id: run-i714-court-dinov3-dpt-b00-v2-aligned256
type: run
title: DINOv3+DPT B00 KP7 v2（256px整合・評価失敗）
issue: 714
provider: codex
date: '2026-08-13'
status: failed
session: 019ff617-dfb3-7182-8c2a-1d0705cf3ff0
config:
  model: dinov3 + dpt + lora
  loss: focal BCE, kp positive_weight 1.0
  data: synthetic_court B00 KP7, train short-side 256, validation short-side 256
  max_epochs: 50
  seed: 714
metrics:
  validation_events: 50
  best_val_kp_mean_dist_px: 42.74757385253906
  final_val_kp_mean_dist_px: 43.69489669799805
  diagnostic_epoch: 11
  diagnostic_sample_count: 408
  diagnostic_matched_mean_court_000_px: 42.14761072573455
  diagnostic_matched_mean_court_001_px: 37.585114880858804
  held_out_test_kp_mean_dist_px: null
repro:
  commit: 1b58215aca5eeffcbc34f7f3b7a7ccb53c5dd254
  branch: feat/issue-714-court-data-composition-v2
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: "PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.tasks.court_detection.scripts.train data/source=synthetic_court data/processing=kp model/encoder=dinov3 model/decoder=dpt training=lora data.batch_size=4 data.num_workers=4 data.augmentation.train_scales='[256]' data.augmentation.val_short_side=256 training.trainer.max_epochs=50 training.checkpoint.monitor=val/kp_mean_dist training.checkpoint.mode=min training.early_stopping.monitor=val/kp_mean_dist training.early_stopping.mode=min training.early_stopping.patience=10 training.qualitative_logging.enabled=false run.seed=714 run.output_dir=issue714/i714-court-dinov3-dpt-b00-v2-aligned256 run.test_after_fit=true"
artifacts:
  log: .training_queue/logs/1786553477246458595_2135737_i714-court-dinov3-dpt-b00-v2-aligned256.log
  job: .training_queue/failed/1786553477246458595_2135737_i714-court-dinov3-dpt-b00-v2-aligned256.job
  diagnostic: .codex/tasks/issue-714/logs/v2-epoch11-channel-diagnostic.json
parents:
- run-i714-court-dinov3-dpt-b00-v1
relations: []
tags:
- court-detection
- synthetic-court
- b00
- dinov3
- dpt
- kp7
- negative
---

## 考察 / Findings

### 要約

DINOv3 + DPT + LoRAをtrain/validationともshort-side 256で50 epoch完走した。
best `val/kp_mean_dist`は42.7476 pxで32 pxのvalidation合格条件を満たさず、さらに
test-after-fitのbest checkpoint復元がPyTorch 2.6の`weights_only=True`既定値により
失敗したため、queue上はfailedとなった。

### アーキテクチャ詳細

full B00、synthetic KP7、batch size 4、seed 714、最大50 epochを使用した。プロセス開始時の
実装はRGBを正方形へ歪める旧geometryを読み込んでいた。epoch 45のbest checkpointは保存
されたが、Lightning checkpoint内の`omegaconf.DictConfig`をweights-only loaderが拒否し、
held-out prediction bundleは生成されなかった。

### メトリクスの解釈

TensorBoardの50 validation eventでbestは42.7476 px、最終は43.6949 pxだった。epoch 11
checkpointを408 validation samplesへFP32で診断した結果、7 channelすべてでthresholded
peakが出力され、`samples_with_no_peak = 0`だった。court-000/court-001のmatched meanは
42.1476/37.5851 pxである。これは診断値であり、欠落したheld-out test metricの代替ではない。

### アーキテクチャ⇄メトリクスの因果考察

診断では単一semantic classや第二courtの一律消失ではなく、複数channelが同時に外れる
camera-condition依存のlong tailが見られた。正方形resizeによるaspect-ratio distortionは
全channelの座標対応を同時に悪化させうるため、共有geometry修正の根拠になった。ただし
epoch 11の診断だけでは全誤差をgeometryへ帰属できない。

### 既存実験との比較

v1のbest validation 134.1354 pxから42.7476 pxへ改善し、各channelのpeak欠落も観測され
なかった。一方、validation閾値32 pxを10.7476 px超過し、held-out test値も取得できて
いないため、v1/v2はいずれも受け入れ証拠にはならない。

### 次に有効な実験

aspect ratioを保存するshared geometry、KP positive weight 32、validation short-side 512を
組み合わせ、trusted local best checkpointを`weights_only=False`で復元してheld-out test
predictionまで保存する。
