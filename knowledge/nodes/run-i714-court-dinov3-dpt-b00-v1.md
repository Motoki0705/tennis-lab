---
id: run-i714-court-dinov3-dpt-b00-v1
type: run
title: DINOv3+DPT B00 KP7 v1（未重み付け・中断）
issue: 714
provider: codex
date: '2026-08-13'
status: failed
session: 019ff617-dfb3-7182-8c2a-1d0705cf3ff0
config:
  model: dinov3 + dpt + lora
  loss: focal BCE, kp positive_weight 1.0
  data: synthetic_court B00 KP7, train short-side 256, validation short-side 512
  max_epochs: 50
  seed: 714
metrics:
  validation_events: 20
  best_val_kp_mean_dist_px: 134.1353759765625
  held_out_test_kp_mean_dist_px: null
repro:
  commit: 3b6cfb300881f60312c22119a5ecfbe6f4605231
  branch: feat/issue-714-court-data-composition-v2
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: "PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.tasks.court_detection.scripts.train data/source=synthetic_court data/processing=kp model/encoder=dinov3 model/decoder=dpt training=lora data.batch_size=4 data.num_workers=4 data.augmentation.train_scales='[256]' data.augmentation.val_short_side=512 training.trainer.max_epochs=50 training.checkpoint.monitor=val/kp_mean_dist training.checkpoint.mode=min training.early_stopping.monitor=val/kp_mean_dist training.early_stopping.mode=min training.early_stopping.patience=10 training.qualitative_logging.enabled=false run.seed=714 run.output_dir=issue714/i714-court-dinov3-dpt-b00-v1 run.test_after_fit=true"
artifacts:
  log: .training_queue/logs/1786552462805520400_2099219_i714-court-dinov3-dpt-b00-v1.log
  job: .training_queue/failed/1786552462805520400_2099219_i714-court-dinov3-dpt-b00-v1.job
parents: []
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

DINOv3 + DPT + LoRAをB00 synthetic KP7で学習した最初のrun。epoch 20途中で
SIGTERMを受け、queue exit code 143で終了した。20回のvalidation event中の最良値は
`val/kp_mean_dist = 134.1354 px`で、32 pxのvalidation合格条件を満たさなかった。

### アーキテクチャ詳細

full B00、batch size 4、train short-side 256、validation short-side 512、seed 714、
最大50 epochの設定で、KP7 focal BCEのpositive weightingは導入前だった。runは
commit `3b6cfb30`を再現情報として保持する。中断によりtest-after-fitは実行されず、
held-out prediction bundleも存在しない。

### メトリクスの解釈

TensorBoardの20個のvalidation scalarから得た最良値134.1354 pxは、途中runの観測値で
あり、完走runのheld-out性能ではない。`test/kp_mean_dist`がないため、このrunを品質合格の
証拠には使えない。

### アーキテクチャ⇄メトリクスの因果考察

観測上、疎なGaussian positiveに対して未重み付けの画素損失を使う構成では、validation
距離が高止まりした。ただし外部SIGTERMで完走していないため、未重み付けだけを失敗原因と
断定はできない。このrunはpositive weighting導入前の診断baselineとして扱う。

### 既存実験との比較

後続の `run-i714-court-dinov3-dpt-b00-v2-aligned256` は同じbackbone/decoderを完走し、
best validationを42.7476 pxまで改善した。validation解像度も異なるため、改善量を
positive weighting以外の単一要因へ帰属させない。

### 次に有効な実験

shared geometryのaspect ratioを保存し、positive weightingを明示した同一seedのrunで、
512 px validationとheld-out test predictionを最後まで取得する。
