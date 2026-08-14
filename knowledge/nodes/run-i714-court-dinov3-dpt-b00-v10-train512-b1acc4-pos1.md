---
id: run-i714-court-dinov3-dpt-b00-v10-train512-b1acc4-pos1
type: run
title: DINOv3+DPT B00 KP7 v10（train 512 / batch 1 accumulation 4）
issue: 714
provider: codex
session: 019ff617-dfb3-7182-8c2a-1d0705cf3ff0
date: '2026-08-14'
status: failed
config:
  model: dinov3 + dpt + lora
  loss: focal BCE, kp positive_weight 1.0
  data: synthetic_court B00 KP7, default augmentation, rescaled crop, train and validation
    short-side 512, micro-batch 1, gradient accumulation 4
  max_epochs: 50
  seed: 714
metrics:
  best_val_kp_mean_dist_px: 148.33570861816406
  final_val_kp_mean_dist_px: 152.56988525390625
  held_out_test_kp_mean_dist_px: 146.04077172265943
  held_out_test_kp_mean_dist_normalized: 0.13993170547686767
  held_out_test_median_px: 31.67340620234529
  held_out_test_p90_px: 433.7886949348966
  held_out_test_fraction_gt_40px: 0.4836209052263066
repro:
  commit: bb12e43baad47f81f1d0c3ccedaf2a92bf000c09
  branch: feat/issue-714-court-data-composition-v2
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.court_detection.scripts.train data/source=synthetic_court data/processing=kp
    model/encoder=dinov3 model/decoder=dpt training=lora loss.kp.positive_weight=1.0
    data.batch_size=1 data.num_workers=4 data.augmentation.train_scales='[512]' data.augmentation.val_short_side=512
    training.trainer.accumulate_grad_batches=4 training.trainer.max_epochs=50 training.checkpoint.monitor=val/kp_mean_dist
    training.checkpoint.mode=min training.early_stopping.monitor=val/kp_mean_dist
    training.early_stopping.mode=min training.early_stopping.patience=10 training.qualitative_logging.enabled=false
    run.seed=714 run.output_dir=issue714/i714-court-dinov3-dpt-b00-v10-train512-b1acc4-pos1
    run.test_after_fit=true
artifacts:
  run_dir: knowledge/runs/run-i714-court-dinov3-dpt-b00-v10-train512-b1acc4-pos1
  predictions: knowledge/runs/run-i714-court-dinov3-dpt-b00-v10-train512-b1acc4-pos1/pred_test.npz
  log: .training_queue/logs/1786671913188158727_1481979_i714-court-dinov3-dpt-b00-v10-train512-b1acc4-pos1.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/issue714/i714-court-dinov3-dpt-b00-v10-train512-b1acc4-pos1/logs/version_0
  tb_logdir: outputs/issue714/i714-court-dinov3-dpt-b00-v10-train512-b1acc4-pos1/logs/version_0
  curves: knowledge/runs/run-i714-court-dinov3-dpt-b00-v10-train512-b1acc4-pos1/curves.png
parents:
- run-i714-court-dinov3-dpt-b00-v9-train512-b2acc2-pos1
relations:
- to: run-i714-court-dinov3-dpt-b00-v6-rescaled-crop
  rel: compares
tags:
- court-detection
- synthetic-court
- b00
- dinov3
- dpt
- kp7
- train-512
- gradient-accumulation
- negative
---

## 考察 / Findings

### 要約

v9のCUDA driver errorを避けるためmicro-batchを1、gradient accumulationを4とし、
train/validation short-side 512の一変数検証を完走させた。best validationは
148.3357 px、held-out testは146.0408 pxで、32/40 pxの合格条件を満たさず不採用とする。

### アーキテクチャ詳細

DINOv3 ViT-B/16 + DPT + rank-8 LoRA、synthetic semantic KP7、default
augmentation、crop short-side再正規化、`positive_weight=1`、seed 714を使用した。
micro-batch 1と4-step accumulationによりv6と同じeffective batch size 4を維持した。
50 epochを完走し、best checkpointはepoch 47、held-out test 278 samplesを評価した。

### メトリクスの解釈

best/final validationは148.3357/152.5699 px、held-out testは146.0408 px
（normalized 0.139932）だった。保存predictionからの再計算は145.8586 px、median
31.6734 px、p90 433.7887 px、40 px超48.36%である。すべての数値配列と距離はfiniteで、
missing peakは0だった。train距離も最終91.4033 pxに留まり、validationとのgapだけでなく
train局在自体がv6より悪い。

### アーキテクチャ⇄メトリクスの因果考察

semantic channel別meanは131.47--160.27 px、court slot別meanは149.6201/143.0168 pxで、
特定classや第二courtだけの系統的collapseはない。1,946 sample-channelのうち1,925で4 peakが
有効で、残りも1--3 peakを保持した。したがって失敗はpeak消失ではなく局在全般の悪化である。
高解像度化で入力detailは増えたが、同じ50 epoch・optimizer update数ではtrain距離が十分に
下がらず、期待したlong-tail改善を得られなかったという観測に整合する。

### 既存実験との比較

品質baseline v6のbest validation 101.9095 px、test 107.4665 pxに対し、v10はそれぞれ
46.4262 px、38.5743 px悪化した。v8/v9で発生したCUDA driver errorはmicro-batch 1で
回避できたため、v10はtrain512条件の品質を判定できる完走runである。train512を続ける
追加resource調整は停止する。

### 次に有効な実験

v6のtrain256へ戻し、geometry修正後には未検証の`data/augmentation=light`だけを変更する。
v5のlight条件はcrop short-side再正規化前だったため、その悪化だけでは修正後lightを反証
できない。milder crop/affine/perspectiveでv6のview/coverage依存long tailを縮められるかを、
同じvalidation/test閾値とseedで検証する。
