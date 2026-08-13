---
id: run-i714-court-dinov3-dpt-b00-v6-rescaled-crop
type: run
title: DINOv3+DPT B00 KP7 v6（crop short-side再正規化）
issue: 714
provider: codex
session: 019ff617-dfb3-7182-8c2a-1d0705cf3ff0
date: '2026-08-14'
status: failed
config:
  model: dinov3 + dpt + lora
  loss: focal BCE, kp positive_weight 1.0
  data: synthetic_court B00 KP7, default augmentation, rescaled crop, train short-side
    256, validation short-side 512
  max_epochs: 50
  seed: 714
metrics:
  best_val_kp_mean_dist_px: 101.90948486328125
  final_val_kp_mean_dist_px: 104.86366271972656
  held_out_test_kp_mean_dist_px: 107.46652116165382
  held_out_test_kp_mean_dist_normalized: 0.10297106356089845
  held_out_test_median_px: 11.97334098815918
  held_out_test_p90_px: 383.564703369141
  held_out_test_fraction_gt_40px: 0.36084021005251316
repro:
  commit: 6ff6b6d5aa67704c9105512fc79157f50df83b23
  branch: feat/issue-714-court-data-composition-v2
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.court_detection.scripts.train data/source=synthetic_court data/processing=kp
    model/encoder=dinov3 model/decoder=dpt training=lora loss.kp.positive_weight=1.0
    data.batch_size=4 data.num_workers=4 data.augmentation.train_scales='[256]' data.augmentation.val_short_side=512
    training.trainer.max_epochs=50 training.checkpoint.monitor=val/kp_mean_dist training.checkpoint.mode=min
    training.early_stopping.monitor=val/kp_mean_dist training.early_stopping.mode=min
    training.early_stopping.patience=10 training.qualitative_logging.enabled=false
    run.seed=714 run.output_dir=issue714/i714-court-dinov3-dpt-b00-v6-rescaled-crop
    run.test_after_fit=true
artifacts:
  run_dir: knowledge/runs/run-i714-court-dinov3-dpt-b00-v6-rescaled-crop
  predictions: knowledge/runs/run-i714-court-dinov3-dpt-b00-v6-rescaled-crop/pred_test.npz
  log: .training_queue/logs/1786640463430687264_1141103_i714-court-dinov3-dpt-b00-v6-rescaled-crop.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/issue714/i714-court-dinov3-dpt-b00-v6-rescaled-crop/logs/version_0
  diagnostic: .codex/tasks/issue-714/logs/v6-test-prediction-diagnostic.json
  curves: knowledge/runs/run-i714-court-dinov3-dpt-b00-v6-rescaled-crop/curves.png
  tb_logdir: outputs/issue714/i714-court-dinov3-dpt-b00-v6-rescaled-crop/logs/version_0
parents:
- run-i714-court-dinov3-dpt-b00-v5-light
relations: []
tags:
- court-detection
- synthetic-court
- b00
- dinov3
- dpt
- kp7
- rescaled-crop
- negative
---

## 考察 / Findings

### 要約

aspect-ratioを保存したrandom cropをtrain short-side 256へ再正規化する修正を加え、v4の
default augmentation条件で50 epoch学習した。best validationは101.9095 px、held-out testは
107.4665 pxで、v4より改善したが32/40 pxの合格条件には届かず不採用とする。

### アーキテクチャ詳細

DINOv3 ViT-B/16 + DPT + rank-8 LoRA、synthetic semantic KP7、batch size 4、default
augmentation、positive weight 1、validation short-side 512、seed 714を使用した。geometryは
sourceを短辺256へresizeしrandom cropした後、cropの縦横比を保存して短辺256へ戻す。
best checkpointはepoch 44で、278 test samplesを評価した。

### メトリクスの解釈

best/final validationは101.9095/104.8637 px、held-out testは107.4665 px
（normalized 0.102971）だった。保存predictionからの再計算は107.3951 pxである。median
11.9733 px、p90 383.5647 px、40 px超36.08%で、中央値は小さいがlong tailが残る。
train距離は27.9071 pxまで低下し、validationとの差は依然大きい。

### アーキテクチャ⇄メトリクスの因果考察

v4比でbest validationは16.3494 px、testは9.2428 px改善し、crop後scale不整合が一因だった
ことを支持する。全semantic channelのmeanは100.43–119.21 px、court slot別は109.0111/
106.1742 pxで、collapseはない。`far_baseline`は70.2308 px、`complex_center`は134.8763 px
で、修正後もcamera/coverage依存long tailが主な残差である。

### 既存実験との比較

v5 light augmentationの167.5116/171.5415 pxから大幅に改善し、v4の118.2589/
116.7093 pxも上回った。一方、旧square geometryかつ256px validationのv2 42.7476 pxとは
評価条件が異なり、v6は512pxで事前合意した品質閾値を満たしていない。

### 次に有効な実験

v6をbaselineに、疎なGaussian正例を適度に強調する`positive_weight=4`だけを変更する。
weight 32は過補償だったが1ではbackground優勢が残りうるため、中間の4でfalse-positiveを
増やし過ぎずpeak localizationとlong-tailを改善できるかを同じvalidation/test契約で検証する。
