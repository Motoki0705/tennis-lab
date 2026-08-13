---
id: run-i714-court-dinov3-dpt-b00-v5-light
type: run
title: DINOv3+DPT B00 KP7 v5（light augmentation）
issue: 714
provider: codex
session: 019ff617-dfb3-7182-8c2a-1d0705cf3ff0
date: '2026-08-14'
status: failed
config:
  model: dinov3 + dpt + lora
  loss: focal BCE, kp positive_weight 1.0
  data: synthetic_court B00 KP7, light augmentation, train short-side 256, validation
    short-side 512
  max_epochs: 50
  seed: 714
metrics:
  validation_events: 20
  best_val_kp_mean_dist_px: 167.51162719726562
  final_val_kp_mean_dist_px: 180.6582489013672
  held_out_test_kp_mean_dist_px: 171.54146045730707
  held_out_test_kp_mean_dist_normalized: 0.16436566907700884
  held_out_test_median_px: 94.33623504638672
  held_out_test_p90_px: 471.2086608886719
  held_out_test_fraction_gt_40px: 0.6096524131032758
repro:
  commit: 572df05a0035efe0bd5463e95816151fdfa0ea57
  branch: feat/issue-714-court-data-composition-v2
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.court_detection.scripts.train data/source=synthetic_court data/processing=kp
    data/augmentation=light model/encoder=dinov3 model/decoder=dpt training=lora loss.kp.positive_weight=1.0
    data.batch_size=4 data.num_workers=4 data.augmentation.train_scales='[256]' data.augmentation.val_short_side=512
    training.trainer.max_epochs=50 training.checkpoint.monitor=val/kp_mean_dist training.checkpoint.mode=min
    training.early_stopping.monitor=val/kp_mean_dist training.early_stopping.mode=min
    training.early_stopping.patience=10 training.qualitative_logging.enabled=false
    run.seed=714 run.output_dir=issue714/i714-court-dinov3-dpt-b00-v5-light run.test_after_fit=true
artifacts:
  run_dir: knowledge/runs/run-i714-court-dinov3-dpt-b00-v5-light
  predictions: knowledge/runs/run-i714-court-dinov3-dpt-b00-v5-light/pred_test.npz
  log: .training_queue/logs/1786633891521434968_1068721_i714-court-dinov3-dpt-b00-v5-light.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/issue714/i714-court-dinov3-dpt-b00-v5-light/logs/version_0
  diagnostic: .codex/tasks/issue-714/logs/v5-test-prediction-diagnostic.json
  curves: knowledge/runs/run-i714-court-dinov3-dpt-b00-v5-light/curves.png
  tb_logdir: outputs/issue714/i714-court-dinov3-dpt-b00-v5-light/logs/version_0
parents:
- run-i714-court-dinov3-dpt-b00-v4-geometry-pos1
relations: []
tags:
- court-detection
- synthetic-court
- b00
- dinov3
- dpt
- kp7
- light-augmentation
- negative
---

## 考察 / Findings

### 要約

v4からaugmentation presetだけを`default`から`light`へ変更した。early stoppingで20回の
validation後に終了し、best validation 167.5116 px、held-out test 171.5415 pxとなった。
v4の118.2589/116.7093 pxから明確に悪化し、32/40 pxの合格条件を満たさないため不採用とする。

### アーキテクチャ詳細

DINOv3 ViT-B/16 + DPT + rank-8 LoRA、synthetic semantic KP7、batch size 4、train
short-side 256、validation short-side 512、positive weight 1、seed 714はv4と同一である。
変更したlight presetはcrop scaleを0.6–1.0、rotationを8度、translationを6%、affine
scaleを0.9–1.1、shearを5度へ抑え、perspective・photometric変換も緩和する。best
checkpointはepoch 9だった。

### メトリクスの解釈

best/final validationは167.5116/180.6582 px、held-out testは171.5415 px
（normalized 0.164366）だった。保存predictionから同じnearest-peak metricを再計算すると
171.5105 pxである。medianは94.3362 px、p90は471.2087 px、40 px超は60.97%で、v4の
17.5604/382.9532 px/42.31%よりlong tailだけでなく中央傾向も悪化した。

### アーキテクチャ⇄メトリクスの因果考察

全7 semantic channelで全サンプルに4 peakがあり、court slot別meanは171.9629/171.1686 px、
unique assignmentでも180.9309/181.6280 pxなので、classや第二courtの欠落ではない。
`far_baseline`は126.1943 px、`complex_center`は205.0196 pxで、v4と同じ視点依存性がより
強く現れた。軽いaugmentationがtrain分布への適合を改善するという仮説は反証され、B00の
camera/coverage変動に対して強いdefault augmentationが必要なregularizerだったと解釈する。

### 既存実験との比較

v4比でbest validationは49.2527 px、testは54.8322 px悪化した。v3のpositive weight 32
（186.4902 px）よりはvalidationが18.9785 px良いが、v4には大きく劣る。したがって次runは
augmentationをdefaultへ戻し、別の一変数を検証する。

### 次に有効な実験

v4/v5後のgeometry監査で、aspect-ratio保存cropをtrain short-sideへ再正規化しておらず、
設定上256でもcrop後の短辺が256未満になることが判明した。crop後に短辺256へresizeし、
縦横比を保存する修正を加えたv6を実行する。v4のdefault augmentation、model、KP loss、
validation解像度、seedを固定し、train scale contractの修正だけを検証する。
