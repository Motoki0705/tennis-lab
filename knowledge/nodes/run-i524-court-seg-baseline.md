---
id: run-i524-court-seg-baseline
type: run
title: court seg / 凍結バックボーン baseline (SSLなし)
issue: 524
provider: claude
date: 2026-06-18
status: done
config:
  model: court_seg_dinov3_detr
  loss: seg
  data: court_seg
  backbone_freeze: true
metrics:
  best_val_miou: 0.517
artifacts:
  log: .training_queue/logs/1781749340334577228_60616_issue524_court_seg_baseline.log
  output_dir: outputs/court_detection/issue524_seg_baseline
parents: []
relations: []
tags: [court-detection, dinov3, segmentation, baseline]
---

## 考察 / Findings

DINOv3 バックボーンを**完全凍結**したまま court segmentation を学習した baseline
（SSL 事前学習なし）。best `val/miou ≈ 0.517`。SSL の効果を測る対照として使う。

SSL 済みバックボーンとの比較は `run-i524-court-seg-ssl` を参照。
