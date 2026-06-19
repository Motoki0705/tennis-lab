---
id: run-i524-court-seg-ssl
type: run
title: court seg / 凍結 SSL バックボーン
issue: 524
provider: claude
date: 2026-06-18
status: done
config:
  model: court_seg_dinov3_detr
  loss: seg
  data: court_seg
  backbone_freeze: true
  backbone_checkpoint: outputs/dino_ssl/issue524_vitb16_lora/court_backbone_vitb16.pth
metrics:
  best_val_miou: 0.800
artifacts:
  log: .training_queue/logs/1781749340351622947_60628_issue524_court_seg_ssl.log
  output_dir: outputs/court_detection/issue524_seg_ssl
parents: [run-i524-convert-backbone, run-i524-court-seg-baseline]
relations:
  - {to: run-i524-court-seg-baseline, rel: improves}
tags: [court-detection, dinov3, segmentation, ssl]
---

## 考察 / Findings

**本 issue の主要結果。** テニスコート画像で SSL 済みの DINOv3 バックボーンを凍結したまま
court segmentation を学習。best `val/miou ≈ 0.800` で、SSL なし baseline (`0.517`) から
**+0.28 の大幅改善**。

→ **ドメイン特化 SSL（テニスコート画像での LoRA DINOv3 SSL）は、バックボーン完全凍結という
厳しい条件下でも下流 court segmentation を強く押し上げる。** バックボーン微調整なしで効くため、
他の下流タスク（ball detection / plcs など）でも同じ SSL バックボーンを試す価値が高い。
