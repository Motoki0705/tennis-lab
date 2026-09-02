---
id: group-court-align-b00-augmentation-pilot
type: group
title: B00実heatmap augmentation pilot
members:
- run-court-align-aug-pilot-scale-v1
- run-court-align-aug-pilot-appearance-v1
- run-court-align-aug-pilot-structure-v1
- run-court-align-aug-pilot-combined-v1
- run-court-align-b00-eval-aug-pilot-scale-v1
- run-court-align-b00-eval-aug-pilot-appearance-v1
- run-court-align-b00-eval-aug-pilot-structure-v1
- run-court-align-b00-eval-aug-pilot-combined-v1
parents:
- group-court-align-b00-real-heatmap-eval
- group-court-align-kp14-sigma-ablation
tags: [court-alignment, kp14, augmentation, b00, real-heatmap]
---

## 考察 / Findings

## まとめ

σ=2.0を固定した4条件のpilotでは、B00実heatmapのF1はscale=0、appearance=0.5、structure=0.5、combined=0だった。appearanceはmatched center=0.59pxで最良、structureは3.86px、scale/combinedは未マッチpenaltyだった。単純な全augmentation同時適用はsynthetic test F1も0.667まで低下し、採用しない。

2面のうち1面だけ対応できたため、次はappearanceを軸にstructureのdropout/false-line強度を下げた段階的構成を試す。評価referenceは従来通りaccepted alignmentで、独立GTではない。
