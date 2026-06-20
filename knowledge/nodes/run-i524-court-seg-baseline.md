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
  curves: knowledge/runs/run-i524-court-seg-baseline/curves.png
  tb_logdir: outputs/court_detection/issue524_seg_baseline/logs/version_0
parents: []
relations: []
tags:
- court-detection
- dinov3
- segmentation
- baseline
---

## 考察 / Findings

### 要約
SSL 事前学習なしで DINOv3 バックボーンを完全凍結したまま court segmentation を学習した baseline。best `val/miou ≈ 0.517`。

### アーキテクチャ詳細
`model=court_seg_dinov3_detr`, `loss=seg`, `data=court_seg`, `backbone_freeze=true`（SSL なしの素の DINOv3 を凍結）。

### メトリクスの解釈
best `val/miou ≈ 0.517`。curves（loss / miou）では学習途中で miou が一度ピーク後に崩れる挙動が見え、凍結特徴のみでは頭打ち。

### アーキテクチャ⇄メトリクスの因果考察
バックボーンを凍結しているため seg ヘッドは汎用 DINOv3 特徴に依存。ドメイン特化していない特徴ではコート構造を十分に分離できず miou が伸びない。

### 既存実験との比較
SSL の効果を測る対照。SSL 済みバックボーンの [[run-i524-court-seg-ssl]]（`0.800`）と直接比較される。

### 次に有効な実験
同条件で SSL 済みバックボーンに差し替え、凍結下での SSL 効果を測る（[[run-i524-court-seg-ssl]]）。
