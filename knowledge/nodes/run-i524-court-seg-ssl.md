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
  best_val_miou: 0.8
artifacts:
  log: .training_queue/logs/1781749340351622947_60628_issue524_court_seg_ssl.log
  output_dir: outputs/court_detection/issue524_seg_ssl
  curves: knowledge/runs/run-i524-court-seg-ssl/curves.png
  tb_logdir: outputs/court_detection/issue524_seg_ssl/logs/version_0
parents:
- run-i524-convert-backbone
- run-i524-court-seg-baseline
relations:
- to: run-i524-court-seg-baseline
  rel: improves
tags:
- court-detection
- dinov3
- segmentation
- ssl
---

## 考察 / Findings

### 要約
本 issue の主要結果。テニスコート画像で SSL 済みの DINOv3 を凍結したまま court seg を学習し、best `val/miou ≈ 0.800`。SSL なし baseline (`0.517`) から +0.28 の大幅改善。

### アーキテクチャ詳細
baseline と同一の `court_seg_dinov3_detr` / `loss=seg` / `data=court_seg` / `backbone_freeze=true`。違いはバックボーンが [[run-i524-convert-backbone]] のドメイン特化 SSL 重み（`court_backbone_vitb16.pth`）である点のみ。

### メトリクスの解釈
best `val/miou ≈ 0.800`。baseline (`0.517`) から +0.28。バックボーン以外を揃えた A/B なので差分は SSL に帰属する。

### アーキテクチャ⇄メトリクスの因果考察
ドメイン特化 SSL がコート画像の構造を捉える特徴を獲得し、凍結下でも seg ヘッドが強い特徴を利用できる。バックボーン微調整なしで効くのが要点。

### 既存実験との比較
親 baseline [[run-i524-court-seg-baseline]] を凍結条件のまま大幅改善（`improves`）。入力バックボーンは [[run-i524-convert-backbone]] 経由。

### 次に有効な実験
ドメイン特化 SSL は完全凍結という厳しい条件下でも下流を強く押し上げる。微調整なしで効くため、ball detection / plcs など他下流でも同 SSL バックボーンを試す価値が高い。
