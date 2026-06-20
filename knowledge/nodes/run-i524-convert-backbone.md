---
id: run-i524-convert-backbone
type: run
title: SSL teacher → court backbone 変換 (LoRA merge)
issue: 524
provider: claude
date: 2026-06-18
status: done
config:
  script: convert_dinov3_ssl_backbone
metrics:
  merged_lora_modules: 24
  lora_scaling: 2
  tensors: 188
artifacts:
  log: .training_queue/logs/1781749340317616658_60604_issue524_convert_backbone.log
  output_dir: outputs/dino_ssl/issue524_vitb16_lora/court_backbone_vitb16.pth
parents:
- run-i524-ssl-lora-vitb16
relations: []
tags:
- court-detection
- dinov3
- ssl
- lora
---

## 考察 / Findings

### 要約
SSL teacher の LoRA をマージし、court detection のバックボーンとして読み込める `.pth` を生成する変換ステップ。

### アーキテクチャ詳細
`script=convert_dinov3_ssl_backbone`。teacher checkpoint の LoRA 24 モジュールを `scaling=2` でマージし、188 tensor の `court_backbone_vitb16.pth` を出力。学習ではなく変換だが SSL→下流をつなぐ必須中間成果物なので 1 ノードとして記録（TensorBoard なし）。

### メトリクスの解釈
`merged_lora_modules=24`, `lora_scaling=2`, `tensors=188`。学習指標ではなく変換の検証値。

### アーキテクチャ⇄メトリクスの因果考察
LoRA 重みをベースに焼き込むことで、下流側は追加依存なしに通常の ViT-B/16 として凍結ロードできる。

### 既存実験との比較
親 [[run-i524-ssl-lora-vitb16]] の出力を受け、子 [[run-i524-court-seg-ssl]] の入力（凍結バックボーン）になる橋渡しノード。

### 次に有効な実験
出力 `court_backbone_vitb16.pth` を凍結バックボーンとして court seg に投入（[[run-i524-court-seg-ssl]]）。
