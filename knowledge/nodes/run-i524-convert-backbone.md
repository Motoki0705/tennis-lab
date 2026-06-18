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
parents: [run-i524-ssl-lora-vitb16]
relations: []
tags: [court-detection, dinov3, ssl, lora]
---

## 考察 / Findings

SSL で得た teacher checkpoint の **LoRA 24 モジュールを scaling=2 でマージ**し、court
detection のバックボーンとして読み込める `.pth`（188 tensor）を生成。これは学習ではなく
変換ステップだが、SSL→下流をつなぐ必須の中間成果物なので 1 ノードとして記録する。

出力 `court_backbone_vitb16.pth` を凍結バックボーンとして使うのが `run-i524-court-seg-ssl`。
