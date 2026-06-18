---
id: run-i524-ssl-lora-vitb16
type: run
title: DINOv3 ViT-B/16 LoRA SSL (tennis court)
issue: 524
provider: claude
date: 2026-06-18
status: done
config:
  model: dinov3_vitb16_lora
  data: wikimedia_tennis_court/images
  iters: 25000
  batch_size_per_gpu: 4
metrics:
  total_loss: 15.93
  dino_global_crops_loss: 11.05
  ibot_loss: 5.05
artifacts:
  log: .training_queue/logs/1781749340298963914_60592_issue524_ssl_lora_vitb16.log
  output_dir: outputs/dino_ssl/issue524_vitb16_lora
parents: []
relations: []
tags: [court-detection, dinov3, ssl, lora]
---

## 考察 / Findings

`third_party/dinov3` の ViT-B/16 を **LoRA で自己教師あり (SSL) 学習**。データは
`data/dino_ssl/wikimedia_tennis_court/images`（テニスコート画像）。25000 iter で
`total_loss` は約 15.9 に収束（dino / ibot loss 主体）。teacher checkpoint を出力し、
これを下流の court detection 用バックボーンへ変換する（→ `run-i524-convert-backbone`）。

SSL 自体の損失値は他タスクと直接比較できないため、価値は**下流タスクでの効果**で測る。
その結論は `run-i524-court-seg-ssl` を参照。
