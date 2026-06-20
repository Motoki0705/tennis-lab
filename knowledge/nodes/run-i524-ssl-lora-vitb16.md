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
tags:
- court-detection
- dinov3
- ssl
- lora
---

## 考察 / Findings

### 要約
ViT-B/16 をテニスコート画像で LoRA 自己教師あり学習し、下流バックボーン用の teacher checkpoint を生成。価値は下流効果で測る。

### アーキテクチャ詳細
`third_party/dinov3` の ViT-B/16 を LoRA で SSL 学習。`model=dinov3_vitb16_lora`, `data=wikimedia_tennis_court/images`, `iters=25000`, `batch_size_per_gpu=4`。TensorBoard ではなく `training_metrics.json`（iteration ログ）に記録。

### メトリクスの解釈
25000 iter で `total_loss ≈ 15.9`（`dino_global_crops_loss 11.05` / `ibot_loss 5.05` 主体）に収束。SSL 損失値自体は他タスクと直接比較できない。

### アーキテクチャ⇄メトリクスの因果考察
LoRA で軽量に DINO/iBOT 目的を最適化。損失は下流転移の質を保証する間接指標に過ぎず、絶対値の良し悪しは判断材料にならない。

### 既存実験との比較
本ノードの teacher を [[run-i524-convert-backbone]] が下流バックボーンへ変換し、[[run-i524-court-seg-ssl]] が baseline との差で価値を確定する。

### 次に有効な実験
SSL 自体の価値は下流タスクでの効果（[[run-i524-court-seg-ssl]]）で測る。効けば ball detection / plcs など他下流でも同 SSL バックボーンを試す。
