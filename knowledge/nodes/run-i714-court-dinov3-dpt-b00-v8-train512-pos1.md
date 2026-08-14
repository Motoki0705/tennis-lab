---
id: run-i714-court-dinov3-dpt-b00-v8-train512-pos1
type: run
title: DINOv3+DPT B00 KP7 v8（train 512 / batch 4 resource probe）
issue: 714
provider: codex
session: 019ff617-dfb3-7182-8c2a-1d0705cf3ff0
date: '2026-08-14'
status: failed
config:
  model: dinov3 + dpt + lora
  loss: focal BCE, kp positive_weight 1.0
  data: synthetic_court B00 KP7, default augmentation, rescaled crop, train and
    validation short-side 512, batch size 4
metrics: {}
repro:
  commit: 08734ca3e94fb8b53c0803a8d2e6b69110f2cb72
  branch: feat/issue-714-court-data-composition-v2
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.court_detection.scripts.train data/source=synthetic_court data/processing=kp
    model/encoder=dinov3 model/decoder=dpt training=lora loss.kp.positive_weight=1.0
    data.batch_size=4 data.num_workers=4 data.augmentation.train_scales='[512]' data.augmentation.val_short_side=512
    training.trainer.max_epochs=50 training.checkpoint.monitor=val/kp_mean_dist training.checkpoint.mode=min
    training.early_stopping.monitor=val/kp_mean_dist training.early_stopping.mode=min
    training.early_stopping.patience=10 training.qualitative_logging.enabled=false
    run.seed=714 run.output_dir=issue714/i714-court-dinov3-dpt-b00-v8-train512-pos1
    run.test_after_fit=true
artifacts:
  run_dir: knowledge/runs/run-i714-court-dinov3-dpt-b00-v8-train512-pos1
  log: .training_queue/logs/1786671265085013680_1471448_i714-court-dinov3-dpt-b00-v8-train512-pos1.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/issue714/i714-court-dinov3-dpt-b00-v8-train512-pos1/logs/version_0
parents:
- run-i714-court-dinov3-dpt-b00-v6-rescaled-crop
relations: []
tags:
- court-detection
- synthetic-court
- b00
- dinov3
- dpt
- kp7
- train-512
- resource-failure
- negative
---

## 考察 / Findings

### 要約

v6からtrain short-sideだけを256から512へ変更したところ、sanity validationは通過したが
最初のtraining batchのbackward中に`CUDA driver error: device not ready`で終了した。品質を
評価できるrunではなく、batch size 4では16 GB GPUの実行余力が不足するというresource probe
として記録する。

### アーキテクチャ詳細

DINOv3 ViT-B/16 + DPT + rank-8 LoRA、positive weight 1、default augmentation、
train/validation short-side 512、batch size 4、seed 714を使用した。CUDA bf16 mixed precisionで
sanity validation 2 batchを完了後、epoch 0の最初のtraining stepで失敗した。

### メトリクスの解釈

optimizer stepとvalidation epochを完了しておらず、品質metric・checkpoint・test predictionは
存在しない。このrunを32/40 px品質契約の成否には数えない。

### アーキテクチャ⇄メトリクスの因果考察

forwardのみのsanity validationはbatch size 4で成功し、勾配を保持するtraining forward中に
CUDA driver errorとなった。終了後の`nvidia-smi`はGPUを正常に列挙し、使用memoryは115 MiB
だったため、恒久的なdevice故障よりtraining activation/gradientの瞬間的なresource不足に
整合する。ただし例外が明示的な`CUDA out of memory`ではないため、原因を断定しない。

### 既存実験との比較

train short-side 256 / batch size 4のv6・v7は学習を完了している。512へ上げたこのrunだけが
最初のtraining stepで失敗したため、解像度仮説を捨てずにmicro-batchを下げる根拠とする。

### 次に有効な実験

train/validation short-side 512とeffective batch size 4を維持し、micro-batchを2、
`accumulate_grad_batches=2`へ変更する。これは品質仮説ではなく、同じ実験を16 GB GPUで実行する
ためのresource adaptationとして扱う。
