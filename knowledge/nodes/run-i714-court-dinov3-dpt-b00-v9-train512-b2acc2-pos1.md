---
id: run-i714-court-dinov3-dpt-b00-v9-train512-b2acc2-pos1
type: run
title: DINOv3+DPT B00 KP7 v9（train 512 / batch 2 accumulation 2）
issue: 714
provider: codex
session: 019ff617-dfb3-7182-8c2a-1d0705cf3ff0
date: '2026-08-14'
status: failed
config:
  model: dinov3 + dpt + lora
  loss: focal BCE, kp positive_weight 1.0
  data: synthetic_court B00 KP7, default augmentation, rescaled crop, train and validation
    short-side 512, micro-batch 2, gradient accumulation 2
metrics: {}
repro:
  commit: 3f643ec2cbf4ca16ef3c034cd6c958e37bafe2de
  branch: feat/issue-714-court-data-composition-v2
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.court_detection.scripts.train data/source=synthetic_court data/processing=kp
    model/encoder=dinov3 model/decoder=dpt training=lora loss.kp.positive_weight=1.0
    data.batch_size=2 data.num_workers=4 data.augmentation.train_scales='[512]' data.augmentation.val_short_side=512
    training.trainer.accumulate_grad_batches=2 training.trainer.max_epochs=50 training.checkpoint.monitor=val/kp_mean_dist
    training.checkpoint.mode=min training.early_stopping.monitor=val/kp_mean_dist
    training.early_stopping.mode=min training.early_stopping.patience=10 training.qualitative_logging.enabled=false
    run.seed=714 run.output_dir=issue714/i714-court-dinov3-dpt-b00-v9-train512-b2acc2-pos1
    run.test_after_fit=true
artifacts:
  run_dir: knowledge/runs/run-i714-court-dinov3-dpt-b00-v9-train512-b2acc2-pos1
  log: .training_queue/logs/1786671464695442495_1475187_i714-court-dinov3-dpt-b00-v9-train512-b2acc2-pos1.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/issue714/i714-court-dinov3-dpt-b00-v9-train512-b2acc2-pos1/logs/version_0
  curves: knowledge/runs/run-i714-court-dinov3-dpt-b00-v9-train512-b2acc2-pos1/curves.png
  tb_logdir: outputs/issue714/i714-court-dinov3-dpt-b00-v9-train512-b2acc2-pos1/logs/version_0
parents:
- run-i714-court-dinov3-dpt-b00-v8-train512-pos1
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

v8の512px品質仮説とeffective batch size 4を保ち、micro-batch 2とgradient accumulation 2へ
変更した。epoch 0の224/1,374 training batchまで進んだ後、v8と同じ`CUDA driver error:
device not ready`で終了し、品質評価には到達しなかった。

### アーキテクチャ詳細

DINOv3 ViT-B/16 + DPT + rank-8 LoRA、positive weight 1、default augmentation、
train/validation short-side 512、micro-batch 2、gradient accumulation 2、seed 714を使用した。
bf16 mixed precisionでsanity validationを完了し、約3分39秒trainingを実行した。

### メトリクスの解釈

validation epoch、checkpoint、test predictionを生成する前に終了したため品質metricはない。
training lossは実行中に低下したが、未完了epochの途中値を品質証拠として扱わない。

### アーキテクチャ⇄メトリクスの因果考察

batch 4のv8より長く動作したことはresource圧縮の効果と整合するが、同じCUDA例外が再発した。
CPUで同じ設定の先頭226 DataLoader batchを監査すると、最大padded shapeは
`[2,3,720,928]`、失敗番号付近は`[2,3,512,736–992]`で、設定されたcrop ratio 0.5–2.0の
範囲内だった。単純な不正shapeは確認できず、micro-batch 2でも高解像度trainingのCUDA実行
余力が不足する可能性を残す。

### 既存実験との比較

v8は最初のtraining batchで失敗したが、v9は224 batchまで進んだ。train 256 / batch 4の
v6・v7は完走しているため、モデルやデータ全般ではなく512px training固有のresource制約で
ある可能性が高い。ただし明示的なOOMではないため断定しない。

### 次に有効な実験

train/validation short-side 512、effective batch size 4を維持し、micro-batchを1、
`accumulate_grad_batches=4`へ下げて最後のresource adaptationを行う。これも品質アブレーション
ではなく同じ512px実験を成立させるための実行条件変更である。
