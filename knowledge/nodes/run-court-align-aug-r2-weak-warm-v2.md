---
id: run-court-align-aug-r2-weak-warm-v2
type: run
title: court-align-aug-r2-weak-warm-v2
provider: codex
session: 01a05a02-ad55-76c0-9fcb-f250929e59a7
date: '2026-09-02'
status: done
config:
  model: CourtAlignmentCNN（4-down U-Net、KP14 heatmap + 2ch center vote）
  loss: focal heatmap + 0.05 masked Smooth L1 center-vote
  data: b00_weak_structure_v2
  image_size: 256
  train_samples: 2048
  val_samples: 256
  test_samples: 256
  batch_size: 16
  steps_per_epoch: 128
  max_epochs: 30
  learning_rate: 0.0003
  warmup_steps: 128
  seed: 42
  sigma_px: 2.0
  initialization: sigma=2.0 checkpointからstrict model-only warm-start
metrics:
  instance_precision: 0.9922680412371134
  instance_recall: 1.0
  instance_f1: 0.9961190168175938
  instance_count_accuracy: 0.98828125
  instance_count_mae: 0.01171875
  matched_center_mean_error_px: 0.519919042476731
  instance_kp_mean_error_px: 1.6476788143790417
  instance_kp_pck_2px: 0.9961089494163424
  instance_kp_pck_4px: 0.9961089494163424
  sim2_translation_error_px: 0.1138000769722094
  sim2_rotation_error_deg: 0.08316241641059156
  sim2_scale_relative_error: 0.0012445012018137624
repro:
  commit: 5de3a7d45e037a799d4dad0ae1ef3ac3cb24897e
  branch: feat/court-alignment-kp14-pipeline
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HYDRA_FULL_ERROR=1 /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.court_alignment.scripts.train data=b00_weak_structure_v2 paths.checkpoint_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/court-alignment-kp14/outputs
    data.train_samples=2048 data.val_samples=256 data.test_samples=256 data.batch_size=16
    training.steps_per_epoch=128 training.trainer.max_epochs=30 training.learning_rate=3.0e-4
    training.warmup_steps=128 run.seed=42 run.init_weights=court_alignment/ablation_sigma_200/logs/version_0/checkpoints/court-alignment-epoch\\=047.ckpt
    run.output_dir=court_alignment/augmentation_round2/b00_weak_warm30_lr3e4_s42
artifacts:
  run_dir: knowledge/runs/run-court-align-aug-r2-weak-warm-v2
  predictions: knowledge/runs/run-court-align-aug-r2-weak-warm-v2/pred_test.npz
  output_dir: outputs/court_alignment/augmentation_round2/b00_weak_warm30_lr3e4_s42/logs/version_0
  curves: knowledge/runs/run-court-align-aug-r2-weak-warm-v2/curves.png
  tb_logdir: outputs/court_alignment/augmentation_round2/b00_weak_warm30_lr3e4_s42/logs/version_0
parents:
- run-court-align-kp14-ablation-sigma-200
- run-court-align-aug-pilot-appearance-v1
- run-court-align-aug-pilot-structure-v1
relations:
- to: run-court-align-aug-pilot-combined-v1
  rel: supersedes
tags:
- court-alignment
- kp14
- augmentation
- b00
- weak-structure
- warm-start
---

## 考察 / Findings

### 要約

σ=2.0を固定し、appearanceに弱いstructure noiseを追加したround 2学習である。synthetic test F1=0.9961190168175938を保ちながら、後続のB00評価で2面の正しいpose候補を生成した。

### アーキテクチャ詳細

4-down U-NetのKP14 heatmap + 2ch center-vote CNNを用い、σ=2.0、2048/256/256 samples、batch 16、128 steps/epoch、30 epochs、LR 3e-4、warmup 128 steps、seed 42で学習した。clean σ=2.0 epoch 47 checkpointを検証し、model weightsだけをstrict loadした。`b00_weak_structure_v2`はappearanceと3--8 px/m scaleに加え、line dropoutをprobability 0.08、ghost/false lineを0.18で弱く追加する。round 1 combinedの強い構造攪乱やpartial cropは採用していない。

### メトリクスの解釈

bundle `metrics.json` のsynthetic testはF1=0.9961190168175938、KP誤差=1.6476788143790417 px、center誤差=0.519919042476731 px、count accuracy=0.98828125である。`diagnostic_metrics.json`ではTP=385、FP=3、FN=0、visible KP coverage=0.9961089494163424、loss=0.027416229248046875だった。弱い構造noiseでもcleanに近いinstance性能を維持した。

### アーキテクチャ⇄メトリクスの因果考察

観測として、appearance-onlyよりFPが1件増え、KP/center誤差もわずかに悪化したが、F1低下は0.0013未満だった。仮説として、低確率の欠損・ghost lineは幾何表現を壊すほど強くなく、B00固有の欠損や平行線へ対応するための不変性を与えた。

### 既存実験との比較

親のclean σ=2.0はF1=0.9973684210526316、round 1 appearance/structureはそれぞれ0.827103/0.760046、round 1 combinedは0.667418だった。本runはround 1の構造強度を下げ、warm-startと学習予算拡大によりsynthetic F1を0.9961190168175938まで回復した。appearance-only round 2のF1=0.9974093264248705より僅かに低いが、B00の2面pose候補では本runが有望だった。

### 次に有効な実験

B00の複数courtで各semantic channelが複数peakを必要とするため、checkpointを変えずdecoder `max_peaks`を4から8へ拡張し、synthetic保持と実B00成功を同時に確認する。
