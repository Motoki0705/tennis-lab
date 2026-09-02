---
id: run-court-align-aug-r2-appearance-warm-v1
type: run
title: court-align-aug-r2-appearance-warm-v1
provider: codex
session: 01a05a02-ad55-76c0-9fcb-f250929e59a7
date: '2026-09-02'
status: done
config:
  model: CourtAlignmentCNN（4-down U-Net、KP14 heatmap + 2ch center vote）
  loss: focal heatmap + 0.05 masked Smooth L1 center-vote
  data: b00_appearance_v1
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
  instance_precision: 0.9948320413436692
  instance_recall: 1.0
  instance_f1: 0.9974093264248705
  instance_count_accuracy: 0.9921875
  instance_count_mae: 0.0078125
  matched_center_mean_error_px: 0.48443875948039744
  instance_kp_mean_error_px: 1.5569069280147785
  instance_kp_pck_2px: 0.996352140077821
  instance_kp_pck_4px: 0.996352140077821
  sim2_translation_error_px: 0.11526801662410718
  sim2_rotation_error_deg: 0.07989881155877798
  sim2_scale_relative_error: 0.0012843319784037908
repro:
  commit: 5de3a7d45e037a799d4dad0ae1ef3ac3cb24897e
  branch: feat/court-alignment-kp14-pipeline
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HYDRA_FULL_ERROR=1 /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.court_alignment.scripts.train data=b00_appearance_v1 paths.checkpoint_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/court-alignment-kp14/outputs
    data.train_samples=2048 data.val_samples=256 data.test_samples=256 data.batch_size=16
    training.steps_per_epoch=128 training.trainer.max_epochs=30 training.learning_rate=3.0e-4
    training.warmup_steps=128 run.seed=42 run.init_weights=court_alignment/ablation_sigma_200/logs/version_0/checkpoints/court-alignment-epoch\\=047.ckpt
    run.output_dir=court_alignment/augmentation_round2/b00_appearance_warm30_lr3e4_s42
artifacts:
  run_dir: knowledge/runs/run-court-align-aug-r2-appearance-warm-v1
  predictions: knowledge/runs/run-court-align-aug-r2-appearance-warm-v1/pred_test.npz
  output_dir: outputs/court_alignment/augmentation_round2/b00_appearance_warm30_lr3e4_s42/logs/version_0
  curves: knowledge/runs/run-court-align-aug-r2-appearance-warm-v1/curves.png
  tb_logdir: outputs/court_alignment/augmentation_round2/b00_appearance_warm30_lr3e4_s42/logs/version_0
parents:
- run-court-align-kp14-ablation-sigma-200
- run-court-align-aug-pilot-appearance-v1
relations:
- to: run-court-align-aug-pilot-combined-v1
  rel: supersedes
tags:
- court-alignment
- kp14
- augmentation
- b00
- appearance
- warm-start
---

## 考察 / Findings

### 要約

σ=2.0を固定し、appearance augmentationだけを強化したround 2学習である。strict model-only warm-startと学習予算拡大により、synthetic testでF1=0.9974093264248705を回復し、B00を評価できるcheckpointを得た。

### アーキテクチャ詳細

4-down U-NetのKP14 heatmap + 2ch center-vote CNNを用い、σ=2.0、2048/256/256 samples、batch 16、128 steps/epoch、30 epochs、LR 3e-4、warmup 128 steps、seed 42で学習した。初期値は`run-court-align-kp14-ablation-sigma-200`のepoch 47 checkpointで、座標正規化メタデータを検証した上でmodel weightsだけをstrict loadした。`b00_appearance_v1`はscaleを3--8 px/mへ広げ、line morphology（主にdilation）、Gaussian blur、foreground amplitude/gamma、speckleを加えるが、line dropoutとghost lineは加えない。

### メトリクスの解釈

bundle `metrics.json` のsynthetic testはF1=0.9974093264248705、KP誤差=1.5569069280147785 px、center誤差=0.48443875948039744 px、count accuracy=0.9921875である。`diagnostic_metrics.json`ではTP=385、FP=2、FN=0、visible KP coverage=0.996352140077821、loss=0.02674102783203125だった。augmentationを加えてもinstance検出は崩壊していない。

### アーキテクチャ⇄メトリクスの因果考察

観測として、scratchだったround 1 appearanceのsynthetic F1=0.827103から大幅に回復した。仮説として、clean σ=2.0表現をwarm-startで保持し、低いLRと2倍のsample/step予算でappearance差だけを段階的に学習したため、domain robustnessとcourt geometryの両方を保てた。

### 既存実験との比較

親のclean σ=2.0はF1=0.9973684210526316、KP誤差=0.8209754559458519 pxであり、本runはF1を維持した一方、augmentationによりKP誤差は1.5569069280147785 pxへ増えた。round 1 appearanceはB00 F1=0.5まで改善したがsynthetic性能が低く、round 1 combinedはB00 F1=0だった。本runは後続B00評価でもF1=0.5となる。

### 次に有効な実験

appearanceを基盤に、line dropoutとghost lineを弱い確率で追加して2面目のsemantic KPを補強する。decoder容量は学習変更とは分離して評価する。
