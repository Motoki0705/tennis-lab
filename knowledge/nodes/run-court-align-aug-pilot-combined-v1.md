---
id: run-court-align-aug-pilot-combined-v1
type: run
title: court-align-aug-pilot-combined-v1
provider: codex
session: 01a05a02-ad55-76c0-9fcb-f250929e59a7
date: '2026-09-02'
status: done
config:
  data: b00_v1
metrics:
  instance_precision: 0.589641
  instance_recall: 0.768831
  instance_f1: 0.667418
  instance_count_accuracy: 0.542969
  instance_count_mae: 0.457031
  matched_center_mean_error_px: 2.987897
  instance_kp_mean_error_px: 143.126471
  instance_kp_pck_2px: 0.602383
  instance_kp_pck_4px: 0.602383
  sim2_translation_error_px: 83.511046
  sim2_rotation_error_deg: 41.707057
  sim2_scale_relative_error: 0.232687
repro:
  commit: 5de3a7d45e037a799d4dad0ae1ef3ac3cb24897e
  branch: feat/court-alignment-kp14-pipeline
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HYDRA_FULL_ERROR=1 /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.court_alignment.scripts.train data=b00_v1 data.train_samples=1024
    data.val_samples=256 data.test_samples=256 data.batch_size=16 training.steps_per_epoch=64
    training.trainer.max_epochs=20 training.learning_rate=1.0e-3 training.warmup_steps=256
    run.seed=42 run.output_dir=court_alignment/augmentation_pilot/b00_combined_v1_warm20_s42
artifacts:
  run_dir: knowledge/runs/run-court-align-aug-pilot-combined-v1
  predictions: knowledge/runs/run-court-align-aug-pilot-combined-v1/pred_test.npz
  output_dir: outputs/court_alignment/augmentation_pilot/b00_combined_v1_warm20_s42/logs/version_0
  curves: knowledge/runs/run-court-align-aug-pilot-combined-v1/curves.png
  tb_logdir: outputs/court_alignment/augmentation_pilot/b00_combined_v1_warm20_s42/logs/version_0
parents:
- run-court-align-kp14-ablation-sigma-200
relations:
- to: group-court-align-kp14-sigma-ablation
  rel: compares
tags:
- court-alignment
- kp14
- augmentation
- b00
---

## 考察 / Findings

### 要約

scale・appearance・structureを同時に適用した `b00_v1`。B00 F1=0で、augmentationを単純に重ねると分布が広がりすぎ、今回の20 epoch予算では失敗した。

### アーキテクチャ詳細

4-down U-NetのKP14 heatmap + 2ch center-vote CNN。1024/256/256 samples、batch 16、64 steps/epoch、20 epochs、σ=2.0。既存checkpointを使わずscratch学習。

### メトリクスの解釈

synthetic testはF1=0.6674、KP誤差=143.13px、center誤差=2.99px、count accuracy=0.5430、Sim(2)回転誤差=41.71°、scale相対誤差=0.2327、test loss=0.2269。B00はTP=0/FP=2/FN=2、F1=0で、2 instancesのcountだけが一致した。

### アーキテクチャ⇄メトリクスの因果考察

各augmentation単独よりsynthetic性能が大幅に悪化しており、同時適用による難易度過多が観測された。仮説として、scale OOD・線欠損・false line・振幅変動の組合せが、KP heatmapのinstance分離を壊している。

### 既存実験との比較

identity σ=2.0はsynthetic F1=0.9974、B00 F1=0。combinedはB00で改善せず、appearance単独のF1=0.5を下回った。

### 次に有効な実験

combinedを採用せず、appearanceを基軸にstructureを弱く追加する段階的なcurriculumまたはaugmentation確率の縮小を試す。
