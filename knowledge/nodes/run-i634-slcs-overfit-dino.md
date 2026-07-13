---
id: run-i634-slcs-overfit-dino
type: run
title: SLCS単一clip過学習（DINOあり）
issue: 634
provider: codex
session: 019f55e6-8819-7e63-8481-72f9effc4079
date: '2026-07-13'
status: done
config:
  model: small
  dataset: tennis_clip/clip_000
  windows: 13
  overfit: true
  require_dino: true
  dino_backbone: dinov3_vitb16_pretrain_lvd1689m
  dino_frame_stride: 10
  batch_size: 1
  epochs: 100
  learning_rate: 0.0003
metrics:
  player_position_error_m: 0.47036
  player_position_error_median_m: 0.375771
  player_angular_error_deg: 7.76192
  player_angular_error_median_deg: 4.076817
  player_position_accuracy_0.3m: 0.388436
  player_position_accuracy_0.5m: 0.628664
  player_position_accuracy_1.0m: 0.929153
  player_position_accuracy_2.0m: 0.995114
  player_angle_accuracy_10deg: 0.796824
  player_angle_accuracy_15deg: 0.879072
  player_angle_accuracy_30deg: 0.949919
  player_position_pred_b_m: 0.511841
  player_rotation_pred_b_deg: 11.863464
  player_position_conf_error_corr: 0.509571
  player_rotation_conf_error_corr: 0.669746
  ball_position_error_m: 1.954024
  ball_position_error_median_m: 1.611596
  ball_position_accuracy_0.3m: 0.021335
  ball_position_accuracy_0.5m: 0.082588
  ball_position_accuracy_1.0m: 0.283551
  ball_position_accuracy_2.0m: 0.611149
  ball_position_pred_b_m: 1.461908
  ball_position_conf_error_corr: 0.525916
repro:
  commit: 8ed508912eff3e6e9f532d60d8e2e4ee01e328b8
  branch: feat/issue-634-slcs-overfit
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.tasks.slcs.scripts.train
    model=small data.dataset_root=data/tennis_multivew/processed/tennis_clip/dataset
    data.split_file=data/tennis_multivew/processed/tennis_clip/dataset/splits.json
    data.overfit=true data.require_dino=true data.batch_size=1 data.num_workers=2
    run.output_dir=outputs/slcs/i634_overfit_dino training.trainer.max_epochs=100
    training.trainer.check_val_every_n_epoch=10 training.trainer.log_every_n_steps=13
    training.early_stopping.enabled=false training.learning_rate=3e-4 training.warmup_steps=20
    training.checkpoint.save_top_k=1
artifacts:
  run_dir: knowledge/runs/run-i634-slcs-overfit-dino
  predictions: knowledge/runs/run-i634-slcs-overfit-dino/pred_test.npz
  log: .training_queue/logs/1783894093467941821_1161920_i634_slcs_overfit_dino.log
  output_dir: outputs/slcs/i634_overfit_dino/logs/version_0
  curves: knowledge/runs/run-i634-slcs-overfit-dino/curves.png
  tb_logdir: outputs/slcs/i634_overfit_dino/logs/version_0
parents:
- run-i634-slcs-overfit-no-dino
relations:
- to: run-i634-slcs-overfit-no-dino
  rel: compares
tags:
- slcs
- overfit
- single-clip
- dino
---

## 考察 / Findings

### 要約

公式事前学習DINOv3 ViT-B/16のsparse patch tokenを10フレーム間隔で加え、DINOなしbaselineと同じ13 windowを100 epochs学習した。testはplayer位置0.470 m、yaw 7.762°、ball位置1.954 mとなり、3指標すべてでDINOなしを上回った。

### アーキテクチャ詳細

DINOなしrunと同じ390K parameterのSLCS small、batch size 1、LR 3e-4、warmup 20 stepsを使用した。差分は `data.require_dino=true` のみで、256x448入力から得た448 patch tokenを10フレーム間隔でcross-attentionした。DINO backbone自体は学習せず、事前計算tokenを用いた。

### メトリクスの解釈

同一データ評価でDINOなしに対し、player位置誤差は0.484→0.470 m（2.8%改善）、yawは9.849→7.762°（21.2%改善）、ball位置は2.105→1.954 m（7.2%改善）した。player 0.5 m以内は62.9%、ball 2 m以内は61.1%だった。汎化性能を示す値ではない。

### アーキテクチャ⇄メトリクスの因果考察

同じ2D観測だけでは曖昧な選手の向きとballのdepthに対し、画像patchのappearance/contextが追加情報になった可能性がある（仮説）。特にyawの改善が大きく、visual tokenが人物の向き推定に寄与した可能性を示す。一方、ball誤差は約2 mであり、sparse DINO tokenだけでは微小ballのframe単位3D軌道を完全には記憶できていない。

### 既存実験との比較

親run `run-i634-slcs-overfit-no-dino` と比較して主要3指標を改善した。同一seed・同一model・同一window・同一optimizerであるため、この比較内ではDINO tokenの有無が主要差分である。

### 次に有効な実験

複数recordingを収集してrecording単位の非重複splitでDINOの改善が再現するか検証する。ballについてはDINO sampling stride、ball-aware visual token、smoothness loss weightのablationを優先する。
