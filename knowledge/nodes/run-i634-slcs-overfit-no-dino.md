---
id: run-i634-slcs-overfit-no-dino
type: run
title: SLCS単一clip過学習（DINOなし）
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
  require_dino: false
  batch_size: 1
  epochs: 100
  learning_rate: 0.0003
metrics:
  player_position_error_m: 0.484077
  player_position_error_median_m: 0.395474
  player_angular_error_deg: 9.848976
  player_angular_error_median_deg: 5.326686
  player_position_accuracy_0.3m: 0.350977
  player_position_accuracy_0.5m: 0.6193
  player_position_accuracy_1.0m: 0.942182
  player_position_accuracy_2.0m: 0.995114
  player_angle_accuracy_10deg: 0.734935
  player_angle_accuracy_15deg: 0.831026
  player_angle_accuracy_30deg: 0.942997
  player_position_pred_b_m: 0.523439
  player_rotation_pred_b_deg: 13.857314
  player_position_conf_error_corr: 0.478934
  player_rotation_conf_error_corr: 0.466868
  ball_position_error_m: 2.105404
  ball_position_error_median_m: 1.725266
  ball_position_accuracy_0.3m: 0.017894
  ball_position_accuracy_0.5m: 0.072953
  ball_position_accuracy_1.0m: 0.26084
  ball_position_accuracy_2.0m: 0.562973
  ball_position_pred_b_m: 1.493816
  ball_position_conf_error_corr: 0.513161
repro:
  commit: 8ed508912eff3e6e9f532d60d8e2e4ee01e328b8
  branch: feat/issue-634-slcs-overfit
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.tasks.slcs.scripts.train
    model=small data.dataset_root=data/tennis_multivew/processed/tennis_clip/dataset
    data.split_file=data/tennis_multivew/processed/tennis_clip/dataset/splits.json
    data.overfit=true data.require_dino=false data.batch_size=1 data.num_workers=2
    run.output_dir=outputs/slcs/i634_overfit_no_dino training.trainer.max_epochs=100
    training.trainer.check_val_every_n_epoch=10 training.trainer.log_every_n_steps=13
    training.early_stopping.enabled=false training.learning_rate=3e-4 training.warmup_steps=20
    training.checkpoint.save_top_k=1
artifacts:
  run_dir: knowledge/runs/run-i634-slcs-overfit-no-dino
  predictions: knowledge/runs/run-i634-slcs-overfit-no-dino/pred_test.npz
  log: .training_queue/logs/1783894087426433674_1161831_i634_slcs_overfit_no_dino.log
  output_dir: outputs/slcs/i634_overfit_no_dino/logs/version_0
  curves: knowledge/runs/run-i634-slcs-overfit-no-dino/curves.png
  tb_logdir: outputs/slcs/i634_overfit_no_dino/logs/version_0
parents: []
relations: []
tags:
- slcs
- overfit
- single-clip
- no-dino
---

## 考察 / Findings

### 要約

797フレームの単一clipから作った13 windowをtrain/val/testで共有し、DINOなしのSLCS smallを100 epochs学習した。testのplayer位置誤差は0.484 m、yaw誤差は9.849°まで低下した一方、ball位置誤差は2.105 mに留まった。

### アーキテクチャ詳細

入力はplayer 2D pose、ball UV、court keypointsで、`data.require_dino=false` によりvisual streamを無効化した。390K parameterのsmall model、batch size 1、13 steps/epoch、LR 3e-4、warmup 20 steps、early stoppingなしで明示的に同一windowを記憶させた。

### メトリクスの解釈

train/val/testが同一データのため、数値は汎化性能ではなくモデルと学習経路のmemorization能力を示す。playerは94.2%が1 m以内、yawは73.5%が10°以内に入った。ballは1 m以内が26.1%で、playerより記憶が難しい。

### アーキテクチャ⇄メトリクスの因果考察

player poseとcourt keypointsはplayer 3D位置・yawに直接対応する強い観測であり、visual streamなしでも同一clipを記憶できた。ballはvisibility欠損があり、jerk smoothnessとconfidence weightingを含む目的関数がframe単位の完全記憶より滑らかな近似を優先した可能性がある（仮説）。

### 既存実験との比較

SLCSの登録済み実runが存在しなかったため、本runをIssue #634のbaselineとする。

### 次に有効な実験

DINOありrunを同一seed・同一window・同一optimizer条件で比較し、visual tokenだけの寄与を測る。ballの完全記憶を検証する場合は、smoothness lossを無効化した独立ablationが必要。
