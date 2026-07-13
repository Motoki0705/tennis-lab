---
id: run-i634-slcs-overfit-split-dino
type: run
title: SLCS単一clip過学習（完全分離trunk・DINOあり）
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
  dino_patch_grid: 16x28
  dino_patch_downsample_factor: 1
  shared_layers: 0
  position_layers: 2
  rotation_layers: 2
  batch_size: 1
  epochs: 100
  learning_rate: 0.0003
metrics:
  player_position_error_m: 0.512462
  player_position_error_median_m: 0.405513
  player_angular_error_deg: 5.751387
  player_angular_error_median_deg: 3.030268
  player_position_accuracy_0.3m: 0.341205
  player_position_accuracy_0.5m: 0.605863
  player_position_accuracy_1.0m: 0.918567
  player_position_accuracy_2.0m: 0.995114
  player_angle_accuracy_10deg: 0.855863
  player_angle_accuracy_15deg: 0.918567
  player_angle_accuracy_30deg: 0.976384
  player_position_pred_b_m: 0.524816
  player_rotation_pred_b_deg: 10.504
  player_position_conf_error_corr: 0.630997
  player_rotation_conf_error_corr: 0.431764
  ball_position_error_m: 1.615541
  ball_position_error_median_m: 1.216975
  ball_position_accuracy_0.3m: 0.0468
  ball_position_accuracy_0.5m: 0.138334
  ball_position_accuracy_1.0m: 0.38541
  ball_position_accuracy_2.0m: 0.703372
  ball_position_pred_b_m: 1.249097
  ball_position_conf_error_corr: 0.449122
repro:
  commit: 56445ad063a9a87e6453f4d095c084f8efb0b532
  branch: feat/issue-634-dino-compress-split-trunks
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m src.tasks.slcs.scripts.train
    model=small data.dataset_root=/home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/tennis_clip/dataset
    data.split_file=/home/kamimura/projects/tennis-lab/data/tennis_multivew/processed/tennis_clip/dataset/splits.json
    data.overfit=true data.batch_size=1 data.num_workers=2 training.trainer.max_epochs=100
    training.trainer.check_val_every_n_epoch=10 training.trainer.log_every_n_steps=13
    training.early_stopping.enabled=false training.learning_rate=3e-4 training.warmup_steps=20
    training.checkpoint.save_top_k=1 data.require_dino=true model.num_shared_layers=0
    model.num_position_layers=2 model.num_rotation_layers=2 run.output_dir=outputs/slcs/i634_overfit_split_dino
artifacts:
  run_dir: knowledge/runs/run-i634-slcs-overfit-split-dino
  predictions: knowledge/runs/run-i634-slcs-overfit-split-dino/pred_test.npz
  log: .training_queue/logs/1783905015875114222_1219210_i634_slcs_overfit_split_dino.log
  output_dir: outputs/slcs/i634_overfit_split_dino/logs/version_0
  curves: knowledge/runs/run-i634-slcs-overfit-split-dino/curves.png
  tb_logdir: outputs/slcs/i634_overfit_split_dino/logs/version_0
parents:
- run-i634-slcs-overfit-dino
- run-i634-slcs-overfit-split-no-dino
relations:
- to: run-i634-slcs-overfit-dino
  rel: compares
- to: run-i634-slcs-overfit-split-no-dino
  rel: compares
tags:
- slcs
- overfit
- single-clip
- dino
- split-trunk
---

## 考察 / Findings

### 要約

完全分離trunkへ非圧縮DINO tokenを入力した。共有/DINO baseline比でyawは7.762→5.751°、ball位置は1.954→1.616mへ大幅改善し、今回のyaw/ball最良runとなった。一方player位置は0.470→0.512mへ悪化した。

### アーキテクチャ詳細

共有0層、position/rotation各2層で、各task branchが448 DINO patchへ独立にcross-attentionする。position branchはplayer/ball位置、rotation branchはyawとそれぞれのuncertaintyを出力する。train/val/testは同一13 window、100 epochs、seed 42である。

### メトリクスの解釈

DINO baseline比でyawは25.9%、ball位置は17.3%改善した。yaw 10°以内は79.7%→85.6%、ball 2m以内は61.1%→70.3%となった。player位置は9.0%悪化し、0.5m以内も62.9%→60.6%へ低下した。
収束曲線ではvalidation lossが滑らかに低下し、ball 1m accuracyは終盤に約0.39へ到達した後ほぼ頭打ちとなった。

### アーキテクチャ⇄メトリクスの因果考察

rotation branchが位置損失から隔離された状態で高解像度appearanceを専有でき、yawの負の転移が軽減された可能性が高い。ballもrotation勾配の影響を受けないposition branchで改善した。一方player位置悪化は、完全分離が共有表現の正の転移まで除いた可能性を示す（仮説）。また711K対390Kのparameter差は改善側の交絡要因である。

### 既存実験との比較

親 `run-i634-slcs-overfit-dino` よりyaw/ballが改善し、`run-i634-slcs-overfit-split-no-dino` よりもyaw 8.319→5.751°、ball 1.827→1.616m、player位置0.543→0.512mと全3指標が改善した。完全分離でもDINOの寄与は残る。

### 次に有効な実験

shared 1層 + task 1層でplayer位置の正の転移を回復できるか確認する。さらにtask branch幅を調整したparameter-matched共有trunk対照で、分離効果と容量効果を切り分ける。
