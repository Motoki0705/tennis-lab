---
id: run-i634-slcs-overfit-dino-down2-shared
type: run
title: SLCS単一clip過学習（DINO 2×2圧縮・共有trunk）
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
  dino_patch_grid: 8x14
  dino_patch_downsample_factor: 2
  shared_layers: 2
  position_layers: 0
  rotation_layers: 0
  batch_size: 1
  epochs: 100
  learning_rate: 0.0003
metrics:
  player_position_error_m: 0.46716
  player_position_error_median_m: 0.385692
  player_angular_error_deg: 8.54665
  player_angular_error_median_deg: 4.316201
  player_position_accuracy_0.3m: 0.355863
  player_position_accuracy_0.5m: 0.645358
  player_position_accuracy_1.0m: 0.946661
  player_position_accuracy_2.0m: 0.995114
  player_angle_accuracy_10deg: 0.770358
  player_angle_accuracy_15deg: 0.855456
  player_angle_accuracy_30deg: 0.947068
  player_position_pred_b_m: 0.508337
  player_rotation_pred_b_deg: 12.746138
  player_position_conf_error_corr: 0.462213
  player_rotation_conf_error_corr: 0.658682
  ball_position_error_m: 1.745198
  ball_position_error_median_m: 1.474864
  ball_position_accuracy_0.3m: 0.029594
  ball_position_accuracy_0.5m: 0.102546
  ball_position_accuracy_1.0m: 0.333104
  ball_position_accuracy_2.0m: 0.664143
  ball_position_pred_b_m: 1.360781
  ball_position_conf_error_corr: 0.478117
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
    training.checkpoint.save_top_k=1 data.require_dino=true model.dino_patch_downsample_factor=2
    run.output_dir=outputs/slcs/i634_overfit_dino_down2_shared
artifacts:
  run_dir: knowledge/runs/run-i634-slcs-overfit-dino-down2-shared
  predictions: knowledge/runs/run-i634-slcs-overfit-dino-down2-shared/pred_test.npz
  log: .training_queue/logs/1783905015753095149_1219154_i634_slcs_overfit_dino_down2_shared.log
  output_dir: outputs/slcs/i634_overfit_dino_down2_shared/logs/version_0
  curves: knowledge/runs/run-i634-slcs-overfit-dino-down2-shared/curves.png
  tb_logdir: outputs/slcs/i634_overfit_dino_down2_shared/logs/version_0
parents:
- run-i634-slcs-overfit-dino
relations:
- to: run-i634-slcs-overfit-dino
  rel: compares
tags:
- slcs
- overfit
- single-clip
- dino
- patch-compression
- shared-trunk
---

## 考察 / Findings

### 要約

DINO patch gridを16×28（448 token）から8×14（112 token）へbilinear downsampleしてから次元圧縮した。DINO baseline比でplayer位置は0.470→0.467m、ball位置は1.954→1.745mへ改善し、yawは7.762→8.547°へ悪化した。token数を1/4にしても主要なvisual情報は保たれた。

### アーキテクチャ詳細

共有trunk 2層の従来small構成とDINO ViT-B/16の事前計算tokenを使用した。唯一の差分は `dino_patch_downsample_factor=2` であり、元の768次元特徴gridへbilinear補間を適用した後に64次元へ射影する。train/val/testは同一13 window、100 epochs、seed 42である。

### メトリクスの解釈

baseline比でplayer位置は0.7%、ball位置は10.7%改善し、yawは10.1%悪化した。ball 2m以内は61.1%→66.4%へ増加した。同一windowへの過学習評価であり、汎化性能ではない。
収束曲線ではvalidation lossが終盤まで単調に低下し、ball 0.5/1.0m accuracyも概ね上昇しており、崩壊は見られない。

### アーキテクチャ⇄メトリクスの因果考察

局所4 patchの補間集約が微小なballに対するノイズ低減として働いた可能性がある（仮説）。一方、playerの細かな向きに必要な局所appearanceが平均化され、yawが悪化した可能性がある（仮説）。1/4 tokenでもposition系が維持・改善したことから、共有trunkのcross-attentionには空間冗長性があったと解釈できる。

### 既存実験との比較

親run `run-i634-slcs-overfit-dino` の448 token以外は同条件である。player位置0.467mは本アブレーション中の最良値、ball位置1.745mもbaselineを上回ったが、yaw最良ではない。

### 次に有効な実験

複数recordingの非重複splitで圧縮のball改善が再現するか確認する。yawについては一律2×2でなく、player周辺patchを高解像度で残すadaptive poolingが候補となる。
