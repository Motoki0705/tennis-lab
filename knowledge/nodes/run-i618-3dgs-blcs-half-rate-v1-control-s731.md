---
id: run-i618-3dgs-blcs-half-rate-v1-control-s731
type: run
title: 3DGS×BLCS 1/12混合用 real-only control seed 731
issue: 618
provider: codex
session: 019f984c-8bc1-7041-8e1d-362a5b11daa2
date: '2026-07-26'
status: done
config:
  model: conv_next_unet
  initialization: ckpt/ball_detection/run-i618-convnext-v2-ft-epoch13.ckpt
  initialization_sha256: cd7927ad27e53ddd6aa77df28eca3c5e674552461ccda083a41e99e629857892
  loss: focal_bce_gamma_2
  data: TrackNet games 1-8 real-only, game 9 validation
  batch_size: 6
  synthetic_per_batch: 0
  synthetic_batch_period: 2
  steps_per_epoch: 655
  epochs: 8
  seed: 731
  learning_rate: 1.0e-05
metrics:
  best_validation_epoch: 1
  best_val_f1: 0.6729001402854919
  best_val_precision: 0.6375375390052795
  best_val_recall: 0.7124161124229431
  best_val_mean_distance_px: 2.283163547515869
  best_val_loss: 0.00040736846858635545
  final_val_f1: 0.6478205919265747
  final_val_precision: 0.6118735074996948
  final_val_recall: 0.6882550120353699
  final_val_mean_distance_px: 2.3588125705718994
  exact_match_to_c11_control: true
repro:
  commit: ac9e640903a6dfaecb65fc980f5dcf408bbcd589
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: .venv/bin/python -m src.tasks.ball_detection.scripts.train --config-name
    train_3dgs_blcs_v1 data.synthetic_batch_period=2 data.synthetic_per_batch=0 run.output_dir=outputs/ball_detection/3dgs_blcs_half_rate_v1/control/seed_731
artifacts:
  run_dir: knowledge/runs/run-i618-3dgs-blcs-half-rate-v1-control-s731
  log: .training_queue/logs/1785010253167130980_2991586_i618_3dgs_blcs_half_rate_v1_control_s731.log
  output_dir: outputs/ball_detection/3dgs_blcs_half_rate_v1/control/seed_731/logs/version_0
  checkpoint: outputs/ball_detection/3dgs_blcs_half_rate_v1/control/seed_731/logs/version_0/checkpoints/3dgs-blcs-epoch=01.ckpt
  checkpoint_sha256: f91e2ac4b0390ef65e1ab4788bc74eaab9e44ea10b6e655b3ebe34303cd3c254
  live_monitor: .codex-loop/C12_LIVE_MONITOR.json
  curves: knowledge/runs/run-i618-3dgs-blcs-half-rate-v1-control-s731/curves.png
  tb_logdir: outputs/ball_detection/3dgs_blcs_half_rate_v1/control/seed_731/logs/version_0
parents:
- run-i618-3dgs-blcs-v1-control-s731
- run-i618-blcs-b00-full-scale-v1
relations:
- to: run-i618-3dgs-blcs-real-baseline-v1
  rel: compares
tags:
- ball_detection
- 3dgs-blcs
- paired-control
- real-only
- synthetic-half-rate
- seed-731
- validation-only
---

## 考察 / Findings

### 要約

C12の1/12 synthetic treatmentと同じ更新後コードでreal-only controlを
再実行した。game-9 best validation F1はepoch 1の**0.672900**で、C11 controlと
全8 epochのvalidation scalarが完全一致した。これによりperiod追加が
synthetic無効時のcontrol挙動を変えていないことを確認した。

### アーキテクチャ詳細

ConvNeXt U-Netを指定checkpointから242/242 tensor完全一致で初期化し、
TrackNet games 1--8のみをbatch 6、655 step/epoch、8 epoch、AdamW、
lr `1e-5`で学習した。`synthetic_per_batch=0`のため全31,440 windowがrealで、
`synthetic_batch_period=2`は構成上存在するがsamplingへ影響しない。
`run.test_after_fit=false`によりgame 10は実行していない。

### メトリクスの解釈

best epoch 1はprecision 0.637538、recall 0.712416、F1 0.672900、
平均距離2.283164 px、loss 0.000407368だった。epoch 0--7のF1は
0.655832、0.672900、0.627012、0.608530、0.639120、0.625119、
0.659005、0.647821で、epoch 1以後はbestを更新しなかった。

### アーキテクチャ⇄メトリクスの因果考察

C11 controlとの完全一致は、seed、データ順、augmentation、optimizer、
budgetが再現され、period条件がsynthetic count 0で挙動を変えないという
実装意図を支持する。これはC12 treatment差を1/12 synthetic置換へ限定する
ためのnegative controlであり、synthetic効果そのものはまだ判断しない。

### 既存実験との比較

親の`run-i618-3dgs-blcs-v1-control-s731`とbest値だけでなく全epochの
loss/precision/recall/F1/mean distanceが一致した。C02配備モデル値とは
inference/aggregation protocolが異なるため直接比較による改善主張はしない。

### 次に有効な実験

既に直列実行中の同一seed・同一budgetのC12 1/12 treatmentを完了させ、
game-9 best F1がこのcontrolの0.672900を超えるかだけでfrozen gateを判定する。
