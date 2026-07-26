---
id: run-i618-3dgs-blcs-half-rate-v1-control-s1931
type: run
title: 3DGS×BLCS 1/12混合用 real-only control seed 1931
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
  seed: 1931
  learning_rate: 1.0e-05
metrics:
  best_validation_epoch: 6
  best_val_f1: 0.6697261929512024
  best_val_precision: 0.6337226629257202
  best_val_recall: 0.710067093372345
  best_val_mean_distance_px: 2.318206787109375
  best_val_loss: 0.00040415258263237774
  final_val_f1: 0.6524980068206787
  final_val_precision: 0.6186466217041016
  final_val_recall: 0.6902684569358826
  final_val_mean_distance_px: 2.3339955806732178
repro:
  commit: ac9e640903a6dfaecb65fc980f5dcf408bbcd589
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: .venv/bin/python -m src.tasks.ball_detection.scripts.train --config-name
    train_3dgs_blcs_v1 data.synthetic_batch_period=2 data.synthetic_per_batch=0 run.seed=1931
    run.output_dir=outputs/ball_detection/3dgs_blcs_half_rate_v1/control/seed_1931
artifacts:
  run_dir: knowledge/runs/run-i618-3dgs-blcs-half-rate-v1-control-s1931
  log: .training_queue/logs/1785020735325179432_3033838_i618_3dgs_blcs_half_rate_v1_control_s1931.log
  output_dir: outputs/ball_detection/3dgs_blcs_half_rate_v1/control/seed_1931/logs/version_0
  checkpoint: outputs/ball_detection/3dgs_blcs_half_rate_v1/control/seed_1931/logs/version_0/checkpoints/3dgs-blcs-epoch=06.ckpt
  checkpoint_sha256: d6a2bf6419f27ae4a4eb1f315e38e3a65139322da30c48a055e8103055a1d079
  live_monitor: .codex-loop/C12_LIVE_MONITOR.json
  curves: knowledge/runs/run-i618-3dgs-blcs-half-rate-v1-control-s1931/curves.png
  tb_logdir: outputs/ball_detection/3dgs_blcs_half_rate_v1/control/seed_1931/logs/version_0
parents:
- run-i618-3dgs-blcs-half-rate-v1-control-s731
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
- seed-1931
- validation-only
---

## 考察 / Findings

### 要約

C12の1/12 synthetic treatmentに対する2番目のreal-only controlをseed 1931で
完走した。game-9 best validation F1はepoch 6の**0.669726**であり、同じ
protocolのseed 731 control（0.672900）との差は`-0.003174`だった。

### アーキテクチャ詳細

ConvNeXt U-Netを指定checkpointから242/242 tensor完全一致で初期化し、
TrackNet games 1--8のみをbatch 6、655 step/epoch、8 epoch、AdamW、
lr `1e-5`で学習した。`synthetic_per_batch=0`のため
`synthetic_batch_period=2`はsamplingへ影響しない。
`run.test_after_fit=false`によりgame 10は実行していない。

### メトリクスの解釈

best epoch 6はprecision 0.633723、recall 0.710067、F1 0.669726、
平均距離2.318207 px、loss 0.000404153だった。epoch 0--7のF1は
0.583147、0.647928、0.606695、0.654419、0.642574、0.636536、
0.669726、0.652498で、epoch 6がtop-1として保持された。

### アーキテクチャ⇄メトリクスの因果考察

同じ初期化・実データ・optimizer・budgetでもseedにより初期epochの
validation軌跡とbest epochが変動した。これは仮説として、データ順と
augmentation乱数によるrun間分散を示す。したがってseed 731単独の
treatment改善を再現性ありと判断せず、同一seedのpaired treatmentとの差を
評価する必要がある。

### 既存実験との比較

親の`run-i618-3dgs-blcs-half-rate-v1-control-s731`に対してbest F1は
0.672900から0.669726へ0.003174低下し、平均距離は2.283164 pxから
2.318207 pxへ0.035043 px増加した。C02配備モデル値とは
inference/aggregation protocolが異なるため、直接の改善主張には使わない。

### 次に有効な実験

既に直列実行中のseed 1931 C12 1/12 treatmentを完了させ、game-9 best F1を
このcontrolの0.669726とpaired比較する。その後もfrozen queue順を維持して
seed 3253 pairを完了し、3 seedの符号とaggregate paired deltaを判定する。
