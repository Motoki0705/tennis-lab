---
id: run-i618-3dgs-blcs-v1-control-s731
type: run
title: 3DGS×BLCS paired control seed 731
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
repro:
  commit: ac9e640903a6dfaecb65fc980f5dcf408bbcd589
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: .venv/bin/python -m src.tasks.ball_detection.scripts.train --config-name
    train_3dgs_blcs_v1 data.synthetic_per_batch=0 run.output_dir=outputs/ball_detection/3dgs_blcs_paired_v1/control/seed_731
artifacts:
  run_dir: knowledge/runs/run-i618-3dgs-blcs-v1-control-s731
  log: .training_queue/logs/1784996113568780634_2936440_i618_3dgs_blcs_v1_control_s731.log
  output_dir: outputs/ball_detection/3dgs_blcs_paired_v1/control/seed_731/logs/version_0
  checkpoint: outputs/ball_detection/3dgs_blcs_paired_v1/control/seed_731/logs/version_0/checkpoints/3dgs-blcs-epoch=01.ckpt
  checkpoint_sha256: 67746c160e1bc3de998c9b75129eb791f74ee9057a13cabe70805bc309736c1f
  live_monitor: .codex-loop/C11_LIVE_MONITOR.json
  curves: knowledge/runs/run-i618-3dgs-blcs-v1-control-s731/curves.png
  tb_logdir: outputs/ball_detection/3dgs_blcs_paired_v1/control/seed_731/logs/version_0
parents:
- run-i618-convnext-v2-ft
- run-i618-3dgs-blcs-real-baseline-v1
relations:
- to: run-i618-blcs-b00-full-scale-v1
  rel: controls_for
tags:
- ball_detection
- 3dgs-blcs
- paired-control
- real-only
- seed-731
---

## 考察 / Findings

### 要約

配備済み ConvNeXt U-Net checkpoint を初期値として、TrackNet games 1--8
だけを8 epoch再学習したseed 731 controlは、game 9 validationのbest F1
**0.672900**をepoch 1で得た。宣言済みtop-1 checkpointを保存し、game 10
final testは実行も参照もしていない。

### アーキテクチャ詳細

親runの`conv_next_unet`（MDD 2ch、8 frame、72x128 heatmap）を変更せず、
AdamW、batch 6、655 step/epoch、lr `1e-5`、warmup 200、同一augmentationで
8 epoch学習した。初期checkpoint SHA-256は
`cd7927ad27e53ddd6aa77df28eca3c5e674552461ccda083a41e99e629857892`。
本armは全batchをreal 6 windowとし、paired treatmentだけがreal 5 +
C10 synthetic 1になる。`run.test_after_fit=false`によりgame 10を隔離した。

### メトリクスの解釈

validation F1のepoch 0--7軌跡は0.655832、0.672900、0.627012、
0.608530、0.639120、0.625119、0.659005、0.647821だった。best epoch 1は
precision 0.637538、recall 0.712416、平均距離2.283164 px、loss
0.000407368。最終epochはF1 0.647821で、epoch間変動が大きいため、
事前宣言どおりval/F1 top-1を比較checkpointとする。

### アーキテクチャ⇄メトリクスの因果考察

このrun単独はsynthetic効果を示さず、paired treatmentの反実仮想を固定する
controlである。C02の配備モデルgame-9 F1 0.712628は別の固定manifest評価
protocolであり、本runのtraining-loop集計と直接比較して改善を主張しない。
同一training loop、seed、予算でtreatmentとの差だけを解釈する。

### 既存実験との比較

`run-i618-convnext-v2-ft`のbyte-identical checkpointから開始し、
`run-i618-3dgs-blcs-real-baseline-v1`と同じvalidation sourceを使うが、
本runは再学習後のqualification controlである。C10 full-scale synthetic
datasetは本armでは一切sampleせず、relationだけで後続treatmentとの対応を
明示する。

### 次に有効な実験

同じseed 731、8 epoch、655 step/epochで実行中のreal 5 + synthetic 1
treatmentを完走し、game-9 best validation F1が本controlの0.672900を
上回るかだけで1/6 mixingを次seedへ進めるか判断する。
