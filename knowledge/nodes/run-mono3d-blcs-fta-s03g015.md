---
id: run-mono3d-blcs-fta-s03g015
type: run
title: mono3d_blcs_ftA_s03g015
issue: 593
provider: claude
session: dd9d56e0-5f2d-4d9d-a8c7-c80704a806b3
date: '2026-07-06'
status: done
config:
  model: multiview_axial_base
  loss: trajectory_position + reprojection(0.1) + axis_weights[1,4,1] + smoothness(0.3)
    + gravity(0.15)
  data: chunked_multiview_sequence C=1
metrics:
  mean_position_error_m: 3.012269
  mean_x_error_m: 0.872541
  mean_y_error_m: 2.574402
  mean_z_error_m: 0.633132
  mean_endpoint_error_m: 4.604888
  position_accuracy_0_3m: 0.008894
  position_accuracy_0_6m: 0.052284
  position_accuracy_1_2m: 0.226789
  endpoint_accuracy_0_5m: 0.01
  endpoint_accuracy_1m: 0.03
repro:
  commit: d36b7dc50522453fa13778e7d31756a733fa0403
  branch: feat/issue-593-physics-prior
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.blcs.scripts.train --config-name train_chunked data.scene_dir=data/blcs_broadcast
    data.chunk.chunks_dir=data/blcs_broadcast/chunks data.num_views_range=[1,1] data.camera_mode=random
    data.num_court_kp=14 data.num_workers=2 data.chunk.generation_workers=8 data.chunk.epochs_per_chunk=20
    camera=broadcast training.position_axis_weights=[1.0,4.0,1.0] training.learning_rate=3e-5
    training.trainer.max_epochs=60 training.trainer.check_val_every_n_epoch=5 training.qualitative_logging.enabled=false
    training.early_stopping.enabled=false run.gpus=1 training.smoothness_loss_weight=0.3
    training.gravity_loss_weight=0.15 run.init_weights=outputs/blcs/blcs_multiview_axial/logs/version_2/checkpoints/last.ckpt
artifacts:
  run_dir: knowledge/runs/run-mono3d-blcs-fta-s03g015
  predictions: knowledge/runs/run-mono3d-blcs-fta-s03g015/pred_test.npz
  log: .training_queue/logs/1783270565709415066_465705_mono3d_blcs_ftA_s03g015.log
  output_dir: outputs/blcs/blcs_multiview_axial/logs/version_4
  curves: knowledge/runs/run-mono3d-blcs-fta-s03g015/curves.png
  tb_logdir: outputs/blcs/blcs_multiview_axial/logs/version_4
parents:
- run-mono3d-blcs-bcast-v3-simfix
relations:
- to: run-mono3d-blcs-bcast-v4-physics
  rel: contradicts
- to: run-mono3d-blcs-bcast-v3-simfix
  rel: compares
tags:
- monocular
- broadcast
- physics-prior
- finetune
- negative-result
---

## 考察 / Findings

### 要約
v3 simfix checkpoint からの軽量物理prior fine-tune (smoothness 0.3 / gravity 0.15)。real clip では Y jerk を 0.322→0.168 に半減したが、in-dist test では Y error が 1.615→2.574m に悪化し、採用不可。

### アーキテクチャ詳細
`run.init_weights=outputs/blcs/.../version_2/checkpoints/last.ckpt` で v3 simfix から weight-only fine-tune。LR 3e-5、60ep、early stopping off。smoothness は全軸 uniform、gravity 0.15。

### メトリクスの解釈
test/pos_error 3.012m、x/y/z=0.873/2.574/0.633m。v3 simfix (1.845m, y=1.615m) から大幅悪化。real clip では jerkY 0.322→0.168 と smoothness 目的は達成したが、|Y|>15m は 10→11 で改善しない。

### アーキテクチャ⇄メトリクスの因果考察
fine-tuneでも uniform smoothness は depth drift を誘発しやすい。重力weightが弱く、かつ z jerk も同時に抑えるため、弾道形状を安定させるよりも全体を滑らかにする方向へ寄ったと考える。

### 既存実験との比較
[[run-mono3d-blcs-bcast-v4-physics]] の from-scratch 振幅崩壊は避けたが、Y drift が増えた。[[run-mono3d-blcs-ftb-s06g03]] は重力とsmoothnessを強めてYを改善するが、別の軸圧縮が出る。

### 次に有効な実験
uniform smoothness ではなく、Z jerk を切った axis-weighted smoothness を試す。実際に [[run-mono3d-blcs-ftc-axis-s04g03]] がこの方向の比較 run。
