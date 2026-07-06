---
id: run-mono3d-blcs-ftb-s06g03
type: run
title: mono3d_blcs_ftB_s06g03
issue: 593
provider: claude
session: dd9d56e0-5f2d-4d9d-a8c7-c80704a806b3
date: '2026-07-06'
status: done
config:
  model: multiview_axial_base
  loss: trajectory_position + reprojection(0.1) + axis_weights[1,4,1] + smoothness(0.6)
    + gravity(0.30)
  data: chunked_multiview_sequence C=1
metrics:
  mean_position_error_m: 2.018878
  mean_x_error_m: 0.865109
  mean_y_error_m: 1.538739
  mean_z_error_m: 0.369998
  mean_endpoint_error_m: 3.262414
  position_accuracy_0_3m: 0.020162
  position_accuracy_0_6m: 0.12663
  position_accuracy_1_2m: 0.467527
  endpoint_accuracy_0_5m: 0.01
  endpoint_accuracy_1m: 0.09
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
    training.early_stopping.enabled=false run.gpus=1 training.smoothness_loss_weight=0.6
    training.gravity_loss_weight=0.30 run.init_weights=outputs/blcs/blcs_multiview_axial/logs/version_2/checkpoints/last.ckpt
artifacts:
  run_dir: knowledge/runs/run-mono3d-blcs-ftb-s06g03
  predictions: knowledge/runs/run-mono3d-blcs-ftb-s06g03/pred_test.npz
  log: .training_queue/logs/1783270565725317355_465720_mono3d_blcs_ftB_s06g03.log
  output_dir: outputs/blcs/blcs_multiview_axial/logs/version_5
  curves: knowledge/runs/run-mono3d-blcs-ftb-s06g03/curves.png
  tb_logdir: outputs/blcs/blcs_multiview_axial/logs/version_5
parents:
- run-mono3d-blcs-bcast-v3-simfix
relations:
- to: run-mono3d-blcs-fta-s03g015
  rel: compares
- to: run-mono3d-blcs-bcast-v3-simfix
  rel: compares
tags:
- monocular
- broadcast
- physics-prior
- finetune
---

## 考察 / Findings

### 要約
v3 simfix からの uniform 物理prior fine-tune (smoothness 0.6 / gravity 0.30)。in-dist では Y error を 1.615→1.539m に小改善し、real clip では Y jerk を 0.322→0.107 (-67%) まで削る。一方で X error が 0.511→0.865m に悪化し、重力曲率も -0.0097→-0.0033 と平坦化したため採用は保留。

### アーキテクチャ詳細
[[run-mono3d-blcs-fta-s03g015]] と同じ v3 checkpoint からの weight-only fine-tune。smoothness/gravity を 2倍にし、全軸 uniform jerk penalty のまま実行した。

### メトリクスの解釈
test/pos_error 2.019m、x/y/z=0.865/1.539/0.370m。Yだけ見れば v3 より良いが、Xが大きく悪化して全体では v3 の 1.845m に届かない。real clip では |Y|>15m が 10→8、Y_in 89.8%→90.7% と少し改善。

### アーキテクチャ⇄メトリクスの因果考察
smoothness強化で depth jitter は大きく減るが、全軸にかけると X の鋭い実運動と Z の弾道曲率も一緒に平坦化される。重力termはあるが、jerk termがZの曲率変化を抑えるため median Δ²z が 0 に寄った。

### 既存実験との比較
[[run-mono3d-blcs-fta-s03g015]] よりY driftは改善。[[run-mono3d-blcs-ftc-axis-s04g03]] はこの結果を受け、Z軸smoothnessを0にして弾道の競合を減らす。

### 次に有効な実験
axisごとの prior 制御を標準化する。X/Yだけsmoothness、Zはgravityに任せる構成、または bounce/free-flight mask でZ jerkを局所的に許容する構成が有効。
