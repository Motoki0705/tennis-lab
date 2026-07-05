---
id: run-mono3d-blcs-ftc-axis-s04g03
type: run
title: mono3d_blcs_ftC_axis_s04g03
issue: 593
provider: claude
session: 22fa49a8-b654-4af8-99ce-6f29e7453a00
date: '2026-07-06'
status: done
config:
  model: multiview_axial_base
  loss: trajectory_position + reprojection(0.1) + axis_weights[1,4,1] + smoothness(0.4,
    axis=[1,1,0]) + gravity(0.30)
  data: chunked_multiview_sequence C=1
metrics:
  mean_position_error_m: 1.946813
  mean_x_error_m: 0.690332
  mean_y_error_m: 1.583612
  mean_z_error_m: 0.408921
  mean_endpoint_error_m: 3.74724
  position_accuracy_0_3m: 0.055685
  position_accuracy_0_6m: 0.234722
  position_accuracy_1_2m: 0.563967
  endpoint_accuracy_0_5m: 0.01
  endpoint_accuracy_1m: 0.12
repro:
  commit: 93c5baceba9f1aa90e6f8dc303e86a01cafb76a4
  branch: worktree-i593-axis-smoothness
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.blcs.scripts.train --config-name train_chunked data.scene_dir=/home/kamimura/projects/tennis-lab/data/blcs_broadcast
    data.chunk.chunks_dir=/home/kamimura/projects/tennis-lab/data/blcs_broadcast/chunks
    data.num_views_range=[1,1] data.camera_mode=random data.num_court_kp=14 data.num_workers=2
    data.chunk.generation_workers=8 data.chunk.epochs_per_chunk=20 camera=broadcast
    training.position_axis_weights=[1.0,4.0,1.0] training.learning_rate=3e-5 training.trainer.max_epochs=60
    training.trainer.check_val_every_n_epoch=5 training.qualitative_logging.enabled=false
    training.early_stopping.enabled=false training.smoothness_loss_weight=0.4 training.gravity_loss_weight=0.3
    training.smoothness_axis_weights=[1,1,0] run.output_dir=/home/kamimura/projects/tennis-lab/outputs/blcs/blcs_multiview_axial
    run.init_weights=/home/kamimura/projects/tennis-lab/outputs/blcs/blcs_multiview_axial/logs/version_2/checkpoints/last.ckpt
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-mono3d-blcs-ftc-axis-s04g03
  predictions: knowledge/runs/run-mono3d-blcs-ftc-axis-s04g03/pred_test.npz
  log: .training_queue/logs/1783274918987051955_506543_mono3d_blcs_ftC_axis_s04g03.log
  output_dir: outputs/blcs/blcs_multiview_axial/logs/version_6
  curves: knowledge/runs/run-mono3d-blcs-ftc-axis-s04g03/curves.png
  tb_logdir: outputs/blcs/blcs_multiview_axial/logs/version_6
parents:
- run-mono3d-blcs-bcast-v3-simfix
relations:
- to: run-mono3d-blcs-ftb-s06g03
  rel: compares
- to: run-mono3d-blcs-bcast-v3-simfix
  rel: compares
tags:
- monocular
- broadcast
- physics-prior
- finetune
- real-clip
---

## 考察 / Findings

### 要約
axis-weighted smoothness (`[1,1,0]`) で Z jerk を外し、gravity term に高さ曲率を任せた BLCS fine-tune。in-dist は v3 simfix 1.845m に対して 1.947m と +0.10m 悪化だが、real clip では Y jerk 0.322→0.136 (-58%)、|Y|>15m 10→7、Y-in-court 89.8%→90.9% と実映像の主要症状を改善した。現時点の BLCS physics prior では最もバランスが良い。

### アーキテクチャ詳細
v3 simfix checkpoint から weight-only fine-tune。smoothness 0.4 / gravity 0.3 / `smoothness_axis_weights=[1,1,0]`、LR 3e-5、60ep。PR #605 の axis-weighted smoothness 実装を使い、Z軸 jerk と重力曲率の競合を避けた。

### メトリクスの解釈
in-dist: pos_error 1.947m、x/y/z=0.690/1.584/0.409m。v3より Y は 1.615→1.584 と微改善、X/Zは悪化。real clip: ball Y jerk 0.322→0.136、Z jerk 0.088→0.023、median Δ²z -0.0097→-0.0054。ftBほど平坦ではないが、baselineよりは重力曲率が弱い。可視化 `outputs/tennis_scene/tennis_clip_physics_final/blcs_viz.mp4` は非空で、ネット越えの軌跡は確認できる。

### アーキテクチャ⇄メトリクスの因果考察
Z jerkを外すことで ftB の過平坦化は緩和したが、gravity termだけでは baseline の Δ²z=-g に完全には戻らない。これは単純な全フレームgravity priorが bounce/検出欠損/補間区間を区別しておらず、free-flight以外のフレームで誤った制約になるためと考える。

### 既存実験との比較
[[run-mono3d-blcs-ftb-s06g03]] より in-dist pos_error は良い (2.019→1.947) が、real clip Y jerk は少し大きい (0.107→0.136)。[[run-mono3d-blcs-bcast-v3-simfix]] より in-dist全体はわずかに悪いが、実clipの外れ値とdepth jitterは改善。

### 次に有効な実験
採用するなら「実clip安定化のための optional fine-tune ckpt」と位置づける。次は (1) free-flight/bounce mask 付き gravity、(2) amplitude-preserving penalty (scale/variance regularizer)、(3) best-checkpoint selection を `val/pos_error_m` ではなく実clip jitter proxy と組み合わせる、の順で詰める。
