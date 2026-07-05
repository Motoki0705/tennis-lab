---
id: run-mono3d-blcs-bcast-v2-yw4
type: run
title: mono3d_blcs_bcast_v2_yw4
issue: 593
provider: claude
session: dd9d56e0-5f2d-4d9d-a8c7-c80704a806b3
date: '2026-07-05'
status: done
config:
  model: multiview_axial_base
  loss: trajectory_position(axis_weights=[1,4,1])+reprojection0.1
  data: chunked_multiview_sequence_bs4 (C=1, broadcast, court_kp=14)
metrics:
  mean_position_error_m: 1.462467
  mean_x_error_m: 0.419424
  mean_y_error_m: 1.264561
  mean_z_error_m: 0.250169
  mean_endpoint_error_m: 3.644223
  position_accuracy_0_3m: 0.166555
  position_accuracy_0_6m: 0.389252
  position_accuracy_1_2m: 0.658611
  endpoint_accuracy_0_5m: 0.07
  endpoint_accuracy_1m: 0.2
repro:
  commit: 6479c948d6329a4ec6258c93585042c80386869c
  branch: feat/mono3d-broadcast-pipeline
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.blcs.scripts.train --config-name train_chunked data.scene_dir=data/blcs_broadcast
    data.chunk.chunks_dir=data/blcs_broadcast/chunks data.num_views_range=[1,1] data.camera_mode=random
    data.num_court_kp=14 data.num_workers=2 data.chunk.generation_workers=8 camera=broadcast
    training.position_axis_weights=[1.0,4.0,1.0] run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-mono3d-blcs-bcast-v2-yw4
  predictions: knowledge/runs/run-mono3d-blcs-bcast-v2-yw4/pred_test.npz
  log: .training_queue/logs/1783211749537294235_43817_mono3d_blcs_bcast_v2_yw4.log
  output_dir: outputs/blcs/blcs_multiview_axial/logs/version_1
  curves: knowledge/runs/run-mono3d-blcs-bcast-v2-yw4/curves.png
  tb_logdir: outputs/blcs/blcs_multiview_axial/logs/version_1
parents:
- run-mono3d-blcs-bcast
relations:
- to: run-mono3d-blcs-bcast
  rel: supersedes
tags:
- blcs
- monocular
- broadcast
- court-kp-14
- depth-ambiguity
- axis-weights
- deploy-candidate
---

## 考察 / Findings

### 要約
v1 の唯一の変更として `training.position_axis_weights=[1,4,1]` を追加した run。test **mean_position_error 2.99m → 1.46m(-51%)**、Y(視線方向深度)は **2.64m → 1.26m**。単一変数実験として、正規化座標損失の Y 過小重みが単眼 BLCS の主要ボトルネックだったことを確定させた。

### アーキテクチャ詳細
[[run-mono3d-blcs-bcast]] と完全同一の model/data/学習設定。差分は position Smooth L1 への軸別重み [1,4,1] のみ(Y の正規化スケール 11.885 vs X 5.485 の二乗比 ≈4.7 の近似補正)。

### メトリクスの解釈
X 0.81→0.42、Z 0.52→0.25 と、Y 以外も改善している(Y 重み付けで全体の勾配配分が健全化したためと解釈)。@1.2m accuracy 0.235→0.659。100 epochs 完走時点で val/pos_error_m は 1.55 まで低下し続けており、**early stopping は発火しないまま max_epochs 到達 = まだ収束していない**。endpoint(着地点)は 4.54→3.64m と改善幅が小さく、依然課題。

### アーキテクチャ⇄メトリクスの因果考察
v1 分析で確定した「1m あたりの Y 損失が X の 0.21 倍」という機械的不均衡を重みで補正した結果であり、因果は明確。endpoint の改善が鈍いのは、着地点がラリー端(可視性が低く外挿になりがちな区間)に集中するためと推測(仮説)。

### 既存実験との比較
[[run-mono3d-blcs-bcast]](2.99m)に対し全軸で改善し、性能で置換(supersedes)。deploy 候補として ckpt/blcs/last.ckpt に配置。

### 次に有効な実験
- **学習延長**(max_epochs 200 + early stopping)— val がまだ下降中のため最も確実な上積み。
- endpoint 改善: 着地点近傍の時間重み付け、または不可視区間の物理外挿の明示化。
- seq_len_range [128,384] で長文脈化(v1 分析での次点候補)。

<!-- run `mono3d_blcs_bcast_v2_yw4` の結果と考察を書く。parents/tags も埋め、 主要 metrics は frontmatter と一致させること。 -->
