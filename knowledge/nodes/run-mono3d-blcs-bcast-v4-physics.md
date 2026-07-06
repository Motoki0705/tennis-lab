---
id: run-mono3d-blcs-bcast-v4-physics
type: run
title: mono3d_blcs_bcast_v4_physics
issue: 593
provider: claude
session: dd9d56e0-5f2d-4d9d-a8c7-c80704a806b3
date: '2026-07-06'
status: done
config:
  model: multiview_axial_base
  loss: trajectory_position + reprojection(0.1) + axis_weights[1,4,1] + smoothness(1.0)
    + gravity(0.5)
  data: chunked_multiview_sequence C=1
metrics:
  mean_position_error_m: 2.437528
  mean_x_error_m: 1.291481
  mean_y_error_m: 1.676458
  mean_z_error_m: 0.427551
  mean_endpoint_error_m: 4.340782
  position_accuracy_0_3m: 0.004424
  position_accuracy_0_6m: 0.03173
  position_accuracy_1_2m: 0.19019
  endpoint_accuracy_0_5m: 0.0
  endpoint_accuracy_1m: 0.04
repro:
  commit: be5e653b79b7a090f095f457c4986239637951c4
  branch: feat/issue-593-physics-prior
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.blcs.scripts.train --config-name train_chunked data.scene_dir=data/blcs_broadcast
    data.chunk.chunks_dir=data/blcs_broadcast/chunks data.num_views_range=[1,1] data.camera_mode=random
    data.num_court_kp=14 data.num_workers=2 data.chunk.generation_workers=8 data.chunk.epochs_per_chunk=20
    camera=broadcast training.position_axis_weights=[1.0,4.0,1.0] training.smoothness_loss_weight=1.0
    training.gravity_loss_weight=0.5 training.trainer.max_epochs=200 training.trainer.check_val_every_n_epoch=5
    training.qualitative_logging.enabled=false run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-mono3d-blcs-bcast-v4-physics
  predictions: knowledge/runs/run-mono3d-blcs-bcast-v4-physics/pred_test.npz
  log: .training_queue/logs/1783265179863554457_429668_mono3d_blcs_bcast_v4_physics.log
  output_dir: outputs/blcs/blcs_multiview_axial/logs/version_3
  curves: knowledge/runs/run-mono3d-blcs-bcast-v4-physics/curves.png
  tb_logdir: outputs/blcs/blcs_multiview_axial/logs/version_3
parents:
- run-mono3d-blcs-bcast-v3-simfix
relations:
- to: run-mono3d-blcs-bcast-v3-simfix
  rel: compares
tags:
- monocular
- broadcast
- physics-prior
- negative-result
---

## 考察 / Findings

### 要約
[[run-mono3d-blcs-bcast-v3-simfix]] へ jerk smoothness(1.0) + gravity curvature(0.5) を足して 200ep from-scratch 学習した初回物理prior run。prior は jitter と depth correlation には効いたが、from-scratch では物理lossが序盤から強く、軌道振幅を縮める逃げ方を学び in-dist pos_error は 1.845m→2.438m に悪化した。

### アーキテクチャ詳細
モデル/データは v3 simfix と同じ multiview_axial_base, C=1, court_kp=14, broadcast camera。差分は `smoothness_loss_weight=1.0` と `gravity_loss_weight=0.5` を追加した点。重力targetはこの時点では固定値相当で、後続PR commitで `rally.output_fps` / `physics.gravity` から導出する形へ改善された。

### メトリクスの解釈
test/pos_error 2.438m、軸別 x/y/z=1.291/1.676/0.428m。v3 simfix の 0.511/1.615/0.369m と比べ、Yはほぼ横ばいだが X が大きく悪化。endpoint 4.34m も悪化。

### アーキテクチャ⇄メトリクスの因果考察
保存予測の後解析では、jerk は 0.026→0.010、z acceleration std も低下し、Y/Z correlation は改善した。一方で X/Y/Z の振幅比が 0.71/0.87/0.88 へ縮み、速い横移動や高い弾道を過小化した。これは「物理prior自体が無効」ではなく、from-scratch で位置教師と物理priorのバランスが悪く、滑らかな小振幅解に逃げた結果と解釈する。

### 既存実験との比較
[[run-mono3d-blcs-bcast-v3-simfix]] より in-dist 精度は明確に悪化。後続の [[run-mono3d-blcs-fta-s03g015]] / [[run-mono3d-blcs-ftb-s06g03]] / [[run-mono3d-blcs-ftc-axis-s04g03]] はこの失敗を受け、v3 checkpoint からの weight-only fine-tune に切り替えた。

### 次に有効な実験
物理priorは from-scratch 主目的ではなく、収束済みモデルの後段fine-tune/regularizerとして使う。さらに Z 軸 jerk は gravity term と競合するため、axis-weighted smoothness で z を外す、または free-flight/bounce mask を導入する。
