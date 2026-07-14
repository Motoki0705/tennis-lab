---
id: run-deploy-multiview-blcs-v3-simfix-c3-6-v2
type: run
title: deploy_multiview_blcs_v3_simfix_c3_6_v2
issue: 593
provider: codex
session: 019f5ddc-9e4a-7d50-85e6-0db5262b88b1
date: '2026-07-14'
status: done
config:
  model: multiview_axial_base
  loss: trajectory_position + reprojection(0.1) + axis_weights[1,4,1]
  data: chunked_multiview_sequence C=3-6 court_kp=14
metrics:
  mean_position_error_m: 1.064595
  mean_x_error_m: 0.426141
  mean_y_error_m: 0.856086
  mean_z_error_m: 0.186533
  mean_endpoint_error_m: 2.024551
  position_accuracy_0_3m: 0.213767
  position_accuracy_0_6m: 0.534419
  position_accuracy_1_2m: 0.759561
  endpoint_accuracy_0_5m: 0.26
  endpoint_accuracy_1m: 0.53
repro:
  commit: fd46ed0b09c6807ffda7a3d1aa4004a458213cdd
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.blcs.scripts.train --config-name train_chunked data.num_court_kp=14
    data.num_views_range=[3,6] data.camera_mode=random data.batch_size=4 data.seq_len_range=[64,256]
    data.num_workers=2 data.chunk.generation_workers=8 data.chunk.epochs_per_chunk=20
    training.position_axis_weights=[1.0,4.0,1.0] training.trainer.max_epochs=200 training.trainer.check_val_every_n_epoch=5
    training.qualitative_logging.enabled=false training.early_stopping.enabled=false
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-deploy-multiview-blcs-v3-simfix-c3-6-v2
  predictions: knowledge/runs/run-deploy-multiview-blcs-v3-simfix-c3-6-v2/pred_test.npz
  log: .training_queue/logs/1783986831913101106_1649558_deploy_multiview_blcs_v3_simfix_c3_6_v2.log
  output_dir: outputs/blcs/blcs_multiview_axial/logs/version_9
  checkpoint: ckpt/blcs/run-multiview-blcs-v3-simfix-c3-6-epoch129.ckpt
  curves: knowledge/runs/run-deploy-multiview-blcs-v3-simfix-c3-6-v2/curves.png
  tb_logdir: outputs/blcs/blcs_multiview_axial/logs/version_9
parents:
- run-mono3d-blcs-bcast-v3-simfix
relations:
- to: run-mono3d-blcs-bcast-v3-simfix
  rel: supersedes
tags:
- blcs
- multiview
- deploy
- court-kp-14
- chunked
- sim-to-real
---

## 考察 / Findings

### 要約
knowledge 上に存在しなかった BLCS の3カメラ対応 deploy run。単眼 simfix baseline [[run-mono3d-blcs-bcast-v3-simfix]] を基礎に3--6 cameraへ拡張し、test位置誤差 **1.065m**、endpoint **2.025m**を得た。val loss 最良の epoch 129 を `ckpt/` に配備する。

### アーキテクチャ詳細
`multiview_axial_base` 51.9M、camera/time axial attention、court keypoint 14点。chunked synthetic dataでcamera数を3--6からランダム選択し、simfix後のball observation augmentation、reprojection weight 0.1、court長手方向を重視するaxis weight [1,4,1]を使用。200 epochs、物理prior/GANは無効。

### メトリクスの解釈
test position mean 1.065m、軸別 X/Y/Z は0.426/0.856/0.187m、endpoint 2.025m。1.2m以内75.96%、endpoint 1m以内53.0%。checkpoint callback の `val/loss` 最良は epoch 129 の0.0242で、last epochより良いためbest checkpointを選択した。

### アーキテクチャ⇄メトリクスの因果考察
複数視点により単眼で支配的だったcourt長手Yのdepth曖昧性が減ったと考える（仮説）。実際、親の単眼runに対してY errorは1.615→0.856mへほぼ半減し、全体誤差も1.845→1.065mへ改善した。

### 既存実験との比較
[[run-mono3d-blcs-bcast-v3-simfix]] 比でposition 1.845→1.065m、endpoint 3.408→2.025m、各軸すべて改善した。ただしcamera presetもbroadcast単眼からdefault multiviewへ変わるため、改善すべてをcamera数だけへ帰属はできない。

### 次に有効な実験
`meiji_3cam` 実推論でcamera別ball coverageと3D trajectoryを確認する。検出欠損時のteleportが残る場合は、model変更より先にmultiview confidence-aware observation maskを評価する。
