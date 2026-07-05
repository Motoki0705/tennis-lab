---
id: run-mono3d-plcs-bcast
type: run
title: mono3d_plcs_bcast
issue: 593
provider: claude
session: dd9d56e0-5f2d-4d9d-a8c7-c80704a806b3
date: '2026-07-05'
status: done
config:
  model: multiview_axial_split
  loss: canonical_rot
  data: chunked_singleview_sequence
metrics:
  position_error_m: 0.312996
  position_error_std_m: 0.291672
  position_error_median_m: 0.245921
  angular_error_deg: 29.706078
  angular_error_std_deg: 35.326874
  angular_error_median_deg: 16.195539
  x_error_m: 0.108428
  y_error_m: 0.262894
  z_error_m: 0.050593
  position_accuracy: 0.854323
  angle_accuracy: 0.461786
  position_accuracy_0.5m: 0.854323
  position_accuracy_1m: 0.973266
  position_accuracy_2m: 0.995792
  angle_accuracy_10deg: 0.315923
  angle_accuracy_15deg: 0.461786
  angle_accuracy_30deg: 0.694288
repro:
  commit: 4a416b2a07f67518448e669ba5e985668373cdf4
  branch: feat/mono3d-broadcast-pipeline
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split model.num_layers=0
    model.num_task_layers=6 data=chunked_singleview_sequence data.scene_dir=data/plcs_broadcast
    data.chunk.chunks_dir=data/plcs_broadcast/chunks data.batch_size=8 data.seq_len_range=[64,256]
    data.num_court_kp=14 data.num_workers=2 loss=canonical_rot loss.position_weight=8.0
    loss.canonical_pose_weight=0.0 loss.joint_angle_weight=0.0 loss.torsion_angle_weight=0.0
    loss.torso_twist_weight=0.0 loss.bone_length_weight=0.0 training.trainer.accumulate_grad_batches=1
    training.trainer.max_epochs=200 training.early_stopping.enabled=false camera=broadcast
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-mono3d-plcs-bcast
  predictions: knowledge/runs/run-mono3d-plcs-bcast/pred_test.npz
  log: .training_queue/logs/1783207757918559209_7684_mono3d_plcs_bcast.log
  output_dir: outputs/plcs/plcs_multiview_axial_split/logs/version_21
  curves: knowledge/runs/run-mono3d-plcs-bcast/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial_split/logs/version_21
parents:
- run-i590-courtkp14
relations:
- to: run-i590-courtkp14
  rel: compares
tags:
- plcs
- monocular
- broadcast
- court-kp-14
- sim-to-real
- split-trunk
- chunked
---

## 考察 / Findings

### 要約
i590 の 14 点 recipe を**単眼 broadcast カメラ(C=1)**へ移した run。位置 **0.313m**(median 0.246m)/ 回転 **29.7°**(median 16.2°)。位置は @1m accuracy 0.973 で単眼でも実用域だが、回転は multiview の 6.28° から大きく後退し、単眼視点の向き曖昧性(フリップ外れ値、std 35.3°)が支配的。

### アーキテクチャ詳細
モデル・損失は [[run-i590-courtkp14]] と同一(`multiview_axial_split` H=0/S=6、`canonical_rot` の aux 全 0、`position_weight=8.0`、200 epochs)。差分はデータのみ: `chunked_singleview_sequence`(各 scene からランダム 1 カメラを選ぶ C=1 サンプリング)+ `camera=broadcast`(PR #592 の高所望遠 2 カメラ preset、`data/plcs_broadcast` 事前生成 + chunk 生成も broadcast)。

### メトリクスの解釈
位置誤差の軸内訳は X 0.108 / Y 0.263 / Z 0.051 と、視線方向(コート長手 Y)が支配的 — 単眼の深度不確実性がそのまま出ている。ただし選手は地面平面上にいるためコート KP のホモグラフィで強く拘束され、@0.5m 0.854 / @1m 0.973 を維持。回転は mean 29.7° に対し median 16.2° で、分布の裾(おそらく 180° 系フリップ)が mean を押し上げている。@30° は 0.694。

### アーキテクチャ⇄メトリクスの因果考察
位置の健全さは「planar court token + 地面接地の事前」で単眼でも幾何が決まるためと解釈できる。回転の後退は、multiview では2視点の見えの差で解消していた前後向きの曖昧さが、単眼 2D pose だけでは原理的に残るため(仮説)。canonical_rot の angle loss はフリップ抑制に効くが、単眼では入力情報自体が不足している。

### 既存実験との比較
[[run-i590-courtkp14]](C=2 multiview, 0.189m / 6.28°)に対し、位置 +0.124m・回転 +23.4° の後退。低下幅は回転に集中しており、単眼化のコストは「向き推定」に局在すると言える。

### 次に有効な実験
- 時系列の移動方向 prior(速度ベクトルと向きの整合)を loss か後処理で入れ、フリップ外れ値を削る。
- 単眼向けに rotation 側 trunk の容量/augmentation(可視性 dropout)再調整。
- 実映像でのフリップ頻度を tennis_scene 出力で定性確認し、後処理(時系列平滑化/多数決)の効果を測る。
