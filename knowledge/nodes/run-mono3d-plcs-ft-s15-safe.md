---
id: run-mono3d-plcs-ft-s15-safe
type: run
title: mono3d_plcs_ft_s15_safe
issue: 593
provider: claude
session: 22fa49a8-b654-4af8-99ce-6f29e7453a00
date: '2026-07-06'
status: done
config:
  model: multiview_axial_split
  loss: canonical_rot + position_smoothness(1.5)
  data: chunked_singleview_sequence
metrics:
  position_error_m: 0.471198
  position_error_std_m: 0.328859
  position_error_median_m: 0.375576
  angular_error_deg: 19.6987
  angular_error_std_deg: 25.656738
  angular_error_median_deg: 12.077632
  x_error_m: 0.175645
  y_error_m: 0.393904
  z_error_m: 0.05634
  position_accuracy: 0.614715
  angle_accuracy: 0.580776
  position_accuracy_0.5m: 0.614715
  position_accuracy_1m: 0.93367
  position_accuracy_2m: 0.998128
  angle_accuracy_10deg: 0.418166
  angle_accuracy_15deg: 0.580776
  angle_accuracy_30deg: 0.841377
repro:
  commit: d36b7dc50522453fa13778e7d31756a733fa0403
  branch: feat/issue-593-physics-prior
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_split model.num_layers=0
    model.num_task_layers=6 data=chunked_singleview_sequence data.scene_dir=data/plcs_broadcast
    data.chunk.chunks_dir=data/plcs_broadcast/chunks data.batch_size=6 data.seq_len_range=[64,192]
    data.num_court_kp=14 data.num_workers=2 data.chunk.generation_workers=3 data.chunk.epochs_per_chunk=30
    loss=canonical_rot loss.position_weight=8.0 loss.position_smoothness_weight=1.5
    loss.canonical_pose_weight=0.0 loss.joint_angle_weight=0.0 loss.torsion_angle_weight=0.0
    loss.torso_twist_weight=0.0 loss.bone_length_weight=0.0 training.learning_rate=3e-5
    training.trainer.max_epochs=50 training.trainer.check_val_every_n_epoch=5 training.qualitative_logging.enabled=false
    training.early_stopping.enabled=false camera=broadcast run.gpus=1 run.init_weights=ckpt/plcs/last.ckpt
artifacts:
  run_dir: knowledge/runs/run-mono3d-plcs-ft-s15-safe
  predictions: knowledge/runs/run-mono3d-plcs-ft-s15-safe/pred_test.npz
  log: .training_queue/logs/1783274156982033821_497703_mono3d_plcs_ft_s15_safe.log
  output_dir: outputs/plcs/plcs_multiview_axial_split/logs/version_24
  curves: knowledge/runs/run-mono3d-plcs-ft-s15-safe/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial_split/logs/version_24
parents:
- run-mono3d-plcs-bcast-v2-simfix
relations:
- to: run-mono3d-plcs-bcast-v2-simfix
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
PLCS v2 simfix checkpoint から position jerk smoothness(1.5) を加えた memory-safe fine-tune。in-dist position は 0.345→0.471m と悪化したが、real clip では far player の >10m/s speed spike が 20→9 に減少した。位置精度とのトレードオフが大きく、標準ckpt置換ではなく出力平滑化/後処理の設計材料として扱うべき。

### アーキテクチャ詳細
モデルは [[run-mono3d-plcs-bcast-v2-simfix]] と同じ multiview_axial_split。`loss.position_smoothness_weight=1.5` を追加し、yaw/rotation にはsmoothnessをかけない。OOM回避のため batch_size=6, seq_len_range=[64,192], generation_workers=3 で実行し、`run.init_weights=ckpt/plcs/last.ckpt` から weight-only fine-tune。

### メトリクスの解釈
test/position_error 0.471m (median 0.376m)、angular 19.7° (median 12.1°)。baseline v2 simfix は 0.345m / 21.1° なので、角度は微改善だが位置は明確に悪化。real clip combined run では p1 speed p99 16.0→15.5、>10m/s 20→9、>15m/s 10→9。改善はあるがピーク速度は 31.2→37.0m/s と悪化するフレームも残る。

### アーキテクチャ⇄メトリクスの因果考察
jerk prior は連続フレームの細かな揺れを抑える一方、実際のサーブ前/方向転換/検出信頼度崩壊に伴う局所的な不連続を一律に扱うため、位置を丸めて in-dist error を落とす。real clip ではスパイク数は減るが、confidence崩壊区間で大きい外れは残る。

### 既存実験との比較
[[run-mono3d-plcs-bcast-v2-simfix]] より標準評価では劣る。実clip安定化は一部確認できたが、ckpt差し替えで解くより、pipeline出力側のconfidence-aware smootherや短区間補間として実装する方が筋が良い。

### 次に有効な実験
PLCS training loss ではなく、推論後の confidence-aware temporal filter を追加する。ViTPose confidence、player visibility、速度上限、短い欠損補間を条件にし、通常フレームは baseline 位置を保持する設計が有効。
