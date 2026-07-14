---
id: run-i647-plcs-kp-small-100ep
type: run
title: i647_plcs_kp_small_100ep
issue: 647
provider: codex
session: 019f6013-0a29-7183-87ae-e4c221387139
date: '2026-07-14'
status: done
config:
  model: multiview_axial_small
  loss: canonical_rot
  data: chunked_multiview_sequence_bs8
metrics:
  position_error_m: 1.209411
  position_error_std_m: 0.74213
  position_error_median_m: 1.088133
  angular_error_deg: 46.495831
  angular_error_std_deg: 40.557697
  angular_error_median_deg: 34.667301
  x_error_m: 0.461709
  y_error_m: 1.004716
  z_error_m: 0.095505
  position_accuracy: 0.148486
  angle_accuracy: 0.222058
  position_accuracy_0.5m: 0.148486
  position_accuracy_1m: 0.427365
  position_accuracy_2m: 0.863977
  angle_accuracy_10deg: 0.139708
  angle_accuracy_15deg: 0.222058
  angle_accuracy_30deg: 0.434066
repro:
  commit: f26addb77c3b6c3657f10caa8917328f22ab0274
  branch: feat/issue-647-court-line-token
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_small data=chunked_multiview_sequence_bs8
    data.scene_dir=data/plcs_broadcast data.chunk.chunks_dir=data/plcs_broadcast/chunks
    camera=broadcast data.batch_size=8 data.num_workers=4 data.num_views_range=[1,1]
    data.seq_len_range=[64,256] data.chunk.generation_workers=6 data.chunk.epochs_per_chunk=30
    loss=canonical_rot loss.position_weight=8.0 loss.canonical_pose_weight=0.0 loss.joint_angle_weight=0.0
    loss.torsion_angle_weight=0.0 loss.torso_twist_weight=0.0 loss.bone_length_weight=0.0
    training.trainer.max_epochs=100 training.trainer.check_val_every_n_epoch=5 training.qualitative_logging.enabled=false
    training.early_stopping.enabled=false run.output_dir=outputs/plcs/issue647_kp_small_100ep
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i647-plcs-kp-small-100ep
  predictions: knowledge/runs/run-i647-plcs-kp-small-100ep/pred_test.npz
  log: .training_queue/logs/1784028363072084109_1964989_i647_plcs_kp_small_100ep.log
  output_dir: outputs/plcs/issue647_kp_small_100ep/logs/version_0
  curves: knowledge/runs/run-i647-plcs-kp-small-100ep/curves.png
  tb_logdir: outputs/plcs/issue647_kp_small_100ep/logs/version_0
parents: []
relations:
- to: run-i647-plcs-line-small-100ep
  rel: compares
- to: run-i647-plcs-line-moderate-v2-100ep
  rel: compares
tags:
- plcs
- court-kp
- baseline
---

## 考察 / Findings

### 要約

line版と同じbroadcast split、small axial規模、seed、100 epochで学習したKP対照。test位置 `1.209m`、角度 `46.50°` で、line版より明確に良好だった。

### 比較

line strong [[run-i647-plcs-line-small-100ep]] に対して位置誤差を `7.236m → 1.209m`、角度誤差を `99.07° → 46.50°` へ低減した。line moderate [[run-i647-plcs-line-moderate-v2-100ep]] に対しても同様に優位である。

### 解釈

固定semanticを持つcourt KPはsingle-view camera geometryの手掛かりとして強く、MVPの可変line flattenはその代替に至らない。KP後方互換経路が機能することと、比較条件の健全性を示すbaselineである。
