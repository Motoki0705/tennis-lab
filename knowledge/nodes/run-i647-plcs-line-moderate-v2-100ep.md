---
id: run-i647-plcs-line-moderate-v2-100ep
type: run
title: i647_plcs_line_moderate_v2_100ep
issue: 647
provider: codex
session: 019f6013-0a29-7183-87ae-e4c221387139
date: '2026-07-14'
status: done
config:
  model: multiview_axial_line_small
  loss: canonical_rot
  data: chunked_multiview_sequence_line_bs8
metrics:
  position_error_m: 6.9105
  position_error_std_m: 3.698879
  position_error_median_m: 6.744992
  angular_error_deg: 95.976753
  angular_error_std_deg: 48.182827
  angular_error_median_deg: 97.134392
  x_error_m: 2.553534
  y_error_m: 5.883515
  z_error_m: 0.0977
  position_accuracy: 0.008444
  angle_accuracy: 0.052678
  position_accuracy_0.5m: 0.008444
  position_accuracy_1m: 0.026213
  position_accuracy_2m: 0.093699
  angle_accuracy_10deg: 0.025331
  angle_accuracy_15deg: 0.052678
  angle_accuracy_30deg: 0.112413
repro:
  commit: f26addb77c3b6c3657f10caa8917328f22ab0274
  branch: feat/issue-647-court-line-token
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_line_small data=chunked_multiview_sequence_line_bs8
    data.scene_dir=data/plcs_broadcast data.chunk.chunks_dir=data/plcs_broadcast/chunks
    camera=broadcast data.batch_size=8 data.num_workers=4 data.num_views_range=[1,1]
    data.seq_len_range=[64,256] data.chunk.generation_workers=6 data.chunk.epochs_per_chunk=30
    loss=canonical_rot loss.position_weight=8.0 loss.canonical_pose_weight=0.0 loss.joint_angle_weight=0.0
    loss.torsion_angle_weight=0.0 loss.torso_twist_weight=0.0 loss.bone_length_weight=0.0
    training.trainer.max_epochs=100 training.trainer.check_val_every_n_epoch=5 training.qualitative_logging.enabled=false
    training.early_stopping.enabled=false run.output_dir=outputs/plcs/issue647_line_moderate_v2_100ep
    run.gpus=1 +data.court_line.augmentation.partial_erasure_prob=0.5 +data.court_line.augmentation.max_partial_erasures=3
    +data.court_line.augmentation.occlusion_prob=0.35 +data.court_line.augmentation.max_occlusions=2
    +data.court_line.augmentation.false_positive_prob=0.2 +data.court_line.augmentation.max_false_positive_lines=1
    +data.court_line.augmentation.blur_prob=0.2 +data.court_line.augmentation.morphology_prob=0.25
    +data.court_line.augmentation.far_dropout_prob=0.15 +data.court_line.augmentation.near_only_prob=0.05
artifacts:
  run_dir: knowledge/runs/run-i647-plcs-line-moderate-v2-100ep
  predictions: knowledge/runs/run-i647-plcs-line-moderate-v2-100ep/pred_test.npz
  log: .training_queue/logs/1784028362684010282_1964946_i647_plcs_line_moderate_v2_100ep.log
  output_dir: outputs/plcs/issue647_line_moderate_v2_100ep/logs/version_0
  curves: knowledge/runs/run-i647-plcs-line-moderate-v2-100ep/curves.png
  tb_logdir: outputs/plcs/issue647_line_moderate_v2_100ep/logs/version_0
parents:
- run-i647-plcs-line-small-100ep
relations:
- to: run-i647-plcs-kp-small-100ep
  rel: compares
tags:
- plcs
- court-line
- ransac
- augmentation
- negative-result
---

## 考察 / Findings

### 要約

全line-map劣化を維持し、重畳確率と最大個数だけを約半分へ下げたPLCS 100 epoch run。test位置 `6.910m`、角度 `95.98°` でstrongより小幅改善したが、未学習の範囲を出なかった。

### 増強差分

部分欠損 `0.5`、遮蔽 `0.35`、false-positive `0.2`、blur `0.2`、morphology `0.25`、far dropout `0.15`、near-only `0.05`。model、seed、broadcast split、batch、epochはstrongと同一である。

### 比較と解釈

strong [[run-i647-plcs-line-small-100ep]] 比で位置は `7.236m → 6.910m`、角度は `99.07° → 95.98°` と改善した。一方KP [[run-i647-plcs-kp-small-100ep]] の `1.209m / 46.50°` には大差で劣る。単なる増強確率ではなく、semantic IDなし・deterministic sort + flattenという表現が部分線分集合の変動を吸収できない可能性が高い。

### 次の実験

線分単位の共有MLP + pooling/set attention、またはcleanから劣化を段階的に増やすcurriculumを比較する。MVP contract自体は維持し、confidence/mask/semantic IDは追加しない。
