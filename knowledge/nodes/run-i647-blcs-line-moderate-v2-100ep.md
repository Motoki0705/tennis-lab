---
id: run-i647-blcs-line-moderate-v2-100ep
type: run
title: i647_blcs_line_moderate_v2_100ep
issue: 647
provider: codex
session: 019f6013-0a29-7183-87ae-e4c221387139
date: '2026-07-14'
status: done
config:
  model: multiview_axial_line_small
  data: chunked_multiview_sequence_line_bs4
metrics:
  mean_position_error_m: 8.426843
  mean_x_error_m: 3.337863
  mean_y_error_m: 7.131818
  mean_z_error_m: 0.488518
  mean_endpoint_error_m: 9.555126
  position_accuracy_0_3m: 0.000519
  position_accuracy_0_6m: 0.004672
  position_accuracy_1_2m: 0.020911
  endpoint_accuracy_0_5m: 0.0
  endpoint_accuracy_1m: 0.02
repro:
  commit: f26addb77c3b6c3657f10caa8917328f22ab0274
  branch: feat/issue-647-court-line-token
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.blcs.scripts.train model=multiview_axial_line_small data=chunked_multiview_sequence_line_bs4
    data.scene_dir=data/blcs_broadcast data.chunk.chunks_dir=data/blcs_broadcast/chunks
    camera=broadcast data.batch_size=8 data.num_workers=4 data.num_views_range=[1,1]
    data.seq_len_range=[64,256] data.chunk.generation_workers=8 data.chunk.epochs_per_chunk=20
    training.position_axis_weights=[1.0,4.0,1.0] training.reprojection_loss_weight=0.1
    training.trainer.max_epochs=100 training.trainer.check_val_every_n_epoch=5 training.qualitative_logging.enabled=false
    training.early_stopping.enabled=false run.output_dir=outputs/blcs/issue647_line_moderate_v2_100ep
    run.gpus=1 +data.court_line.augmentation.partial_erasure_prob=0.5 +data.court_line.augmentation.max_partial_erasures=3
    +data.court_line.augmentation.occlusion_prob=0.35 +data.court_line.augmentation.max_occlusions=2
    +data.court_line.augmentation.false_positive_prob=0.2 +data.court_line.augmentation.max_false_positive_lines=1
    +data.court_line.augmentation.blur_prob=0.2 +data.court_line.augmentation.morphology_prob=0.25
    +data.court_line.augmentation.far_dropout_prob=0.15 +data.court_line.augmentation.near_only_prob=0.05
artifacts:
  run_dir: knowledge/runs/run-i647-blcs-line-moderate-v2-100ep
  predictions: knowledge/runs/run-i647-blcs-line-moderate-v2-100ep/pred_test.npz
  log: .training_queue/logs/1784028362876224806_1964966_i647_blcs_line_moderate_v2_100ep.log
  output_dir: outputs/blcs/issue647_line_moderate_v2_100ep/logs/version_0
  curves: knowledge/runs/run-i647-blcs-line-moderate-v2-100ep/curves.png
  tb_logdir: outputs/blcs/issue647_line_moderate_v2_100ep/logs/version_0
parents:
- run-i647-blcs-line-small-100ep
relations:
- to: run-i647-blcs-kp-small-100ep
  rel: compares
tags:
- blcs
- court-line
- ransac
- augmentation
- negative-result
---

## 考察 / Findings

### 要約

増強重畳を弱めたBLCS 100 epoch run。test位置 `8.427m`、endpoint `9.555m` で、strongに対してendpointのみ小幅改善し、位置は悪化した。

### 比較と解釈

strong [[run-i647-blcs-line-small-100ep]] の位置 `8.144m` / endpoint `9.905m` に対し、moderateは `8.427m` / `9.555m`。差は一貫した改善ではなく、KP [[run-i647-blcs-kp-small-100ep]] の位置 `1.828m` / endpoint `3.046m` からも大きく離れる。したがって強度調整だけでは解決せず、可変部分線分を固定順flattenする表現が主要な制約と考える。

### 次の実験

line-wise set encoderとaugmentation curriculumを独立にablationし、clean line上限も測定する。線分入力は端点UVのみというcontractを維持する。
