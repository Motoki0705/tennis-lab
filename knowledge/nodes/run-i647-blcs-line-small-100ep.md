---
id: run-i647-blcs-line-small-100ep
type: run
title: i647_blcs_line_small_100ep
issue: 647
provider: codex
session: 019f6013-0a29-7183-87ae-e4c221387139
date: '2026-07-14'
status: done
config:
  model: multiview_axial_line_small
  data: chunked_multiview_sequence_line_bs4
metrics:
  mean_position_error_m: 8.144406
  mean_x_error_m: 3.2849
  mean_y_error_m: 6.874434
  mean_z_error_m: 0.548385
  mean_endpoint_error_m: 9.905495
  position_accuracy_0_3m: 0.000572
  position_accuracy_0_6m: 0.004294
  position_accuracy_1_2m: 0.019823
  endpoint_accuracy_0_5m: 0.0
  endpoint_accuracy_1m: 0.01
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
    training.early_stopping.enabled=false run.output_dir=outputs/blcs/issue647_line_small_100ep
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i647-blcs-line-small-100ep
  predictions: knowledge/runs/run-i647-blcs-line-small-100ep/pred_test.npz
  log: .training_queue/logs/1784027400496446355_1950816_i647_blcs_line_small_100ep.log
  output_dir: outputs/blcs/issue647_line_small_100ep/logs/version_0
  curves: knowledge/runs/run-i647-blcs-line-small-100ep/curves.png
  tb_logdir: outputs/blcs/issue647_line_small_100ep/logs/version_0
parents: []
relations:
- to: run-i647-blcs-line-moderate-v2-100ep
  rel: compares
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

RANSAC court-line tokenと強いline-map増強を用いたBLCSの100 epoch baseline。train/validation/testは完走したが、test位置誤差 `8.144m`、endpoint誤差 `9.905m` で未学習だった。

### アーキテクチャと増強

`multiview_axial_line_small`（13.0M params）、broadcast single-view、`[court token, ball token]`の2-token camera-axis入力。PLCS strongと同じ強い部分欠損・遮蔽・false-positive・blur/morphology・遠方欠損を適用した。

### メトリクスと因果考察

test位置誤差はx `3.285m`、y `6.874m`、z `0.548m` で、主にコート長手方向の推定に失敗した。`1.2m`以内精度も `1.98%` に留まる。court token依存性テストは通過しているため未接続ではなく、線分欠損とsort位置変動に対してflatten embeddingが不安定な可能性が高い。

### 比較と次の実験

増強を弱めた [[run-i647-blcs-line-moderate-v2-100ep]]、同一split/model規模のKP対照 [[run-i647-blcs-kp-small-100ep]] と比較する。次はline-wise set encoderまたはaugmentation curriculumが妥当である。
