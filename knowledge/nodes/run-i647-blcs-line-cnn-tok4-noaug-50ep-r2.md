---
id: run-i647-blcs-line-cnn-tok4-noaug-50ep-r2
type: run
title: BLCS CNN line map 4 token・no augmentation・50 epoch
issue: 647
provider: codex
session: 019f6013-0a29-7183-87ae-e4c221387139
date: '2026-07-15'
status: done
config:
  model: multiview_axial_line_small
  data: chunked_multiview_sequence_line_bs4
metrics:
  mean_position_error_m: 8.476074
  mean_x_error_m: 3.301062
  mean_y_error_m: 7.184183
  mean_z_error_m: 0.531576
  mean_endpoint_error_m: 10.842196
  position_accuracy_0_3m: 0.000604
  position_accuracy_0_6m: 0.003823
  position_accuracy_1_2m: 0.018443
  endpoint_accuracy_0_5m: 0.01
  endpoint_accuracy_1m: 0.01
repro:
  commit: 46474cb82a1d2de36f20272d668385ab3bcbd41f
  branch: exp/issue-647-cnn-line-train
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.blcs.scripts.train model=multiview_axial_line_small data=chunked_multiview_sequence_line_bs4
    model.num_line_map_tokens=4 data.scene_dir=/home/kamimura/projects/tennis-lab/data/blcs_broadcast
    data.chunk.chunks_dir=/home/kamimura/projects/tennis-lab/data/blcs_broadcast/chunks
    camera=broadcast data.batch_size=8 data.num_workers=4 data.num_views_range=[1,1]
    data.seq_len_range=[64,256] data.chunk.generation_workers=8 data.chunk.epochs_per_chunk=20
    training.position_axis_weights=[1.0,4.0,1.0] training.reprojection_loss_weight=0.1
    training.trainer.max_epochs=50 training.trainer.check_val_every_n_epoch=5 training.qualitative_logging.enabled=false
    training.early_stopping.enabled=false run.output_dir=outputs/blcs/issue647_line_cnn_tok4_noaug_50ep_r2
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i647-blcs-line-cnn-tok4-noaug-50ep-r2
  predictions: knowledge/runs/run-i647-blcs-line-cnn-tok4-noaug-50ep-r2/pred_test.npz
  log: .training_queue/logs/1784098371787594066_2377891_i647_blcs_line_cnn_tok4_noaug_50ep_r2.log
  output_dir: outputs/blcs/issue647_line_cnn_tok4_noaug_50ep_r2/logs/version_0
  curves: knowledge/runs/run-i647-blcs-line-cnn-tok4-noaug-50ep-r2/curves.png
  tb_logdir: outputs/blcs/issue647_line_cnn_tok4_noaug_50ep_r2/logs/version_0
parents:
- run-i647-blcs-line-noaug-50ep
relations:
- to: run-i647-blcs-line-cnn-tok1-noaug-50ep-r2
  rel: compares
- to: run-i647-blcs-line-cnn-tok16-noaug-50ep-r2
  rel: compares
- to: run-i647-blcs-kp-small-100ep
  rel: compares
tags:
- blcs
- court-line
- cnn
- line-map
- four-token
- no-augmentation
- negative-result
---

## 考察 / Findings

### 要約

clean binary court-line mapを軽量CNNで`2 x 2`の4 court tokenへ圧縮した。test位置誤差は`8.476m`、endpoint誤差は`10.842m`で1 tokenよりわずかに良いが、位置精度`1.2m`以内は`1.84%`に留まり、空間tokenを4個へ増やしても学習は成立しなかった。

### アーキテクチャ詳細

1 token条件と同じ`multiview_axial_line_small`、`160 x 90` line map、3段depthwise-separable CNN、broadcast single-view、50 epochを用い、唯一の主要差分を`model.num_line_map_tokens=4`とした。CNN feature mapを決定論的な`2 x 2`領域で平均poolし、row-majorの4 court tokenとしてball tokenとともにaxial attentionへ入力する。ball観測とcourt-line augmentationはいずれも無効である。

### メトリクスの解釈

test `mean_position_error_m=8.476`、`mean_endpoint_error_m=10.842`、`position_accuracy_1_2m=1.84%`である。validation位置誤差は約`8.9m`、lossは約`0.53`で頭打ちとなり、train値の大きな変動に対してvalidationの改善は限定的だった。

### アーキテクチャ⇄メトリクスの因果考察

`2 x 2`分割で粗い空間位置を保持しても、1 token比の改善は小さい。court lineの投影幾何は線の傾き・交点・消失方向など領域境界をまたぐ関係に依存するため、cellごとの平均特徴と同一type IDだけでは対応関係を十分に符号化できない可能性がある（仮説）。

### 既存実験との比較

1 token [[run-i647-blcs-line-cnn-tok1-noaug-50ep-r2]] の位置`8.543m` / endpoint`11.040m`に対し、4 tokenは`0.067m` / `0.197m`だけ改善した。RANSAC no-augmentation [[run-i647-blcs-line-noaug-50ep]] の`8.606m` / `11.583m`よりは良いが、KP対照 [[run-i647-blcs-kp-small-100ep]] の`1.828m` / `3.046m`との差はほぼ埋まっていない。

### 次に有効な実験

`4 x 4`の16 token条件で、さらに細かな空間分割が位置誤差を一貫して改善するか確認する。それでもKPとの差が残る場合は、token数の追加ではなく、明示的な2D位置encodingまたは少量データoverfit試験を優先する。
