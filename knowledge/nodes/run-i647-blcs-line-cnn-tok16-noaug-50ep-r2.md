---
id: run-i647-blcs-line-cnn-tok16-noaug-50ep-r2
type: run
title: BLCS CNN line map 16 token・no augmentation・50 epoch
issue: 647
provider: codex
session: 019f6013-0a29-7183-87ae-e4c221387139
date: '2026-07-15'
status: done
config:
  model: multiview_axial_line_small
  data: chunked_multiview_sequence_line_bs4
metrics:
  mean_position_error_m: 8.322783
  mean_x_error_m: 3.330198
  mean_y_error_m: 7.064492
  mean_z_error_m: 0.423708
  mean_endpoint_error_m: 12.262795
  position_accuracy_0_3m: 0.000838
  position_accuracy_0_6m: 0.003949
  position_accuracy_1_2m: 0.018609
  endpoint_accuracy_0_5m: 0.0
  endpoint_accuracy_1m: 0.0
repro:
  commit: 46474cb82a1d2de36f20272d668385ab3bcbd41f
  branch: exp/issue-647-cnn-line-train
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.blcs.scripts.train model=multiview_axial_line_small data=chunked_multiview_sequence_line_bs4
    model.num_line_map_tokens=16 data.scene_dir=/home/kamimura/projects/tennis-lab/data/blcs_broadcast
    data.chunk.chunks_dir=/home/kamimura/projects/tennis-lab/data/blcs_broadcast/chunks
    camera=broadcast data.batch_size=8 data.num_workers=4 data.num_views_range=[1,1]
    data.seq_len_range=[64,256] data.chunk.generation_workers=8 data.chunk.epochs_per_chunk=20
    training.position_axis_weights=[1.0,4.0,1.0] training.reprojection_loss_weight=0.1
    training.trainer.max_epochs=50 training.trainer.check_val_every_n_epoch=5 training.qualitative_logging.enabled=false
    training.early_stopping.enabled=false run.output_dir=outputs/blcs/issue647_line_cnn_tok16_noaug_50ep_r2
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i647-blcs-line-cnn-tok16-noaug-50ep-r2
  predictions: knowledge/runs/run-i647-blcs-line-cnn-tok16-noaug-50ep-r2/pred_test.npz
  log: .training_queue/logs/1784098371808517307_2377906_i647_blcs_line_cnn_tok16_noaug_50ep_r2.log
  output_dir: outputs/blcs/issue647_line_cnn_tok16_noaug_50ep_r2/logs/version_0
  curves: knowledge/runs/run-i647-blcs-line-cnn-tok16-noaug-50ep-r2/curves.png
  tb_logdir: outputs/blcs/issue647_line_cnn_tok16_noaug_50ep_r2/logs/version_0
parents:
- run-i647-blcs-line-noaug-50ep
relations:
- to: run-i647-blcs-line-cnn-tok1-noaug-50ep-r2
  rel: compares
- to: run-i647-blcs-line-cnn-tok4-noaug-50ep-r2
  rel: compares
- to: run-i647-blcs-kp-small-100ep
  rel: compares
tags:
- blcs
- court-line
- cnn
- line-map
- sixteen-token
- no-augmentation
- negative-result
---

## 考察 / Findings

### 要約

clean binary court-line mapを軽量CNNで`4 x 4`の16 court tokenへ圧縮した。3条件中の最良test位置誤差`8.323m`を得たが、endpoint誤差は最悪の`12.263m`であり、空間token増加による一貫した改善は確認できなかった。

### アーキテクチャ詳細

1/4 token条件と同じ`multiview_axial_line_small`、`160 x 90` line map、3段depthwise-separable CNN、broadcast single-view、50 epochを用い、`model.num_line_map_tokens=16`のみを主要差分とした。CNN feature mapを決定論的な`4 x 4`領域で平均poolし、row-majorの16 court tokenとしてball tokenとともにaxial attentionへ入力する。ball観測とcourt-line augmentationはいずれも無効である。

### メトリクスの解釈

test `mean_position_error_m=8.323`、`mean_endpoint_error_m=12.263`、`position_accuracy_1_2m=1.86%`である。位置誤差は3条件で最小だが、endpoint accuracyは`0.5m`・`1m`とも`0%`だった。validation位置誤差は約`9.1m`で横ばい、lossも約`0.57`で頭打ちとなった。

### アーキテクチャ⇄メトリクスの因果考察

細かなgrid tokenが平均位置誤差をわずかに下げた一方、軌道端点の安定性は悪化した。局所line特徴を増やすだけでは各tokenの明示的なgrid位置がモデルへ与えられず、同じcourt type IDを持つtoken間の空間対応を安定して利用できていない可能性がある（仮説）。ただし単一seedの小差なので、位置改善をtoken数の因果効果と断定できない。

### 既存実験との比較

1 token [[run-i647-blcs-line-cnn-tok1-noaug-50ep-r2]]、4 token [[run-i647-blcs-line-cnn-tok4-noaug-50ep-r2]] の位置`8.543m` / `8.476m`に対し、本runは`8.323m`だった。一方endpointは`11.040m` / `10.842m`から`12.263m`へ悪化した。RANSAC no-augmentation [[run-i647-blcs-line-noaug-50ep]] より位置は`0.283m`良いが、KP対照 [[run-i647-blcs-kp-small-100ep]] の`1.828m`には遠い。

### 次に有効な実験

現構成でtoken数をさらに増やす優先度は低い。まず小規模な固定sampleへのoverfitでline mapから正しい軌道を記憶できるか検証し、表現能力を確認する。表現可能ならgrid座標encodingを各court tokenへ付与するablation、表現不能ならhomographyなど明示的な幾何推定との統合を検討する。
