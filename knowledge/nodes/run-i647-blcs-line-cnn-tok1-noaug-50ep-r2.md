---
id: run-i647-blcs-line-cnn-tok1-noaug-50ep-r2
type: run
title: BLCS CNN line map 1 token・no augmentation・50 epoch
issue: 647
provider: codex
session: 019f6013-0a29-7183-87ae-e4c221387139
date: '2026-07-15'
status: done
config:
  model: multiview_axial_line_small
  data: chunked_multiview_sequence_line_bs4
metrics:
  mean_position_error_m: 8.543202
  mean_x_error_m: 3.373078
  mean_y_error_m: 7.256043
  mean_z_error_m: 0.414404
  mean_endpoint_error_m: 11.039667
  position_accuracy_0_3m: 0.000625
  position_accuracy_0_6m: 0.003748
  position_accuracy_1_2m: 0.020381
  endpoint_accuracy_0_5m: 0.0
  endpoint_accuracy_1m: 0.0
repro:
  commit: 46474cb82a1d2de36f20272d668385ab3bcbd41f
  branch: exp/issue-647-cnn-line-train
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.blcs.scripts.train model=multiview_axial_line_small data=chunked_multiview_sequence_line_bs4
    model.num_line_map_tokens=1 data.scene_dir=/home/kamimura/projects/tennis-lab/data/blcs_broadcast
    data.chunk.chunks_dir=/home/kamimura/projects/tennis-lab/data/blcs_broadcast/chunks
    camera=broadcast data.batch_size=8 data.num_workers=4 data.num_views_range=[1,1]
    data.seq_len_range=[64,256] data.chunk.generation_workers=8 data.chunk.epochs_per_chunk=20
    training.position_axis_weights=[1.0,4.0,1.0] training.reprojection_loss_weight=0.1
    training.trainer.max_epochs=50 training.trainer.check_val_every_n_epoch=5 training.qualitative_logging.enabled=false
    training.early_stopping.enabled=false run.output_dir=outputs/blcs/issue647_line_cnn_tok1_noaug_50ep_r2
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i647-blcs-line-cnn-tok1-noaug-50ep-r2
  predictions: knowledge/runs/run-i647-blcs-line-cnn-tok1-noaug-50ep-r2/pred_test.npz
  log: .training_queue/logs/1784098371767570988_2377876_i647_blcs_line_cnn_tok1_noaug_50ep_r2.log
  output_dir: outputs/blcs/issue647_line_cnn_tok1_noaug_50ep_r2/logs/version_0
  curves: knowledge/runs/run-i647-blcs-line-cnn-tok1-noaug-50ep-r2/curves.png
  tb_logdir: outputs/blcs/issue647_line_cnn_tok1_noaug_50ep_r2/logs/version_0
parents:
- run-i647-blcs-line-noaug-50ep
relations:
- to: run-i647-blcs-line-cnn-tok4-noaug-50ep-r2
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
- one-token
- no-augmentation
- negative-result
---

## 考察 / Findings

### 要約

RANSAC線分への離散化を廃止し、clean binary court-line mapを軽量CNNで直接1 court tokenへ圧縮した。test位置誤差は`8.543m`、endpoint誤差は`11.040m`で、RANSAC no-augmentationより小幅に改善したものの、3D軌道は学習できなかった。

### アーキテクチャ詳細

`multiview_axial_line_small`と`chunked_multiview_sequence_line_bs4`を使用した。`160 x 90`のline mapを3段のstride-2 depthwise-separable CNN（channels `16,32,64`）へ通し、全空間を決定論的に平均poolして1 tokenへ射影する。camera-time入力はball token 1個とcourt token 1個で、broadcast single-view、50 epoch、ball観測とcourt-lineのaugmentationはいずれも無効である。

### メトリクスの解釈

test `mean_position_error_m=8.543`、`mean_endpoint_error_m=11.040`、`position_accuracy_1_2m=2.04%`であり、実用的な復元精度ではない。収束曲線ではvalidation位置誤差が約`9m`でほぼ横ばいとなり、lossも約`0.57`で頭打ちになったため、50 epoch内で有効なcourt geometryを獲得した兆候は弱い。

### アーキテクチャ⇄メトリクスの因果考察

RANSACの抽出順・線分端点の不安定性を除いても性能が崩壊したため、それだけを失敗原因とは説明できない。1 tokenへのglobal poolingが線の絶対位置と局所的な交差関係を強く圧縮し、単眼ball UVを世界座標へ写すための射影幾何を保持できなかった可能性がある（仮説）。

### 既存実験との比較

親のRANSAC no-augmentation [[run-i647-blcs-line-noaug-50ep]] は位置`8.606m` / endpoint`11.583m`であり、本runはそれぞれ`0.063m` / `0.543m`改善したが、差は小さい。KP対照 [[run-i647-blcs-kp-small-100ep]] の`1.828m` / `3.046m`には大きく及ばない。4/16 token条件との比較は [[group-i647-blcs-line-cnn-token-ablation]] にまとめる。

### 次に有効な実験

4/16 tokenで空間分割を保持した条件と比較し、global poolingだけがボトルネックかを切り分ける。token数を増やしても改善しない場合は、少量データへのoverfit試験でline-map経路が幾何を表現可能かを先に検証し、学習可能性と汎化の問題を分離する。
