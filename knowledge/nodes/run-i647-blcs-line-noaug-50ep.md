---
id: run-i647-blcs-line-noaug-50ep
type: run
title: i647_blcs_line_noaug_50ep
issue: 647
provider: codex
session: 019f6013-0a29-7183-87ae-e4c221387139
date: '2026-07-15'
status: done
config:
  model: multiview_axial_line_small
  data: chunked_multiview_sequence_line_bs4
metrics:
  mean_position_error_m: 8.606204
  mean_x_error_m: 3.393909
  mean_y_error_m: 7.281614
  mean_z_error_m: 0.612032
  mean_endpoint_error_m: 11.582884
  position_accuracy_0_3m: 0.000382
  position_accuracy_0_6m: 0.002485
  position_accuracy_1_2m: 0.018225
  endpoint_accuracy_0_5m: 0.0
  endpoint_accuracy_1m: 0.02
repro:
  commit: aeb0a66b56034195e4b9a62a15b17d8d2af58be7
  branch: feat/issue-647-court-line-token
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.blcs.scripts.train model=multiview_axial_line_small data=chunked_multiview_sequence_line_bs4
    data.scene_dir=data/blcs_broadcast data.chunk.chunks_dir=data/blcs_broadcast/chunks
    camera=broadcast data.batch_size=8 data.num_workers=4 data.num_views_range=[1,1]
    data.seq_len_range=[64,256] data.chunk.generation_workers=8 data.chunk.epochs_per_chunk=20
    training.position_axis_weights=[1.0,4.0,1.0] training.reprojection_loss_weight=0.1
    training.trainer.max_epochs=50 training.trainer.check_val_every_n_epoch=5 training.qualitative_logging.enabled=false
    training.early_stopping.enabled=false data.court_line.augmentation.enabled=false
    run.output_dir=outputs/blcs/issue647_line_noaug_50ep run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-i647-blcs-line-noaug-50ep
  predictions: knowledge/runs/run-i647-blcs-line-noaug-50ep/pred_test.npz
  log: .training_queue/logs/1784091243559411910_2303578_i647_blcs_line_noaug_50ep.log
  output_dir: outputs/blcs/issue647_line_noaug_50ep/logs/version_0
  curves: knowledge/runs/run-i647-blcs-line-noaug-50ep/curves.png
  tb_logdir: outputs/blcs/issue647_line_noaug_50ep/logs/version_0
parents:
- run-i647-blcs-line-small-100ep
relations:
- to: run-i647-blcs-line-moderate-v2-100ep
  rel: compares
- to: run-i647-blcs-kp-small-100ep
  rel: compares
tags:
- blcs
- court-line
- ransac
- no-line-augmentation
- negative-result
---

## 考察 / Findings

### 要約

court-line map augmentationを完全に無効化し、1 px固定のclean mapからRANSAC線分を抽出して50 epoch学習した。test位置誤差は`8.606m`、endpoint誤差は`11.583m`で、増強を除いてもline入力の学習は成立しなかった。

### アーキテクチャ詳細

`multiview_axial_line_small`（約13.0M parameters）と`chunked_multiview_sequence_line_bs4`を使用した。各camera-timeのbinary court-line mapをiterative RANSACで最大12本の有限線分へ変換し、決定論的sort後の端点UVをflatten MLPで1 court tokenへ圧縮する。`data.court_line.augmentation.enabled=false`により、線幅変動、部分欠損、遮蔽、false-positive、blur、morphology、far dropout、near-onlyをすべて停止した。ball観測augmentationは既存比較条件どおり有効である。

### メトリクスの解釈

test `mean_position_error_m=8.606`、`mean_endpoint_error_m=11.583`、`position_accuracy_1_2m=1.82%`であり、実用的な3D軌道を復元できていない。50 epoch終了時もvalidation位置誤差は約`9.19m`で頭打ちとなり、単なる学習時間不足ではなく表現上の失敗を示す。

### アーキテクチャ⇄メトリクスの因果考察

previewでは同一のclean binary mapからでもRANSAC seedにより抽出本数と端点が変化した。観測された線のsemantic identityを持たないまま、確率的な部分線分集合をsort + flatten位置へ割り当てるため、court tokenの各入力次元が安定した幾何を表さないことが主要因という仮説を支持する。line map自体をCNNで直接圧縮すれば、この離散化と順序不安定性を回避できる。

### 既存実験との比較

strong増強の[[run-i647-blcs-line-small-100ep]]は位置`8.144m` / endpoint`9.905m`、moderate増強の[[run-i647-blcs-line-moderate-v2-100ep]]は`8.427m` / `9.555m`だった。no-augmentationは`8.606m` / `11.583m`でいずれも改善せず、KP対照[[run-i647-blcs-kp-small-100ep]]の`1.828m` / `3.046m`から大きく離れる。したがって、line-map劣化だけを失敗原因とは説明できない。

### 次に有効な実験

RANSACと固定順線分flattenを廃し、clean binary line mapを軽量CNNで直接1 court tokenへ圧縮するBLCSを同じbroadcast single-view条件で学習する。最初のablationではcourt-line mapとball観測の両augmentationを無効化し、encoder自体が幾何を学習できる上限を測る。
