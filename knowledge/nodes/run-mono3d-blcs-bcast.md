---
id: run-mono3d-blcs-bcast
type: run
title: mono3d_blcs_bcast
issue: 593
provider: claude
session: dd9d56e0-5f2d-4d9d-a8c7-c80704a806b3
date: '2026-07-05'
status: done
config:
  model: multiview_axial_base
  loss: trajectory_position+reprojection0.1
  data: chunked_multiview_sequence_bs4 (C=1, broadcast, court_kp=14)
metrics:
  mean_position_error_m: 2.986075
  mean_x_error_m: 0.811207
  mean_y_error_m: 2.640812
  mean_z_error_m: 0.520961
  mean_endpoint_error_m: 4.543103
  position_accuracy_0_3m: 0.016452
  position_accuracy_0_6m: 0.074495
  position_accuracy_1_2m: 0.235489
  endpoint_accuracy_0_5m: 0.03
  endpoint_accuracy_1m: 0.05
repro:
  commit: ef76d3d60d237ae9cceafeda5411f51757b4f81d
  branch: main
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.blcs.scripts.train --config-name train_chunked data.scene_dir=data/blcs_broadcast
    data.chunk.chunks_dir=data/blcs_broadcast/chunks data.num_views_range=[1,1] data.camera_mode=random
    data.num_court_kp=14 data.num_workers=2 data.chunk.generation_workers=8 camera=broadcast
    run.gpus=1
artifacts:
  run_dir: knowledge/runs/run-mono3d-blcs-bcast
  predictions: knowledge/runs/run-mono3d-blcs-bcast/pred_test.npz
  log: .training_queue/logs/1783207757902116131_7669_mono3d_blcs_bcast.log
  output_dir: outputs/blcs/blcs_multiview_axial/logs/version_0
  curves: knowledge/runs/run-mono3d-blcs-bcast/curves.png
  tb_logdir: outputs/blcs/blcs_multiview_axial/logs/version_0
parents: []
relations: []
tags:
- blcs
- monocular
- broadcast
- court-kp-14
- depth-ambiguity
- baseline
---

## 考察 / Findings

### 要約
単眼 broadcast カメラでの BLCS 初回 run(knowledge 上初の BLCS ノード)。test **mean_position_error 2.99m** で、内訳 X 0.81 / **Y 2.64** / Z 0.52m と視線方向(コート長手 Y)の深度誤差が支配的。early stopping(val/pos_error_m, patience 10)が epoch 29 で発火した真のプラトーであり、単純な学習延長では改善しない。

### アーキテクチャ詳細
`train_chunked` 既定の `multiview_axial_base`(51.9M)を `num_views_range=[1,1]`(C=1)で学習。`camera=broadcast`(PR #592)、`data/blcs_broadcast`(1000 scenes)+ broadcast chunk 生成。court KP は 14 点契約。reprojection loss 0.1 は可視フレームのみに作用。

### メトリクスの解釈
pred_test 分析(100 scenes / 14,162 frames): Y は GT との相関 0.917 と高いが、回帰 `pred_y = 0.824*gt_y - 1.54` の**レンジ圧縮 + 系統的 -Y バイアス**(negative 75.7%、奥側 GT 上位 20% で平均 -3.89m の過小予測)。ball 不可視フレームは position 5.28m と可視 2.50m の約 2 倍悪く、metrics は不可視フレームも含んで集計している。

### アーキテクチャ⇄メトリクスの因果考察
正規化座標(COURT_COORD_SCALE: X 5.485 / Y 11.885 / Z 1.07)の Smooth L1 では、**1m あたりの損失が Y は X の約 0.21 倍**になり、Y 誤差が構造的に軽視される(機械的要因、確定)。また単眼では reprojection が視線方向の深度を拘束できない(幾何的要因、確定)。レンジ圧縮は「損失が軽い軸は平均側へ縮む」回帰の典型挙動と整合する(解釈)。

### 既存実験との比較
BLCS の既存 knowledge ノードが無いため直接比較対象なし。multiview 設定との対照は今後の課題。

### 次に有効な実験
- **`training.position_axis_weights=[1,4,1]`**(本 run の分析を受けて実装済み)で Y の損失重みをスケール二乗比相当へ補正 → v2 run で検証中。
- 効果不足なら: seq_len 延長([128,384])、reprojection weight 低減、physics/velocity 補助損失。

<!-- run `mono3d_blcs_bcast` の結果と考察を書く。parents/tags も埋め、 主要 metrics は frontmatter と一致させること。 -->
