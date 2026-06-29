---
id: run-i576-camtoken-posw8
type: run
title: i576_camtoken_posw8
issue: 576
provider: claude
session: 163b1591-40c5-4551-8b36-29e97a8931bc
date: '2026-06-28'
status: done
config:
  model: multiview_axial_camtoken
  loss: canonical_rot
  data: chunked_multiview_sequence_bs8
metrics:
  position_error_m: 0.312978
  position_error_std_m: 0.255575
  position_error_median_m: 0.248927
  angular_error_deg: 8.622585
  angular_error_std_deg: 7.563326
  angular_error_median_deg: 6.976696
  x_error_m: 0.131702
  y_error_m: 0.257414
  z_error_m: 0.042678
  position_accuracy: 0.830525
  angle_accuracy: 0.840834
  position_accuracy_0.5m: 0.830525
  position_accuracy_1m: 0.977792
  position_accuracy_2m: 1.0
  angle_accuracy_10deg: 0.664585
  angle_accuracy_15deg: 0.840834
  angle_accuracy_30deg: 0.98009
repro:
  commit: 27c9c73053baedffd2e2152520cf35443d170c57
  branch: exp/issue-576-camera-token-split
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_camtoken data.batch_size=8
    training.trainer.accumulate_grad_batches=1 data=chunked_multiview_sequence_bs8
    data.seq_len_range=[64,256] loss=canonical_rot training.trainer.max_epochs=200
    training.early_stopping.enabled=false training.trainer.check_val_every_n_epoch=10
    run.gpus=1 loss.position_weight=8.0
artifacts:
  run_dir: knowledge/runs/run-i576-camtoken-posw8
  predictions: knowledge/runs/run-i576-camtoken-posw8/pred_test.npz
  log: .training_queue/logs/1782618120727989065_75059_i576_camtoken_posw8.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/plcs/plcs_multiview_axial_camtoken/logs/version_2
  curves: knowledge/runs/run-i576-camtoken-posw8/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial_camtoken/logs/version_2
parents:
- run-i576-camtoken-s0-h12
relations:
- to: run-i576-camtoken-posw4
  rel: compares
- to: run-i545-s6-h0-auxoff-posw8
  rel: compares
tags:
- plcs
- shared-trunk
- readout-split
- camtoken
- position-weight
- chunked
---

## 考察 / Findings

### 要約
camtoken に position_weight=8 を掛けると、posw=1 baseline（[[run-i576-camtoken-s0-h12]]）から**位置 0.564→0.313m（-0.25m, ≈45%）・回転 10.18→8.62°**。位置は posw4（0.353m）よりさらに改善し camtoken 系で最良、回転は posw4（8.45°）よりわずかに悪化（8.62°）。共有 trunk + readout 分離でも posw 増で位置 0.31m・回転 8.6° の同時達成が可能で、separate-trunk（0.166m/8.46°）に回転は並び位置は ~0.15m 差まで接近。

### アーキテクチャ詳細
`multiview_axial_camtoken`（fully-shared S=0/H=12, pose←cam0 / rot←cam1）。baseline からの差分は `loss.position_weight=8.0` のみ。

### メトリクスの解釈
位置 mean 0.313m / median 0.249m、回転 mean 8.62° / median 6.98°。位置@0.5m=0.831（camtoken 系最高）、角@15°=0.841。x 0.132m / y 0.257m / z 0.043m。posw4 比で y 0.294→0.257m と長手方向がさらに縮み位置改善を牽引、一方回転 median は 6.02→6.98° と微増。

### アーキテクチャ⇄メトリクスの因果考察
posw を 4→8 に上げると容量配分がさらに position 側へ寄り位置が伸びる一方、rotation への配分が削られ回転が頭打ち〜微増に転じた、と解釈する（仮説）。位置改善幅は posw4→posw8 で 0.353→0.313m（-0.04m）と posw1→posw4 の -0.21m より縮小しており、**位置のリターンは逓減**。回転とのトレードオフ点は posw4〜8 の間にある。

### 既存実験との比較
- **camtoken posw1 / posw4 [[run-i576-camtoken-s0-h12]] / [[run-i576-camtoken-posw4]]**: 0.564m/10.18° → 0.353m/8.45° → (posw8) 0.313m/8.62°。位置は単調改善、回転は posw4 が底。
- **separate-trunk posw8 [[run-i545-s6-h0-auxoff-posw8]]** 0.166m/8.46°。→ 回転は同等、位置は依然 ~0.15m 差。trunk 分離の位置優位は posw でも完全には埋まらない。

### 次に有効な実験
- 位置最優先なら posw>8 の飽和点探索だが逓減のためリターンは小。
- 残る位置 gap は trunk 表現の競合由来と見られるため、readout 分離 × 浅い共有（S5/H2）や separate-trunk への readout 分離併用で 0.2m 切りを狙うのが本筋。
- 回転重視用途では posw4 が camtoken 系のスイートスポット（8.45°/0.353m）。
