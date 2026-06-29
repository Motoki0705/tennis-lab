---
id: run-i576-camtoken-posw4
type: run
title: i576_camtoken_posw4
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
  position_error_m: 0.352758
  position_error_std_m: 0.257672
  position_error_median_m: 0.294512
  angular_error_deg: 8.453344
  angular_error_std_deg: 9.006538
  angular_error_median_deg: 6.015355
  x_error_m: 0.130389
  y_error_m: 0.294313
  z_error_m: 0.047071
  position_accuracy: 0.789408
  angle_accuracy: 0.853533
  position_accuracy_0.5m: 0.789408
  position_accuracy_1m: 0.974903
  position_accuracy_2m: 0.998703
  angle_accuracy_10deg: 0.705826
  angle_accuracy_15deg: 0.853533
  angle_accuracy_30deg: 0.974057
repro:
  commit: 27c9c73053baedffd2e2152520cf35443d170c57
  branch: exp/issue-576-camera-token-split
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: MPLBACKEND=Agg PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python
    -m src.tasks.plcs.scripts.train model=multiview_axial_camtoken data.batch_size=8
    training.trainer.accumulate_grad_batches=1 data=chunked_multiview_sequence_bs8
    data.seq_len_range=[64,256] loss=canonical_rot training.trainer.max_epochs=200
    training.early_stopping.enabled=false training.trainer.check_val_every_n_epoch=10
    run.gpus=1 loss.position_weight=4.0
artifacts:
  run_dir: knowledge/runs/run-i576-camtoken-posw4
  predictions: knowledge/runs/run-i576-camtoken-posw4/pred_test.npz
  log: .training_queue/logs/1782618120707958414_75044_i576_camtoken_posw4.log
  output_dir: /home/kamimura/projects/tennis-lab/outputs/plcs/plcs_multiview_axial_camtoken/logs/version_1
  curves: knowledge/runs/run-i576-camtoken-posw4/curves.png
  tb_logdir: outputs/plcs/plcs_multiview_axial_camtoken/logs/version_1
parents:
- run-i576-camtoken-s0-h12
relations:
- to: run-i576-camtoken-posw8
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
camtoken（共有 trunk + readout 分離）に position_weight=4 を掛けると、posw=1 baseline（[[run-i576-camtoken-s0-h12]]）から**位置 0.564→0.353m（-0.21m, ≈37%）・回転 10.18→8.45° と両方大幅改善**。回転は separate-trunk 並み（~8.5°）に到達し、位置も separate-trunk posw8（0.166m）との差を 3 倍→2 倍弱に縮めた。安価な共有 trunk でも position_weight で位置⇔回転を同時に押せる。

### アーキテクチャ詳細
`multiview_axial_camtoken`（fully-shared S=0/H=12, hidden=512/layers=12/heads=8, pose←cam0 / rot←cam1 の readout 分離）。baseline からの差分は `loss.position_weight=4.0` のみ（他は canonical_rot / chunked_multiview_sequence_bs8 / batch=8 / seq_len[64,256] / 200ep / early-stop OFF と同一）。

### メトリクスの解釈
位置 mean 0.353m / median 0.295m、回転 mean 8.45° / median 6.02°。位置@0.5m=0.789（baseline 0.542 から大幅向上）、角@15°=0.854。x 0.130m / y 0.294m / z 0.047m と y（court 長手方向）誤差が支配的なのは baseline と同傾向だが全軸で縮小。回転 median 6.02° は #545+#560 でも上位。

### アーキテクチャ⇄メトリクスの因果考察
position_weight を上げると共有 trunk の容量配分が position 側に寄り、readout 分離で rotation が別トークンに逃げているぶん回転を犠牲にせず位置だけ伸ばせた、と解釈する（仮説）。回転がむしろ baseline より改善したのは、posw 増で position が安定し共有表現の競合が緩和された副次効果の可能性。

### 既存実験との比較
- **camtoken posw=1 [[run-i576-camtoken-s0-h12]]** 0.564m / 10.18° → posw4 で 0.353m / 8.45°。位置・回転とも改善。
- **separate-trunk posw8 [[run-i545-s6-h0-auxoff-posw8]]** 0.166m / 8.46°。→ 回転はほぼ同等（8.45 vs 8.46）だが位置は依然 ~0.19m 差。
- 安価枠（trunk 1 本）で回転は separate-trunk に並び、位置の gap のみ残る構図。

### 次に有効な実験
- posw8（[[run-i576-camtoken-posw8]]）と対で位置の伸びしろ／回転トレードオフを確認（→ posw8 は位置 0.313m まで改善、回転 8.62° で微増）。
- 位置をさらに詰めるなら readout 分離を浅い共有（S5/H2）に重ねる、もしくは posw>8 の飽和点探索。
