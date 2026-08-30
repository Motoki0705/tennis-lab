---
id: run-i801-dref-pose-beta005-cardw010-s42-r1
type: run
title: PLCS tracking + pose soft cardinality 0.1（seed 42）
issue: 801
provider: codex
session: 01a04915-27a4-7b62-9f6b-34275561fded
date: '2026-08-30'
status: done
config:
  model: track_query_ablation_d_v2_selector
  architecture: track_query_ablation_d
  loss: tracking_all_outputs_beta01_reprojection
  data: plcs/multi_object_camera_view_v2
  rotation_weight: 0.05
  angle_weight: 0.05
  canonical_pose_weight: 1.0
  reprojection_weight: 1.0
  presence_inactive_weight: 0.25
  match_presence_inactive_weight: 0.25
  cardinality_weight: 0.1
  seed: 42
  seq_len: 128
  num_views: 6
  batch_size: 8
  accumulate_grad_batches: 4
  epochs: 40
  warmup_steps: 400
  precision: bf16-mixed
  cswa_backend: cuda
metrics:
  loss: 1.055286
  loss_position: 0.16348
  loss_rotation: 0.285918
  loss_presence: 0.612155
  loss_track_smoothness: 0.0
  loss_angle: 0.307863
  loss_canonical_pose: 0.011119
  loss_reprojection: 0.117207
  loss_cardinality: 1.216362
  position_error: 0.471408
  presence_precision: 0.496731
  presence_recall: 0.998325
  presence_f1: 0.658805
  lifecycle_presence_f1: 0.658805
  birth_frame_error: 13.342726
  death_frame_error: 15.840462
  query_reuse_count: 0.16
  illegal_overlap_count: 0.0
  segment_id_switches: 45.48
  id_switches: 45.48
  duplicate_active_tracks: 264.720001
  missed_gt_frames: 2.48
  inactive_query_false_positives: 1409.599976
  angular_error_deg: 37.01741
  heading_error_deg: 37.080002
  position_error_m: 5.251515
  x_error_m: 1.960625
  y_error_m: 4.3925
  z_error_m: 0.374766
  y_sign_accuracy: 0.672344
  reference_index_0_position_error_m: 7.277138
  reference_index_1_position_error_m: 5.645833
  reference_index_2_position_error_m: 4.652344
  reference_index_3_position_error_m: 5.611607
  reference_index_4_position_error_m: 5.453125
  canonical_mpjpe_m: 0.185483
  world_mpjpe_m: 5.262976
  reprojection_error_px: 182.428665
  behind_camera_fraction: 0.000221
  reference_index_5_position_error_m: 5.666667
repro:
  commit: 9ffae3d34d34fe93b41a4f7e64e60238b3254cd4
  branch: experiments/plcs-tracking-pose-beta005
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.plcs.scripts.train --config-name train_tracking_pose model=track_query_ablation_d_v2_selector
    court_keypoints=camera_view_v2 model.cswa.backend=cuda paths.data_root=/home/kamimura/projects/tennis-lab/data
    data.scene_dir=plcs/multi_object_camera_view_v2 data.seq_len_range=\[128,128\]
    data.num_views_range=\[6,6\] data.batch_size=8 data.num_workers=16 data.evaluation_reference_camera_id=camera_2
    loss.cardinality_weight=0.1 training.compile.enabled=false training.trainer.precision=bf16-mixed
    training.trainer.accumulate_grad_batches=4 training.trainer.max_epochs=40 training.warmup_steps=400
    training.trainer.check_val_every_n_epoch=5 training.trainer.enable_progress_bar=false
    training.trainer.enable_model_summary=false training.early_stopping.enabled=false
    training.qualitative_logging.enabled=false run.seed=42 run.fast_dev_run=false
    run.test_after_fit=true run.output_dir=plcs/i801_dref_pose_beta005_cardw010_s42_r1
artifacts:
  run_dir: knowledge/runs/run-i801-dref-pose-beta005-cardw010-s42-r1
  predictions: knowledge/runs/run-i801-dref-pose-beta005-cardw010-s42-r1/pred_test.npz
  output_dir: outputs/plcs/i801_dref_pose_beta005_cardw010_s42_r1/logs/version_0
  curves: knowledge/runs/run-i801-dref-pose-beta005-cardw010-s42-r1/curves.png
  tb_logdir: outputs/plcs/i801_dref_pose_beta005_cardw010_s42_r1/logs/version_0
parents:
- run-i801-dref-pose-beta005-s42-r1
relations:
- to: run-i801-dref-pose-beta005-cardw025-s42-r1
  rel: compares
- to: run-i801-dref-pose-beta005-inact1-match025-s42-r1
  rel: compares
tags:
- plcs
- tracking
- canonical-pose
- reprojection
- camera-view-v2
- cardinality
- soft-count
- beta005
- seed-42
---

## 考察 / Findings

### 要約

各 frame の query presence probability の総和を GT active 人数へ近づける soft cardinality loss を
`cardinality_weight=0.1` で追加した。親 run に対して duplicate は `357.60→264.72` と減ったが、
F1、inactive-query false positive、position、angle、canonical pose、reprojection は悪化した。
予測 soft active 数も GT `1.6593` に対して `3.3321` のままで、人数過剰は解消していない。

### アーキテクチャ詳細

`track_query_ablation_d_v2_selector` の各 query から position、rotation、presence、17-joint canonical
pose を予測し、assignment 後の active target に canonical pose と multi-view reprojection loss を適用する。
本 run では既存の per-query presence BCE に加え、valid frame ごとに
`sum_q(sigmoid(presence_logit_q))` と GT active 人数の Smooth L1 を計算する assignment 非依存の
soft cardinality loss を追加した。`rotation_weight=angle_weight=0.05`、canonical / reprojection weight
`1.0`、presence / matching inactive weight `0.25`、camera-view-v2、T=128、V=6、effective batch=32、
seed 42、CUDA CSWA は比較条件と共通である。40 epoch、warmup 400 step で学習した。

### メトリクスの解釈

test precision / recall / F1 は `0.4967 / 0.9983 / 0.6588`、ID switch `45.48`、duplicate
`264.72`、inactive-query false positive `1409.60`、missed GT frame `2.48` だった。recall はほぼ
1.0 だが precision は約 0.50 であり、不要 query の過剰発火が残っている。test prediction から
frame 平均を算出すると、GT active 数 `1.6593` に対して soft count は `3.3321`、threshold 0.5 の
hard count は `3.3498` だった。

pose は position `5.2515m`、angular `37.0174deg`、heading `37.0800deg`、canonical MPJPE
`0.1855m`、reprojection `182.43px` だった。test raw cardinality loss は `1.21636`、重み込み寄与は
`0.12164` である。angle の重み込み寄与は `0.05 × 0.30786 = 0.01539` で、test total
`1.05529` の約 1.46% に留まり、angle 支配は起きていない。

収束曲線では val loss は epoch 34 の `1.03418` が最小で、epoch 39 は `1.04573`、最終 train
loss は `0.60933` だった。終盤の val 悪化は小さい一方、train / val gap があり、人数制約を追加しても
validation 上の query calibration は改善し続けていない。

### アーキテクチャ⇄メトリクスの因果考察

観測として、cardinality loss を加えても soft / hard count は親 run より減らず、inactive false positive と
F1 も改善しなかった。一方で duplicate は減ったため、`0.1` は出力軌跡の重なりの一部を変えたが、不要
query を inactive にする解には導かなかった。

以下は仮説である。総和だけを制約する loss は query ごとの役割を特定せず、複数 query が中間的または高い
probability を共有する解も許す。このため per-query assignment / BCE と勾配が競合し、人数誤差を下げるより
位置・pose 表現を悪化させた可能性がある。ただし親 run は 100 epoch、本 run は 40 epoch なので、親との差を
cardinality loss だけの因果効果とは断定できない。

### 既存実験との比較

親 `run-i801-dref-pose-beta005-s42-r1` に対し、ID switch は `46.24→45.48`、duplicate は
`357.60→264.72`、missed は `15.36→2.48` と改善した。一方、F1 は `0.6639→0.6588`、inactive
false positive は `1361.92→1409.60`、position は `5.0050→5.2515m`、angular は
`33.4878→37.0174deg`、canonical MPJPE は `0.1796→0.1855m`、reprojection は
`160.50→182.43px` と悪化した。soft count も `3.1777→3.3321` で、総合改善ではない。

直接比較できる `run-i801-dref-pose-beta005-cardw025-s42-r1` は weight 以外を揃えた run である。
`0.1→0.25` では soft count `3.3321→3.2800` と inactive false positive
`1409.60→1384.00` は僅かに減ったが、ID switch `45.48→60.08`、duplicate
`264.72→369.60`、missed `2.48→10.40`、position `5.2515→5.3116m`、canonical MPJPE
`0.1855→0.1942m` は悪化したため、有望な dose response はない。

`run-i801-dref-pose-beta005-inact1-match025-s42-r1` に対しては、F1
`0.6449→0.6588`、duplicate `352.12→264.72`、inactive false positive
`1483.20→1409.60` は改善したが、ID switch は `28.96→45.48` と悪化した。独立 BCE 増量と
soft cardinality のどちらも tracking と pose を同時には改善していない。

### 次に有効な実験

soft-count weight の追加増量は止め、query ごとの不要発火へ直接勾配を与える施策を検証する。候補は matched / unmatched
query を分けた presence supervision、または top-k / set-level な人数制約である。次の run では GT 人数別の
soft / hard count 分布、inactive-query false positive、duplicate、ID switch、missed と pose 指標を同時に保存し、
人数平均だけを下げて recall や pose を損なう解を採用しない。
