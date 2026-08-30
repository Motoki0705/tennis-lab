---
id: run-i801-dref-pose-beta005-cardw025-s42-r1
type: run
title: PLCS tracking + pose soft cardinality 0.25（seed 42）
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
  cardinality_weight: 0.25
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
  loss: 1.221764
  loss_position: 0.167659
  loss_rotation: 0.300293
  loss_presence: 0.622204
  loss_track_smoothness: 0.0
  loss_angle: 0.323023
  loss_canonical_pose: 0.012209
  loss_reprojection: 0.097956
  loss_cardinality: 1.162284
  position_error: 0.480923
  presence_precision: 0.500156
  presence_recall: 0.993067
  presence_f1: 0.660903
  lifecycle_presence_f1: 0.660903
  birth_frame_error: 13.615099
  death_frame_error: 16.522406
  query_reuse_count: 0.16
  illegal_overlap_count: 0.0
  segment_id_switches: 60.080002
  id_switches: 60.080002
  duplicate_active_tracks: 369.600006
  missed_gt_frames: 10.4
  inactive_query_false_positives: 1384.0
  angular_error_deg: 38.032356
  heading_error_deg: 38.200001
  position_error_m: 5.311617
  x_error_m: 1.9425
  y_error_m: 4.49375
  z_error_m: 0.363203
  y_sign_accuracy: 0.674688
  reference_index_0_position_error_m: 7.710526
  reference_index_1_position_error_m: 5.710417
  reference_index_2_position_error_m: 4.660156
  reference_index_3_position_error_m: 5.904018
  reference_index_4_position_error_m: 5.4375
  canonical_mpjpe_m: 0.194177
  world_mpjpe_m: 5.323865
  reprojection_error_px: 163.846695
  behind_camera_fraction: 0.000349
  reference_index_5_position_error_m: 5.819445
repro:
  commit: 9ffae3d34d34fe93b41a4f7e64e60238b3254cd4
  branch: experiments/plcs-tracking-pose-beta005
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.plcs.scripts.train --config-name train_tracking_pose model=track_query_ablation_d_v2_selector
    court_keypoints=camera_view_v2 model.cswa.backend=cuda paths.data_root=/home/kamimura/projects/tennis-lab/data
    data.scene_dir=plcs/multi_object_camera_view_v2 data.seq_len_range=\[128,128\]
    data.num_views_range=\[6,6\] data.batch_size=8 data.num_workers=16 data.evaluation_reference_camera_id=camera_2
    loss.cardinality_weight=0.25 training.compile.enabled=false training.trainer.precision=bf16-mixed
    training.trainer.accumulate_grad_batches=4 training.trainer.max_epochs=40 training.warmup_steps=400
    training.trainer.check_val_every_n_epoch=5 training.trainer.enable_progress_bar=false
    training.trainer.enable_model_summary=false training.early_stopping.enabled=false
    training.qualitative_logging.enabled=false run.seed=42 run.fast_dev_run=false
    run.test_after_fit=true run.output_dir=plcs/i801_dref_pose_beta005_cardw025_s42_r1
artifacts:
  run_dir: knowledge/runs/run-i801-dref-pose-beta005-cardw025-s42-r1
  predictions: knowledge/runs/run-i801-dref-pose-beta005-cardw025-s42-r1/pred_test.npz
  output_dir: outputs/plcs/i801_dref_pose_beta005_cardw025_s42_r1/logs/version_0
  curves: knowledge/runs/run-i801-dref-pose-beta005-cardw025-s42-r1/curves.png
  tb_logdir: outputs/plcs/i801_dref_pose_beta005_cardw025_s42_r1/logs/version_0
parents:
- run-i801-dref-pose-beta005-s42-r1
relations:
- to: run-i801-dref-pose-beta005-cardw010-s42-r1
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

soft cardinality loss を `cardinality_weight=0.25` に増やした。`0.1` run より soft active 数と
inactive-query false positive は僅かに減ったが、ID switch、duplicate、missed、position、angle、canonical
pose は悪化した。GT active 数 `1.6593` に対する soft count はなお `3.2800` であり、人数過剰を解消せず、
tracking + pose の総合改善にはならなかった。

### アーキテクチャ詳細

`track_query_ablation_d_v2_selector` の各 query から position、rotation、presence、17-joint canonical
pose を予測し、assignment 後の active target に canonical pose と multi-view reprojection loss を適用する。
valid frame ごとの query presence probability の総和と GT active 人数に Smooth L1 を適用する
assignment 非依存の soft cardinality loss を、既存 per-query presence BCE に加えた。

`run-i801-dref-pose-beta005-cardw010-s42-r1` との差は `cardinality_weight=0.25` のみである。
`rotation_weight=angle_weight=0.05`、canonical / reprojection weight `1.0`、presence / matching inactive
weight `0.25`、camera-view-v2、T=128、V=6、effective batch=32、seed 42、CUDA CSWA、40 epoch、
warmup 400 step は同一である。

### メトリクスの解釈

test precision / recall / F1 は `0.5002 / 0.9931 / 0.6609`、ID switch `60.08`、duplicate
`369.60`、inactive-query false positive `1384.00`、missed GT frame `10.40` だった。test prediction
の frame 平均は、GT active 数 `1.6593` に対して soft count `3.2800`、threshold 0.5 の hard count
`3.3088` である。僅かな人数抑制は観測できるが、依然として約 2 倍の query が active である。

pose は position `5.3116m`、angular `38.0324deg`、heading `38.2000deg`、canonical MPJPE
`0.1942m`、reprojection `163.85px` だった。test raw cardinality loss は `1.16228`、重み込み寄与は
`0.29057` である。angle の重み込み寄与は `0.05 × 0.32302 = 0.01615` で、test total
`1.22176` の約 1.32% に留まり、angle 支配は起きていない。weight が異なるため、test total loss 自体は
`0.1` run と直接優劣比較できない。

収束曲線では val loss は epoch 24 の `1.21282` が最小で、epoch 39 は `1.22150`、最終 train
loss は `0.63340` だった。終盤はほぼ頭打ちで、train / val gap も残る。val cardinality loss は終盤も
約 `1.13` であり、学習崩壊はないが人数誤差が十分に縮んでいない。

### アーキテクチャ⇄メトリクスの因果考察

観測として、weight を `0.1→0.25` に上げると soft / hard count と inactive false positive は僅かに下がった。
しかしその変化は GT との差を埋めるには小さく、ID switch、duplicate、missed と pose は同時に悪化した。
したがって、この範囲では「soft count の圧力を強めれば tracking 全体が単調に改善する」という仮説は支持されない。

以下は仮説である。総和だけを合わせる loss はどの query を下げるべきか指定しないため、identity assignment を
安定化できず、複数 query 間に勾配を広く配ることで position / pose head と競合した可能性がある。また、
reprojection `163.85px` だけは `0.1` より良いが、canonical / world pose と tracking が悪化しているため、
再投影単独の改善を表現全体の改善とはみなせない。

### 既存実験との比較

親 `run-i801-dref-pose-beta005-s42-r1` に対し、missed は `15.36→10.40` と減り、reprojection は
`160.50→163.85px` と近い。一方、F1 は `0.6639→0.6609`、ID switch は `46.24→60.08`、
duplicate は `357.60→369.60`、inactive false positive は `1361.92→1384.00`、position は
`5.0050→5.3116m`、angular は `33.4878→38.0324deg`、canonical MPJPE は
`0.1796→0.1942m` と悪化した。soft count も親の `3.1777` より高い `3.2800` である。親は
100 epoch、本 run は 40 epoch なので、親との差を cardinality loss だけの効果とは断定しない。

直接比較の `run-i801-dref-pose-beta005-cardw010-s42-r1` に対し、soft count は
`3.3321→3.2800`、hard count は `3.3498→3.3088`、inactive false positive は
`1409.60→1384.00` と僅かに減った。一方、ID switch は `45.48→60.08`、duplicate は
`264.72→369.60`、missed は `2.48→10.40`、position は `5.2515→5.3116m`、canonical MPJPE は
`0.1855→0.1942m` と悪化した。weight 増量に有望な dose response は観測されない。

`run-i801-dref-pose-beta005-inact1-match025-s42-r1` に対しては、F1
`0.6449→0.6609` と inactive false positive `1483.20→1384.00` は改善したが、ID switch
`28.96→60.08`、duplicate `352.12→369.60`、missed `8.08→10.40`、position
`5.2208→5.3116m`、canonical MPJPE `0.1854→0.1942m` は悪化した。soft cardinality と
inactive BCE 増量のいずれも、presence、identity、pose を同時には改善していない。

### 次に有効な実験

soft cardinality weight の追加増量は行わない。次は query ごとの不要発火を直接識別できる matched / unmatched
presence supervision、または離散人数に近い top-k / set-level 制約を小規模に検証する。評価では平均人数だけでなく
GT 人数別の active count 分布、precision / recall、inactive false positive、duplicate、ID switch、missed と
position / pose を同時に比較し、identity や pose を犠牲にした count 低下を改善と判定しない。
