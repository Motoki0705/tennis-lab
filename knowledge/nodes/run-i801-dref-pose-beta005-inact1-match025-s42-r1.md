---
id: run-i801-dref-pose-beta005-inact1-match025-s42-r1
type: run
title: PLCS tracking + pose inactive BCE 1.0 / matching inactive 0.25（seed 42）
issue: 801
provider: codex
session: 01a04915-27a4-7b62-9f6b-34275561fded
date: '2026-08-29'
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
  presence_inactive_weight: 1.0
  match_presence_inactive_weight: 0.25
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
  loss: 1.321407
  loss_position: 0.161868
  loss_rotation: 0.310727
  loss_presence: 1.012053
  loss_track_smoothness: 0.0
  loss_angle: 0.337002
  loss_canonical_pose: 0.011401
  loss_reprojection: 0.103699
  position_error: 0.467245
  presence_precision: 0.481341
  presence_recall: 0.994581
  presence_f1: 0.644877
  lifecycle_presence_f1: 0.644877
  birth_frame_error: 14.656651
  death_frame_error: 17.630266
  query_reuse_count: 0.08
  illegal_overlap_count: 0.0
  segment_id_switches: 28.959999
  id_switches: 28.959999
  duplicate_active_tracks: 352.119995
  missed_gt_frames: 8.08
  inactive_query_false_positives: 1483.199951
  angular_error_deg: 38.678158
  heading_error_deg: 38.509998
  position_error_m: 5.220839
  x_error_m: 1.955625
  y_error_m: 4.3875
  z_error_m: 0.332188
  y_sign_accuracy: 0.674219
  reference_index_0_position_error_m: 7.319901
  reference_index_1_position_error_m: 5.464583
  reference_index_2_position_error_m: 4.494141
  reference_index_3_position_error_m: 5.575893
  reference_index_4_position_error_m: 5.399457
  canonical_mpjpe_m: 0.185416
  world_mpjpe_m: 5.231677
  reprojection_error_px: 170.540543
  behind_camera_fraction: 0.000193
  reference_index_5_position_error_m: 5.451389
repro:
  commit: 9ffae3d34d34fe93b41a4f7e64e60238b3254cd4
  branch: experiments/plcs-tracking-pose-beta005
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.plcs.scripts.train --config-name train_tracking_pose model=track_query_ablation_d_v2_selector
    court_keypoints=camera_view_v2 model.cswa.backend=cuda paths.data_root=/home/kamimura/projects/tennis-lab/data
    data.scene_dir=plcs/multi_object_camera_view_v2 'data.seq_len_range=[128,128]'
    'data.num_views_range=[6,6]' data.batch_size=8 data.num_workers=16 data.evaluation_reference_camera_id=camera_2
    loss.presence_inactive_weight=1.0 loss.match_presence_inactive_weight=0.25 training.compile.enabled=false
    training.trainer.precision=bf16-mixed training.trainer.accumulate_grad_batches=4
    training.trainer.max_epochs=40 training.warmup_steps=400 training.trainer.check_val_every_n_epoch=5
    training.trainer.enable_progress_bar=false training.trainer.enable_model_summary=false
    training.early_stopping.enabled=false training.qualitative_logging.enabled=false
    run.seed=42 run.fast_dev_run=false run.test_after_fit=true run.output_dir=plcs/i801_dref_pose_beta005_inact1_match025_s42_r1
artifacts:
  run_dir: knowledge/runs/run-i801-dref-pose-beta005-inact1-match025-s42-r1
  predictions: knowledge/runs/run-i801-dref-pose-beta005-inact1-match025-s42-r1/pred_test.npz
  output_dir: outputs/plcs/i801_dref_pose_beta005_inact1_match025_s42_r1/logs/version_0
  curves: knowledge/runs/run-i801-dref-pose-beta005-inact1-match025-s42-r1/curves.png
  tb_logdir: outputs/plcs/i801_dref_pose_beta005_inact1_match025_s42_r1/logs/version_0
parents:
- run-i801-dref-pose-beta005-s42-r1
relations:
- to: run-i801-dref-pose-beta005-inact05-s42
  rel: compares
- to: run-i801-dref-pose-beta005-matchp0-s42
  rel: compares
- to: run-i801-dref-pose-beta005-matchinact0-s42
  rel: compares
tags:
- plcs
- tracking
- canonical-pose
- reprojection
- camera-view-v2
- presence
- inactive-weight
- split-matching-weight
- beta005
- seed-42
---

## 考察 / Findings

### 要約

最終 presence BCE の inactive weight を `1.0` に上げ、Hungarian matching 側の inactive
penalty は `0.25` に固定して 40 epoch 学習した。ID switch は親 run の `46.24` から
`28.96` へ減ったが、inactive-query false positive は `1361.92` から `1483.20` へ増え、
F1 と全 pose 指標も悪化した。matching から重みを分離しても、inactive BCE の増量だけでは
query 過剰発火を解消できなかった。

### アーキテクチャ詳細

`track_query_ablation_d_v2_selector` の各 track query から position、rotation、presence、
17-joint canonical pose を予測し、assignment 後の active target に canonical pose と multi-view
reprojection loss を適用する。camera-view-v2、T=128、V=6、effective batch=32、seed 42、
CUDA CSWA、`rotation_weight=angle_weight=0.05`、canonical / reprojection weight `1.0` は親 run と同じ。

親 run では最終 BCE と matching の inactive weight がともに `0.25` だった。本 run は
`presence_inactive_weight=1.0` の一方で `match_presence_inactive_weight=0.25` を明示し、最終 BCE の
inactive 抑制だけを強めた。最大 epoch は親の 100 から 40、warmup は 400 step であり、学習長・schedule
にも差があるため、親との差分すべてを weight だけの因果効果とはみなせない。

### メトリクスの解釈

test precision / recall / F1 は `0.4813 / 0.9946 / 0.6449`、inactive-query false positive
`1483.20`、duplicate `352.12`、ID switch `28.96`、missed GT frame `8.08` だった。
高 recall に対して precision が低く、抑制対象だった inactive query の過剰発火が残っている。

pose は position `5.2208m`、angular `38.6782deg`、heading `38.5100deg`、canonical MPJPE
`0.1854m`、reprojection `170.54px` だった。test total loss `1.32141` のうち presence は
`1.01205`（約 76.6%）を占める。一方、angle の重み込み寄与は `0.05 × 0.33700 = 0.01685`
（約 1.28%）であり、angle 支配は起きていない。

収束曲線では train loss が `1.20860→0.56498` と低下した一方、val loss の最小は最初の評価
step 124 の `1.24814` で、最終 step 999 は `1.30293` だった。特に train presence loss は
`0.66393→0.31525` と低下したのに対し、val presence loss は `0.79793→0.97526` と上昇した。
val position は `6.97m` から step 624 の `5.59m` まで改善後、最終 `5.80m`、val angular は
`36.17–39.10deg` の範囲で振動しており、presence を中心に強い train / val 乖離が観測された。

### アーキテクチャ⇄メトリクスの因果考察

観測として、matching の inactive penalty を `0.25` に固定した状態でも、最終 BCE の inactive weight を
`1.0` に上げると inactive-query false positive と presence loss は改善せず、precision / F1 と pose が
悪化した。したがって「matching assignment への副作用を除けば inactive BCE の増量が過剰発火を
単調に抑える」という仮説は、この run では支持されない。

以下は仮説である。各 query の独立 BCE は GT の人数を直接制約しないため、複数 query が同時に高い
presence を出す解を十分に罰していない可能性がある。また、train / val presence loss の逆方向の推移から、
weight 増量が学習データの logit 分離を強めても汎化時の calibration を改善しなかった可能性がある。
ID switch 改善についても、40 epoch と schedule の差が混在するため weight 分離だけの効果とは断定しない。

### 既存実験との比較

親 `run-i801-dref-pose-beta005-s42-r1` に対し、ID switch は `46.24→28.96`、missed は
`15.36→8.08`、duplicate は `357.60→352.12` と改善した。一方、F1 は `0.6639→0.6449`、
inactive false positive は `1361.92→1483.20`、position は `5.0050→5.2208m`、angular は
`33.4878→38.6782deg`、canonical MPJPE は `0.1796→0.1854m` と悪化し、総合改善ではない。

coupled weight `0.5` の `run-i801-dref-pose-beta005-inact05-s42` と比べても、ID switch
`44.40→28.96` 以外は F1 `0.6624→0.6449`、duplicate `307.20→352.12`、inactive false positive
`1394.88→1483.20`、canonical MPJPE `0.1836→0.1854m` と悪化した。

`run-i801-dref-pose-beta005-matchp0-s42` / `run-i801-dref-pose-beta005-matchinact0-s42` に対しても
ID switch は `34.20 / 40.40→28.96` と少ないが、F1 は `0.6621 / 0.6633→0.6449`、duplicate は
`243.60 / 332.72→352.12`、inactive false positive は `1380.48 / 1373.44→1483.20` と悪い。
4 施策の中で identity 指標の一部だけを改善しても、query activation と pose を同時に改善できていない。

### 次に有効な実験

inactive BCE weight の追加増量は行わず、GT active 人数と query presence probability の総和を直接比較する
微分可能な cardinality loss を次候補とする。これは今回の観測から確立した解決策ではなく、「独立 BCE が
人数制約を持たないことが過剰発火の一因」という上記仮説を検証する実験である。まず weight の小さい一点で、
親 run と同じ inference threshold を用い、F1、inactive-query false positive、duplicate、ID switch、missed、
position / pose を同時評価する。改善が query 抑制によるものか確認するため、予測 active query 数と GT 人数の
frame 単位分布も保存して比較する。
