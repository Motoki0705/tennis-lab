---
id: run-i801-eval-beta005-presence-head-inact05-bestvalf1-thr050
type: run
title: presence head inactive 0.5 の best-val-F1 評価（threshold 0.5）
issue: 801
provider: codex
session: 01a04915-27a4-7b62-9f6b-34275561fded
date: '2026-08-30'
status: done
config:
  model: track_query_ablation_d_v2_selector
  loss: tracking_all_outputs_beta01_reprojection
  data: plcs/multi_object_camera_view_v2
  source_checkpoint: plcs/i801_dref_pose_beta005_presence_head_inact05_s42/logs/version_0/checkpoints/plcs-epoch=01.ckpt
  source_checkpoint_epoch: 1
  source_checkpoint_selection: best val/presence_f1
  evaluation_only: true
  fine_tune_mode: presence_head
  presence_inactive_weight: 0.5
  presence_threshold: 0.5
  rotation_weight: 0.05
  angle_weight: 0.05
metrics:
  loss: 1.025135
  loss_position: 0.148652
  loss_rotation: 0.236882
  loss_presence: 0.75027
  loss_track_smoothness: 0.0
  loss_angle: 0.252914
  loss_canonical_pose: 0.010317
  loss_reprojection: 0.091406
  position_error: 0.433046
  presence_precision: 0.50706
  presence_recall: 0.984138
  presence_f1: 0.664791
  lifecycle_presence_f1: 0.664791
  birth_frame_error: 14.798079
  death_frame_error: 17.194836
  query_reuse_count: 0.08
  illegal_overlap_count: 0.0
  segment_id_switches: 39.48
  id_switches: 39.48
  duplicate_active_tracks: 361.440002
  missed_gt_frames: 22.879999
  inactive_query_false_positives: 1344.640015
  angular_error_deg: 33.26778
  heading_error_deg: 33.450001
  position_error_m: 4.93163
  x_error_m: 1.833125
  y_error_m: 4.191875
  z_error_m: 0.218437
  y_sign_accuracy: 0.713906
  reference_index_0_position_error_m: 5.304276
  reference_index_1_position_error_m: 5.422917
  reference_index_2_position_error_m: 4.632812
  reference_index_3_position_error_m: 5.033482
  reference_index_4_position_error_m: 4.964334
  canonical_mpjpe_m: 0.175777
  world_mpjpe_m: 4.944825
  reprojection_error_px: 155.340607
  behind_camera_fraction: 0.000177
  reference_index_5_position_error_m: 4.500868
repro:
  commit: 9ffae3d34d34fe93b41a4f7e64e60238b3254cd4
  branch: experiments/plcs-tracking-pose-beta005
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 /home/kamimura/projects/tennis-lab/.venv/bin/python
    -c "import torch; import pytorch_lightning as pl; from src.tasks.plcs.training.tracking_lightning_module
    import PLCSTrackingLightningModule as M; from src.tasks.plcs.data.tracking_datamodule
    import PLCSTrackingDataModule as D; p='/home/kamimura/projects/tennis-lab/.claude/worktrees/plcs-tracking-pose-beta005/outputs/plcs/i801_dref_pose_beta005_presence_head_inact05_s42/logs/version_0/checkpoints/plcs-epoch=01.ckpt';
    b=torch.load(p,map_location='cpu',weights_only=False); c=b['hyper_parameters']['config'];
    c.tracking_metrics.presence_threshold=0.5; c.data.num_workers=16; m=M.load_from_checkpoint(p,config=c,map_location='cpu',weights_only=False);
    d=D(c); t=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',logger=False,enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False);
    print(t.test(m,datamodule=d))"
artifacts:
  run_dir: knowledge/runs/run-i801-eval-beta005-presence-head-inact05-bestvalf1-thr050
  predictions: knowledge/runs/run-i801-eval-beta005-presence-head-inact05-bestvalf1-thr050/pred_test.npz
  curves: knowledge/runs/run-i801-eval-beta005-presence-head-inact05-bestvalf1-thr050/curves.png
  tb_logdir: outputs/plcs/i801_dref_pose_beta005_presence_head_cnll005_s42/logs/version_0
parents:
- run-i801-dref-pose-beta005-presence-head-inact05-s42
relations:
- to: run-i801-eval-beta005-e69-thr050-r1
  rel: compares
- to: run-i801-eval-beta005-presence-head-inact1-bestvalf1-thr050
  rel: compares
tags:
- plcs
- tracking
- pose
- presence
- fine-tune
- evaluation
- threshold-050
- beta005
---

## 考察 / Findings

### 要約

inactive weight `0.5` のpresence-head fine-tuneから、最良 `val/presence_f1` のepoch 1を
threshold `0.5`で評価した。epoch 69 baselineよりinactive FPとID switchは少し減ったが、
F1は`0.664791`に留まり、GT 1–3人時の4-query collapseはほぼ維持されたため総合改善ではない。

### アーキテクチャ詳細

親の `run-i801-dref-pose-beta005-presence-head-inact05-s42` はbeta005 epoch 69を初期値に、
presence headだけを `presence_inactive_weight=0.5` でfine-tuneした。本runはそのうち
`val/presence_f1` が最大だったepoch 1 checkpointを、モデル更新なしで同じtest splitへ適用した。
`rotation_weight=0.05`、`angle_weight=0.05` とpose trunkは不変で、評価thresholdは`0.5`である。

### メトリクスの解釈

precision / recall / F1は `0.507060 / 0.984138 / 0.664791`、ID switch `39.48`、
duplicate `361.44`、missed `22.88`、inactive FP `1344.64` だった。pose metricはposition
`4.931630 m`、angular `33.267780°`、canonical MPJPE `0.175777 m`、reprojection
`155.340607 px` である。GT 0人時は予測active数平均`0.0319`、全query inactive率`98.12%`へ
改善したが、GT 1–3人時は平均`3.874`、4-query全発火率`93.83%`だった。評価専用runなので
独自の学習曲線はなく、収束は親training nodeの`curves.png`を参照する。

### アーキテクチャ⇄メトリクスの因果考察

inactive BCEを強めたpresence headは、選手がいないフレームのlogitを下げる方向には機能した。
一方、人数が1–3人のフレームでは各queryが同時に高い状態を十分に分離できていない。
pose系parameterは凍結されていてもpresenceによるassignment / metric gatingが変わるため、
pose metricの小差はpose head更新ではなく評価対象対応の変化による可能性がある。

### 既存実験との比較

同じtest split・threshold `0.5` のepoch 69評価に対し、precisionは
`0.505068→0.507060`、inactive FPは `1361.28→1344.64`、ID switchは `40.04→39.48`、
duplicateは `366.08→361.44` と小幅改善した。一方、recallは `0.990172→0.984138`、
missedは `13.76→22.88` と悪化し、F1差は`+0.000377`に留まる。同一threshold比較なので、
このtrade-offはthreshold校正ではなくpresence-head checkpointの効果として解釈できる。

### 次に有効な実験

inactive weight `1.0` のbest-val-F1 checkpointを同じthreshold `0.5`で比較し、
GT 0人抑制がGT 1–3人のquery分離へ波及するか確認する。単純weight増加でcollapseが残る場合は、
GT人数を直接扱うquery-permutation-invariantなcount lossをpresence headへ追加する。
