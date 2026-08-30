---
id: run-i801-eval-beta005-presence-head-inact1-bestvalf1-thr050
type: run
title: presence head inactive 1.0 の best-val-F1 評価（threshold 0.5）
issue: 801
provider: codex
session: 01a04915-27a4-7b62-9f6b-34275561fded
date: '2026-08-30'
status: done
config:
  model: track_query_ablation_d_v2_selector
  loss: tracking_all_outputs_beta01_reprojection
  data: plcs/multi_object_camera_view_v2
  source_checkpoint: plcs/i801_dref_pose_beta005_presence_head_inact1_s42/logs/version_0/checkpoints/plcs-epoch=03.ckpt
  source_checkpoint_epoch: 3
  source_checkpoint_selection: best val/presence_f1
  evaluation_only: true
  fine_tune_mode: presence_head
  presence_inactive_weight: 1.0
  presence_threshold: 0.5
  rotation_weight: 0.05
  angle_weight: 0.05
metrics:
  loss: 1.134163
  loss_position: 0.148652
  loss_rotation: 0.236882
  loss_presence: 0.859298
  loss_track_smoothness: 0.0
  loss_angle: 0.252914
  loss_canonical_pose: 0.010317
  loss_reprojection: 0.091406
  position_error: 0.433046
  presence_precision: 0.512172
  presence_recall: 0.980543
  presence_f1: 0.66843
  lifecycle_presence_f1: 0.66843
  birth_frame_error: 14.29149
  death_frame_error: 16.311823
  query_reuse_count: 0.16
  illegal_overlap_count: 0.0
  segment_id_switches: 39.0
  id_switches: 39.0
  duplicate_active_tracks: 342.480011
  missed_gt_frames: 28.24
  inactive_query_false_positives: 1313.920044
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
    import PLCSTrackingDataModule as D; p='/home/kamimura/projects/tennis-lab/.claude/worktrees/plcs-tracking-pose-beta005/outputs/plcs/i801_dref_pose_beta005_presence_head_inact1_s42/logs/version_0/checkpoints/plcs-epoch=03.ckpt';
    b=torch.load(p,map_location='cpu',weights_only=False); c=b['hyper_parameters']['config'];
    c.tracking_metrics.presence_threshold=0.5; c.data.num_workers=16; m=M.load_from_checkpoint(p,config=c,map_location='cpu',weights_only=False);
    d=D(c); t=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',logger=False,enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False);
    print(t.test(m,datamodule=d))"
artifacts:
  run_dir: knowledge/runs/run-i801-eval-beta005-presence-head-inact1-bestvalf1-thr050
  predictions: knowledge/runs/run-i801-eval-beta005-presence-head-inact1-bestvalf1-thr050/pred_test.npz
  curves: knowledge/runs/run-i801-eval-beta005-presence-head-inact1-bestvalf1-thr050/curves.png
  tb_logdir: outputs/plcs/i801_dref_pose_beta005_presence_head_cnll005_s42/logs/version_0
parents:
- run-i801-dref-pose-beta005-presence-head-inact1-s42
relations:
- to: run-i801-eval-beta005-e69-thr050-r1
  rel: compares
- to: run-i801-eval-beta005-presence-head-inact05-bestvalf1-thr050
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

inactive weight `1.0` のpresence-head fine-tuneから最良 `val/presence_f1` のepoch 3を
threshold `0.5`で評価すると、比較した3 checkpoint中ではF1 `0.668430`、ID switch `39.00`、
duplicate `342.48`で最良だった。ただしGT 1–3人時の4-query全発火率は`90.40%`で、未解決である。

### アーキテクチャ詳細

親の `run-i801-dref-pose-beta005-presence-head-inact1-s42` はbeta005 epoch 69を初期値に、
presence headだけを `presence_inactive_weight=1.0` でfine-tuneした。本runは
`val/presence_f1` 最大のepoch 3 checkpointを同じtest splitで評価したもので、学習は行わない。
pose構成と `rotation_weight=0.05`、`angle_weight=0.05` を維持し、thresholdは`0.5`に固定した。

### メトリクスの解釈

precision / recall / F1は `0.512172 / 0.980543 / 0.668430`、ID switch `39.00`、
duplicate `342.48`、missed `28.24`、inactive FP `1313.92` だった。pose側はposition
`4.931630 m`、angular `33.267780°`、canonical MPJPE `0.175777 m`、reprojection
`155.340607 px` である。GT 0人時は予測active数平均`0.0099`、全query inactive率`99.44%`だが、
GT 1–3人時は平均`3.820`、4-query全発火率`90.40%`である。評価専用runのため独自の
`curves.png`はなく、学習収束は親nodeの曲線を参照する。

### アーキテクチャ⇄メトリクスの因果考察

inactive penaltyを`1.0`へ上げたことで、特にGT 0人時の不要発火がほぼ消え、precision、F1、
duplicateは`0.5`より改善した。しかしGT 1–3人時にはpresence headの全queryが同様に高いという
条件付きcollapseが残る。これは仮説だが、固定trunkの特徴が「誰もいない」を識別できても、
存在する選手数に応じてqueryを分ける情報を十分提供していないため、headのclass weightだけでは解けない。

### 既存実験との比較

同じtest split・threshold `0.5` のepoch 69評価に対し、precisionは
`0.505068→0.512172`、F1は `0.664414→0.668430`、ID switchは `40.04→39.00`、
duplicateは `366.08→342.48`、inactive FPは `1361.28→1313.92`へ改善した。一方、recallは
`0.990172→0.980543`、missedは `13.76→28.24`へ悪化した。inactive `0.5` 評価に対しても
F1、ID、duplicate、inactive FPは改善したがmissedは増えた。同一threshold比較により、これらは
threshold trade-offではなくcheckpoint差であり、総合的には最良のPareto点だがtracking問題の解消ではない。

### 次に有効な実験

pose trunkを凍結する安全性は維持しつつ、GT人数を直接最適化するPoisson-binomial exact count NLLを
低weightで加える。threshold `0.5`、同一test split、GT人数別active数を固定評価し、
GT 1–3人時の4-query collapseを減らせない施策はF1が小幅改善しても採用しない。
