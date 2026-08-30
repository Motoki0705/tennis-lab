---
id: run-i801-eval-beta005-e69-thr094-r1
type: run
title: beta005 epoch 69 の threshold 0.94 再評価
issue: 801
provider: codex
session: 01a04915-27a4-7b62-9f6b-34275561fded
date: '2026-08-29'
status: done
config:
  model: track_query_ablation_d_v2_selector
  loss: tracking_all_outputs_beta01_reprojection
  data: plcs/multi_object_camera_view_v2
  source_checkpoint: plcs/i801_dref_pose_beta005_s42_r1/logs/version_0/checkpoints/plcs-epoch=69.ckpt
  source_checkpoint_epoch: 69
  source_checkpoint_selection: best val/loss
  evaluation_only: true
  presence_threshold: 0.94
  rotation_weight: 0.05
  angle_weight: 0.05
metrics:
  loss: 0.885384
  loss_position: 0.14839
  loss_rotation: 0.24
  loss_presence: 0.610579
  loss_track_smoothness: 0.0
  loss_angle: 0.2563
  loss_canonical_pose: 0.010312
  loss_reprojection: 0.091288
  position_error: 0.432328
  presence_precision: 0.592326
  presence_recall: 0.827092
  presence_f1: 0.687786
  lifecycle_presence_f1: 0.687786
  birth_frame_error: 29.511911
  death_frame_error: 31.677248
  query_reuse_count: 0.08
  illegal_overlap_count: 0.0
  segment_id_switches: 27.92
  id_switches: 27.92
  duplicate_active_tracks: 115.080002
  missed_gt_frames: 265.040009
  inactive_query_false_positives: 775.679993
  angular_error_deg: 33.491806
  heading_error_deg: 33.259998
  position_error_m: 4.924719
  x_error_m: 1.838125
  y_error_m: 4.183125
  z_error_m: 0.218437
  y_sign_accuracy: 0.714219
  reference_index_0_position_error_m: 5.304276
  reference_index_1_position_error_m: 5.422917
  reference_index_2_position_error_m: 4.632812
  reference_index_3_position_error_m: 4.872768
  reference_index_4_position_error_m: 4.964334
  canonical_mpjpe_m: 0.175703
  world_mpjpe_m: 4.937574
  reprojection_error_px: 155.214096
  behind_camera_fraction: 0.000177
  reference_index_5_position_error_m: 4.500868
repro:
  commit: 9ffae3d34d34fe93b41a4f7e64e60238b3254cd4
  branch: experiments/plcs-tracking-pose-beta005
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: /home/kamimura/projects/tennis-lab/.venv/bin/python -c "import torch; import
    pytorch_lightning as pl; from src.tasks.plcs.training.tracking_lightning_module
    import PLCSTrackingLightningModule as M; from src.tasks.plcs.data.tracking_datamodule
    import PLCSTrackingDataModule as D; p='/home/kamimura/projects/tennis-lab/.claude/worktrees/plcs-tracking-pose-beta005/outputs/plcs/i801_dref_pose_beta005_s42_r1/logs/version_0/checkpoints/plcs-epoch=69.ckpt';
    b=torch.load(p,map_location='cpu',weights_only=False); c=b['hyper_parameters']['config'];
    c.tracking_metrics.presence_threshold=0.94; c.data.num_workers=16; m=M.load_from_checkpoint(p,config=c,map_location='cpu',weights_only=False);
    d=D(c); t=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',logger=False,enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False);
    print(t.test(m,datamodule=d))"
artifacts:
  run_dir: knowledge/runs/run-i801-eval-beta005-e69-thr094-r1
  predictions: knowledge/runs/run-i801-eval-beta005-e69-thr094-r1/pred_test.npz
parents:
- run-i801-dref-pose-beta005-s42-r1
relations:
- to: run-i801-eval-beta005-e69-thr050-r1
  rel: compares
tags: [plcs, tracking, pose, evaluation, threshold-094, beta005, epoch-69]
---

## 考察 / Findings

### 要約

親 run の最良 `val/loss` checkpoint（epoch 69）を presence threshold `0.94` で再評価した。
epoch 69 の threshold `0.5` 評価に対して F1、ID switch、duplicate は改善したが、recall が低下し、
missed GT frames が大幅に増えたため、tracking 全体として採用できる改善ではない。

### アーキテクチャ詳細

`run-i801-dref-pose-beta005-s42-r1` の epoch 69 checkpoint を読み込み、モデルや重みを更新せず
同じ test split を評価した。tracking + canonical pose + reprojection 構成、
`rotation_weight=0.05`、`angle_weight=0.05` は親 run から不変で、推論時の
`presence_threshold` だけを `0.94` とした評価専用 run である。

### メトリクスの解釈

tracking の precision / recall / F1 は `0.592326 / 0.827092 / 0.687786`、ID switch は
`27.92`、duplicate active tracks は `115.08`、missed GT frames は `265.04`、inactive query
false positives は `775.68` だった。pose 側は position `4.924719 m`、angular `33.491806°`、
heading `33.259998°`、canonical MPJPE `0.175703 m`、reprojection `155.214096 px` である。
評価専用 run で学習系列を持たないため、収束曲線は生成対象外とした。

### アーキテクチャ⇄メトリクスの因果考察

同一 checkpoint の threshold `0.5` 評価と比べて inactive query false positives と duplicate が
減ったことは、presence logit の判定を厳しくした直接の効果と整合する。一方、recall の低下と
missed GT frames の増加は、本来 active である query まで除外したことを示す。したがって、
threshold を上げるだけでは query の過剰発火を解消せず、false positive を false negative へ
置き換える trade-off に留まる、というのがこの run の観測である。

### 既存実験との比較

同じ epoch 69 checkpoint を threshold `0.5` で評価した
`run-i801-eval-beta005-e69-thr050-r1` に対し、F1 は `0.664414→0.687786`、ID switch は
`40.04→27.92`、duplicate は `366.08→115.08`、inactive query false positives は
`1361.28→775.68` と改善した。しかし recall は `0.990172→0.827092`、missed GT frames は
`13.76→265.04` と大幅に悪化した。pose metrics は同一であり、この直接比較では差を threshold
変更へ帰属できる。一方、親の学習 run の最終 checkpoint 評価との比較は checkpoint を epoch 69 へ
変更すると同時に threshold も `0.5→0.94` としているため、その差を threshold 単独の効果とは解釈できない。

### 次に有効な実験

threshold は `0.5` に固定し、presence head の学習側で inactive query を分離する施策を比較する。
F1 だけでなく precision / recall、ID switch、duplicate、inactive query false positives、missed GT frames
を同時に評価し、false positive 削減を missed 増加へ付け替えていないことを採用条件とする。
