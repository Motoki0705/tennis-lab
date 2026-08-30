---
id: run-i801-eval-beta005-e69-thr050-r1
type: run
title: beta005 epoch 69 の threshold 0.5 再評価
issue: 801
provider: codex
session: 01a04915-27a4-7b62-9f6b-34275561fded
date: '2026-08-30'
status: done
config:
  model: track_query_ablation_d_v2_selector
  loss: tracking_all_outputs_beta01_reprojection
  data: plcs/multi_object_camera_view_v2
  source_checkpoint: plcs/i801_dref_pose_beta005_s42_r1/logs/version_0/checkpoints/plcs-epoch=69.ckpt
  source_checkpoint_epoch: 69
  source_checkpoint_selection: best val/loss
  evaluation_only: true
  presence_threshold: 0.5
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
  presence_precision: 0.505068
  presence_recall: 0.990172
  presence_f1: 0.664414
  lifecycle_presence_f1: 0.664414
  birth_frame_error: 12.941874
  death_frame_error: 15.65583
  query_reuse_count: 0.24
  illegal_overlap_count: 0.0
  segment_id_switches: 40.040001
  id_switches: 40.040001
  duplicate_active_tracks: 366.079987
  missed_gt_frames: 13.76
  inactive_query_false_positives: 1361.280029
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
  command: CUDA_VISIBLE_DEVICES=0 /home/kamimura/projects/tennis-lab/.venv/bin/python
    -c "import torch; import pytorch_lightning as pl; from src.tasks.plcs.training.tracking_lightning_module
    import PLCSTrackingLightningModule as M; from src.tasks.plcs.data.tracking_datamodule
    import PLCSTrackingDataModule as D; p='/home/kamimura/projects/tennis-lab/.claude/worktrees/plcs-tracking-pose-beta005/outputs/plcs/i801_dref_pose_beta005_s42_r1/logs/version_0/checkpoints/plcs-epoch=69.ckpt';
    b=torch.load(p,map_location='cpu',weights_only=False); c=b['hyper_parameters']['config'];
    c.tracking_metrics.presence_threshold=0.5; c.data.num_workers=16; m=M.load_from_checkpoint(p,config=c,map_location='cpu',weights_only=False);
    d=D(c); t=pl.Trainer(accelerator='gpu',devices=1,precision='bf16-mixed',logger=False,enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False);
    print(t.test(m,datamodule=d))"
artifacts:
  run_dir: knowledge/runs/run-i801-eval-beta005-e69-thr050-r1
  predictions: knowledge/runs/run-i801-eval-beta005-e69-thr050-r1/pred_test.npz
parents:
- run-i801-dref-pose-beta005-s42-r1
relations:
- to: run-i801-eval-beta005-presence-head-inact05-bestvalf1-thr050
  rel: compares
- to: run-i801-eval-beta005-presence-head-inact1-bestvalf1-thr050
  rel: compares
tags: [plcs, tracking, pose, evaluation, threshold-050, beta005, epoch-69]
---

## 考察 / Findings

### 要約

親 run の最良 `val/loss` checkpoint（epoch 69）を presence threshold `0.5` で再評価した。
pose は position error `4.924719 m`、canonical MPJPE `0.175703 m` まで改善しているが、
presence は recall `0.990172` に対して precision `0.505068` で、4 query の過剰発火が残った。

### アーキテクチャ詳細

`run-i801-dref-pose-beta005-s42-r1` の epoch 69 checkpointを読み込み、モデルや重みは更新せず
同じ test splitを評価した。tracking + canonical pose + reprojection構成、
`rotation_weight=0.05`、`angle_weight=0.05` は親 runから不変で、推論時の
`presence_threshold` だけを明示的に `0.5` とした評価専用 runである。

### メトリクスの解釈

pose側は position `4.924719 m`、angular `33.491806°`、heading `33.259998°`、
canonical MPJPE `0.175703 m`、reprojection `155.214096 px` だった。tracking側は
precision / recall / F1 が `0.505068 / 0.990172 / 0.664414`、ID switch `40.04`、
duplicate active tracks `366.08`、missed GT frames `13.76`、inactive query false positives
`1361.28` である。12,800 valid frameの診断では、GT 0人時の予測active数は平均`0.0517`
（全query inactive率`97.09%`）だが、GT 1–3人時は平均`3.910`、4-query全発火率`96.05%`だった。
評価専用 runでTensorBoard系列を持たないため、`curves.png` は生成対象外である。

### アーキテクチャ⇄メトリクスの因果考察

epoch 69選択により親 runの最終stateよりposeとID switchは改善した一方、threshold `0.5` は
GT 1–3人のフレームでほぼ全queryをactiveと判定する。これは観測上、angle支配ではなく
presence logitのquery分離不足がtrackingの律速であることを示す。GT 0人時には抑制できているため、
単純なglobal biasだけでなく、選手が存在する状況で必要人数へqueryを分ける能力が不足しているという仮説である。

### 既存実験との比較

親 runの最終testに対し、positionは `5.005014→4.924719 m`、canonical MPJPEは
`0.179558→0.175703 m`、reprojectionは `160.501266→155.214096 px`、ID switchは
`46.24→40.04` と改善した。F1は `0.663940→0.664414` とほぼ同等だが、duplicateは
`357.60→366.08` に悪化した。後続のpresence-head評価とも同じtest split・threshold `0.5`を使うため、
以後の差はthreshold変更ではなくcheckpoint / presence-head学習の差として公平に比較できる。

### 次に有効な実験

このepoch 69 stateを固定し、presence headだけを学習してposeを保持したまま過剰発火を抑える。
評価はthreshold `0.5`を固定し、F1だけでなくGT人数別active数、duplicate、inactive FP、missedを同時に比較する。
