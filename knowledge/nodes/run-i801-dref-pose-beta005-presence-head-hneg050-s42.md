---
id: run-i801-dref-pose-beta005-presence-head-hneg050-s42
type: run
title: presence head hard-negative focal 0.5（seed 42）
issue: 801
provider: codex
session: 01a04915-27a4-7b62-9f6b-34275561fded
date: '2026-08-30'
status: done
config:
  model: track_query_ablation_d_v2_selector
  training_config: train_tracking_pose_presence_head
  init_weights: plcs/i801_dref_pose_beta005_s42_r1/logs/version_0/checkpoints/plcs-epoch=69.ckpt
  fine_tune_mode: presence_head
  rotation_weight: 0.05
  angle_weight: 0.05
  presence_inactive_weight: 0.5
  match_presence_weight: 0.0
  match_presence_inactive_weight: 0.25
  cardinality_weight: 0.0
  cardinality_nll_weight: 0.0
  presence_hard_negative_gamma: 2.0
  presence_hard_negative_weight: 0.5
  learning_rate: 0.001
  max_epochs: 8
  sequence_length: 128
  num_views: 6
  seed: 42
metrics:
  loss: 1.361272
  loss_position: 0.148668
  loss_rotation: 0.236838
  loss_presence: 0.639899
  loss_track_smoothness: 0.0
  loss_angle: 0.25287
  loss_canonical_pose: 0.010316
  loss_reprojection: 0.089208
  loss_presence_hard_negative: 0.897391
  position_error: 0.433102
  presence_precision: 0.510098
  presence_recall: 0.982154
  presence_f1: 0.667055
  lifecycle_presence_f1: 0.667055
  birth_frame_error: 14.27298
  death_frame_error: 16.257999
  query_reuse_count: 0.16
  illegal_overlap_count: 0.0
  segment_id_switches: 39.52
  id_switches: 39.52
  duplicate_active_tracks: 351.679993
  missed_gt_frames: 25.92
  inactive_query_false_positives: 1327.040039
  angular_error_deg: 33.260555
  heading_error_deg: 33.450001
  position_error_m: 4.931891
  x_error_m: 1.8325
  y_error_m: 4.191875
  z_error_m: 0.218594
  y_sign_accuracy: 0.714219
  reference_index_0_position_error_m: 5.310855
  reference_index_1_position_error_m: 5.422917
  reference_index_2_position_error_m: 4.636719
  reference_index_3_position_error_m: 5.040179
  reference_index_4_position_error_m: 4.964334
  canonical_mpjpe_m: 0.175773
  world_mpjpe_m: 4.945139
  reprojection_error_px: 153.38942
  behind_camera_fraction: 0.000179
  reference_index_5_position_error_m: 4.503472
repro:
  commit: 9ffae3d34d34fe93b41a4f7e64e60238b3254cd4
  branch: experiments/plcs-tracking-pose-beta005
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    /home/kamimura/projects/tennis-lab/.venv/bin/python -m src.tasks.plcs.scripts.train
    --config-name train_tracking_pose_presence_head model=track_query_ablation_d_v2_selector
    court_keypoints=camera_view_v2 model.cswa.backend=cuda paths.data_root=/home/kamimura/projects/tennis-lab/data
    paths.checkpoint_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/plcs-tracking-pose-beta005/outputs
    data.scene_dir=plcs/multi_object_camera_view_v2 data.seq_len_range=\[128,128\]
    data.num_views_range=\[6,6\] data.batch_size=8 data.num_workers=16 data.evaluation_reference_camera_id=camera_2
    run.init_weights="plcs/i801_dref_pose_beta005_s42_r1/logs/version_0/checkpoints/plcs-epoch\=69.ckpt"
    loss.match_presence_weight=0.0 loss.match_presence_inactive_weight=0.25 loss.presence_inactive_weight=0.5
    loss.cardinality_weight=0.0 loss.cardinality_nll_weight=0.0 loss.presence_hard_negative_gamma=2.0
    loss.rotation_weight=0.05 loss.angle_weight=0.05 training.learning_rate=1e-3 training.compile.enabled=false
    training.trainer.precision=bf16-mixed training.trainer.accumulate_grad_batches=4
    training.trainer.max_epochs=8 training.warmup_steps=10 training.trainer.check_val_every_n_epoch=1
    training.trainer.enable_progress_bar=false training.trainer.enable_model_summary=false
    training.qualitative_logging.enabled=false run.gpus=1 run.seed=42 run.fast_dev_run=false
    run.test_after_fit=true loss.presence_hard_negative_weight=0.5 run.output_dir=plcs/i801_dref_pose_beta005_presence_head_hneg050_s42
artifacts:
  run_dir: knowledge/runs/run-i801-dref-pose-beta005-presence-head-hneg050-s42
  predictions: knowledge/runs/run-i801-dref-pose-beta005-presence-head-hneg050-s42/pred_test.npz
  output_dir: outputs/plcs/i801_dref_pose_beta005_presence_head_hneg050_s42/logs/version_0
  curves: knowledge/runs/run-i801-dref-pose-beta005-presence-head-hneg050-s42/curves.png
  tb_logdir: outputs/plcs/i801_dref_pose_beta005_presence_head_hneg050_s42/logs/version_0
parents:
- run-i801-dref-pose-beta005-presence-head-inact05-s42
relations:
- to: run-i801-dref-pose-beta005-presence-head-hneg025-s42
  rel: compares
- to: run-i801-eval-beta005-e69-thr050-r1
  rel: compares
- to: run-i801-eval-beta005-presence-head-inact1-bestvalf1-thr050
  rel: compares
tags: [plcs, tracking, pose, presence, fine-tune, hard-negative, focal, beta005]
---

## 考察 / Findings

### 要約

hard-negative focal weight `0.5`は、source epoch 69とinactive `0.5` baselineの双方に対してprecision、F1、duplicate、inactive FPを改善した。ただしGT 1–3人時の4-query全発火率は`92.25%`で、inact1 bestの`90.40%`にも届かない中間改善に留まり、tracking問題は未解決である。

### アーキテクチャ詳細

beta005 epoch 69 checkpointからpresence headだけを8 epoch更新し、親 `run-i801-dref-pose-beta005-presence-head-inact05-s42` の設定へinactive target用hard-negative項を追加した。`presence_inactive_weight=0.5`、`match_presence_weight=0.0`、`match_presence_inactive_weight=0.25`、hard-negativeの `gamma=2.0`、weight `0.5` で、cardinality系lossは無効である。tracking + pose構成と `rotation_weight=0.05`、`angle_weight=0.05`、長さ128、6 view、seed 42は比較runと共通である。

### メトリクスの解釈

testのprecision / recall / F1は `0.510098 / 0.982154 / 0.667055`、ID switch `39.52`、duplicate `351.68`、missed `25.92`、inactive FP `1327.04` だった。予測bundleをthreshold `0.5`で人数別に再集計すると、GT 0人時は予測active数平均`0.0108`、全query inactive率`99.44%`である。しかしGT 1–3人時は予測active数平均`3.8437`、4-query全発火率`92.25%`であり、特にGT 2人時は`99.67%`のフレームで4 queryが全発火した。pose側はposition `4.931891 m`、angular `33.260555°`、canonical MPJPE `0.175773 m`、reprojection `153.389420 px`で、weight適用後のangle寄与約`0.01264`は総loss `1.361272`の約`0.9%`に過ぎない。`curves.png`では凍結したpose validation値は一定で、validation lossはstep 49で大きく低下し、step 74の反発後は最終stepまで緩やかに低下した。一方best validation F1はepoch 2（step 49）であり、lossとF1の最良点は一致しない。

### アーキテクチャ⇄メトリクスの因果考察

weight `0.25`より強いhard-negative項はGT 0人時の不要logitと全体のinactive FPをさらに抑え、precisionとduplicateを改善したと考えられる。しかしGT 1–3人の場面ではほぼ全queryが連動してactiveになり、人数に応じたquery選択は学習できていない。仮説として、独立query単位のfocal reweightingは難しいnegativeへ勾配を集中できても、同一フレームの総active数やquery間排他性を表現しないためである。presence head限定更新によりpose値がweight `0.25` runと完全に一致している点は、pose保持とpresence改善を分離する設計の有効性を示す。

### 既存実験との比較

親のinactive `0.5` training runに対し、precisionは `0.505291→0.510098`、F1は `0.664566→0.667055`、duplicateは `366.72→351.68`、inactive FPは `1356.80→1327.04`へ改善したが、ID switchは `38.28→39.52`、missedは `15.04→25.92`へ悪化した。source epoch 69評価に対してはF1 `+0.002641`、duplicate `-14.40`、inactive FP `-34.24`、ID switch `-0.52`で、missedは`+12.16`である。weight `0.25`比ではF1 `+0.001660`、duplicate `-8.88`、inactive FP `-13.28`の代わりにmissedが`+1.68`増えた。inact1 best-val-F1評価はF1 `0.668430`、duplicate `342.48`、inactive FP `1313.92`、GT 1–3人時の4-query全発火率`90.40%`で本runより良いが、missed `28.24`は本runより`2.32`多い。したがってfocal `0.5`は改善方向だが、既存Pareto点を更新していない。

### 次に有効な実験

focal weightの増加だけではGT人数条件付きcollapseを解けないため、presence headを固定または限定更新したまま、GT人数を直接教師とするPoisson-binomial exact-count NLLを低weightで再検証する。validation選択もpresence F1単独ではなく、GT 1–3人時の4-query全発火率とmissedの制約を含む複合基準にし、同一threshold `0.5`でinact1 bestと比較する。
