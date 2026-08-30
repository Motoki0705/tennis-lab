---
id: run-i801-dref-pose-beta005-presence-head-hneg025-s42
type: run
title: presence head hard-negative focal 0.25（seed 42）
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
  presence_hard_negative_weight: 0.25
  learning_rate: 0.001
  max_epochs: 8
  sequence_length: 128
  num_views: 6
  seed: 42
metrics:
  loss: 1.214243
  loss_position: 0.148668
  loss_rotation: 0.236838
  loss_presence: 0.677724
  loss_track_smoothness: 0.0
  loss_angle: 0.25287
  loss_canonical_pose: 0.010316
  loss_reprojection: 0.089208
  loss_presence_hard_negative: 1.055362
  position_error: 0.433102
  presence_precision: 0.507921
  presence_recall: 0.983308
  presence_f1: 0.665395
  lifecycle_presence_f1: 0.665395
  birth_frame_error: 14.228735
  death_frame_error: 16.274672
  query_reuse_count: 0.16
  illegal_overlap_count: 0.0
  segment_id_switches: 39.52
  id_switches: 39.52
  duplicate_active_tracks: 360.559998
  missed_gt_frames: 24.24
  inactive_query_false_positives: 1340.319946
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
    run.test_after_fit=true loss.presence_hard_negative_weight=0.25 run.output_dir=plcs/i801_dref_pose_beta005_presence_head_hneg025_s42
artifacts:
  run_dir: knowledge/runs/run-i801-dref-pose-beta005-presence-head-hneg025-s42
  predictions: knowledge/runs/run-i801-dref-pose-beta005-presence-head-hneg025-s42/pred_test.npz
  output_dir: outputs/plcs/i801_dref_pose_beta005_presence_head_hneg025_s42/logs/version_0
  curves: knowledge/runs/run-i801-dref-pose-beta005-presence-head-hneg025-s42/curves.png
  tb_logdir: outputs/plcs/i801_dref_pose_beta005_presence_head_hneg025_s42/logs/version_0
parents:
- run-i801-dref-pose-beta005-presence-head-inact05-s42
relations:
- to: run-i801-dref-pose-beta005-presence-head-hneg050-s42
  rel: compares
- to: run-i801-eval-beta005-e69-thr050-r1
  rel: compares
- to: run-i801-eval-beta005-presence-head-inact1-bestvalf1-thr050
  rel: compares
tags: [plcs, tracking, pose, presence, fine-tune, hard-negative, focal, beta005]
---

## 考察 / Findings

### 要約

presence head限定fine-tuneへgamma `2.0`、weight `0.25`のhard-negative focal項を加えると、inactive `0.5` baselineよりduplicateとinactive FPは減ったが、missedは増えた。GT 1–3人時の4-query全発火率はなお`93.85%`であり、過剰発火collapseは解消していない。

### アーキテクチャ詳細

親の `run-i801-dref-pose-beta005-presence-head-inact05-s42` と同じく、beta005 epoch 69 checkpointを初期値としてpresence headだけを更新した。最終BCEは `presence_inactive_weight=0.5`、Hungarian matchingは `match_presence_weight=0.0`、`match_presence_inactive_weight=0.25` とし、inactive targetを重点化するhard-negative項を `gamma=2.0`、weight `0.25` で追加した。cardinality系lossは無効である。tracking + pose構成、`rotation_weight=0.05`、`angle_weight=0.05`、長さ128、6 view、seed 42を維持し、8 epoch学習した。

### メトリクスの解釈

testのprecision / recall / F1は `0.507921 / 0.983308 / 0.665395`、ID switch `39.52`、duplicate `360.56`、missed `24.24`、inactive FP `1340.32` だった。予測bundleの12,800 valid frameをthreshold `0.5`で再集計すると、GT 0人時は予測active数平均`0.0160`、全query inactive率`99.11%`まで抑えられた。一方GT 1–3人時は予測active数平均`3.8668`、4-query全発火率`93.85%`である。pose側はposition `4.931891 m`、angular `33.260555°`、canonical MPJPE `0.175773 m`、reprojection `153.389420 px`。raw angle loss `0.252870`にweight `0.05`を掛けた寄与は約`0.01264`で、総loss `1.214243`の約`1.0%`に留まる。`curves.png`では凍結したposition / angleのvalidation値は一定で、validation lossはstep 49で急減後に一度反発し、最終stepまで緩やかに低下した。ただしbest validation F1はepoch 2（step 49）で、後半のloss低下はF1改善に結びつかなかった。

### アーキテクチャ⇄メトリクスの因果考察

hard-negative項により特にGT 0人時の不要発火が抑えられ、duplicateとinactive FPが小幅に減ったと考えられる。一方、GT 1–3人時には複数queryのlogitが同時に高いままである。これは仮説だが、targetごとのnegative重点化はglobalな抑制には効いても、フレーム内で必要なquery数を決める制約やquery間の役割分担を与えないため、条件付き4-query collapseを解けない。presence headだけを更新したためpose精度を維持できた点は狙いどおりである。

### 既存実験との比較

親のinactive `0.5` training runに対し、F1は `0.664566→0.665395`、duplicateは `366.72→360.56`、inactive FPは `1356.80→1340.32`へ改善した。一方ID switchは `38.28→39.52`、missedは `15.04→24.24`へ悪化した。source epoch 69評価に対してもF1は`+0.000981`、duplicateは`-5.52`、inactive FPは`-20.96`だが、missedは`+10.48`である。weight `0.5`の兄弟runは本runよりF1 `+0.001660`、duplicate `-8.88`、inactive FP `-13.28`とさらに改善するがmissedは`+1.68`増える。inact1 best-val-F1評価はF1 `0.668430`、duplicate `342.48`、GT 1–3人時の4-query全発火率`90.40%`で本runより良い一方、missed `28.24`は悪い。

### 次に有効な実験

hard-negative weightを`0.5`へ上げた兄弟runでもcollapseが残るため、同じpresence-head限定条件でGT人数を直接最適化するPoisson-binomial exact-count NLLを低weightで加えるのが有効である。選択基準にはF1だけでなくGT 1–3人時の4-query全発火率、duplicate、inactive FP、missedを含め、pose不変も確認する。
