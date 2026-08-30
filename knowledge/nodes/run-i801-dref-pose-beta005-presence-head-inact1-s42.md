---
id: run-i801-dref-pose-beta005-presence-head-inact1-s42
type: run
title: presence head 限定 fine-tune（inactive 1.0、seed 42）
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
  presence_inactive_weight: 1.0
  match_presence_weight: 0.0
  match_presence_inactive_weight: 0.25
  learning_rate: 0.001
  max_epochs: 20
  sequence_length: 128
  num_views: 6
  seed: 42
metrics:
  loss: 1.151637
  loss_position: 0.148668
  loss_rotation: 0.236838
  loss_presence: 0.878959
  loss_track_smoothness: 0.0
  loss_angle: 0.25287
  loss_canonical_pose: 0.010316
  loss_reprojection: 0.089208
  position_error: 0.433102
  presence_precision: 0.50865
  presence_recall: 0.981903
  presence_f1: 0.665743
  lifecycle_presence_f1: 0.665743
  birth_frame_error: 14.332254
  death_frame_error: 16.251999
  query_reuse_count: 0.16
  illegal_overlap_count: 0.0
  segment_id_switches: 40.119999
  id_switches: 40.119999
  duplicate_active_tracks: 356.559998
  missed_gt_frames: 26.32
  inactive_query_false_positives: 1330.719971
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
    loss.match_presence_weight=0.0 loss.match_presence_inactive_weight=0.25 loss.rotation_weight=0.05
    loss.angle_weight=0.05 training.learning_rate=1e-3 training.compile.enabled=false
    training.trainer.precision=bf16-mixed training.trainer.accumulate_grad_batches=4
    training.trainer.max_epochs=20 training.warmup_steps=25 training.trainer.check_val_every_n_epoch=2
    training.trainer.enable_progress_bar=false training.trainer.enable_model_summary=false
    training.qualitative_logging.enabled=false run.gpus=1 run.seed=42 run.fast_dev_run=false
    run.test_after_fit=true loss.presence_inactive_weight=1.0 run.output_dir=plcs/i801_dref_pose_beta005_presence_head_inact1_s42
artifacts:
  run_dir: knowledge/runs/run-i801-dref-pose-beta005-presence-head-inact1-s42
  predictions: knowledge/runs/run-i801-dref-pose-beta005-presence-head-inact1-s42/pred_test.npz
  output_dir: outputs/plcs/i801_dref_pose_beta005_presence_head_inact1_s42/logs/version_0
  curves: knowledge/runs/run-i801-dref-pose-beta005-presence-head-inact1-s42/curves.png
  tb_logdir: outputs/plcs/i801_dref_pose_beta005_presence_head_inact1_s42/logs/version_0
parents:
- run-i801-dref-pose-beta005-s42-r1
relations:
- to: run-i801-dref-pose-beta005-presence-head-inact05-s42
  rel: compares
- to: run-i801-dref-pose-beta005-inact1-match025-s42-r1
  rel: compares
tags: [plcs, tracking, pose, presence, fine-tune, beta005]
---

## 考察 / Findings

### 要約

親 run の epoch 69 から presence head だけを inactive weight `1.0` で fine-tuneすると、poseを完全に保持したままF1は`0.665743`、precisionは`0.508650`まで上がった。一方、duplicate active tracksは`356.56`、inactive query false positivesは`1330.72`で、過剰発火は依然として大きい。

### アーキテクチャ詳細

`run-i801-dref-pose-beta005-s42-r1` の epoch 69 checkpointを初期値とし、`fine_tune_mode=presence_head` によってpresence headだけを20 epoch更新した。presence以外の173 tensorは学習前後で bitwise 不変であることを確認している。inactive側の最終BCEを `presence_inactive_weight=1.0` とし、Hungarian matchingは `match_presence_weight=0.0`、`match_presence_inactive_weight=0.25` に分離した。複合lossの `rotation_weight=0.05`、`angle_weight=0.05`、長さ128、6 view、seed 42はinactive `0.5` runと同一である。

### メトリクスの解釈

test のpresence precision / recall / F1は `0.508650 / 0.981903 / 0.665743`、ID switchは`40.12`、duplicate active tracksは`356.56`、missed GT framesは`26.32`だった。inactive query false positivesは`1330.72`であり、inactive penaltyを強めてもprecisionは約0.51に留まる。pose側はposition error `4.931891 m`、angular error `33.260555°`、canonical MPJPE `0.175773 m`、reprojection error `153.389420 px`で、inactive `0.5` runと完全に一致する。raw angle loss `0.252870` に対する重みは `0.05` であり、angleは総lossを支配していない。記録metricsはepoch 19の最終in-memory stateに対するtest結果であり、`last.ckpt`は保存条件上epoch 5なので同一stateではない。`curves.png` では位置・角度のvalidation値は凍結により一定だが、validation lossはstep 50の約`1.072`からほぼ一貫して増加し約`1.113`に達した。best validation F1はepoch 3で、長く学習する根拠は得られない。

### アーキテクチャ⇄メトリクスの因果考察

presence headだけを更新したことで、inactive penaltyを強めてもpose経路を壊さずにprecisionを調整できた。`0.5`より強いinactive BCEがfalse positiveとduplicateをわずかに減らした一方、recall低下とmissed増加を招いており、単純なclass weight変更の典型的なtrade-offが現れたという仮説である。必要なactive query数を直接学習していないため、query過剰発火という構造的な失敗は残り、weightの増加だけではtracking精度の問題を解消できない。

### 既存実験との比較

親 run に対し、F1は`0.663940`から`0.665743`、precisionは`0.504449`から`0.508650`、ID switchは`46.24`から`40.12`、duplicateは`357.60`から`356.56`、inactive FPは`1361.92`から`1330.72`へ改善した。一方、recallは`0.989816`から`0.981903`、missedは`15.36`から`26.32`へ悪化した。full-model inactive `1.0` の `run-i801-dref-pose-beta005-inact1-match025-s42-r1` と比べると、F1 `0.644877`、position `5.2208 m`、angular `38.6782°`、canonical MPJPE `0.1854 m`、inactive FP `1483.20`より良いが、ID `28.96`、duplicate `352.12`、missed `8.08`より悪い。head-only inactive `0.5` に対してはF1・precision・duplicate・inactive FPがわずかに良い反面、ID・recall・missedが悪い。

### 次に有効な実験

validation F1が最大だったepoch 3 checkpointを同じtest手順で評価し、最終in-memory stateとの差を確認する。その後、pose系173 tensorを凍結する設計は維持しつつ、フレーム単位のGT人数を対象にしたactive-query cardinality lossをpresence headへ加え、duplicateとinactive FPを直接抑える。checkpoint選択もF1単独ではなく、recall下限を課した上でduplicate・missedを含む指標にする必要がある。
