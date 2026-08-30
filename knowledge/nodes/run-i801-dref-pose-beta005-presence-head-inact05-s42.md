---
id: run-i801-dref-pose-beta005-presence-head-inact05-s42
type: run
title: presence head 限定 fine-tune（inactive 0.5、seed 42）
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
  learning_rate: 0.001
  max_epochs: 20
  sequence_length: 128
  num_views: 6
  seed: 42
metrics:
  loss: 1.040507
  loss_position: 0.148668
  loss_rotation: 0.236838
  loss_presence: 0.767828
  loss_track_smoothness: 0.0
  loss_angle: 0.25287
  loss_canonical_pose: 0.010316
  loss_reprojection: 0.089208
  position_error: 0.433102
  presence_precision: 0.505291
  presence_recall: 0.990016
  presence_f1: 0.664566
  lifecycle_presence_f1: 0.664566
  birth_frame_error: 14.249825
  death_frame_error: 16.102072
  query_reuse_count: 0.16
  illegal_overlap_count: 0.0
  segment_id_switches: 38.279999
  id_switches: 38.279999
  duplicate_active_tracks: 366.720001
  missed_gt_frames: 15.04
  inactive_query_false_positives: 1356.800049
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
    run.test_after_fit=true loss.presence_inactive_weight=0.5 run.output_dir=plcs/i801_dref_pose_beta005_presence_head_inact05_s42
artifacts:
  run_dir: knowledge/runs/run-i801-dref-pose-beta005-presence-head-inact05-s42
  predictions: knowledge/runs/run-i801-dref-pose-beta005-presence-head-inact05-s42/pred_test.npz
  output_dir: outputs/plcs/i801_dref_pose_beta005_presence_head_inact05_s42/logs/version_0
  curves: knowledge/runs/run-i801-dref-pose-beta005-presence-head-inact05-s42/curves.png
  tb_logdir: outputs/plcs/i801_dref_pose_beta005_presence_head_inact05_s42/logs/version_0
parents:
- run-i801-dref-pose-beta005-s42-r1
relations:
- to: run-i801-dref-pose-beta005-presence-head-inact1-s42
  rel: compares
- to: run-i801-dref-pose-beta005-inact05-s42
  rel: compares
- to: run-i801-dref-pose-beta005-inact1-match025-s42-r1
  rel: compares
tags: [plcs, tracking, pose, presence, fine-tune, beta005]
---

## 考察 / Findings

### 要約

親 run の epoch 69 から presence head だけを inactive weight `0.5` で fine-tune すると、pose を完全に保持したまま F1 は `0.664566`、ID switch は `38.28` になった。一方、duplicate active tracks は `366.72`、inactive query false positives は `1356.80` で、query の過剰発火は解消していない。

### アーキテクチャ詳細

`run-i801-dref-pose-beta005-s42-r1` の epoch 69 checkpoint を初期値とし、`fine_tune_mode=presence_head` によって presence head だけを20 epoch更新した。presence以外の173 tensorは学習前後で bitwise 不変であることを確認している。最終 BCE の `presence_inactive_weight=0.5` とし、Hungarian matching は `match_presence_weight=0.0`、`match_presence_inactive_weight=0.25` に分離した。trackingとposeの複合lossでは `rotation_weight=0.05`、`angle_weight=0.05` を維持し、入力は長さ128、6 view、seed 42である。

### メトリクスの解釈

test の presence precision / recall / F1 は `0.505291 / 0.990016 / 0.664566`、ID switch は `38.28`、duplicate active tracks は `366.72`、missed GT frames は `15.04` だった。precisionが約0.51に留まり、inactive query false positivesも `1356.80` あるため、高recallは余分なqueryをactiveにすることで得られている。pose側は position error `4.931891 m`、angular error `33.260555°`、canonical MPJPE `0.175773 m`、reprojection error `153.389420 px` である。raw angle loss `0.252870` に対する重みは `0.05` なので、angleが総lossを支配する状態ではない。記録metricsはepoch 19の最終in-memory stateに対するtest結果である。`curves.png` では位置・角度のvalidation値は凍結により一定で、validation lossはstep 100付近を底に上昇して約`1.018`で頭打ちとなった。best validation F1もepoch 1で、後半の更新に改善は見られない。

### アーキテクチャ⇄メトリクスの因果考察

presence head以外を凍結したため、position・rotation・canonical pose・reprojectionの出力はfine-tuneで変化せず、full-model更新時に起きたpose悪化を避けられた。ID switch低下はpresence logitの調整で一部の不要trackが抑制された結果という仮説が立つ。しかしper-query BCEだけではフレームごとの必要人数を直接制約しないため、recallを維持したまま複数queryが同時発火する解に残り、duplicate数はむしろ増えた。したがってこれはpresence calibrationの限定的な改善であり、trackingとposeの同時解決ではない。

### 既存実験との比較

親 run は F1 `0.663940`、ID switch `46.24`、duplicate `357.60`、missed `15.36`、inactive FP `1361.92` だった。本runはF1を`+0.000626`、ID switchを`-7.96`、missedを`-0.32`、inactive FPを`-5.12`改善したが、duplicateは`+9.12`悪化した。full-model inactive `0.5` の `run-i801-dref-pose-beta005-inact05-s42` と比べると、F1・ID・poseは良い一方、duplicateとmissedは悪い。head-only inactive `1.0` と比べると、本runはID switchとmissedが良く、inactive `1.0` はF1・precision・duplicate・inactive FPがわずかに良いというPareto関係である。

### 次に有効な実験

validation F1が最大だったepoch 1 checkpointを同じtest手順で評価し、後半の過学習分を切り分ける。その上でbackbone / pose headを凍結したまま、GT人数を直接使うフレーム単位のactive-query cardinality loss、またはpresence logitの校正lossを導入し、F1だけでなくduplicate・inactive FP・missedを含む選択基準で比較するのが有効である。
