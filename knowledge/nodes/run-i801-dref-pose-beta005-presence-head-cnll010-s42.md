---
id: run-i801-dref-pose-beta005-presence-head-cnll010-s42
type: run
title: presence head exact cardinality NLL 0.10（seed 42）
issue: 801
provider: codex
session: 01a04915-27a4-7b62-9f6b-34275561fded
date: '2026-08-30'
status: done
config:
  model: track_query_ablation_d_v2_selector
  training_config: train_tracking_pose_presence_head
  loss: tracking_all_outputs_beta01_reprojection
  data: plcs/multi_object_camera_view_v2
  init_weights: plcs/i801_dref_pose_beta005_s42_r1/logs/version_0/checkpoints/plcs-epoch=69.ckpt
  fine_tune_mode: presence_head
  rotation_weight: 0.05
  angle_weight: 0.05
  presence_inactive_weight: 0.5
  match_presence_weight: 0.0
  match_presence_inactive_weight: 0.25
  cardinality_weight: 0.0
  cardinality_nll_weight: 0.1
  presence_threshold: 0.5
  learning_rate: 0.001
  max_epochs: 8
  sequence_length: 128
  num_views: 6
  seed: 42
metrics:
  loss: 1.284831
  loss_position: 0.148668
  loss_rotation: 0.236838
  loss_presence: 0.737438
  loss_track_smoothness: 0.0
  loss_angle: 0.25287
  loss_canonical_pose: 0.010316
  loss_reprojection: 0.089208
  loss_cardinality_nll: 2.747144
  position_error: 0.433102
  presence_precision: 0.506873
  presence_recall: 0.983955
  presence_f1: 0.664627
  lifecycle_presence_f1: 0.664627
  birth_frame_error: 14.231176
  death_frame_error: 16.266655
  query_reuse_count: 0.16
  illegal_overlap_count: 0.0
  segment_id_switches: 40.48
  id_switches: 40.48
  duplicate_active_tracks: 361.519989
  missed_gt_frames: 23.24
  inactive_query_false_positives: 1344.959961
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
    loss.cardinality_weight=0.0 loss.rotation_weight=0.05 loss.angle_weight=0.05 training.learning_rate=1e-3
    training.compile.enabled=false training.trainer.precision=bf16-mixed training.trainer.accumulate_grad_batches=4
    training.trainer.max_epochs=8 training.warmup_steps=10 training.trainer.check_val_every_n_epoch=1
    training.trainer.enable_progress_bar=false training.trainer.enable_model_summary=false
    training.qualitative_logging.enabled=false run.gpus=1 run.seed=42 run.fast_dev_run=false
    run.test_after_fit=true loss.cardinality_nll_weight=0.1 run.output_dir=plcs/i801_dref_pose_beta005_presence_head_cnll010_s42
artifacts:
  run_dir: knowledge/runs/run-i801-dref-pose-beta005-presence-head-cnll010-s42
  predictions: knowledge/runs/run-i801-dref-pose-beta005-presence-head-cnll010-s42/pred_test.npz
  output_dir: outputs/plcs/i801_dref_pose_beta005_presence_head_cnll010_s42/logs/version_0
  curves: knowledge/runs/run-i801-dref-pose-beta005-presence-head-cnll010-s42/curves.png
  tb_logdir: outputs/plcs/i801_dref_pose_beta005_presence_head_cnll010_s42/logs/version_0
parents:
- run-i801-dref-pose-beta005-s42-r1
relations:
- to: run-i801-eval-beta005-e69-thr050-r1
  rel: compares
- to: run-i801-dref-pose-beta005-presence-head-inact05-s42
  rel: compares
- to: run-i801-dref-pose-beta005-presence-head-cnll005-s42
  rel: compares
tags:
- plcs
- tracking
- pose
- presence
- fine-tune
- cardinality-nll
- beta005
- seed-42
---

## 考察 / Findings

### 要約

exact cardinality NLLを`0.10`へ増やしてpresence headを8 epoch fine-tuneしたが、
GT 1–3人時の4-query全発火率は`94.53%`でcollapseを解消できなかった。F1 `0.664627`、
ID switch `40.48`も総合改善を示さず、weight `0.05`と同様に不採用とする。

### アーキテクチャ詳細

`run-i801-dref-pose-beta005-s42-r1` のepoch 69から、pose trunkを凍結してpresence headだけを更新した。
Poisson-binomial exact count NLLはquery permutationに不変で、GT人数の確率へ直接作用する。
`cardinality_nll_weight=0.10`以外は`.05` runと同一で、`presence_inactive_weight=0.5`、
`match_presence_weight=0.0`、soft cardinality weight `0.0`、T=128、V=6、seed 42である。
`rotation_weight=0.05`、`angle_weight=0.05`も維持した。

### メトリクスの解釈

raw cardinality NLLは`2.747144`、重み込み寄与は約`0.274714`である。precision / recall / F1は
`0.506873 / 0.983955 / 0.664627`、ID switch `40.48`、duplicate `361.52`、missed
`23.24`、inactive FP `1344.96`だった。pose metricはposition `4.931891 m`、angular
`33.260555°`、canonical MPJPE `0.175773 m`、reprojection `153.389420 px`で`.05` runと一致する。
GT 0人時は予測active数平均`0.0202`、全query inactive率`99.01%`だが、GT 1–3人時は平均
`3.877`、4-query全発火率`94.53%`だった。
`curves.png`では凍結したposition / angularのvalidation値は一定で、validation total lossはstep 50を
底に上昇し、後半も初期値へ戻っていない。weight増加による安定したvalidation収束は観測されなかった。

### アーキテクチャ⇄メトリクスの因果考察

NLL寄与を`.05`の約2倍にしても、主な変化はGT 0人時の共通抑制とGT 1–3人時のごく小さい
active数低下だった。count objectiveは期待人数を制約できても、どのqueryがどのplayerを表すべきかという
identity情報を持たない。固定trunk上でquery logitが同時に動く条件では、weight増加だけでは対称な
4-query active解から抜けられないという仮説が、観測されたcollapse率と整合する。

### 既存実験との比較

同じtest split・threshold `0.5` のepoch 69評価に対し、precisionは
`0.505068→0.506873`、duplicateは `366.08→361.52`、inactive FPは `1361.28→1344.96`へ
小幅改善した。一方、recallは `0.990172→0.983955`、missedは `13.76→23.24`、ID switchは
`40.04→40.48`へ悪化し、F1差も`+0.000213`に過ぎない。NLL `.05`に対してF1とduplicateは
僅かに良いが、ID switchとmissedが悪く、GT 1–3人collapse率も`95.53→94.53%`の微減だけである。
同一thresholdの公平比較でtracking + poseの優位性はなく、総合不採用と判断する。

### 次に有効な実験

exact count NLLの追加増量は打ち切る。matched / unmatched queryのhard negativeへ直接作用する
focalまたはmargin loss、あるいはpresence専用branchでqueryごとのidentity特徴を作り、
GT 1–3人時の4-query全発火率を主要gateとして評価する。
