---
id: run-i801-dref-pose-beta005-presence-head-cnll005-s42
type: run
title: presence head exact cardinality NLL 0.05（seed 42）
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
  cardinality_nll_weight: 0.05
  presence_threshold: 0.5
  learning_rate: 0.001
  max_epochs: 8
  sequence_length: 128
  num_views: 6
  seed: 42
metrics:
  loss: 1.154819
  loss_position: 0.148668
  loss_rotation: 0.236838
  loss_presence: 0.742962
  loss_track_smoothness: 0.0
  loss_angle: 0.25287
  loss_canonical_pose: 0.010316
  loss_reprojection: 0.089208
  loss_cardinality_nll: 2.783588
  position_error: 0.433102
  presence_precision: 0.505878
  presence_recall: 0.985797
  presence_f1: 0.664182
  lifecycle_presence_f1: 0.664182
  birth_frame_error: 14.493006
  death_frame_error: 16.311443
  query_reuse_count: 0.16
  illegal_overlap_count: 0.0
  segment_id_switches: 39.360001
  id_switches: 39.360001
  duplicate_active_tracks: 363.839996
  missed_gt_frames: 20.4
  inactive_query_false_positives: 1348.640015
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
    run.test_after_fit=true loss.cardinality_nll_weight=0.05 run.output_dir=plcs/i801_dref_pose_beta005_presence_head_cnll005_s42
artifacts:
  run_dir: knowledge/runs/run-i801-dref-pose-beta005-presence-head-cnll005-s42
  predictions: knowledge/runs/run-i801-dref-pose-beta005-presence-head-cnll005-s42/pred_test.npz
  output_dir: outputs/plcs/i801_dref_pose_beta005_presence_head_cnll005_s42/logs/version_0
  curves: knowledge/runs/run-i801-dref-pose-beta005-presence-head-cnll005-s42/curves.png
  tb_logdir: outputs/plcs/i801_dref_pose_beta005_presence_head_cnll005_s42/logs/version_0
parents:
- run-i801-dref-pose-beta005-s42-r1
relations:
- to: run-i801-eval-beta005-e69-thr050-r1
  rel: compares
- to: run-i801-dref-pose-beta005-presence-head-inact05-s42
  rel: compares
- to: run-i801-dref-pose-beta005-presence-head-cnll010-s42
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

beta005 epoch 69からpresence headだけをexact cardinality NLL weight `0.05`で8 epoch
fine-tuneした。GT 0人時の不要発火は改善したが、GT 1–3人時は`95.53%`のframeで4 queryが
全発火し、F1 `0.664182`もbaselineを上回らなかったため総合不採用とする。

### アーキテクチャ詳細

`run-i801-dref-pose-beta005-s42-r1` のepoch 69を初期値とし、pose trunkを含む173 tensorを凍結して
65 parameterのpresence headだけを更新した。既存のassignment済みpresence BCEに加え、各frameで
4 queryのBernoulli確率からGT人数のPoisson-binomial確率を厳密計算する、query-permutation-invariantな
`cardinality_nll_weight=0.05` を追加した。`presence_inactive_weight=0.5`、
`match_presence_weight=0.0`、soft cardinality weight `0.0`、T=128、V=6、seed 42である。
pose lossは `rotation_weight=0.05`、`angle_weight=0.05`を維持した。

### メトリクスの解釈

testのraw cardinality NLLは`2.783588`で、総lossへの重み込み寄与は約`0.139179`だった。
precision / recall / F1は `0.505878 / 0.985797 / 0.664182`、ID switch `39.36`、
duplicate `363.84`、missed `20.40`、inactive FP `1348.64`である。pose metricはposition
`4.931891 m`、angular `33.260555°`、canonical MPJPE `0.175773 m`、reprojection
`153.389420 px`だった。GT 0人時の予測active数は平均`0.0230`、全query inactive率`98.83%`へ
改善した一方、GT 1–3人時は平均`3.891`、4-query全発火率`95.53%`だった。
`curves.png`では凍結したposition / angularのvalidation値は全stepで一定で、validation total lossは
最初のstep 25が最小、その後は概ね増加した。8 epoch内でcount学習がvalidation収束を改善した証拠はない。

### アーキテクチャ⇄メトリクスの因果考察

exact count NLLは人数分布へ直接勾配を与えるが、本runではGT 0人時の共通bias低下には効いても、
選手が存在するframeでquery間の対称性を十分に破れなかった。これは仮説だが、固定trunk上で各queryの
presence logitが強く相関し、count NLL `0.05`が8 epochでは個々のqueryを別identityへ割り当てる情報を
持たないためである。pose parameterは凍結されており、pose metricの小差はpresenceに依存する
assignment / gatingの変化として解釈するのが妥当である。

### 既存実験との比較

同じtest split・threshold `0.5` のepoch 69評価に対し、inactive FPは
`1361.28→1348.64`、ID switchは `40.04→39.36`、duplicateは `366.08→363.84`へ小幅改善した。
一方、recallは `0.990172→0.985797`、F1は `0.664414→0.664182`、missedは
`13.76→20.40`と悪化した。inactive `0.5` のbest-val-F1評価に対してもF1とduplicateが悪く、
同一thresholdの公平比較で優位性はない。GT 1–3人のcollapse率もepoch 69の`96.05%`から
`95.53%`への微減に留まるため、本施策は採用しない。

### 次に有効な実験

weight `0.1`でcount勾配不足かを一度切り分ける。ただし同じGT 1–3人collapseが残る場合、
count-only objectiveの増量は打ち切り、matched / unmatched queryを区別するhard-negative loss、
またはpresence専用feature branchでquery identityを分離する施策へ移る。
