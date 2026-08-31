---
id: run-i801-dref-pose-beta005-presence-comp-centered-pair010m05-s42
type: run
title: zero-mean competition + pairwise 0.1 / margin 0.5（seed 42）
issue: 801
provider: codex
session: 01a04915-27a4-7b62-9f6b-34275561fded
date: '2026-08-31'
status: done
config:
  model: track_query_ablation_d_v2_selector
  training_config: train_tracking_pose_presence_competition_centered
  init_weights: plcs/i801_dref_pose_beta005_presence_head_hneg050_s42/logs/version_0/checkpoints/plcs-epoch=01.ckpt
  fine_tune_mode: presence_competition
  presence_competition: deepsets_centered
  presence_inactive_weight: 0.5
  match_presence_weight: 0.0
  match_presence_inactive_weight: 0.25
  presence_hard_negative_weight: 0.0
  presence_pairwise_weight: 0.1
  presence_pairwise_margin: 0.5
  cardinality_weight: 0.0
  cardinality_nll_weight: 0.0
  rotation_weight: 0.05
  angle_weight: 0.05
  learning_rate: 0.001
  max_epochs: 8
  sequence_length: 128
  num_views: 6
  seed: 42
metrics:
  loss: 0.968175
  position_error_m: 4.93005
  angular_error_deg: 33.64315
  presence_precision: 0.505125
  presence_recall: 0.983619
  presence_f1: 0.66786
  id_switches: 0.12
  duplicate_active_tracks: 48.05
  missed_gt_frames: 3.56
  inactive_query_false_positives: 172.08
  canonical_mpjpe_m: 0.174601
  reprojection_error_px: 153.20363
  loss_presence_pairwise: 0.472619
  exact_count_accuracy: 0.225781
  gt1_all_four_active_rate: 0.883213
  gt2_all_four_active_rate: 0.999409
  gt3_all_four_active_rate: 1.0
  stable_hardest_gap_mean: -0.165185
  stable_zero_margin_violation_rate: 0.71472
  stable_pairwise_hinge_margin_050: 0.477536
repro:
  commit: a4279f40c422e930a62869b10c212c0e4f669d53
  branch: experiments/plcs-tracking-pose-beta005
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    /home/kamimura/projects/tennis-lab/.venv/bin/python -m src.tasks.plcs.scripts.train
    --config-name train_tracking_pose_presence_competition_centered model=track_query_ablation_d_v2_selector
    court_keypoints=camera_view_v2 model.cswa.backend=cuda paths.data_root=/home/kamimura/projects/tennis-lab/data
    paths.checkpoint_root=/home/kamimura/projects/tennis-lab/.claude/worktrees/plcs-tracking-pose-beta005/outputs
    data.scene_dir=plcs/multi_object_camera_view_v2 data.seq_len_range=\[128,128\]
    data.num_views_range=\[6,6\] data.batch_size=8 data.num_workers=16 data.evaluation_reference_camera_id=camera_2
    run.init_weights="plcs/i801_dref_pose_beta005_presence_head_hneg050_s42/logs/version_0/checkpoints/plcs-epoch\=01.ckpt"
    loss.match_presence_weight=0.0 loss.match_presence_inactive_weight=0.25 loss.presence_inactive_weight=0.5
    loss.cardinality_weight=0.0 loss.cardinality_nll_weight=0.0 loss.presence_hard_negative_weight=0.0
    loss.presence_pairwise_weight=0.1 loss.presence_pairwise_margin=0.5 loss.rotation_weight=0.05
    loss.angle_weight=0.05 training.learning_rate=1e-3 training.compile.enabled=false
    training.trainer.precision=bf16-mixed training.trainer.accumulate_grad_batches=4
    training.trainer.max_epochs=8 training.warmup_steps=10 training.trainer.check_val_every_n_epoch=1
    training.trainer.enable_progress_bar=false training.trainer.enable_model_summary=false
    training.early_stopping.enabled=false training.qualitative_logging.enabled=false
    run.gpus=1 run.seed=42 run.fast_dev_run=false run.test_after_fit=true run.output_dir=plcs/i801_dref_pose_beta005_presence_comp_centered_pair010m05_s42
artifacts:
  run_dir: knowledge/runs/run-i801-dref-pose-beta005-presence-comp-centered-pair010m05-s42
  predictions: knowledge/runs/run-i801-dref-pose-beta005-presence-comp-centered-pair010m05-s42/pred_test.npz
  log: .training_queue/logs/1788160289518477410_3562690_i801_dref_pose_beta005_presence_comp_centered_pair010m05_s42.log
  output_dir: outputs/plcs/i801_dref_pose_beta005_presence_comp_centered_pair010m05_s42/logs/version_0
  curves: knowledge/runs/run-i801-dref-pose-beta005-presence-comp-centered-pair010m05-s42/curves.png
  tb_logdir: outputs/plcs/i801_dref_pose_beta005_presence_comp_centered_pair010m05_s42/logs/version_0
parents:
- run-i801-dref-pose-beta005-presence-head-hneg050-s42
relations:
- to: run-i801-dref-pose-beta005-presence-comp-centered-hneg050-s42
  rel: compares
- to: run-i801-dref-pose-beta005-presence-head-pair010m05-s42
  rel: compares
- to: run-i801-eval-beta005-presence-head-hneg050-e01-thr050-currentmetrics
  rel: compares
tags: [plcs, tracking, pose, presence, competition, deepsets, zero-mean, centered, pairwise, ranking, beta005, seed-42]
---

## 考察 / Findings

### 要約

zero-mean competitionへpairwise weight `0.1` / margin `0.5`を加えたが、最終F1 `0.667860`、exact-count `0.225781`、GT 1人all4 `0.883213`で不採用とする。hinge平均は僅かに縮んだ一方、人数精度はcentered hard-negativeにもcontrolにも届かなかった。

### アーキテクチャ詳細

hard-negative epoch 1 sourceからbias-free `deepsets_centered` branchの3 tensorだけを8 epoch更新した。各stable frameでHungarian-aligned active queryをinactive queryよりmargin `0.5`上へ置く全pair hingeをframe均等平均し、weight `0.1`で加えた。hard-negativeは無効、inactive `0.5`、pairwise `0.1`、cardinality系無効、`rotation_weight=0.05`、`angle_weight=0.05`、T=128、V=6、seed 42である。sourceと共通する175 tensorは全てbitwise同一だった。

### メトリクスの解釈

最終testのprecision / recall / F1は`0.505125 / 0.983619 / 0.667860`、ID switch `0.12`、duplicate `48.05`、missed `3.56`、inactive FP `172.08`だった。raw `loss_presence_pairwise`は`0.472619`、同じpred bundleから公平再集計したstable hinge平均は`0.477536`である。exact-count率は`22.58%`、GT 1 / 2 / 3人all4率は`88.32% / 99.94% / 100%`、stable zero-margin違反は`71.47%`だった。poseはposition `4.930050 m`、angular `33.643150°`、canonical MPJPE `0.174601 m`、reprojection `153.203630 px`である。`curves.png`ではvalidation F1最大`0.674280`、validation loss最小`0.948670`はいずれもepoch 0で、以後改善しなかった。

### アーキテクチャ⇄メトリクスの因果考察

pairwiseはthresholdに依存しないlogit差へ直接作用するため、centered hard-negative最終state比でhingeを`0.478463→0.477536`、zero-margin違反を`71.78%→71.47%`へ僅かに改善した。しかしexact-countは`0.226094→0.225781`、GT 1人all4は`88.07%→88.32%`へ悪化した。仮説として、margin hingeは既に正しく並んだpairの幅を広げても、最も難しいqueryの符号やframe総active数を直接拘束しないため、countへ変換されない。

### 既存実験との比較

best epoch 0の公平評価でも、hinge平均`0.478444`はcontrol `0.502529`より小さい一方、F1は`0.670431<0.671911`、duplicateは`47.35>44.35`、exact-countは`0.230547<0.231016`、GT 1人all4は`84.85%>76.87%`、zero-margin違反は`72.77%>70.03%`だった。従来presence-head pairwiseも既存groupで不採用であり、competition branchとの組合せでも結論を覆さない。

### 次に有効な実験

centered+pairwiseのweight / margin sweepは行わず、DeepSets competition系を終了する。次はhard-negative controlへ戻り、top-k / set cardinalityを構造的に決める機構、またはunmatched queryの符号を直接教師にするassignment-aware objectiveを検討する。
