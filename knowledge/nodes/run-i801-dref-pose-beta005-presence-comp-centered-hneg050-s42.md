---
id: run-i801-dref-pose-beta005-presence-comp-centered-hneg050-s42
type: run
title: zero-mean DeepSets competition + hard-negative 0.5（seed 42）
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
  presence_hard_negative_weight: 0.5
  presence_hard_negative_gamma: 2.0
  presence_pairwise_weight: 0.0
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
  loss: 1.38734
  position_error_m: 4.93005
  angular_error_deg: 33.64315
  presence_precision: 0.505125
  presence_recall: 0.983249
  presence_f1: 0.66786
  id_switches: 0.12
  duplicate_active_tracks: 48.01
  missed_gt_frames: 3.59
  inactive_query_false_positives: 171.96
  canonical_mpjpe_m: 0.174601
  reprojection_error_px: 153.20363
  exact_count_accuracy: 0.226094
  gt1_all_four_active_rate: 0.880734
  gt2_all_four_active_rate: 0.999409
  gt3_all_four_active_rate: 1.0
  stable_hardest_gap_mean: -0.167467
  stable_zero_margin_violation_rate: 0.717757
  stable_pairwise_hinge_margin_050: 0.478463
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
    loss.cardinality_weight=0.0 loss.cardinality_nll_weight=0.0 loss.presence_hard_negative_weight=0.5
    loss.presence_hard_negative_gamma=2.0 loss.presence_pairwise_weight=0.0 loss.rotation_weight=0.05
    loss.angle_weight=0.05 training.learning_rate=1e-3 training.compile.enabled=false
    training.trainer.precision=bf16-mixed training.trainer.accumulate_grad_batches=4
    training.trainer.max_epochs=8 training.warmup_steps=10 training.trainer.check_val_every_n_epoch=1
    training.trainer.enable_progress_bar=false training.trainer.enable_model_summary=false
    training.early_stopping.enabled=false training.qualitative_logging.enabled=false
    run.gpus=1 run.seed=42 run.fast_dev_run=false run.test_after_fit=true run.output_dir=plcs/i801_dref_pose_beta005_presence_comp_centered_hneg050_s42
artifacts:
  run_dir: knowledge/runs/run-i801-dref-pose-beta005-presence-comp-centered-hneg050-s42
  predictions: knowledge/runs/run-i801-dref-pose-beta005-presence-comp-centered-hneg050-s42/pred_test.npz
  log: .training_queue/logs/1788154451810698694_3402102_i801_dref_pose_beta005_presence_comp_centered_hneg050_s42.log
  output_dir: outputs/plcs/i801_dref_pose_beta005_presence_comp_centered_hneg050_s42/logs/version_0
  curves: knowledge/runs/run-i801-dref-pose-beta005-presence-comp-centered-hneg050-s42/curves.png
  tb_logdir: outputs/plcs/i801_dref_pose_beta005_presence_comp_centered_hneg050_s42/logs/version_0
parents:
- run-i801-dref-pose-beta005-presence-head-hneg050-s42
relations:
- to: run-i801-dref-pose-beta005-presence-comp-hneg050-s42
  rel: compares
- to: run-i801-dref-pose-beta005-presence-comp-centered-pair010m05-s42
  rel: compares
- to: run-i801-eval-beta005-presence-head-hneg050-e01-thr050-currentmetrics
  rel: compares
tags: [plcs, tracking, pose, presence, competition, deepsets, zero-mean, centered, hard-negative, beta005, seed-42]
---

## 考察 / Findings

### 要約

uncentered branchの共通gateを除くためresidualをquery方向にzero-mean化したが、最終F1は`0.667860`、exact-count率`0.226094`、GT 1人all4率`0.880734`で不採用とする。共通shiftは除去できても、ranking / countはcontrolを超えなかった。

### アーキテクチャ詳細

親のhard-negative `0.5` epoch 1 checkpointから、`presence_competition=deepsets_centered`のbias-free 3 tensorだけを8 epoch更新した。query-local、frame mean、その差を使うDeepSets residualを出した後、query meanを減算して各frameのresidual総和を0にする。inactive `0.5`、hard-negative `0.5` / gamma `2.0`、pairwise / cardinality無効、`rotation_weight=0.05`、`angle_weight=0.05`、T=128、V=6、seed 42は比較runと共通である。sourceの175 tensorは全てbitwise同一だった。

### メトリクスの解釈

最終testのprecision / recall / F1は`0.505125 / 0.983249 / 0.667860`、ID switch `0.12`、duplicate `48.01`、missed `3.59`、inactive FP `171.96`だった。exact-count率は`22.61%`、GT 1 / 2 / 3人all4率は`88.07% / 99.94% / 100%`、stable hardest gap平均`-0.167467`、zero-margin違反`71.78%`、hinge平均`0.478463`である。poseはposition `4.930050 m`、angular `33.643150°`、canonical MPJPE `0.174601 m`、reprojection `153.203630 px`だった。`curves.png`ではvalidation F1の最大はepoch 2の`0.674096`だが、全8 epochを通じて`0.67327–0.67410`の狭い範囲に留まり、validation lossはepoch 0の`1.314547`から僅かに悪化した。

### アーキテクチャ⇄メトリクスの因果考察

controlとの差分residualのframe mean絶対値は平均`0.00108`、共通成分の二乗エネルギー比は`0.015%`まで下がり、zero-mean contractは実測でも成立した。それでもbest checkpointのexact-countとstable rankingがcontrolより悪い。仮説として、残差和を0にしても「GT active queryを上げ、unmatched queryを下げる」割当情報はbranch入力に無く、任意のquery間再配分が人数に対応しない。共通gateはuncentered失敗の一因だが、主因ではない。

### 既存実験との比較

uncentered最終stateに対しF1は`0.667433→0.667860`、duplicateは`48.16→48.01`、exact-countは`0.224063→0.226094`へ僅かに改善した。しかし同一contractのbest epoch 2はF1`0.669057`、duplicate`47.67`、exact-count`0.227734`、GT 1人all4`86.14%`、zero-margin違反`71.44%`で、controlの`0.671911 / 44.35 / 0.231016 / 76.87% / 70.03%`に全て劣る。zero-mean化だけでは採用水準に届かない。

### 次に有効な実験

centered hard-negative単独は打ち切る。branch容量やepochを増やす前に、同じzero-mean branchへ明示的なpairwise rankingを加え、相対分離を直接最適化する。ただし採用はhinge低下だけでなくF1、exact-count、GT人数別all4、stable zero-margin違反がcontrolを同時に超える場合に限る。
