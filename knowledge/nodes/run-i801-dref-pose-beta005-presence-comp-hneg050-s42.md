---
id: run-i801-dref-pose-beta005-presence-comp-hneg050-s42
type: run
title: DeepSets presence competition + hard-negative 0.5（seed 42）
issue: 801
provider: codex
session: 01a04915-27a4-7b62-9f6b-34275561fded
date: '2026-08-31'
status: done
config:
  model: track_query_ablation_d_v2_selector
  training_config: train_tracking_pose_presence_competition
  init_weights: plcs/i801_dref_pose_beta005_presence_head_hneg050_s42/logs/version_0/checkpoints/plcs-epoch=01.ckpt
  fine_tune_mode: presence_competition
  presence_competition: deepsets
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
  max_epochs: 12
  sequence_length: 128
  num_views: 6
  seed: 42
metrics:
  loss: 1.397517
  position_error_m: 4.93005
  angular_error_deg: 33.64315
  presence_precision: 0.504148
  presence_recall: 0.984181
  presence_f1: 0.667433
  id_switches: 0.12
  duplicate_active_tracks: 48.16
  missed_gt_frames: 3.36
  inactive_query_false_positives: 172.72
  canonical_mpjpe_m: 0.174601
  reprojection_error_px: 153.20363
  exact_count_accuracy: 0.224063
  gt1_all_four_active_rate: 0.888421
  gt2_all_four_active_rate: 0.999409
  gt3_all_four_active_rate: 1.0
  stable_hardest_gap_mean: -0.179868
  stable_zero_margin_violation_rate: 0.731525
  stable_pairwise_hinge_margin_050: 0.479152
repro:
  commit: 1275bdb154f580872a0e571cc0f7f493226434f9
  branch: experiments/plcs-tracking-pose-beta005
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    /home/kamimura/projects/tennis-lab/.venv/bin/python -m src.tasks.plcs.scripts.train
    --config-name train_tracking_pose_presence_competition model=track_query_ablation_d_v2_selector
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
    training.trainer.max_epochs=12 training.warmup_steps=10 training.trainer.check_val_every_n_epoch=1
    training.trainer.enable_progress_bar=false training.trainer.enable_model_summary=false
    training.early_stopping.enabled=false training.qualitative_logging.enabled=false
    run.gpus=1 run.seed=42 run.fast_dev_run=false run.test_after_fit=true run.output_dir=plcs/i801_dref_pose_beta005_presence_comp_hneg050_s42
artifacts:
  run_dir: knowledge/runs/run-i801-dref-pose-beta005-presence-comp-hneg050-s42
  predictions: knowledge/runs/run-i801-dref-pose-beta005-presence-comp-hneg050-s42/pred_test.npz
  log: .training_queue/logs/1788148740388604943_3147109_i801_dref_pose_beta005_presence_comp_hneg050_s42.log
  output_dir: outputs/plcs/i801_dref_pose_beta005_presence_comp_hneg050_s42/logs/version_0
  curves: knowledge/runs/run-i801-dref-pose-beta005-presence-comp-hneg050-s42/curves.png
  tb_logdir: outputs/plcs/i801_dref_pose_beta005_presence_comp_hneg050_s42/logs/version_0
parents:
- run-i801-dref-pose-beta005-presence-head-hneg050-s42
relations:
- to: run-i801-dref-pose-beta005-presence-comp-centered-hneg050-s42
  rel: compares
- to: run-i801-eval-beta005-presence-head-hneg050-e01-thr050-currentmetrics
  rel: compares
tags: [plcs, tracking, pose, presence, competition, deepsets, hard-negative, beta005, seed-42]
---

## 考察 / Findings

### 要約

converged presence logitsへquery-local特徴とframe meanを入力するDeepSets residualを追加し、branchだけを12 epoch学習したが、不採用とする。最終testはF1 `0.667433`、duplicate `48.16`、exact-count率`22.41%`、GT 1人all4率`88.84%`で、source controlを更新せずquery過活性化も残った。

### アーキテクチャ詳細

親のhard-negative `0.5` epoch 1 checkpointから、既存175 state tensorを凍結し、`presence_competition=deepsets`の4 tensorだけを更新した。各query hidden、frame内query mean、その差を結合してMLPへ通し、従来presence logitへ加算するpermutation-equivariant residualである。output biasを持つためquery共通方向のshiftも表現できる。hard-negative `0.5`、`gamma=2.0`、inactive `0.5`を維持し、pairwise / cardinality系は無効、`rotation_weight=0.05`、`angle_weight=0.05`、T=128、V=6、seed 42である。

### メトリクスの解釈

最終testのprecision / recall / F1は`0.504148 / 0.984181 / 0.667433`、ID switch `0.12`、duplicate `48.16`、missed `3.36`、inactive FP `172.72`だった。exact-count率は`0.224063`、GT 1 / 2 / 3人all4率は`88.84% / 99.94% / 100%`である。stable hardest gap平均`-0.179868`、zero-margin違反率`73.15%`で、query順位も成立していない。poseはposition `4.930050 m`、angular `33.643150°`、canonical MPJPE `0.174601 m`、reprojection `153.203630 px`である。`curves.png`ではvalidation F1はepoch 0の`0.672905`が最大で、その後`0.6703–0.6714`へ低下し、validation lossもepoch 1以降反発している。

### アーキテクチャ⇄メトリクスの因果考察

controlとのlogit差をframeごとに分解すると、frame mean成分がresidual二乗エネルギーの`48.22%`を占め、query間residual相関の平均は`0.723`だった。多くのsceneでframe mean shiftが時間方向にほぼ一定であり、このbranchは人数に応じた競合よりscene-levelの共通gateとして使われたと解釈できる。仮説として、shared pooled stateとoutput biasにより全queryを一緒に上下させる近道が存在し、query cardinalityを分ける勾配を吸収した。checkpoint比較ではsourceと共通する175 tensorが全てbitwise同一で、pose保持は意図どおりである。

### 既存実験との比較

同一current-metrics contractで選んだepoch 0評価はF1`0.670431`、duplicate`46.23`、exact-count`0.229219`、GT 1人all4`83.09%`で、control hard-negative epoch 1の`0.671911 / 44.35 / 0.231016 / 76.87%`の全てに劣る。stable zero-margin違反もcontrol `70.03%`から`72.85%`へ悪化した。したがって最終stateだけでなくbest-validation checkpointでも既存Pareto点を更新しない。

### 次に有効な実験

uncentered DeepSetsは打ち切る。共通shiftを構造的に除いたzero-mean residualを同じsourceと評価contractで比較し、それでもexact-count / GT人数別all4 / stable rankingがcontrolを超えない場合は、post-hoc residual branchではなくquery interaction自体へ競合を導入する。
