---
id: run-i801-dref-pose-beta005-presence-head-pair010m05-s42
type: run
title: presence head pairwise ranking 0.1 / margin 0.5（seed 42）
issue: 801
provider: codex
session: 01a04915-27a4-7b62-9f6b-34275561fded
date: '2026-08-31'
status: done
config:
  model: track_query_ablation_d_v2_selector
  training_config: train_tracking_pose_presence_head
  init_weights: plcs/i801_dref_pose_beta005_s42_r1/logs/version_0/checkpoints/plcs-epoch=69.ckpt
  fine_tune_mode: presence_head
  presence_inactive_weight: 0.5
  match_presence_weight: 0.0
  match_presence_inactive_weight: 0.25
  presence_pairwise_weight: 0.1
  presence_pairwise_margin: 0.5
  presence_hard_negative_weight: 0.0
  cardinality_weight: 0.0
  cardinality_nll_weight: 0.0
  rotation_weight: 0.05
  angle_weight: 0.05
  learning_rate: 0.001
  max_epochs: 8
  sequence_length: 128
  num_views: 6
  presence_threshold: 0.5
  duplicate_distance_m: 0.05
  id_switch_distance_m: 0.05
  seed: 42
metrics:
  position_error_m: 4.93005
  angular_error_deg: 33.64315
  presence_precision: 0.503058
  presence_recall: 0.991711
  presence_f1: 0.667681
  id_switches: 0.12
  duplicate_active_tracks: 48.36
  missed_gt_frames: 1.85
  inactive_query_false_positives: 174.56
  canonical_mpjpe_m: 0.174601
  reprojection_error_px: 153.20363
  loss_presence_pairwise: 0.477001
  exact_count_accuracy: 0.218594
  gt1_all_four_active_rate: 0.925862
  gt2_all_four_active_rate: 0.999409
  stable_margin_050_violation_frame_weighted: 0.85928
repro:
  commit: fafff3ae9f950e2bac274aee2b922defe8c59d56
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
    loss.cardinality_weight=0.0 loss.cardinality_nll_weight=0.0 loss.presence_hard_negative_weight=0.0
    loss.presence_pairwise_weight=0.1 loss.presence_pairwise_margin=0.5 loss.rotation_weight=0.05
    loss.angle_weight=0.05 training.learning_rate=1e-3 training.compile.enabled=false
    training.trainer.precision=bf16-mixed training.trainer.accumulate_grad_batches=4
    training.trainer.max_epochs=8 training.warmup_steps=10 training.trainer.check_val_every_n_epoch=1
    training.trainer.enable_progress_bar=false training.trainer.enable_model_summary=false
    training.qualitative_logging.enabled=false run.gpus=1 run.seed=42 run.fast_dev_run=false
    run.test_after_fit=true run.output_dir=plcs/i801_dref_pose_beta005_presence_head_pair010m05_s42
artifacts:
  run_dir: knowledge/runs/run-i801-dref-pose-beta005-presence-head-pair010m05-s42
  predictions: knowledge/runs/run-i801-dref-pose-beta005-presence-head-pair010m05-s42/pred_test.npz
  log: .training_queue/logs/1788135090990133582_2911382_i801_dref_pose_beta005_presence_head_pair010m05_s42.log
  output_dir: outputs/plcs/i801_dref_pose_beta005_presence_head_pair010m05_s42/logs/version_0
  curves: knowledge/runs/run-i801-dref-pose-beta005-presence-head-pair010m05-s42/curves.png
  tb_logdir: outputs/plcs/i801_dref_pose_beta005_presence_head_pair010m05_s42/logs/version_0
parents:
- run-i801-dref-pose-beta005-presence-head-inact05-s42
relations:
- to: run-i801-eval-beta005-presence-head-inact1-bestvalf1-thr050
  rel: compares
- to: run-i801-dref-pose-beta005-presence-head-hneg050-s42
  rel: compares
tags: [plcs, tracking, pose, presence, fine-tune, pairwise, ranking, beta005, seed-42]
---

## 考察 / Findings

### 要約

stableなactive queryをinactive queryより`0.5` logit上へ並べるpairwise lossをweight `0.1`で加えたが、最終epochのF1は`0.667681`、GT 1人時の4-query全発火率は`92.59%`であり、不採用とする。presence head以外は凍結されたままでposeは保持されたが、query過活性化は解けなかった。

### アーキテクチャ詳細

親のinactive weight `0.5` presence-head fine-tuneへ、Hungarian assignment後かつlifecycle transition半径`2`の外側にあるstable frameで、各active / inactive query対へ`ReLU(logit_negative - logit_positive + 0.5)`を課した。frame内でpair平均してからframe間平均し、`presence_pairwise_weight=0.1`とした。cardinality、exact-count NLL、hard-negativeは無効で、`rotation_weight=0.05`、`angle_weight=0.05`、T=128、V=6、seed 42は親と共通である。epoch 69を初期値としてpresence headの2 tensorだけを8 epoch更新した。

### メトリクスの解釈

最終epochのprecision / recall / F1は`0.503058 / 0.991711 / 0.667681`、ID switch `0.12`、duplicate `48.36`、missed `1.85`、inactive FP `174.56`だった。threshold `0.5`でのexact-count率は`21.86%`、GT 1人時の4-query全発火率は`92.59%`、GT 2人時は`99.94%`である。stable pairのmargin `0.5`違反率もframe-weightedで`85.93%`残った。position `4.930050 m`、angular `33.643150°`、canonical MPJPE `0.174601 m`、reprojection `153.203630 px`である。`curves.png`では凍結したposition / angular validation値が一定で、train / validation lossは初期低下後に頭打ちとなった。best validation F1はepoch 0であり、長い更新での改善は観測されない。

### アーキテクチャ⇄メトリクスの因果考察

pairwise項はquery間の相対順位を直接教師にするが、最終stateの`loss_presence_pairwise=0.477001`と高いmargin違反率から、目的の分離は十分形成されていない。仮説として、同一の線形presence headを共有query特徴へ適用する現在の構造では、active / inactive queryを0.5 logit離す勾配がGT人数に応じた競合へ変換されず、全queryが同時に上がる解を避けられない。presence head限定更新のためpose側を壊さなかった点は安全だが、trackingの律速には届かなかった。

### 既存実験との比較

同じcurrent metric contractでbest checkpointのepoch 0を評価した子runはF1`0.668627`、GT 1人all4率`84.60%`で、最終stateより良い。それでもinactive `1.0` epoch 3のF1`0.672165` / GT 1人all4率`78.45%`、hard-negative epoch 1のF1`0.671911` / `76.87%`に劣る。したがって、best checkpointを選んでもpairwise施策は既存Pareto点を更新しない。

### 次に有効な実験

このpairwise weight / marginの増量は行わず、hard-negative epoch 1またはinactive `1.0` epoch 3を起点に、Hungarian assignmentでunmatchedとなったqueryを直接抑えるloss、またはquery間競合を明示する仕組みを試す。採用判定はF1だけでなくGT人数別all4率とexact-count率をprimary gateにする。
