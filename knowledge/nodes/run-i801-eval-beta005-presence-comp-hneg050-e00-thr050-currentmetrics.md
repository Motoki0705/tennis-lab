---
id: run-i801-eval-beta005-presence-comp-hneg050-e00-thr050-currentmetrics
type: run
title: uncentered competition epoch 0 のcurrent metrics公平評価
issue: 801
provider: codex
session: 01a04915-27a4-7b62-9f6b-34275561fded
date: '2026-08-31'
status: done
config:
  model: track_query_ablation_d_v2_selector
  data: plcs/multi_object_camera_view_v2
  source_checkpoint: plcs/i801_dref_pose_beta005_presence_comp_hneg050_s42/logs/version_0/checkpoints/plcs-epoch=00.ckpt
  source_checkpoint_epoch: 0
  evaluation_only: true
  presence_competition: deepsets
  presence_inactive_weight: 0.5
  presence_hard_negative_weight: 0.5
  presence_hard_negative_gamma: 2.0
  presence_threshold: 0.5
  duplicate_distance_m: 0.05
  id_switch_distance_m: 0.05
  rotation_weight: 0.05
  angle_weight: 0.05
  sequence_length: 128
  num_views: 6
  reference_camera_id: camera_2
metrics:
  position_error_m: 4.929797
  angular_error_deg: 33.645799
  presence_precision: 0.507579
  presence_recall: 0.983064
  presence_f1: 0.670431
  id_switches: 0.13
  duplicate_active_tracks: 46.23
  missed_gt_frames: 3.52
  inactive_query_false_positives: 170.36
  canonical_mpjpe_m: 0.174605
  reprojection_error_px: 155.01878
  exact_count_accuracy: 0.229219
  gt1_all_four_active_rate: 0.830895
  gt2_all_four_active_rate: 0.998226
  gt3_all_four_active_rate: 0.972093
  stable_hardest_gap_mean: -0.19967
  stable_zero_margin_violation_rate: 0.728488
  stable_margin_050_violation_rate: 0.954039
  stable_all_pair_violation_rate: 0.467775
  stable_pairwise_hinge_margin_050: 0.486175
repro:
  commit: 1275bdb154f580872a0e571cc0f7f493226434f9
  branch: experiments/plcs-tracking-pose-beta005
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 /home/kamimura/projects/tennis-lab/.venv/bin/python
    -c 'import torch; import pytorch_lightning as pl; from omegaconf import OmegaConf;
    from src.tasks.plcs.training.tracking_lightning_module import PLCSTrackingLightningModule
    as M; from src.tasks.plcs.data.tracking_datamodule import PLCSTrackingDataModule
    as D; p="/home/kamimura/projects/tennis-lab/.claude/worktrees/plcs-tracking-pose-beta005/outputs/plcs/i801_dref_pose_beta005_presence_comp_hneg050_s42/logs/version_0/checkpoints/plcs-epoch=00.ckpt";
    b=torch.load(p,map_location="cpu",weights_only=False); c=b["hyper_parameters"]["config"];
    c.paths.data_root="/home/kamimura/projects/tennis-lab/data"; c.data.scene_dir="plcs/multi_object_camera_view_v2";
    c.data.seq_len_range=[128,128]; c.data.num_views_range=[6,6]; c.data.evaluation_reference_camera_id="camera_2";
    c.data.batch_size=8; c.data.num_workers=16; OmegaConf.update(c,"tracking_metrics.presence_threshold",0.5,force_add=True);
    OmegaConf.update(c,"tracking_metrics.duplicate_distance",0.05,force_add=True);
    OmegaConf.update(c,"tracking_metrics.id_switch_distance",0.05,force_add=True);
    m=M.load_from_checkpoint(p,config=c,map_location="cpu",weights_only=False); d=D(c);
    t=pl.Trainer(accelerator="gpu",devices=1,precision="bf16-mixed",logger=False,enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False);
    print(t.test(m,datamodule=d))'
artifacts:
  run_dir: knowledge/runs/run-i801-eval-beta005-presence-comp-hneg050-e00-thr050-currentmetrics
  predictions: knowledge/runs/run-i801-eval-beta005-presence-comp-hneg050-e00-thr050-currentmetrics/pred_test.npz
  log: .training_queue/logs/1788151512203652860_3197439_i801_eval_beta005_presence_comp_hneg050_e00_thr050_currentmetrics.log
parents:
- run-i801-dref-pose-beta005-presence-comp-hneg050-s42
relations:
- to: run-i801-eval-beta005-presence-head-hneg050-e01-thr050-compcontrol
  rel: compares
- to: run-i801-eval-beta005-presence-comp-centered-e02-thr050-currentmetrics
  rel: compares
- to: run-i801-eval-beta005-presence-head-hneg050-e01-thr050-currentmetrics
  rel: compares
tags: [plcs, tracking, pose, presence, competition, deepsets, evaluation, current-metrics, threshold-050, fair-contract, beta005]
---

## 考察 / Findings

### 要約

validation F1で選んだuncentered DeepSets epoch 0は、同一評価contractのcontrolよりF1、duplicate、exact-count、GT人数別all4、stable rankingが全て悪く、不採用とする。F1は`0.670431`、exact-count率`0.229219`、GT 1人all4率`0.830895`だった。

### アーキテクチャ詳細

`run-i801-dref-pose-beta005-presence-comp-hneg050-s42`のbest-validation-F1 checkpointをevaluation-onlyで再評価した。query-local / frame-pooled / centered query特徴から得たuncentered residualを凍結済みpresence logitsへ加える。評価はcommit `1275bdb1`、threshold `0.5`、duplicate / ID switch距離各`0.05 m`、同一test split、T=128、V=6、reference camera `camera_2`で固定した。sourceと共通する175 checkpoint tensorはbitwise同一である。

### メトリクスの解釈

precision / recall / F1は`0.507579 / 0.983064 / 0.670431`、ID switch `0.13`、duplicate `46.23`、missed `3.52`、inactive FP `170.36`だった。exact-count率は`22.92%`、GT 1 / 2 / 3人all4率は`83.09% / 99.82% / 97.21%`である。stable hardest gap平均`-0.199670`、zero-margin違反`72.85%`、margin 0.5違反`95.40%`、frame-balanced hinge平均`0.486175`で、相対順位も人数判定も不十分である。poseはposition `4.929797 m`、angular `33.645799°`、canonical MPJPE `0.174605 m`、reprojection `155.018780 px`だった。evaluation-onlyのため独自の収束曲線はない。

### アーキテクチャ⇄メトリクスの因果考察

controlとの差分residualではframe共通成分が二乗エネルギーの`48.22%`、query間相関平均が`0.723`で、共通gateとしての挙動が強い。このshiftはscene内で概ね一定であり、人数に応じて余剰queryだけを下げる競合になっていない。仮説として、pooled特徴とoutput biasがsource logitのscene calibrationを調整する近道を提供し、query順位を改善しないままvalidation BCEを下げた。

### 既存実験との比較

control `run-i801-eval-beta005-presence-head-hneg050-e01-thr050-compcontrol`に対し、F1は`0.671911→0.670431`、duplicateは`44.35→46.23`、exact-count率は`0.231016→0.229219`、GT 1人all4率は`76.87%→83.09%`、stable zero-margin違反は`70.03%→72.85%`へ全て悪化した。recallは`0.981537→0.983064`と`0.001527`増え、missedは`4.00→3.52`へ減ったが、過活性化をさらに強めた代償である。pose指標はcontrolと同値で、tracking差はcompetition branchだけに帰属できる。

### 次に有効な実験

uncentered modeは再試行せず、query meanを厳密に0へ落とすcentered residualで共通gateを除く。採用条件はcontrolのF1`0.671911`、duplicate`44.35`、exact-count`0.231016`、GT 1人all4`0.768659`、stable zero-margin違反`0.700344`を同時に超えることとする。
