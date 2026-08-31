---
id: run-i801-eval-beta005-e69-thr050-currentmetrics
type: run
title: beta005 source epoch 69 のcurrent metrics公平評価
issue: 801
provider: codex
session: 01a04915-27a4-7b62-9f6b-34275561fded
date: '2026-08-31'
status: done
config:
  model: track_query_ablation_d_v2_selector
  data: plcs/multi_object_camera_view_v2
  source_checkpoint: plcs/i801_dref_pose_beta005_s42_r1/logs/version_0/checkpoints/plcs-epoch=69.ckpt
  source_checkpoint_epoch: 69
  source_checkpoint_selection: best val/loss
  evaluation_only: true
  presence_threshold: 0.5
  duplicate_distance_m: 0.05
  id_switch_distance_m: 0.05
  rotation_weight: 0.05
  angle_weight: 0.05
  sequence_length: 128
  num_views: 6
  reference_camera_id: camera_2
metrics:
  position_error_m: 4.923846
  angular_error_deg: 33.483538
  presence_precision: 0.502481
  presence_recall: 0.991527
  presence_f1: 0.667258
  id_switches: 0.13
  duplicate_active_tracks: 48.32
  missed_gt_frames: 1.9
  inactive_query_false_positives: 175.28
  canonical_mpjpe_m: 0.174542
  reprojection_error_px: 154.915043
  exact_count_accuracy: 0.220156
  gt1_all_four_active_rate: 0.909497
  gt2_all_four_active_rate: 0.999409
  gt3_all_four_active_rate: 0.989147
  stable_hardest_gap_mean: -0.240028
  stable_zero_margin_violation_rate: 0.7052
  stable_margin_050_violation_rate: 0.92579
  stable_all_pair_violation_rate: 0.46348
repro:
  commit: fafff3ae9f950e2bac274aee2b922defe8c59d56
  branch: experiments/plcs-tracking-pose-beta005
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 /home/kamimura/projects/tennis-lab/.venv/bin/python
    -c 'import torch; import pytorch_lightning as pl; from omegaconf import OmegaConf;
    from src.tasks.plcs.training.tracking_lightning_module import PLCSTrackingLightningModule
    as M; from src.tasks.plcs.data.tracking_datamodule import PLCSTrackingDataModule
    as D; p="/home/kamimura/projects/tennis-lab/.claude/worktrees/plcs-tracking-pose-beta005/outputs/plcs/i801_dref_pose_beta005_s42_r1/logs/version_0/checkpoints/plcs-epoch=69.ckpt";
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
  run_dir: knowledge/runs/run-i801-eval-beta005-e69-thr050-currentmetrics
  predictions: knowledge/runs/run-i801-eval-beta005-e69-thr050-currentmetrics/pred_test.npz
  log: .training_queue/logs/1788143669443712009_3048549_i801_eval_beta005_e69_thr050_currentmetrics.log
parents:
- run-i801-dref-pose-beta005-s42-r1
relations:
- to: run-i801-eval-beta005-presence-head-pair010m05-e00-thr050-currentmetrics
  rel: compares
- to: run-i801-eval-beta005-presence-head-inact1-e03-thr050-currentmetrics
  rel: compares
- to: run-i801-eval-beta005-presence-head-hneg050-e01-thr050-currentmetrics
  rel: compares
tags: [plcs, tracking, pose, presence, evaluation, baseline, current-metrics, threshold-050, fair-contract, beta005]
---

## 考察 / Findings

### 要約

presence-head fine-tune前のbeta005 epoch 69をcurrent metricsで再評価した基準点である。4候補中でrecall`0.991527`とmissed`1.90`は最良だが、F1`0.667258`、GT 1人all4率`90.95%`、exact-count率`22.02%`で、過活性化が強い。

### アーキテクチャ詳細

tracking + canonical pose + reprojectionを`rotation_weight=0.05`、`angle_weight=0.05`で学習したbeta005 sourceのbest `val/loss` epoch 69を、追加学習せず評価した。評価contractはcommit `fafff3ae`、presence threshold `0.5`、duplicate / ID switch距離各`0.05 m`、同一test split、T=128、V=6、reference camera `camera_2`で固定した。本nodeはcurrent aggregationでのfine-tune前baselineである。

### メトリクスの解釈

precision / recall / F1は`0.502481 / 0.991527 / 0.667258`、ID switch `0.13`、duplicate `48.32`、missed `1.90`、inactive FP `175.28`だった。position `4.923846 m`、angular `33.483538°`、canonical MPJPE `0.174542 m`、reprojection `154.915043 px`で、presence-head候補よりpose値はわずかに良い。人数別all4率はGT 1人`90.95%`、GT 2人`99.94%`、GT 3人`98.91%`、exact-count率`22.02%`である。stable hardest gap平均`-0.240028`、zero-margin違反`70.52%`、margin 0.5違反`92.58%`だった。evaluation-onlyのため独自の収束曲線はない。

### アーキテクチャ⇄メトリクスの因果考察

低いpresence抑制のsourceはactive判定を広く保つためrecallとmissedが良い反面、precision、duplicate、inactive FP、人数別all4率が悪い。stable margin 0.5違反は4候補で最小でもthreshold後の人数精度は低く、相対ranking単独ではlogit calibrationとactive query数を保証しないことも示している。

### 既存実験との比較

同一contractのinactive `1.0` epoch 3はF1を`+0.004907`、duplicateを`-3.57`、inactive FPを`-6.80`、GT 1人all4率を`-12.50`ポイント改善したが、missedを`+1.87`増やした。hard-negative epoch 1はF1`+0.004653`、duplicate`-3.97`、inactive FP`-7.32`、GT 1人all4率`-14.08`ポイントで、missedは`+2.10`である。pairwise epoch 0も改善するが両者に及ばない。過去の`run-i801-eval-beta005-e69-thr050-r1`は旧aggregationのため、ID / duplicate等の絶対値を本nodeと直接比較しない。

### 次に有効な実験

recall / missed下限の基準として本checkpointを残し、headlineはinactive `1.0`、過活性抑制はhard-negativeを次の起点とする。新施策は本runのrecall優位をどこまで保てるかも確認しつつ、人数別all4率とexact-count率を直接改善する必要がある。
