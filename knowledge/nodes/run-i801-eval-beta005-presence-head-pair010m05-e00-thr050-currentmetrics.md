---
id: run-i801-eval-beta005-presence-head-pair010m05-e00-thr050-currentmetrics
type: run
title: pairwise 0.1 best-val epoch 0 のcurrent metrics公平評価
issue: 801
provider: codex
session: 01a04915-27a4-7b62-9f6b-34275561fded
date: '2026-08-31'
status: done
config:
  model: track_query_ablation_d_v2_selector
  data: plcs/multi_object_camera_view_v2
  source_checkpoint: plcs/i801_dref_pose_beta005_presence_head_pair010m05_s42/logs/version_0/checkpoints/plcs-epoch=00.ckpt
  source_checkpoint_epoch: 0
  source_checkpoint_selection: best val/presence_f1
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
  position_error_m: 4.929797
  angular_error_deg: 33.645799
  presence_precision: 0.505607
  presence_recall: 0.98512
  presence_f1: 0.668627
  id_switches: 0.13
  duplicate_active_tracks: 46.77
  missed_gt_frames: 3.14
  inactive_query_false_positives: 172.12
  canonical_mpjpe_m: 0.174605
  reprojection_error_px: 155.01878
  exact_count_accuracy: 0.225313
  gt1_all_four_active_rate: 0.84602
  gt2_all_four_active_rate: 0.999409
  gt3_all_four_active_rate: 0.972093
  stable_hardest_gap_mean: -0.233844
  stable_zero_margin_violation_rate: 0.70156
  stable_margin_050_violation_rate: 0.93551
  stable_all_pair_violation_rate: 0.4529
repro:
  commit: fafff3ae9f950e2bac274aee2b922defe8c59d56
  branch: experiments/plcs-tracking-pose-beta005
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 /home/kamimura/projects/tennis-lab/.venv/bin/python
    -c 'import torch; import pytorch_lightning as pl; from omegaconf import OmegaConf;
    from src.tasks.plcs.training.tracking_lightning_module import PLCSTrackingLightningModule
    as M; from src.tasks.plcs.data.tracking_datamodule import PLCSTrackingDataModule
    as D; p="/home/kamimura/projects/tennis-lab/.claude/worktrees/plcs-tracking-pose-beta005/outputs/plcs/i801_dref_pose_beta005_presence_head_pair010m05_s42/logs/version_0/checkpoints/plcs-epoch=00.ckpt";
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
  run_dir: knowledge/runs/run-i801-eval-beta005-presence-head-pair010m05-e00-thr050-currentmetrics
  predictions: knowledge/runs/run-i801-eval-beta005-presence-head-pair010m05-e00-thr050-currentmetrics/pred_test.npz
  log: .training_queue/logs/1788143669187105481_3048473_i801_eval_beta005_presence_head_pair010m05_e00_thr050_currentmetrics.log
parents:
- run-i801-dref-pose-beta005-presence-head-pair010m05-s42
relations:
- to: run-i801-eval-beta005-presence-head-inact1-e03-thr050-currentmetrics
  rel: compares
- to: run-i801-eval-beta005-presence-head-hneg050-e01-thr050-currentmetrics
  rel: compares
- to: run-i801-eval-beta005-e69-thr050-currentmetrics
  rel: compares
tags: [plcs, tracking, pose, presence, evaluation, pairwise, current-metrics, threshold-050, fair-contract, beta005]
---

## 考察 / Findings

### 要約

pairwise trainingのbest `val/presence_f1` checkpoint（epoch 0）をcurrent metricsで公平再評価するとF1は`0.668627`だった。最終epochよりは良いが、GT 1人時の4-query全発火率`84.60%`、exact-count率`22.53%`、stable ranking違反率から、pairwise施策は不採用とする。

### アーキテクチャ詳細

親runでbeta005 epoch 69からpresence headだけをpairwise weight `0.1`、margin `0.5`、inactive weight `0.5`でfine-tuneし、best validation F1のepoch 0を学習せずにtestした。評価contractはcommit `fafff3ae`のcurrent aggregation、threshold `0.5`、duplicate / ID switch距離各`0.05 m`、同一100 sceneのtest split、T=128、V=6、reference camera `camera_2`である。pose側の`rotation_weight=0.05`、`angle_weight=0.05`はsource checkpointから維持した。

### メトリクスの解釈

precision / recall / F1は`0.505607 / 0.985120 / 0.668627`、ID switch `0.13`、duplicate `46.77`、missed `3.14`、inactive FP `172.12`だった。position `4.929797 m`、angular `33.645799°`、canonical MPJPE `0.174605 m`、reprojection `155.018780 px`である。人数別ではGT 1人のall4率`84.60%`、GT 2人`99.94%`、GT 3人`97.21%`で、全体exact-count率は`22.53%`に留まる。Hungarian再matching後のstable frameではhardest gap平均`-0.233844`、zero-margin違反`70.16%`、margin 0.5違反`93.55%`だった。evaluation-onlyのため独自の収束曲線はない。

### アーキテクチャ⇄メトリクスの因果考察

epoch 0は最終stateよりGT 1人時の過活性化を抑えたが、GT 2人では事実上全queryが発火したままである。仮説として、初期更新はpresence calibrationを改善した一方、pairwise目的はquery間の相対順位だけを扱い、thresholdを跨ぐquery総数を直接拘束しないため、人数collapseを解けなかった。hardest gapも負で、stableな正例の最低logitが負例の最高logitを下回るframeが多い。

### 既存実験との比較

親の最終stateに対しF1は`0.667681→0.668627`、GT 1人all4率は`92.59%→84.60%`、exact-count率は`21.86%→22.53%`へ改善した。しかし同一contractのinactive `1.0` epoch 3はF1`0.672165` / all4`78.45%`、hard-negative epoch 1はF1`0.671911` / all4`76.87%`で、pairwiseより良い。stable zero-margin違反もhard-negativeの`70.03%`を更新せず、margin 0.5違反`93.55%`はepoch 69の`92.58%`より悪い。

### 次に有効な実験

pairwise weight / marginの探索は打ち切る。hard-negative epoch 1またはinactive `1.0` epoch 3から、unmatched queryをassignment-awareに抑える目的、またはquery間競合を直接表現する目的へ進み、GT 1人・2人all4率とexact-count率を主要なcheckpoint選択条件にする。
