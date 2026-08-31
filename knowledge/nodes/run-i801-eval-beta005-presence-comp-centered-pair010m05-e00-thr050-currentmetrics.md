---
id: run-i801-eval-beta005-presence-comp-centered-pair010m05-e00-thr050-currentmetrics
type: run
title: zero-mean competition + pairwise epoch 0 の公平評価
issue: 801
provider: codex
session: 01a04915-27a4-7b62-9f6b-34275561fded
date: '2026-08-31'
status: done
config:
  model: track_query_ablation_d_v2_selector
  data: plcs/multi_object_camera_view_v2
  source_checkpoint: plcs/i801_dref_pose_beta005_presence_comp_centered_pair010m05_s42/logs/version_0/checkpoints/plcs-epoch=00.ckpt
  source_checkpoint_epoch: 0
  evaluation_only: true
  presence_competition: deepsets_centered
  presence_inactive_weight: 0.5
  presence_hard_negative_weight: 0.0
  presence_pairwise_weight: 0.1
  presence_pairwise_margin: 0.5
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
  presence_recall: 0.983249
  presence_f1: 0.670431
  id_switches: 0.13
  duplicate_active_tracks: 47.35
  missed_gt_frames: 3.57
  inactive_query_false_positives: 170.32
  canonical_mpjpe_m: 0.174605
  reprojection_error_px: 155.01878
  loss_presence_pairwise: 0.473345
  exact_count_accuracy: 0.230547
  gt1_all_four_active_rate: 0.8485
  gt2_all_four_active_rate: 0.996453
  gt3_all_four_active_rate: 0.974031
  stable_hardest_gap_mean: -0.182539
  stable_zero_margin_violation_rate: 0.727678
  stable_margin_050_violation_rate: 0.965884
  stable_all_pair_violation_rate: 0.478144
  stable_pairwise_hinge_margin_050: 0.478444
repro:
  commit: a4279f40c422e930a62869b10c212c0e4f669d53
  branch: experiments/plcs-tracking-pose-beta005
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 /home/kamimura/projects/tennis-lab/.venv/bin/python
    -c 'import torch; import pytorch_lightning as pl; from omegaconf import OmegaConf;
    from src.tasks.plcs.training.tracking_lightning_module import PLCSTrackingLightningModule
    as M; from src.tasks.plcs.data.tracking_datamodule import PLCSTrackingDataModule
    as D; p="/home/kamimura/projects/tennis-lab/.claude/worktrees/plcs-tracking-pose-beta005/outputs/plcs/i801_dref_pose_beta005_presence_comp_centered_pair010m05_s42/logs/version_0/checkpoints/plcs-epoch=00.ckpt";
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
  run_dir: knowledge/runs/run-i801-eval-beta005-presence-comp-centered-pair010m05-e00-thr050-currentmetrics
  predictions: knowledge/runs/run-i801-eval-beta005-presence-comp-centered-pair010m05-e00-thr050-currentmetrics/pred_test.npz
  log: .training_queue/logs/1788161565945650430_3606325_i801_eval_beta005_presence_comp_centered_pair010m05_e00_thr050_currentmetrics.log
parents:
- run-i801-dref-pose-beta005-presence-comp-centered-pair010m05-s42
relations:
- to: run-i801-eval-beta005-presence-head-hneg050-e01-thr050-compcontrol
  rel: compares
- to: run-i801-eval-beta005-presence-comp-centered-e02-thr050-currentmetrics
  rel: compares
- to: run-i801-eval-beta005-presence-head-pair010m05-e00-thr050-currentmetrics
  rel: compares
tags: [plcs, tracking, pose, presence, competition, deepsets, zero-mean, centered, pairwise, evaluation, current-metrics, threshold-050, fair-contract, beta005]
---

## 考察 / Findings

### 要約

centered+pairwiseのbest-validation epoch 0はhinge平均をcontrolの`0.502529`から`0.478444`へ縮めたが、F1 `0.670431`、duplicate `47.35`、exact-count `0.230547`、GT 1人all4 `0.848500`、zero-margin違反`0.727678`でcontrolより悪い。margin幅だけの改善であり、不採用とする。

### アーキテクチャ詳細

`run-i801-dref-pose-beta005-presence-comp-centered-pair010m05-s42`のepoch 0をevaluation-onlyで再評価した。zero-mean DeepSets branchへstable active / inactive全pairのmargin `0.5` hingeをweight `0.1`で与え、hard-negativeは無効である。評価はcommit `a4279f40`、threshold `0.5`、duplicate / ID switch距離各`0.05 m`、同一test split、T=128、V=6、reference camera `camera_2`で固定した。controlと共通する175 tensorはbitwise同一である。

### メトリクスの解釈

precision / recall / F1は`0.507579 / 0.983249 / 0.670431`、ID switch `0.13`、duplicate `47.35`、missed `3.57`、inactive FP `170.32`だった。exact-count率は`23.05%`、GT 1 / 2 / 3人all4率は`84.85% / 99.65% / 97.40%`である。raw `loss_presence_pairwise`は`0.473345`、公平再集計のstable hinge平均`0.478444`、hardest gap平均`-0.182539`、zero-margin違反`72.77%`、margin 0.5違反`96.59%`だった。poseはposition `4.929797 m`、angular `33.645799°`、canonical MPJPE `0.174605 m`、reprojection `155.018780 px`でcontrolと同値である。evaluation-onlyのため独自の収束曲線はない。

### アーキテクチャ⇄メトリクスの因果考察

pairwise objectiveと同じmargin hingeは低下したが、hardest pairの符号を示すzero-margin違反とthreshold後の人数精度は悪化した。これは平均hingeが「既に正しく並ぶpairの余白拡大」でも下がり、frame内の最難pairやactive数を保証しないためである。zero-mean residualのframe共通エネルギー比は`0.012%`で共通gateは除けており、失敗原因は共通shiftではなく競合教師とcount決定の不一致にある。

### 既存実験との比較

control比ではF1 `-0.001480`、duplicate `+3.00`、inactive FP `+2.36`、exact-count `-0.000469`、GT 1人all4 `+7.98`ポイント、zero-margin違反`+2.73`ポイントである。centered hard-negative epoch 2よりF1 `+0.001374`、exact-count `+0.002813`だが、GT 1人all4は`86.14%→84.85%`の小改善に留まり、controlには届かない。従来presence-head pairwise epoch 0のF1`0.668627`よりは良いが、採用基準はcontrol更新である。

### 次に有効な実験

pairwiseはhinge幅だけを動かしprimary count gateを更新しなかったため、DeepSets + pairwise系列を終了する。hard-negative controlから、予測人数を明示してtop-k queryを選ぶ機構、またはunmatched queryのlogit符号を直接制約するobjectiveへ移る。
