---
id: run-i801-eval-beta005-presence-comp-centered-e02-thr050-currentmetrics
type: run
title: zero-mean competition epoch 2 のcurrent metrics公平評価
issue: 801
provider: codex
session: 01a04915-27a4-7b62-9f6b-34275561fded
date: '2026-08-31'
status: done
config:
  model: track_query_ablation_d_v2_selector
  data: plcs/multi_object_camera_view_v2
  source_checkpoint: plcs/i801_dref_pose_beta005_presence_comp_centered_hneg050_s42/logs/version_0/checkpoints/plcs-epoch=02.ckpt
  source_checkpoint_epoch: 2
  evaluation_only: true
  presence_competition: deepsets_centered
  presence_inactive_weight: 0.5
  presence_hard_negative_weight: 0.5
  presence_hard_negative_gamma: 2.0
  presence_pairwise_weight: 0.0
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
  presence_precision: 0.506005
  presence_recall: 0.983249
  presence_f1: 0.669057
  id_switches: 0.13
  duplicate_active_tracks: 47.67
  missed_gt_frames: 3.54
  inactive_query_false_positives: 171.2
  canonical_mpjpe_m: 0.174605
  reprojection_error_px: 155.01878
  exact_count_accuracy: 0.227734
  gt1_all_four_active_rate: 0.861394
  gt2_all_four_active_rate: 0.998226
  gt3_all_four_active_rate: 0.98876
  stable_hardest_gap_mean: -0.182801
  stable_zero_margin_violation_rate: 0.714416
  stable_margin_050_violation_rate: 0.960012
  stable_all_pair_violation_rate: 0.468542
  stable_pairwise_hinge_margin_050: 0.480906
repro:
  commit: a4279f40c422e930a62869b10c212c0e4f669d53
  branch: experiments/plcs-tracking-pose-beta005
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 /home/kamimura/projects/tennis-lab/.venv/bin/python
    -c 'import torch; import pytorch_lightning as pl; from omegaconf import OmegaConf;
    from src.tasks.plcs.training.tracking_lightning_module import PLCSTrackingLightningModule
    as M; from src.tasks.plcs.data.tracking_datamodule import PLCSTrackingDataModule
    as D; p="/home/kamimura/projects/tennis-lab/.claude/worktrees/plcs-tracking-pose-beta005/outputs/plcs/i801_dref_pose_beta005_presence_comp_centered_hneg050_s42/logs/version_0/checkpoints/plcs-epoch=02.ckpt";
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
  run_dir: knowledge/runs/run-i801-eval-beta005-presence-comp-centered-e02-thr050-currentmetrics
  predictions: knowledge/runs/run-i801-eval-beta005-presence-comp-centered-e02-thr050-currentmetrics/pred_test.npz
  log: .training_queue/logs/1788158952201208723_3530675_i801_eval_beta005_presence_comp_centered_e02_thr050_currentmetrics.log
parents:
- run-i801-dref-pose-beta005-presence-comp-centered-hneg050-s42
relations:
- to: run-i801-eval-beta005-presence-head-hneg050-e01-thr050-compcontrol
  rel: compares
- to: run-i801-eval-beta005-presence-comp-hneg050-e00-thr050-currentmetrics
  rel: compares
- to: run-i801-eval-beta005-presence-comp-centered-pair010m05-e00-thr050-currentmetrics
  rel: compares
tags: [plcs, tracking, pose, presence, competition, deepsets, zero-mean, centered, evaluation, current-metrics, threshold-050, fair-contract, beta005]
---

## 考察 / Findings

### 要約

zero-mean DeepSetsのbest-validation-F1 epoch 2は、共通shift除去に成功したが、F1 `0.669057`、duplicate `47.67`、exact-count `0.227734`、GT 1人all4 `0.861394`でcontrolより悪く、不採用とする。query rankingもstable zero-margin違反`0.714416`へ悪化した。

### アーキテクチャ詳細

`run-i801-dref-pose-beta005-presence-comp-centered-hneg050-s42`のepoch 2をevaluation-onlyで再評価した。bias-free DeepSets residualをquery方向にzero-mean化し、既存presence logitへ加算する。評価はcommit `a4279f40`、threshold `0.5`、duplicate / ID switch距離各`0.05 m`、同一test split、T=128、V=6、reference camera `camera_2`で固定した。`1275bdb1..a4279f40`ではcurrent metric実装に差分がなく、controlと共通の175 checkpoint tensorもbitwise同一である。

### メトリクスの解釈

precision / recall / F1は`0.506005 / 0.983249 / 0.669057`、ID switch `0.13`、duplicate `47.67`、missed `3.54`、inactive FP `171.20`だった。exact-count率は`22.77%`、GT 1 / 2 / 3人all4率は`86.14% / 99.82% / 98.88%`である。stable hardest gap平均`-0.182801`、zero-margin違反`71.44%`、margin 0.5違反`96.00%`、hinge平均`0.480906`で、rankingも改善していない。poseはposition `4.929797 m`、angular `33.645799°`、canonical MPJPE `0.174605 m`、reprojection `155.018780 px`でcontrolと同値だった。evaluation-onlyのため独自の収束曲線はない。

### アーキテクチャ⇄メトリクスの因果考察

controlとの差分のframe mean絶対値は平均`0.00108`、共通エネルギー比は`0.015%`で、BF16丸めを除きzero-mean化は成立した。しかしquery差分を増やすだけでは、どのqueryを人数に応じて抑えるかが学習されず、GT 1人all4率はcontrolより`9.27`ポイント悪い。仮説として、permutation-equivariantなpost-hoc branchにはassignment / lifecycleの明示的な識別信号が不足している。

### 既存実験との比較

uncentered best epoch 0に対し、F1は`0.670431→0.669057`、duplicateは`46.23→47.67`、exact-countは`0.229219→0.227734`、GT 1人all4は`83.09%→86.14%`とさらに悪化した。stable zero-margin違反だけは`72.85%→71.44%`へ改善したが、controlの`70.03%`に届かない。control比ではF1 `-0.002854`、duplicate `+3.32`、inactive FP `+3.24`、exact-count `-0.003281`であり、zero-mean化後もranking / count悪化という結論は変わらない。

### 次に有効な実験

centered hard-negativeは棄却する。centered branchへpairwise weight `0.1` / margin `0.5`を与え、hinge、zero-margin ranking、exact-count、GT人数別all4のどれが実際に動くかを分離して確認する。hingeだけが低下しthreshold後のcountがcontrolを超えなければpairwise案も停止する。
