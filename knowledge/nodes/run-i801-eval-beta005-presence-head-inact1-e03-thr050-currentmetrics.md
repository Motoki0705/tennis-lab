---
id: run-i801-eval-beta005-presence-head-inact1-e03-thr050-currentmetrics
type: run
title: inactive 1.0 best-val epoch 3 のcurrent metrics公平評価
issue: 801
provider: codex
session: 01a04915-27a4-7b62-9f6b-34275561fded
date: '2026-08-31'
status: done
config:
  model: track_query_ablation_d_v2_selector
  data: plcs/multi_object_camera_view_v2
  source_checkpoint: plcs/i801_dref_pose_beta005_presence_head_inact1_s42/logs/version_0/checkpoints/plcs-epoch=03.ckpt
  source_checkpoint_epoch: 3
  source_checkpoint_selection: best val/presence_f1
  evaluation_only: true
  presence_inactive_weight: 1.0
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
  presence_precision: 0.510172
  presence_recall: 0.982668
  presence_f1: 0.672165
  id_switches: 0.13
  duplicate_active_tracks: 44.75
  missed_gt_frames: 3.77
  inactive_query_false_positives: 168.48
  canonical_mpjpe_m: 0.174605
  reprojection_error_px: 155.01878
  exact_count_accuracy: 0.230234
  gt1_all_four_active_rate: 0.784528
  gt2_all_four_active_rate: 0.996453
  gt3_all_four_active_rate: 0.96938
  stable_hardest_gap_mean: -0.236671
  stable_zero_margin_violation_rate: 0.70905
  stable_margin_050_violation_rate: 0.93642
  stable_all_pair_violation_rate: 0.45894
repro:
  commit: fafff3ae9f950e2bac274aee2b922defe8c59d56
  branch: experiments/plcs-tracking-pose-beta005
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 /home/kamimura/projects/tennis-lab/.venv/bin/python
    -c 'import torch; import pytorch_lightning as pl; from omegaconf import OmegaConf;
    from src.tasks.plcs.training.tracking_lightning_module import PLCSTrackingLightningModule
    as M; from src.tasks.plcs.data.tracking_datamodule import PLCSTrackingDataModule
    as D; p="/home/kamimura/projects/tennis-lab/.claude/worktrees/plcs-tracking-pose-beta005/outputs/plcs/i801_dref_pose_beta005_presence_head_inact1_s42/logs/version_0/checkpoints/plcs-epoch=03.ckpt";
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
  run_dir: knowledge/runs/run-i801-eval-beta005-presence-head-inact1-e03-thr050-currentmetrics
  predictions: knowledge/runs/run-i801-eval-beta005-presence-head-inact1-e03-thr050-currentmetrics/pred_test.npz
  log: .training_queue/logs/1788143669280695160_3048509_i801_eval_beta005_presence_head_inact1_e03_thr050_currentmetrics.log
parents:
- run-i801-dref-pose-beta005-presence-head-inact1-s42
relations:
- to: run-i801-eval-beta005-presence-head-pair010m05-e00-thr050-currentmetrics
  rel: compares
- to: run-i801-eval-beta005-presence-head-hneg050-e01-thr050-currentmetrics
  rel: compares
- to: run-i801-eval-beta005-e69-thr050-currentmetrics
  rel: compares
tags: [plcs, tracking, pose, presence, evaluation, inactive-weight, current-metrics, threshold-050, fair-contract, beta005]
---

## 考察 / Findings

### 要約

inactive weight `1.0`のbest-val epoch 3は、current metrics公平比較のheadline bestである。F1`0.672165`、duplicate`44.75`、missed`3.77`を得たが、GT 1人all4率`78.45%`、GT 2人all4率`99.65%`、exact-count率`23.02%`であり、query過活性化は未解決である。

### アーキテクチャ詳細

beta005 epoch 69からpresence headだけを`presence_inactive_weight=1.0`でfine-tuneした既存runの、best `val/presence_f1` epoch 3を学習せず評価した。評価contractはcommit `fafff3ae`、presence threshold `0.5`、duplicate / ID switch距離各`0.05 m`、同一test split、T=128、V=6、reference camera `camera_2`で固定した。pose trunkと`rotation_weight=0.05`、`angle_weight=0.05`はsourceから不変である。

### メトリクスの解釈

precision / recall / F1は`0.510172 / 0.982668 / 0.672165`、ID switch `0.13`、duplicate `44.75`、missed `3.77`、inactive FP `168.48`だった。position `4.929797 m`、angular `33.645799°`、canonical MPJPE `0.174605 m`、reprojection `155.018780 px`である。人数別all4率はGT 1人`78.45%`、GT 2人`99.65%`、GT 3人`96.94%`で、exact-count率は`23.02%`に留まる。stable hardest gap平均は`-0.236671`、zero-margin違反`70.91%`、margin 0.5違反`93.64%`である。evaluation-onlyのため独自の収束曲線はない。

### アーキテクチャ⇄メトリクスの因果考察

inactive BCEを強めたことでepoch 69やpairwiseよりinactive queryの発火が減り、F1を最大化しながらmissedの増加を小さく抑えた。一方、GT 2人ではほぼ必ず4 queryがactiveである。仮説として、このweightは全体calibrationを下げる効果は持つが、同一frame内の必要query数や排他性を表現しないため、queryごとの役割分担を学習できない。

### 既存実験との比較

epoch 69に対しF1は`0.667258→0.672165`、duplicateは`48.32→44.75`、inactive FPは`175.28→168.48`へ改善した一方、recallは`0.991527→0.982668`、missedは`1.90→3.77`へ悪化した。pairwise epoch 0よりF1は`+0.003538`、duplicateは`-2.02`、GT 1人all4率は`-6.15`ポイントである。hard-negative epoch 1よりF1は`+0.000254`、recallは`+0.001130`、missedは`-0.23`でheadlineは本runが良いが、duplicate / inactive FP / exact-count / GT 1人all4率ではhard-negativeがわずかに良い。

### 次に有効な実験

headline基準の起点として本checkpointを保持し、assignment-awareなunmatched-query抑制またはquery間競合を追加する。次の候補はF1`0.672165`を大きく損なわず、GT 1人・2人all4率を明確に下げ、exact-count率を`23.02%`から上げることを採用条件にする。
