---
id: run-i801-eval-beta005-presence-head-hneg050-e01-thr050-currentmetrics
type: run
title: hard-negative 0.5 epoch 1 のcurrent metrics公平評価
issue: 801
provider: codex
session: 01a04915-27a4-7b62-9f6b-34275561fded
date: '2026-08-31'
status: done
config:
  model: track_query_ablation_d_v2_selector
  data: plcs/multi_object_camera_view_v2
  source_checkpoint: plcs/i801_dref_pose_beta005_presence_head_hneg050_s42/logs/version_0/checkpoints/plcs-epoch=01.ckpt
  source_checkpoint_epoch: 1
  evaluation_only: true
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
  presence_precision: 0.510584
  presence_recall: 0.981537
  presence_f1: 0.671911
  id_switches: 0.13
  duplicate_active_tracks: 44.35
  missed_gt_frames: 4.0
  inactive_query_false_positives: 167.96
  canonical_mpjpe_m: 0.174605
  reprojection_error_px: 155.01878
  exact_count_accuracy: 0.231016
  gt1_all_four_active_rate: 0.768659
  gt2_all_four_active_rate: 0.996453
  gt3_all_four_active_rate: 0.968992
  stable_hardest_gap_mean: -0.234167
  stable_zero_margin_violation_rate: 0.70034
  stable_margin_050_violation_rate: 0.93653
  stable_all_pair_violation_rate: 0.45139
repro:
  commit: fafff3ae9f950e2bac274aee2b922defe8c59d56
  branch: experiments/plcs-tracking-pose-beta005
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 /home/kamimura/projects/tennis-lab/.venv/bin/python
    -c 'import torch; import pytorch_lightning as pl; from omegaconf import OmegaConf;
    from src.tasks.plcs.training.tracking_lightning_module import PLCSTrackingLightningModule
    as M; from src.tasks.plcs.data.tracking_datamodule import PLCSTrackingDataModule
    as D; p="/home/kamimura/projects/tennis-lab/.claude/worktrees/plcs-tracking-pose-beta005/outputs/plcs/i801_dref_pose_beta005_presence_head_hneg050_s42/logs/version_0/checkpoints/plcs-epoch=01.ckpt";
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
  run_dir: knowledge/runs/run-i801-eval-beta005-presence-head-hneg050-e01-thr050-currentmetrics
  predictions: knowledge/runs/run-i801-eval-beta005-presence-head-hneg050-e01-thr050-currentmetrics/pred_test.npz
  log: .training_queue/logs/1788143669363120229_3048529_i801_eval_beta005_presence_head_hneg050_e01_thr050_currentmetrics.log
parents:
- run-i801-dref-pose-beta005-presence-head-hneg050-s42
relations:
- to: run-i801-eval-beta005-presence-head-pair010m05-e00-thr050-currentmetrics
  rel: compares
- to: run-i801-eval-beta005-presence-head-inact1-e03-thr050-currentmetrics
  rel: compares
- to: run-i801-eval-beta005-e69-thr050-currentmetrics
  rel: compares
tags: [plcs, tracking, pose, presence, evaluation, hard-negative, focal, current-metrics, threshold-050, fair-contract, beta005]
---

## 考察 / Findings

### 要約

hard-negative focal `0.5`のepoch 1は、current metrics公平比較のbalanced bestである。F1`0.671911`を保ちながらprecision`0.510584`、duplicate`44.35`、inactive FP`167.96`、GT 1人all4率`76.87%`、exact-count率`23.10%`を4候補中で最良にした。ただしGT 2人all4率は`99.65%`で、query過活性化は未解決である。

### アーキテクチャ詳細

beta005 epoch 69からpresence headだけをfine-tuneし、inactive BCE weight `0.5`にgamma `2.0` / weight `0.5`のhard-negative focal項を加えた既存runのepoch 1を評価した。評価contractはcommit `fafff3ae`、threshold `0.5`、duplicate / ID switch距離各`0.05 m`、同一test split、T=128、V=6、reference camera `camera_2`で固定した。pose trunk、`rotation_weight=0.05`、`angle_weight=0.05`はsourceから不変である。

### メトリクスの解釈

precision / recall / F1は`0.510584 / 0.981537 / 0.671911`、ID switch `0.13`、duplicate `44.35`、missed `4.00`、inactive FP `167.96`だった。position `4.929797 m`、angular `33.645799°`、canonical MPJPE `0.174605 m`、reprojection `155.018780 px`である。人数別all4率はGT 1人`76.87%`、GT 2人`99.65%`、GT 3人`96.90%`で、全体exact-count率は比較中最高でも`23.10%`に留まる。stable hardest gap平均`-0.234167`、zero-margin違反`70.03%`、margin 0.5違反`93.65%`である。evaluation-onlyのため独自の収束曲線はない。

### アーキテクチャ⇄メトリクスの因果考察

hard-negative focalは高logitのinactive queryへ重点的に勾配を当てるため、inactive `1.0`より低い全体penaltyでもprecision、duplicate、inactive FPとGT 1人all4率をわずかに改善したと考えられる。しかし人数そのものを教師にせず、GT 2人のall4率がほぼ100%なので、難しいnegativeの局所抑制だけではquery cardinality collapseを解けない。

### 既存実験との比較

epoch 69に対しF1は`0.667258→0.671911`、duplicateは`48.32→44.35`、inactive FPは`175.28→167.96`、GT 1人all4率は`90.95%→76.87%`へ改善したが、missedは`1.90→4.00`へ増えた。headline bestのinactive `1.0` epoch 3に対してF1は`-0.000254`、recallは`-0.001130`、missedは`+0.23`だが、precisionは`+0.000412`、duplicateは`-0.40`、inactive FPは`-0.52`、GT 1人all4率は`-1.59`ポイント、exact-count率は`+0.08`ポイントである。pairwise epoch 0より全体として良い。

### 次に有効な実験

過活性抑制を重視する次施策の起点に本checkpointを使い、assignment-awareなunmatched-query抑制またはquery間競合を追加する。F1を`0.671911`近傍に維持しつつ、GT 1人all4率`76.87%`とGT 2人`99.65%`を明確に下げ、exact-count率`23.10%`を超えることを採用条件にする。
