---
id: run-i801-eval-beta005-presence-head-hneg050-e01-thr050-compcontrol
type: run
title: competition対照 hard-negative epoch 1 の公平再評価
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
  presence_competition: none
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
  stable_zero_margin_violation_rate: 0.700344
  stable_margin_050_violation_rate: 0.936526
  stable_all_pair_violation_rate: 0.451394
  stable_pairwise_hinge_margin_050: 0.502529
repro:
  commit: 1275bdb154f580872a0e571cc0f7f493226434f9
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
  run_dir: knowledge/runs/run-i801-eval-beta005-presence-head-hneg050-e01-thr050-compcontrol
  predictions: knowledge/runs/run-i801-eval-beta005-presence-head-hneg050-e01-thr050-compcontrol/pred_test.npz
  log: .training_queue/logs/1788151512287561758_3197475_i801_eval_beta005_presence_head_hneg050_e01_thr050_compcontrol.log
parents:
- run-i801-dref-pose-beta005-presence-head-hneg050-s42
relations:
- to: run-i801-eval-beta005-presence-head-hneg050-e01-thr050-currentmetrics
  rel: confirms
- to: run-i801-eval-beta005-presence-comp-hneg050-e00-thr050-currentmetrics
  rel: compares
- to: run-i801-eval-beta005-presence-comp-centered-e02-thr050-currentmetrics
  rel: compares
- to: run-i801-eval-beta005-presence-comp-centered-pair010m05-e00-thr050-currentmetrics
  rel: compares
tags: [plcs, tracking, pose, presence, control, evaluation, hard-negative, current-metrics, threshold-050, fair-contract, beta005]
---

## 考察 / Findings

### 要約

competitionアブレーションの固定対照としてhard-negative `0.5` epoch 1を再評価した。F1 `0.671911`、duplicate `44.35`、exact-count率`0.231016`、GT 1人all4率`0.768659`で、uncentered / centered / centered+pairwiseの全候補よりbalancedに良く、対照を採用したまま全DeepSets候補を棄却する。

### アーキテクチャ詳細

presence competitionを持たない既存presence-head checkpointをevaluation-onlyで評価した。inactive `0.5`、hard-negative focal weight `0.5` / gamma `2.0`でpresence headだけを学習済みである。評価はcommit `1275bdb1`、threshold `0.5`、duplicate / ID switch距離各`0.05 m`、同一test split、T=128、V=6、reference camera `camera_2`へ固定した。competition候補はこの175-tensor stateをbitwise保持し、追加branchだけを更新している。

### メトリクスの解釈

precision / recall / F1は`0.510584 / 0.981537 / 0.671911`、ID switch `0.13`、duplicate `44.35`、missed `4.00`、inactive FP `167.96`だった。exact-count率は`23.10%`、GT 1 / 2 / 3人all4率は`76.87% / 99.65% / 96.90%`で、対照自体にも強い過活性化が残る。stable hardest gap平均`-0.234167`、zero-margin違反`70.03%`、margin 0.5違反`93.65%`、frame-balanced hinge平均`0.502529`である。poseはposition `4.929797 m`、angular `33.645799°`、canonical MPJPE `0.174605 m`、reprojection `155.018780 px`だった。evaluation-onlyのため独自の収束曲線はない。

### アーキテクチャ⇄メトリクスの因果考察

query間情報を追加しない単純なlinear presence headでも、hard-negative focalが高logit inactive queryを直接抑えるため、今回のDeepSets residualよりprecision / duplicate / exact-countの均衡が良いと考えられる。一方、GT 2人all4率が`99.65%`であるため、この対照も人数競合を学習したとは言えず、「相対的な最良」であって問題解決ではない。

### 既存実験との比較

既存 `run-i801-eval-beta005-presence-head-hneg050-e01-thr050-currentmetrics` と`pred_test.npz`、`metrics.json`、`diagnostic_metrics.json`のSHA-256が全て一致し、評価再現性を確認した。competition bestとの比較では、uncentered / centered / centered+pairwiseのF1は`0.670431 / 0.669057 / 0.670431`、duplicateは`46.23 / 47.67 / 47.35`、exact-countは`0.229219 / 0.227734 / 0.230547`、GT 1人all4は`0.830895 / 0.861394 / 0.848500`で、全候補が対照より悪い。

### 次に有効な実験

次施策の起点は本checkpointに戻す。post-hoc DeepSets residualの追加探索は止め、query interaction本体、assignment-awareなunmatched抑制、または人数条件付きset predictionを検討する。採用gateにはF1 / duplicateだけでなくGT 1–2人all4率、exact-count率、missed、175 shared tensor / pose保持を含める。
