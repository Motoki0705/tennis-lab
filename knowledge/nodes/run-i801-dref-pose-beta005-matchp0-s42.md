---
id: run-i801-dref-pose-beta005-matchp0-s42
type: run
title: PLCS tracking + pose matching presence 0（seed 42）
issue: 801
provider: codex
session: 01a04915-27a4-7b62-9f6b-34275561fded
date: '2026-08-29'
status: done
config:
  model: track_query_ablation_d_v2_selector
  architecture: track_query_ablation_d
  loss: tracking_all_outputs_beta01_reprojection
  data: plcs/multi_object_camera_view_v2
  rotation_weight: 0.05
  angle_weight: 0.05
  canonical_pose_weight: 1.0
  reprojection_weight: 1.0
  presence_inactive_weight: 0.25
  match_position_weight: 1.0
  match_rotation_weight: 0.5
  match_presence_weight: 0.0
  presence_threshold: 0.5
  seed: 42
  seq_len: 128
  num_views: 6
  batch_size: 8
  accumulate_grad_batches: 4
  effective_batch_size: 32
  epochs: 70
  warmup_steps: 1000
  precision: bf16-mixed
  cswa_backend: cuda
metrics:
  loss: 0.951008
  loss_position: 0.163837
  loss_rotation: 0.241564
  loss_presence: 0.644309
  loss_track_smoothness: 0.0
  loss_angle: 0.25777
  loss_canonical_pose: 0.010835
  loss_reprojection: 0.10706
  position_error: 0.47333
  presence_precision: 0.501278
  presence_recall: 0.994891
  presence_f1: 0.662091
  lifecycle_presence_f1: 0.662091
  birth_frame_error: 14.154241
  death_frame_error: 16.483625
  query_reuse_count: 0.16
  illegal_overlap_count: 0.0
  segment_id_switches: 34.200001
  id_switches: 34.200001
  duplicate_active_tracks: 243.600006
  missed_gt_frames: 7.44
  inactive_query_false_positives: 1380.47998
  angular_error_deg: 33.612911
  heading_error_deg: 33.91
  position_error_m: 5.328853
  x_error_m: 1.882187
  y_error_m: 4.584375
  z_error_m: 0.256719
  y_sign_accuracy: 0.654063
  reference_index_0_position_error_m: 6.525494
  reference_index_1_position_error_m: 5.9
  reference_index_2_position_error_m: 5.289062
  reference_index_3_position_error_m: 5.870536
  reference_index_4_position_error_m: 5.381114
  canonical_mpjpe_m: 0.184695
  world_mpjpe_m: 5.343116
  reprojection_error_px: 172.965805
  behind_camera_fraction: 0.000302
  reference_index_5_position_error_m: 5.385417
  mean_gt_active_players: 1.659297
  mean_predicted_active_queries: 3.311641
  presence_cardinality_mae: 1.664844
repro:
  commit: 9ffae3d34d34fe93b41a4f7e64e60238b3254cd4
  branch: experiments/plcs-tracking-pose-beta005
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.plcs.scripts.train --config-name train_tracking_pose model=track_query_ablation_d_v2_selector
    court_keypoints=camera_view_v2 model.cswa.backend=cuda paths.data_root=/home/kamimura/projects/tennis-lab/data
    data.scene_dir=plcs/multi_object_camera_view_v2 'data.seq_len_range=[128,128]'
    'data.num_views_range=[6,6]' data.batch_size=8 data.num_workers=16 data.evaluation_reference_camera_id=camera_2
    loss.match_presence_weight=0.0 training.compile.enabled=false training.trainer.precision=bf16-mixed
    training.trainer.accumulate_grad_batches=4 training.trainer.max_epochs=70 training.trainer.check_val_every_n_epoch=5
    training.trainer.enable_progress_bar=false training.trainer.enable_model_summary=false
    training.early_stopping.enabled=false training.qualitative_logging.enabled=false
    run.seed=42 run.fast_dev_run=false run.test_after_fit=true run.output_dir=plcs/i801_dref_pose_beta005_matchp0_s42
artifacts:
  run_dir: knowledge/runs/run-i801-dref-pose-beta005-matchp0-s42
  predictions: knowledge/runs/run-i801-dref-pose-beta005-matchp0-s42/pred_test.npz
  output_dir: outputs/plcs/i801_dref_pose_beta005_matchp0_s42/logs/version_0
  curves: knowledge/runs/run-i801-dref-pose-beta005-matchp0-s42/curves.png
  tb_logdir: outputs/plcs/i801_dref_pose_beta005_matchp0_s42/logs/version_0
parents:
- run-i801-dref-pose-beta005-s42-r1
relations:
- to: run-i801-a2-plcs-d-reference
  rel: compares
- to: run-i801-dref-pose-beta005-matchinact0-s42
  rel: compares
tags:
- plcs
- tracking
- canonical-pose
- reprojection
- camera-view-v2
- ablation-d
- hungarian-matching
- presence-match-disabled
- beta005
- seed-42
---

## 考察 / Findings

### 要約

Hungarian assignment の `match_presence_weight` を `0.5` から `0.0` にして、position / rotation
だけで query と GT track を対応付けた。親 run より ID switch、duplicate、missed は減ったが、
presence F1 と inactive false positive は改善せず、position・canonical pose・再投影も悪化した。
tracking tail の一部改善は query の過剰発火を解消した結果ではなく、総合改善とは判断しない。

### アーキテクチャ詳細

4 本の track query から position、rotation、presence、17-joint canonical pose を予測する
`track_query_ablation_d_v2_selector` を用いた。Hungarian assignment 後の active target に position、
wrapped angle、canonical pose、6-view clean keypoint への reprojection を適用する。camera-view-v2、
T=128、V=6、effective batch=32、seed 42、CUDA CSWA、bf16 mixed、
`rotation_weight=angle_weight=0.05`、canonical / reprojection weight `1.0` は親 run と同じである。

差分は assignment の presence cost を完全に無効化した点で、最終 presence BCE は
`presence_inactive_weight=0.25` のまま残る。したがって presence head を直接弱めた実験ではなく、
学習中の query-target 対応だけを position cost `1.0` と rotation cost `0.5` で決める実験である。
最大 epoch は親の 100 に対して 70 なので、親との差には学習長も含まれる。

### メトリクスの解釈

test の precision / recall / F1 は `0.5013 / 0.9949 / 0.6621`、ID switch `34.20`、
duplicate `243.60`、missed `7.44`、inactive-query false positive `1380.48` だった。保存した
test 推論を threshold `0.5` で集計すると、GT は平均 `1.6593` 人 / frame なのに対し予測 active
query は平均 `3.3116`、人数 MAE は `1.6648` である。GT が 2 人以上の全 `6638` frame で
4 query すべてが active であり、低 precision の主因である過剰発火は残っている。

pose は position `5.3289m`、angular `33.6129deg`、heading `33.9100deg`、canonical MPJPE
`0.1847m`、world MPJPE `5.3431m`、reprojection `172.97px` だった。test total loss
`0.95101` に対する重み込み angle の寄与は約 `1.36%`、presence は約 `67.75%` であり、
angle 支配は起きていない。

収束曲線では val loss の最小が step 999 の `0.91007`、最終 step 1749 が `0.93309` だった。
一方、train epoch loss は `1.28348` から `0.63911` まで低下した。val presence loss は最初の
`0.56132` が最小で最終 `0.62116` まで上昇し、val position は step 624 の `5.6471m` を底に
最終 `5.8528m` へ戻った。崩壊ではないが、後半は presence と position に train / val 乖離がある。

### アーキテクチャ⇄メトリクスの因果考察

観測として、presence を assignment から外しても平均 active query 数と inactive false positive は
親より増えた。最終 BCE 自体は変更していないため、matching cost の除去だけでは presence logit の
calibration や query 数を直接制約できないことと整合する。

以下は仮説である。position / rotation だけで対応を選ぶことで、距離に依存する ID・duplicate 指標に
有利な query-target 対応が形成され、tail 指標が減った可能性がある。一方、lifecycle presence が持つ
track 全期間の対応情報を assignment から失ったため、同じ query へ pose / reprojection supervision を
一貫して与えにくくなり、position、canonical MPJPE、reprojection が悪化した可能性がある。tail 指標の
改善と raw active query 数の悪化が同時に起きたため、前者を query 抑制の効果とは解釈しない。

### 既存実験との比較

親 `run-i801-dref-pose-beta005-s42-r1` に対し、ID switch は `46.24→34.20`（26.0%減）、
duplicate は `357.60→243.60`（31.9%減）、missed は `15.36→7.44`（51.6%減）だった。
一方、F1 は `0.6639→0.6621`、inactive false positive は `1361.92→1380.48`、position は
`5.0050→5.3289m`、canonical MPJPE は `0.1796→0.1847m`、reprojection は
`160.50→172.97px` と悪化した。

inactive 部分だけを matching から外した `run-i801-dref-pose-beta005-matchinact0-s42` と比べると、
本 run は ID switch `34.20` 対 `40.40`、duplicate `243.60` 対 `332.72` と少ないが、position
`5.3289m` 対 `5.2222m`、canonical MPJPE `0.1847m` 対 `0.1834m`、reprojection
`172.97px` 対 `161.41px` と悪い。tracking-only の `run-i801-a2-plcs-d-reference` に対しても
F1、ID switch、duplicate、position は劣り、matching presence の全除去は採用できない。

### 次に有効な実験

matching presence は既定の `0.5` に戻す。2 つの matching ablation で raw active query 数が
減らなかったため、次は GT active 人数と presence probability の総和を直接比較する微分可能な
cardinality loss を小さい weight から検証する。threshold `0.5` を固定し、F1、inactive false
positive、duplicate、ID switch、missed と pose 指標に加えて、frame 単位の予測人数分布を保存する。
