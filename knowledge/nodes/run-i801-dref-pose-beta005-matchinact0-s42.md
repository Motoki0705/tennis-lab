---
id: run-i801-dref-pose-beta005-matchinact0-s42
type: run
title: PLCS tracking + pose matching inactive 0（seed 42）
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
  match_presence_weight: 0.5
  match_presence_inactive_weight: 0.0
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
  loss: 0.93564
  loss_position: 0.159763
  loss_rotation: 0.244629
  loss_presence: 0.644394
  loss_track_smoothness: 0.0
  loss_angle: 0.261125
  loss_canonical_pose: 0.010881
  loss_reprojection: 0.095314
  position_error: 0.463785
  presence_precision: 0.502635
  presence_recall: 0.994786
  presence_f1: 0.663253
  lifecycle_presence_f1: 0.663253
  birth_frame_error: 14.174511
  death_frame_error: 16.496052
  query_reuse_count: 0.16
  illegal_overlap_count: 0.0
  segment_id_switches: 40.400002
  id_switches: 40.400002
  duplicate_active_tracks: 332.720001
  missed_gt_frames: 7.6
  inactive_query_false_positives: 1373.439941
  angular_error_deg: 33.948395
  heading_error_deg: 34.360001
  position_error_m: 5.222153
  x_error_m: 1.84625
  y_error_m: 4.48375
  z_error_m: 0.253984
  y_sign_accuracy: 0.660156
  reference_index_0_position_error_m: 6.288651
  reference_index_1_position_error_m: 5.760417
  reference_index_2_position_error_m: 5.109375
  reference_index_3_position_error_m: 5.65625
  reference_index_4_position_error_m: 5.281929
  canonical_mpjpe_m: 0.183367
  world_mpjpe_m: 5.235973
  reprojection_error_px: 161.40625
  behind_camera_fraction: 0.000304
  reference_index_5_position_error_m: 5.104167
  mean_gt_active_players: 1.659297
  mean_predicted_active_queries: 3.303047
  presence_cardinality_mae: 1.655
repro:
  commit: 9ffae3d34d34fe93b41a4f7e64e60238b3254cd4
  branch: experiments/plcs-tracking-pose-beta005
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.venv/bin/python
    -m src.tasks.plcs.scripts.train --config-name train_tracking_pose model=track_query_ablation_d_v2_selector
    court_keypoints=camera_view_v2 model.cswa.backend=cuda paths.data_root=/home/kamimura/projects/tennis-lab/data
    data.scene_dir=plcs/multi_object_camera_view_v2 'data.seq_len_range=[128,128]'
    'data.num_views_range=[6,6]' data.batch_size=8 data.num_workers=16 data.evaluation_reference_camera_id=camera_2
    loss.match_presence_inactive_weight=0.0 training.compile.enabled=false training.trainer.precision=bf16-mixed
    training.trainer.accumulate_grad_batches=4 training.trainer.max_epochs=70 training.trainer.check_val_every_n_epoch=5
    training.trainer.enable_progress_bar=false training.trainer.enable_model_summary=false
    training.early_stopping.enabled=false training.qualitative_logging.enabled=false
    run.seed=42 run.fast_dev_run=false run.test_after_fit=true run.output_dir=plcs/i801_dref_pose_beta005_matchinact0_s42
artifacts:
  run_dir: knowledge/runs/run-i801-dref-pose-beta005-matchinact0-s42
  predictions: knowledge/runs/run-i801-dref-pose-beta005-matchinact0-s42/pred_test.npz
  output_dir: outputs/plcs/i801_dref_pose_beta005_matchinact0_s42/logs/version_0
  curves: knowledge/runs/run-i801-dref-pose-beta005-matchinact0-s42/curves.png
  tb_logdir: outputs/plcs/i801_dref_pose_beta005_matchinact0_s42/logs/version_0
parents:
- run-i801-dref-pose-beta005-s42-r1
relations:
- to: run-i801-a2-plcs-d-reference
  rel: compares
- to: run-i801-dref-pose-beta005-matchp0-s42
  rel: compares
tags:
- plcs
- tracking
- canonical-pose
- reprojection
- camera-view-v2
- ablation-d
- hungarian-matching
- matching-inactive-weight
- beta005
- seed-42
---

## 考察 / Findings

### 要約

Hungarian assignment の presence cost `0.5` は維持し、inactive frame の matching weight だけを
`0.25` から `0.0` にした。親 run より ID switch、duplicate、missed は少し改善したが、F1 と
inactive false positive は改善せず、position・angle・pose も悪化した。matching 時に inactive
区間を無視するだけでは、presence の過剰発火と tracking + pose の両立は解決しない。

### アーキテクチャ詳細

4 本の track query から position、rotation、presence、17-joint canonical pose を予測する
`track_query_ablation_d_v2_selector` を用いた。Hungarian assignment 後の active target に position、
wrapped angle、canonical pose、6-view clean keypoint への reprojection を適用する。camera-view-v2、
T=128、V=6、effective batch=32、seed 42、CUDA CSWA、bf16 mixed、
`rotation_weight=angle_weight=0.05`、canonical / reprojection weight `1.0` は親 run と同じである。

最終 presence BCE の `presence_inactive_weight=0.25` は変更せず、matching 内の weighted presence
BCE だけ `match_presence_inactive_weight=0.0` とした。active weight `1.0` と transition weight
`2.0` は残り、それらを `match_presence_weight=0.5` で position / rotation cost に加える。したがって
inactive 区間は assignment に寄与しないが、active・transition の lifecycle 情報は残る。最大 epoch は
親の 100 に対して 70 なので、親との差には学習長も含まれる。

### メトリクスの解釈

test の precision / recall / F1 は `0.5026 / 0.9948 / 0.6633`、ID switch `40.40`、
duplicate `332.72`、missed `7.60`、inactive-query false positive `1373.44` だった。保存した
test 推論を threshold `0.5` で集計すると、GT は平均 `1.6593` 人 / frame に対して予測 active
query は平均 `3.3030`、人数 MAE は `1.6550` である。GT が 2 人以上の `6638` frame の
`99.85%` で 4 query すべてが active であり、inactive matching cost の除去は query 数を抑えていない。

pose は position `5.2222m`、angular `33.9484deg`、heading `34.3600deg`、canonical MPJPE
`0.1834m`、world MPJPE `5.2360m`、reprojection `161.41px` だった。test total loss
`0.93564` に対する重み込み angle の寄与は約 `1.40%`、presence は約 `68.87%` であり、
angle 支配は起きていない。

収束曲線では val loss の最小が step 874 の `0.91350`、最終 step 1749 が `0.92792` だった。
train epoch loss は `1.28325` から `0.61588` へ低下する一方、val presence loss は最初の
`0.56096` が最小で最終 `0.61975` まで上昇した。val position は step 624 の `5.6753m` を底に
最終 `5.7807m`、val angular は step 1499 の `34.1493deg` が最小で最終 `34.3319deg` だった。
学習崩壊はないが、epoch 35 付近以降は total / presence loss に過学習が見られる。

### アーキテクチャ⇄メトリクスの因果考察

観測として、inactive frame を Hungarian cost から外しても、平均 active query 数と inactive false
positive は親より増え、precision / F1 も改善しなかった。最終 BCE は同じなので、assignment の
inactive penalty だけでは presence logit の calibration や総 query 数を直接制約できない。

以下は仮説である。active・transition の presence cost を残したため、presence cost を全て外した run
より query-target 対応の lifecycle 情報が保たれ、position、canonical MPJPE、reprojection の悪化が
小さく済んだ可能性がある。一方、inactive 区間で「この query はこの target ではない」という情報が
assignment に入らず、長い track 全体の identity 対応が曖昧になったため、親より ID / duplicate が
十分改善せず、過剰発火も残った可能性がある。matching は離散的な対応を変えるため、この仮説は
assignment 自体の可視化なしには断定できない。

### 既存実験との比較

親 `run-i801-dref-pose-beta005-s42-r1` に対し、ID switch は `46.24→40.40`（12.6%減）、
duplicate は `357.60→332.72`（7.0%減）、missed は `15.36→7.60`（50.5%減）だった。
一方、F1 は `0.6639→0.6633`、inactive false positive は `1361.92→1373.44`、position は
`5.0050→5.2222m`、angular は `33.4878→33.9484deg`、canonical MPJPE は
`0.1796→0.1834m` と悪化した。reprojection も `160.50→161.41px` で僅かに悪い。

presence cost を全て外した `run-i801-dref-pose-beta005-matchp0-s42` より、本 run は position
`5.2222m` 対 `5.3289m`、canonical MPJPE `0.1834m` 対 `0.1847m`、reprojection
`161.41px` 対 `172.97px` と良い。一方、ID switch は `40.40` 対 `34.20`、duplicate は
`332.72` 対 `243.60` と多い。tracking-only の `run-i801-a2-plcs-d-reference` に対しても F1、
ID switch、duplicate、position は劣り、この matching 設定も総合的な採用候補ではない。

### 次に有効な実験

`match_presence_weight=0.5`、matching inactive weight `0.25` の既定値へ戻す。2 つの matching
ablation で raw active query 数が減らなかったため、次は GT active 人数と presence probability の
総和を直接比較する微分可能な cardinality loss を小さい weight から検証する。threshold `0.5` を
固定し、F1、inactive false positive、duplicate、ID switch、missed と pose 指標に加えて、frame
単位の予測人数分布を保存し、assignment 変更と presence calibration を混同せず評価する。
