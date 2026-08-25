---
id: run-i801-a2-plcs-reference-paired
type: run
title: PLCS D reference selector strict paired評価（Attempt 2）
issue: 801
provider: codex
session: 01a03207-0be4-72c1-ba60-a051d0d8d9b6
date: '2026-08-25'
status: done
config:
  model: track_query_ablation_d_v2_selector
  task: plcs
  evaluation: reference_counterfactual_paired
  selector_mode: reference
  dataset: plcs/multi_object_camera_view_norm-v2
  seed: 42
  seq_len: 128
  num_views: 6
  scene_count: 100
  same_side_camera: camera_2
  opposite_side_camera: camera_0
  court_keypoint_contract: camera_view_courtkp20_rzpi_v1
  target_frame_contract: reference_camera_court_rzpi_v1
  track_query_rope_contract: time_camera_reference_selector_v1
  checkpoint_sha256: 7a79b61b39ccbf75dbfc1dfcbaac9c010e2315d2038a6e721092039de3d49779
  manifest_digest: 45698da77ae4e666c2976a062e4b0a58809e518a01d090e9de1f479d9a21cae7
  report_digest: a12d29a4d444b24161b27a3b26a266d4469e29001e4c257fdb05fca9a75c0f98
metrics:
  physical_restored_heading_consistency_l2: 1.950773
  physical_restored_position_consistency_error_m: 4.701119
  reference_target_opposite_side_heading_error_deg: 83.947287
  reference_target_opposite_side_position_error_x_m: 1.944177
  reference_target_opposite_side_position_error_y_m: 4.281877
  reference_target_opposite_side_position_error_z_m: 0.411426
  reference_target_opposite_side_reference_index_0_position_error_m: 5.094875
  reference_target_opposite_side_y_sign_accuracy: 0.768398
  reference_target_same_side_heading_error_deg: 91.024517
  reference_target_same_side_position_error_x_m: 1.85429
  reference_target_same_side_position_error_y_m: 4.471955
  reference_target_same_side_position_error_z_m: 0.368826
  reference_target_same_side_reference_index_2_position_error_m: 5.222559
  reference_target_same_side_y_sign_accuracy: 0.671877
repro:
  commit: 71426cb84519d4fa716ac9b221d90b00d26b4e63
  branch: feat/issue-801-reference-camera-rope
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    /home/kamimura/projects/tennis-lab/.venv/bin/python -m src.tasks.plcs.scripts.evaluate_reference_counterfactual
    evaluation.checkpoint_path=/home/kamimura/projects/tennis-lab/.claude/worktrees/issue-801-reference-rope/outputs/plcs/norm-v2/issue-801/t128-v6/ablation-d-reference-selector-eb32-final/logs/version_0/checkpoints/last.ckpt
    paths.data_root=/home/kamimura/projects/tennis-lab/data data.scene_dir=plcs/multi_object_camera_view_norm-v2
    data.batch_size=1 data.num_workers=16 model=track_query_ablation_d_v2_selector
    evaluation.trainer.accelerator=gpu evaluation.trainer.precision=bf16-mixed
artifacts:
  run_dir: knowledge/runs/run-i801-a2-plcs-reference-paired
  predictions: knowledge/runs/run-i801-a2-plcs-reference-paired/pred_test.npz
  reference_counterfactual: knowledge/runs/run-i801-a2-plcs-reference-paired/reference_counterfactual.json
  log: .training_queue/logs/1787652981842495298_4030041_i801_attempt2_plcs_reference_paired_v6_final.log
parents:
- run-i801-a2-plcs-d-reference
relations:
- to: run-i801-a2-plcs-selector-zero-paired
  rel: compares
tags:
- plcs
- reference-camera
- counterfactual
- paired-evaluation
- selector
- ablation-d
- camera-view-v2
---

## 考察 / Findings

### 要約

PLCS reference-selector checkpointを100 test scenesの`camera_2`（same side）と`camera_0`（opposite side）でstrict paired評価した。Y-sign accuracyは`0.6719`対`0.7684`、heading errorは`91.02deg`対`83.95deg`となり、selector効果はsideで反転した。

### アーキテクチャ詳細

親run `run-i801-a2-plcs-d-reference`のD checkpointをfitなしで2 pass評価した。scene/window、6-view order、seed、checkpoint、input contract、physical targetを固定し、指定cameraに合わせたpositionとheading教師を構築した。same-sideはlocal index 2、opposite-sideはindex 0である。

### メトリクスの解釈

same-side X/Y/Z MAEは`1.8543/4.4720/0.3688m`、local-index position errorは`5.2226m`。opposite-sideは`1.9442/4.2819/0.4114m`、`5.0949m`。physical-restored prediction consistencyはposition `4.7011m`、heading L2 `1.9508`で、一意なphysical poseへの一致は弱い。評価runのため収束曲線はない。

### アーキテクチャ⇄メトリクスの因果考察

reference selectorはopposite-sideでzeroよりY-sign、position、headingを改善するが、same-sideでは全て悪化する。selector信号を一貫したreference-frame制御として使うのではなく、sideに依存したshortcutとして利用した可能性がある。ただし単一seedなので仮説である。

### 既存実験との比較

`run-i801-a2-plcs-selector-zero-paired`比で、opposite-sideはY-sign `+0.02768`、local-index position `-0.1434m`、heading `-6.51deg`と良い。same-sideはY-sign `-0.05579`、position `+0.1103m`、heading `+7.25deg`と悪く、physical consistencyも`0.2002m`悪い。

### 次に有効な実験

production v1を維持する。継続時はsame/oppositeを明示的に均衡化したtraining、paired physical consistency loss、複数seedでside反転が再現するかを検証する。
