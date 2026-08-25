---
id: run-i801-a2-plcs-selector-zero-paired
type: run
title: PLCS D selector-zero strict paired評価（Attempt 2）
issue: 801
provider: codex
session: 01a03207-0be4-72c1-ba60-a051d0d8d9b6
date: '2026-08-25'
status: done
config:
  model: track_query_ablation_d_v2_selector_zero
  task: plcs
  evaluation: reference_counterfactual_paired
  selector_mode: selector_zero
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
  checkpoint_sha256: dfc340f8b45ba10c716b474824f3be4a0f2cbfd6788d1fdb352a30bb7a77abd0
  manifest_digest: 45698da77ae4e666c2976a062e4b0a58809e518a01d090e9de1f479d9a21cae7
  report_digest: a4206c755c657ac28aa0e2aea6726789613d7db4765a7748796d7aba93c3bdb1
metrics:
  physical_restored_heading_consistency_l2: 1.917765
  physical_restored_position_consistency_error_m: 4.50089
  reference_target_opposite_side_heading_error_deg: 90.462277
  reference_target_opposite_side_position_error_x_m: 1.856576
  reference_target_opposite_side_position_error_y_m: 4.463129
  reference_target_opposite_side_position_error_z_m: 0.407983
  reference_target_opposite_side_reference_index_0_position_error_m: 5.238275
  reference_target_opposite_side_y_sign_accuracy: 0.740713
  reference_target_same_side_heading_error_deg: 83.770242
  reference_target_same_side_position_error_x_m: 1.784223
  reference_target_same_side_position_error_y_m: 4.379272
  reference_target_same_side_position_error_z_m: 0.347541
  reference_target_same_side_reference_index_2_position_error_m: 5.112237
  reference_target_same_side_y_sign_accuracy: 0.727671
repro:
  commit: 71426cb84519d4fa716ac9b221d90b00d26b4e63
  branch: feat/issue-801-reference-camera-rope
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    /home/kamimura/projects/tennis-lab/.venv/bin/python -m src.tasks.plcs.scripts.evaluate_reference_counterfactual
    evaluation.checkpoint_path=/home/kamimura/projects/tennis-lab/.claude/worktrees/issue-801-reference-rope/outputs/plcs/norm-v2/issue-801/t128-v6/ablation-d-selector-zero-eb32-final/logs/version_0/checkpoints/last.ckpt
    paths.data_root=/home/kamimura/projects/tennis-lab/data data.scene_dir=plcs/multi_object_camera_view_norm-v2
    data.batch_size=1 data.num_workers=16 model=track_query_ablation_d_v2_selector_zero
    evaluation.trainer.accelerator=gpu evaluation.trainer.precision=bf16-mixed
artifacts:
  run_dir: knowledge/runs/run-i801-a2-plcs-selector-zero-paired
  predictions: knowledge/runs/run-i801-a2-plcs-selector-zero-paired/pred_test.npz
  reference_counterfactual: knowledge/runs/run-i801-a2-plcs-selector-zero-paired/reference_counterfactual.json
  log: .training_queue/logs/1787652981925032454_4030076_i801_attempt2_plcs_selector_zero_paired_v6_final.log
parents:
- run-i801-a2-plcs-d-selector-zero
relations:
- to: run-i801-a2-plcs-reference-paired
  rel: compares
tags:
- plcs
- reference-camera
- counterfactual
- paired-evaluation
- selector-zero
- ablation-d
- camera-view-v2
---

## 考察 / Findings

### 要約

PLCS D selector-zero checkpointのstrict paired対照である。same/opposite Y-sign accuracyは`0.7277/0.7407`、heading errorは`83.77/90.46deg`、physical-restored position consistencyは`4.5009m`だった。

### アーキテクチャ詳細

親run `run-i801-a2-plcs-d-selector-zero`をfitなしで評価し、reference-frame targetとcamera-view Court inputは維持した。第三RoPE軸だけを全tokenで`0`とし、reference runと同じ100 scenes、centered T128 window、V6 order、seed、same/opposite camera manifestを使用した。

### メトリクスの解釈

same-side X/Y/Z MAEは`1.7842/4.3793/0.3475m`、local-index position errorは`5.1122m`。opposite-sideは`1.8566/4.4631/0.4080m`、`5.2383m`。physical-restored heading consistency L2は`1.9178`で、2方向の出力は十分一致しない。評価runのため収束曲線はない。

### アーキテクチャ⇄メトリクスの因果考察

selectorなしではsame/oppositeのY-sign差が`0.0130`に収まり、reference selectorの`0.0965`よりside均衡は良い。ただし両sideで高精度という意味ではなく、physical prediction consistencyも`4.5009m`残るため、一意化の成功とは評価できない。

### 既存実験との比較

`run-i801-a2-plcs-reference-paired`よりsame-sideのY-sign、position、headingとphysical consistencyは良いが、opposite-sideでは逆に悪い。このside反転によりselectorの一方向の寄与は棄却される。

### 次に有効な実験

対照runとして保存し、production defaultはv1のままにする。次の候補はpaired physical supervisionと複数seedのside-stratified評価である。
