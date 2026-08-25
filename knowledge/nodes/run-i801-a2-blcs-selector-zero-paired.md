---
id: run-i801-a2-blcs-selector-zero-paired
type: run
title: BLCS D selector-zero strict paired評価（Attempt 2）
issue: 801
provider: codex
session: 01a03207-0be4-72c1-ba60-a051d0d8d9b6
date: '2026-08-25'
status: done
config:
  model: track_query_ablation_d_v2_selector_zero
  task: blcs
  evaluation: reference_counterfactual_paired
  selector_mode: selector_zero
  dataset: blcs/multi_object_camera_view_norm-v2
  seed: 42
  seq_len: 128
  num_views: 6
  scene_count: 100
  same_side_camera: cam_2
  opposite_side_camera: cam_0
  court_keypoint_contract: camera_view_courtkp20_rzpi_v1
  target_frame_contract: reference_camera_court_rzpi_v1
  track_query_rope_contract: time_camera_reference_selector_v1
  checkpoint_sha256: bae09f3628fba881dd8fee11c2a8b82f9be988eb5858eb0636494fcd8e8af25b
  manifest_digest: 61e7ce3b9aca1db325f6399aa1ac7c4f4d10e4caf70feb02d3b16e6ec11faf8d
  report_digest: cab62459a0d671c60286770e8b10159d4913f9ea311ccfcb13d850edd05b443c
metrics:
  physical_restored_position_consistency_error_m: 3.32589
  reference_target_opposite_side_position_error_x_m: 2.406773
  reference_target_opposite_side_position_error_y_m: 5.023587
  reference_target_opposite_side_position_error_z_m: 0.985586
  reference_target_opposite_side_reference_index_0_position_error_m: 6.189918
  reference_target_opposite_side_y_sign_accuracy: 0.762901
  reference_target_same_side_position_error_x_m: 2.489158
  reference_target_same_side_position_error_y_m: 5.175421
  reference_target_same_side_position_error_z_m: 0.985517
  reference_target_same_side_reference_index_2_position_error_m: 6.380284
  reference_target_same_side_y_sign_accuracy: 0.75857
repro:
  commit: 40c21fbad59e040f52c040ae354637c5f3c8975a
  branch: feat/issue-801-reference-camera-rope
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    /home/kamimura/projects/tennis-lab/.venv/bin/python -m src.tasks.blcs.scripts.evaluate_reference_counterfactual
    evaluation.checkpoint_path=/home/kamimura/projects/tennis-lab/.claude/worktrees/issue-801-reference-rope/outputs/blcs/norm-v2/issue-801/t128-v4/ablation-d-selector-zero-eb32-seeded-retry4/logs/version_0/checkpoints/last.ckpt
    paths.data_root=/home/kamimura/projects/tennis-lab/data data.scene_dir=blcs/multi_object_camera_view_norm-v2
    data.batch_size=1 data.num_workers=16 model=track_query_ablation_d_v2_selector_zero
    evaluation.trainer.accelerator=gpu evaluation.trainer.precision=bf16-mixed
artifacts:
  run_dir: knowledge/runs/run-i801-a2-blcs-selector-zero-paired
  predictions: knowledge/runs/run-i801-a2-blcs-selector-zero-paired/pred_test.npz
  reference_counterfactual: knowledge/runs/run-i801-a2-blcs-selector-zero-paired/reference_counterfactual.json
  log: .training_queue/logs/1787643617793137915_3889485_i801_attempt2_blcs_selector_zero_paired_v6_final.log
parents:
- run-i801-d-selector-zero-seeded
relations:
- to: run-i801-a2-blcs-reference-paired
  rel: compares
tags:
- blcs
- reference-camera
- counterfactual
- paired-evaluation
- selector-zero
- ablation-d
- camera-view-v2
---

## 考察 / Findings

### 要約

BLCS Dのreference-frame教師とcamera-view入力を保ち、第三RoPE selectorだけを全`0`にしたcheckpointをstrict paired評価した。same/opposite Y-sign accuracyは`0.7586/0.7629`、physical-restored consistencyは`3.3259m`だった。

### アーキテクチャ詳細

親run `run-i801-d-selector-zero-seeded`と同じD checkpointで、selector軸のみ`selector_zero`である。100 test scenes、T128/V6、seed 42、scene/window/view order、Court/target contractをreference runと揃え、`cam_2`と`cam_0`を指定cameraとして2 pass評価した。

### メトリクスの解釈

same-side X/Y/Z MAEは`2.4892/5.1754/0.9855m`、opposite-sideは`2.4068/5.0236/0.9856m`。local-index position errorはindex 2が`6.3803m`、index 0が`6.1899m`である。評価runのため収束曲線はない。

### アーキテクチャ⇄メトリクスの因果考察

selectorを消してもside別Y-signはほぼ同じであり、Court channelやobject内容だけでも一部のframe識別信号を得られる可能性がある。一方、physical frameに戻した2予測の差は`3.3259m`あり、対称性の一意化は未解決である。

### 既存実験との比較

`run-i801-a2-blcs-reference-paired`に対し、same-side Y-signとopposite-side X/Yは良いが、physical consistency、same-side XYZ、opposite-side Y-sign/Zは悪い。一方向の優位性はなく、第三RoPE selectorの追加寄与は確認できない。

### 次に有効な実験

本runをpaired counterfactual対照として保持し、v1 defaultは変えない。次はreference sideを均衡化した複数seed、またはphysical-frame consistency lossを検証する。
