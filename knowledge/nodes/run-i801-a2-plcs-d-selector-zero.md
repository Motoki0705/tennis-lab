---
id: run-i801-a2-plcs-d-selector-zero
type: run
title: PLCS D camera-view v2 selector-zero（Attempt 2）
issue: 801
provider: codex
session: 01a03207-0be4-72c1-ba60-a051d0d8d9b6
date: '2026-08-25'
status: done
config:
  model: track_query_ablation_d_v2_selector_zero
  architecture: track_query_ablation_d
  task: plcs
  ffn_mode: shared
  mhc_writeback: layer_end
  reference_selector_mode: selector_zero
  court_keypoint_contract: camera_view_courtkp20_rzpi_v1
  target_frame_contract: reference_camera_court_rzpi_v1
  track_query_rope_contract: time_camera_reference_selector_v1
  loss: tracking
  data: plcs/multi_object_camera_view_norm-v2
  seed: 42
  seq_len: 128
  num_views: 6
  batch_size: 8
  accumulate_grad_batches: 4
  effective_batch_size: 32
  epochs: 100
  precision: bf16-mixed
  cswa_backend: cuda
metrics:
  loss: 0.89717
  loss_position: 0.162631
  loss_rotation: 0.248537
  loss_presence: 0.610271
  loss_track_smoothness: 0.0
  position_error: 0.458037
  presence_precision: 0.50701
  presence_recall: 0.992788
  presence_f1: 0.666932
  lifecycle_presence_f1: 0.666932
  birth_frame_error: 13.120378
  death_frame_error: 15.242575
  query_reuse_count: 0.24
  illegal_overlap_count: 0.0
  segment_id_switches: 36.68
  id_switches: 36.68
  duplicate_active_tracks: 296.279999
  missed_gt_frames: 10.64
  inactive_query_false_positives: 1354.079956
  angular_error_deg: 34.144745
  heading_error_deg: 34.32
  position_error_m: 5.141654
  x_error_m: 1.768438
  y_error_m: 4.41
  z_error_m: 0.327188
  y_sign_accuracy: 0.684531
  reference_index_0_position_error_m: 5.764802
  reference_index_1_position_error_m: 5.002083
  reference_index_2_position_error_m: 4.496528
  reference_index_5_position_error_m: 5.985054
  reference_index_3_position_error_m: 4.795759
  reference_index_4_position_error_m: 6.423828
repro:
  commit: 71426cb84519d4fa716ac9b221d90b00d26b4e63
  branch: feat/issue-801-reference-camera-rope
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    /home/kamimura/projects/tennis-lab/.venv/bin/python -m src.tasks.plcs.scripts.train
    --config-name train_tracking court_coordinate_normalization=v2 court_keypoints=camera_view_v2
    model.cswa.backend=cuda paths.data_root=/home/kamimura/projects/tennis-lab/data
    data.scene_dir=plcs/multi_object_camera_view_norm-v2 data.seq_len_range=[128,128]
    data.num_views_range=[6,6] data.batch_size=8 data.num_workers=16 data.evaluation_reference_camera_id=camera_2
    training.compile.enabled=false training.trainer.precision=bf16-mixed training.trainer.accumulate_grad_batches=4
    training.trainer.max_epochs=100 training.trainer.check_val_every_n_epoch=5 training.trainer.enable_progress_bar=false
    training.trainer.enable_model_summary=false training.early_stopping.enabled=false
    training.qualitative_logging.enabled=false run.seed=42 run.fast_dev_run=false
    run.test_after_fit=true model=track_query_ablation_d_v2_selector_zero run.output_dir=plcs/norm-v2/issue-801/t128-v6/ablation-d-selector-zero-eb32-final
artifacts:
  run_dir: knowledge/runs/run-i801-a2-plcs-d-selector-zero
  predictions: knowledge/runs/run-i801-a2-plcs-d-selector-zero/pred_test.npz
  log: .training_queue/logs/1787644646445719253_3910910_i801_attempt2_plcs_d_selector_zero_t128_v6_eb32_e100_final.log
  output_dir: outputs/plcs/norm-v2/issue-801/t128-v6/ablation-d-selector-zero-eb32-final/logs/version_0
  curves: knowledge/runs/run-i801-a2-plcs-d-selector-zero/curves.png
  tb_logdir: outputs/plcs/norm-v2/issue-801/t128-v6/ablation-d-selector-zero-eb32-final/logs/version_0
parents:
- run-i801-d-selector-zero-seeded
relations:
- to: run-i801-a2-plcs-d-reference
  rel: compares
tags:
- plcs
- tracking
- camera-view-v2
- reference-camera
- rope
- selector-zero
- ablation-d
- effective-batch-32
---

## 考察 / Findings

### 要約

PLCS Dのreference-frame教師とv2入力は維持し、第三RoPE selectorだけを全`0`にしたmatched対照を100 epoch学習した。test position errorは`5.1417m`、Y-sign accuracyは`0.6845`、heading errorは`34.32deg`だった。

### アーキテクチャ詳細

`run-i801-a2-plcs-d-reference`との差は`reference_selector_mode=selector_zero`とoutput directoryだけである。dataset、split、seed、T128/V6、micro-batch 8×accumulation 4、CUDA CSWA、bf16 mixed、100 epochを固定し、query/reference/otherの第三RoPE座標をすべて`0`にした。

### メトリクスの解釈

X/Y/Z MAEは`1.7684/4.4100/0.3272m`、presence F1は`0.6669`。ID switchesは`36.68`、duplicate active tracksは`296.28`でreference runより多い。validation曲線は前半で改善し、終盤にposition errorがやや戻る。崩壊はないがtrain/val position-loss gapはreference runと同程度に残る。

### アーキテクチャ⇄メトリクスの因果考察

selectorなしでもposition/headingを同程度まで学習できたため、camera-view Court channelとobject内容がreference識別信号を担った可能性がある。一方、tracking identity指標はreference runより悪く、第三軸が別の形でidentity associationへ影響した可能性は残る。

### 既存実験との比較

`run-i801-a2-plcs-d-reference`よりposition errorは`0.0266m`悪いが、Y-signは`0.00422`、headingは`1.39deg`良い。presence F1も`0.00150`良い一方、ID switchesは`11.92`多く、優位性は混在する。

### 次に有効な実験

本runをselectorのmatched対照として保持し、複数seedとreference local-index均衡samplingで差を分解する。現時点ではv1 production defaultを変えない。
