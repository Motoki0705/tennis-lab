---
id: run-i801-d-reference-seeded
type: run
title: BLCS D camera-view v2 reference selector（seeded matched run）
issue: 801
provider: codex
session: 01a03207-0be4-72c1-ba60-a051d0d8d9b6
date: '2026-08-25'
status: done
config:
  model: track_query_ablation_d_v2_selector
  architecture: track_query_ablation_d
  ffn_mode: shared
  mhc_writeback: layer_end
  reference_selector_mode: reference
  court_keypoint_contract: camera_view_courtkp20_rzpi_v1
  target_frame_contract: reference_camera_court_rzpi_v1
  track_query_rope_contract: time_camera_reference_selector_v1
  loss: tracking
  data: blcs/multi_object_camera_view_norm-v2
  seed: 42
  seq_len: 128
  num_views: 4
  batch_size: 8
  accumulate_grad_batches: 4
  effective_batch_size: 32
  epochs: 100
  precision: bf16-mixed
  cswa_backend: cuda
metrics:
  loss: 0.182473
  loss_position: 0.131362
  loss_position_x: 0.120263
  loss_position_y: 0.241805
  loss_position_z: 0.032019
  loss_presence: 0.05111
  loss_smoothness: 0.0
  loss_gravity: 0.0
  position_error: 0.355952
  presence_precision: 0.956533
  presence_recall: 0.980455
  presence_f1: 0.968175
  lifecycle_presence_f1: 0.968175
  birth_frame_error: 3.546506
  death_frame_error: 5.221581
  query_reuse_count: 1.36
  illegal_overlap_count: 0.0
  segment_id_switches: 24.92
  id_switches: 24.92
  duplicate_active_tracks: 18.639999
  missed_gt_frames: 16.559999
  inactive_query_false_positives: 0.32
  position_error_m: 3.813505
  position_mae_x_m: 1.671875
  position_mae_y_m: 2.9425
  position_mae_z_m: 0.655
  x_error_m: 1.671875
  y_error_m: 2.9425
  z_error_m: 0.655
  y_sign_accuracy: 0.871094
  reference_index_0_position_error_m: 3.41875
  reference_index_1_position_error_m: 2.670387
  reference_index_2_position_error_m: 3.998355
  reference_index_3_position_error_m: 3.289453
repro:
  commit: b392bbcbab877172b74c190af32b4dcc12366853
  branch: feat/issue-801-reference-camera-rope
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: 'CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    /home/kamimura/projects/tennis-lab/.venv/bin/python -c ''import atexit,runpy,torch;
    atexit.register(lambda: print("TRAIN_PEAK_CUDA_ALLOCATED_BYTES="+str(torch.cuda.max_memory_allocated())+"
    TRAIN_PEAK_CUDA_RESERVED_BYTES="+str(torch.cuda.max_memory_reserved()))); runpy.run_module("src.tasks.blcs.scripts.train",run_name="__main__")''
    --config-name train_tracking court_coordinate_normalization=v2 court_keypoints=camera_view_v2
    model=track_query_ablation_d_v2_selector paths.data_root=/home/kamimura/projects/tennis-lab/data
    data.scene_dir=blcs/multi_object_camera_view_norm-v2 data.seq_len_range=[128,128]
    data.num_views_range=[4,4] data.batch_size=8 data.num_workers=16 data.evaluation_reference_camera_id=cam_1
    training.compile.enabled=false training.trainer.accumulate_grad_batches=4 training.trainer.max_epochs=100
    training.trainer.check_val_every_n_epoch=5 training.trainer.enable_progress_bar=false
    training.trainer.enable_model_summary=false training.early_stopping.enabled=false
    run.seed=42 run.fast_dev_run=false run.test_after_fit=true run.output_dir=blcs/norm-v2/issue-801/t128-v4/ablation-d-reference-selector-eb32-seeded-retry4'
artifacts:
  run_dir: knowledge/runs/run-i801-d-reference-seeded
  predictions: knowledge/runs/run-i801-d-reference-seeded/pred_test.npz
  log: .training_queue/logs/1787615604723446824_3420820_i801_normv2_camera_view_d_reference_b8_a4_eb32_e100_gpu0_retry4_seeded.log
  output_dir: outputs/blcs/norm-v2/issue-801/t128-v4/ablation-d-reference-selector-eb32-seeded-retry4/logs/version_0
  curves: knowledge/runs/run-i801-d-reference-seeded/curves.png
  tb_logdir: outputs/blcs/norm-v2/issue-801/t128-v4/ablation-d-reference-selector-eb32-seeded-retry4/logs/version_0
parents:
- run-i786-normv2-large-cuda-ablation-d-b8-a4-eb32-e100-gpu0
relations:
- to: run-i801-d-selector-zero-seeded
  rel: compares
tags:
- blcs
- tracking
- camera-view-v2
- reference-camera
- rope
- selector
- ablation-d
- effective-batch-32
- seeded-replay
---

## 考察 / Findings

### 要約

camera-view v2のreference cameraを第三RoPE軸で明示するD architectureを、seed 42・100 epochで完走した。test `position_error_m=3.8135m`、`y_sign_accuracy=0.8711`であり、完全にmatchedな`selector_zero`よりpositionとY-signの双方で改善しなかった。

### アーキテクチャ詳細

PR #797系列のD（shared FFN / layer-end mHC）を維持し、CourtKP20・3D target・camera extrinsicsを指定reference cameraのproper `I | Rz(pi)` frameへ揃えた。queryとreference objectは第三RoPE座標`0`、other camera objectは`1`とし、v2専用6入力契約で`reference_view_index`を渡す。datasetは`blcs/multi_object_camera_view_norm-v2`、T=128、V=4、micro-batch 8×accumulation 4、bf16 mixed、CUDA CSWA、100 epochである。

### メトリクスの解釈

軸別MAEはX `1.6719m`、Y `2.9425m`、Z `0.6550m`で、Y誤差が支配的だった。presence F1は`0.9682`。reference local index別position errorはindex 0/1/2/3で`3.4188/2.6704/3.9984/3.2895m`である。`curves.png`ではtrain/val position lossがほぼ同じ軌跡で低下し、val position errorも終盤まで緩やかに改善しており、学習崩壊や明瞭な過学習は見られない。

### アーキテクチャ⇄メトリクスの因果考察

第三軸selectorが利用可能でも、単一seedではsymmetric target frameの識別改善は観測されなかった。仮説として、第2軸camera位置とcamera-view Court channelだけでreference側を相当識別でき、追加phase差が冗長だった可能性がある。ただし単一seedの小差なので、selector自体が有害だとは断定しない。

### 既存実験との比較

直接の一変数比較は`run-i801-d-selector-zero-seeded`である。全test target・mask・scene・view集合/順・reference identity/index・transformが一致し、selector modeだけが異なる。zero比でposition errorは`+0.0924m`（`+2.48%`）、Y-sign accuracyは`-0.00656`だった。親の#786 Dはphysical Court datasetなので、数値差をselectorだけへ帰属できない。

### 次に有効な実験

production defaultはv1のまま維持する。selector仮説を継続する場合は、同じdeterministic data contractで3 seeds以上を実行し、opposite-side sampleだけの層別結果とattentionのreference利用率を確認する。
