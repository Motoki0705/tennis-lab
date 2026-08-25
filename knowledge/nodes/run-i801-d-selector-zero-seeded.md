---
id: run-i801-d-selector-zero-seeded
type: run
title: BLCS D camera-view v2 selector-zero（seeded matched run）
issue: 801
provider: codex
session: 01a03207-0be4-72c1-ba60-a051d0d8d9b6
date: '2026-08-25'
status: done
config:
  model: track_query_ablation_d_v2_selector_zero
  architecture: track_query_ablation_d
  ffn_mode: shared
  mhc_writeback: layer_end
  reference_selector_mode: selector_zero
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
  loss: 0.181808
  loss_position: 0.126897
  loss_position_x: 0.116287
  loss_position_y: 0.231279
  loss_position_z: 0.033124
  loss_presence: 0.054911
  loss_smoothness: 0.0
  loss_gravity: 0.0
  position_error: 0.34378
  presence_precision: 0.954905
  presence_recall: 0.981222
  presence_f1: 0.967648
  lifecycle_presence_f1: 0.967648
  birth_frame_error: 4.859987
  death_frame_error: 5.84545
  query_reuse_count: 1.28
  illegal_overlap_count: 0.0
  segment_id_switches: 24.08
  id_switches: 24.08
  duplicate_active_tracks: 15.08
  missed_gt_frames: 15.44
  inactive_query_false_positives: 0.88
  position_error_m: 3.721058
  position_mae_x_m: 1.639375
  position_mae_y_m: 2.8525
  position_mae_z_m: 0.659375
  x_error_m: 1.639375
  y_error_m: 2.8525
  z_error_m: 0.659375
  y_sign_accuracy: 0.877656
  reference_index_0_position_error_m: 3.452344
  reference_index_1_position_error_m: 2.46317
  reference_index_2_position_error_m: 3.643503
  reference_index_3_position_error_m: 3.348438
repro:
  commit: b392bbcbab877172b74c190af32b4dcc12366853
  branch: feat/issue-801-reference-camera-rope
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: 'CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    /home/kamimura/projects/tennis-lab/.venv/bin/python -c ''import atexit,runpy,torch;
    atexit.register(lambda: print("TRAIN_PEAK_CUDA_ALLOCATED_BYTES="+str(torch.cuda.max_memory_allocated())+"
    TRAIN_PEAK_CUDA_RESERVED_BYTES="+str(torch.cuda.max_memory_reserved()))); runpy.run_module("src.tasks.blcs.scripts.train",run_name="__main__")''
    --config-name train_tracking court_coordinate_normalization=v2 court_keypoints=camera_view_v2
    model=track_query_ablation_d_v2_selector_zero paths.data_root=/home/kamimura/projects/tennis-lab/data
    data.scene_dir=blcs/multi_object_camera_view_norm-v2 data.seq_len_range=[128,128]
    data.num_views_range=[4,4] data.batch_size=8 data.num_workers=16 data.evaluation_reference_camera_id=cam_1
    training.compile.enabled=false training.trainer.accumulate_grad_batches=4 training.trainer.max_epochs=100
    training.trainer.check_val_every_n_epoch=5 training.trainer.enable_progress_bar=false
    training.trainer.enable_model_summary=false training.early_stopping.enabled=false
    run.seed=42 run.fast_dev_run=false run.test_after_fit=true run.output_dir=blcs/norm-v2/issue-801/t128-v4/ablation-d-selector-zero-eb32-seeded-retry4'
artifacts:
  run_dir: knowledge/runs/run-i801-d-selector-zero-seeded
  predictions: knowledge/runs/run-i801-d-selector-zero-seeded/pred_test.npz
  log: .training_queue/logs/1787615604815554190_3420855_i801_normv2_camera_view_d_selector_zero_b8_a4_eb32_e100_gpu0_retry4_seeded.log
  output_dir: outputs/blcs/norm-v2/issue-801/t128-v4/ablation-d-selector-zero-eb32-seeded-retry4/logs/version_0
  curves: knowledge/runs/run-i801-d-selector-zero-seeded/curves.png
  tb_logdir: outputs/blcs/norm-v2/issue-801/t128-v4/ablation-d-selector-zero-eb32-seeded-retry4/logs/version_0
parents:
- run-i786-normv2-large-cuda-ablation-d-b8-a4-eb32-e100-gpu0
relations:
- to: run-i801-d-reference-seeded
  rel: compares
tags:
- blcs
- tracking
- camera-view-v2
- reference-camera
- rope
- selector-zero
- ablation-d
- effective-batch-32
- seeded-replay
---

## 考察 / Findings

### 要約

reference-frame教師とCourt入力は維持しつつ第三RoPE軸を全tokenで`0`にした対照run。seed 42・100 epochでtest `position_error_m=3.7211m`、`y_sign_accuracy=0.8777`となり、matched reference selector runをわずかに上回った。

### アーキテクチャ詳細

D trunk、camera-view CourtKP20、reference-frame target、6入力`reference_view_index`、dataset、学習条件はreference runと同一である。唯一の意図した差は`reference_selector_mode=selector_zero`で、query/reference/otherの第三RoPE座標をすべて`0`にする点である。v1 role軸へのfallbackではない。

### メトリクスの解釈

軸別MAEはX `1.6394m`、Y `2.8525m`、Z `0.6594m`、presence F1は`0.9676`。reference local index別position errorは`3.4523/2.4632/3.6435/3.3484m`だった。`curves.png`ではreference runと非常に近い収束曲線を示し、終盤までval position errorが低下している。崩壊や大きな汎化gapはない。

### アーキテクチャ⇄メトリクスの因果考察

selectorなしでもreference-frame教師を学習できたため、camera-view Court channelやview内容が一意化信号として機能した可能性がある。これは第三RoPE軸が不要だという確定結論ではなく、単一seedの本条件で追加寄与が見えなかったという観測である。

### 既存実験との比較

`run-i801-d-reference-seeded`よりposition errorが`0.0924m`（`2.42%`、referenceを分母）低く、Y-sign accuracyは`0.00656`高い。X/Yはzeroが良く、Zとpresence F1はreferenceがわずかに良いため、全指標で一方向の差ではない。親の#786 Dとはdataset target frameが異なるため直接の一変数比較ではない。

### 次に有効な実験

本runをcamera-view v2内の対照として保持し、production v1は変更しない。追加評価を行うなら複数seedとopposite-side層別でselector効果の信頼区間を得る。
