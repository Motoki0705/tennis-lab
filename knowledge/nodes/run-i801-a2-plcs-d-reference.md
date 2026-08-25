---
id: run-i801-a2-plcs-d-reference
type: run
title: PLCS D camera-view v2 reference selector（Attempt 2）
issue: 801
provider: codex
session: 01a03207-0be4-72c1-ba60-a051d0d8d9b6
date: '2026-08-25'
status: done
config:
  model: track_query_ablation_d_v2_selector
  architecture: track_query_ablation_d
  task: plcs
  ffn_mode: shared
  mhc_writeback: layer_end
  reference_selector_mode: reference
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
  loss: 0.883266
  loss_position: 0.162313
  loss_rotation: 0.257283
  loss_presence: 0.592311
  loss_track_smoothness: 0.0
  position_error: 0.455151
  presence_precision: 0.505365
  presence_recall: 0.992618
  presence_f1: 0.665436
  lifecycle_presence_f1: 0.665436
  birth_frame_error: 12.192044
  death_frame_error: 14.70257
  query_reuse_count: 0.24
  illegal_overlap_count: 0.0
  segment_id_switches: 24.76
  id_switches: 24.76
  duplicate_active_tracks: 152.160004
  missed_gt_frames: 9.44
  inactive_query_false_positives: 1365.119995
  angular_error_deg: 35.489464
  heading_error_deg: 35.709999
  position_error_m: 5.115064
  x_error_m: 1.7825
  y_error_m: 4.38625
  z_error_m: 0.340313
  y_sign_accuracy: 0.680313
  reference_index_0_position_error_m: 5.759046
  reference_index_1_position_error_m: 4.902083
  reference_index_2_position_error_m: 4.536458
  reference_index_5_position_error_m: 5.506793
  reference_index_3_position_error_m: 5.015625
  reference_index_4_position_error_m: 6.867188
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
    run.test_after_fit=true model=track_query_ablation_d_v2_selector run.output_dir=plcs/norm-v2/issue-801/t128-v6/ablation-d-reference-selector-eb32-final
artifacts:
  run_dir: knowledge/runs/run-i801-a2-plcs-d-reference
  predictions: knowledge/runs/run-i801-a2-plcs-d-reference/pred_test.npz
  log: .training_queue/logs/1787644646345366163_3910851_i801_attempt2_plcs_d_reference_t128_v6_eb32_e100_final.log
  output_dir: outputs/plcs/norm-v2/issue-801/t128-v6/ablation-d-reference-selector-eb32-final/logs/version_0
  curves: knowledge/runs/run-i801-a2-plcs-d-reference/curves.png
  tb_logdir: outputs/plcs/norm-v2/issue-801/t128-v6/ablation-d-reference-selector-eb32-final/logs/version_0
parents:
- run-i801-d-reference-seeded
relations:
- to: run-i801-a2-plcs-d-selector-zero
  rel: compares
tags:
- plcs
- tracking
- camera-view-v2
- reference-camera
- rope
- selector
- ablation-d
- effective-batch-32
---

## 考察 / Findings

### 要約

PLCSへD architectureとreference-camera第三RoPE軸を適用し、dataset v2、seed 42、T128/V6、effective batch 32、100 epochで完走した。test position errorは`5.1151m`、Y-sign accuracyは`0.6803`、heading errorは`35.71deg`で、matched selector-zeroに一方向の改善を示さなかった。

### アーキテクチャ詳細

PR #797系列のD（shared FFN / layer-end mHC）をPLCSへ適用した。指定cameraのCourtKP20、3D position、headingをproper `I | Rz(pi)` reference frameへ変換し、queryと指定camera objectは第三RoPE座標`0`、other camera objectは`1`とする。CUDA CSWA、bf16 mixed、micro-batch 8×accumulation 4で学習した。

### メトリクスの解釈

X/Y/Z MAEは`1.7825/4.3863/0.3403m`でYが支配的、presence F1は`0.6654`、ID switchesは`24.76`だった。reference local index別position errorはindex 2が最小`4.5365m`、index 4が最大`6.8672m`で偏りがある。収束曲線はposition/angular validation errorが前半で低下し、終盤は横ばいで、崩壊はないがtrain/val position-loss gapが残る。

### アーキテクチャ⇄メトリクスの因果考察

selector runはoverall positionをzeroより`0.0266m`改善しID switchesも`11.92`少ない一方、Y-sign、heading、presence F1は悪い。第三軸がtracking identityへ寄与した可能性はあるが、対称frameの一意化へ直接効いたとは言えない。

### 既存実験との比較

matched `run-i801-a2-plcs-d-selector-zero`に対しposition errorは`0.52%`低いが、Y-signは`0.00422`、headingは`1.39deg`悪い。BLCSのreference/zero比較と同様に効果が指標間で混在している。

### 次に有効な実験

paired評価でside別挙動を確認し、production v1は維持する。継続する場合は3 seeds以上とreference local-index均衡samplingで差の再現性を測る。
