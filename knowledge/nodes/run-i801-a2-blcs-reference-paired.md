---
id: run-i801-a2-blcs-reference-paired
type: run
title: BLCS D reference selector strict paired評価（Attempt 2）
issue: 801
provider: codex
session: 01a03207-0be4-72c1-ba60-a051d0d8d9b6
date: '2026-08-25'
status: done
config:
  model: track_query_ablation_d_v2_selector
  task: blcs
  evaluation: reference_counterfactual_paired
  selector_mode: reference
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
  checkpoint_sha256: 51e49749b9389157c0e975729a329ca2aced003fe0fa4830daf5d1aa334569fc
  manifest_digest: 61e7ce3b9aca1db325f6399aa1ac7c4f4d10e4caf70feb02d3b16e6ec11faf8d
  report_digest: df5f558763f422e35ed63815a0c0f7e72708def179240a32da2229d551312363
metrics:
  physical_restored_position_consistency_error_m: 3.058086
  reference_target_opposite_side_position_error_x_m: 2.535247
  reference_target_opposite_side_position_error_y_m: 5.126343
  reference_target_opposite_side_position_error_z_m: 0.935184
  reference_target_opposite_side_reference_index_0_position_error_m: 6.312042
  reference_target_opposite_side_y_sign_accuracy: 0.766771
  reference_target_same_side_position_error_x_m: 2.472001
  reference_target_same_side_position_error_y_m: 5.097868
  reference_target_same_side_position_error_z_m: 0.944553
  reference_target_same_side_reference_index_2_position_error_m: 6.267955
  reference_target_same_side_y_sign_accuracy: 0.753133
repro:
  commit: 40c21fbad59e040f52c040ae354637c5f3c8975a
  branch: feat/issue-801-reference-camera-rope
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    /home/kamimura/projects/tennis-lab/.venv/bin/python -m src.tasks.blcs.scripts.evaluate_reference_counterfactual
    evaluation.checkpoint_path=/home/kamimura/projects/tennis-lab/.claude/worktrees/issue-801-reference-rope/outputs/blcs/norm-v2/issue-801/t128-v4/ablation-d-reference-selector-eb32-seeded-retry4/logs/version_0/checkpoints/last.ckpt
    paths.data_root=/home/kamimura/projects/tennis-lab/data data.scene_dir=blcs/multi_object_camera_view_norm-v2
    data.batch_size=1 data.num_workers=16 model=track_query_ablation_d_v2_selector
    evaluation.trainer.accelerator=gpu evaluation.trainer.precision=bf16-mixed
artifacts:
  run_dir: knowledge/runs/run-i801-a2-blcs-reference-paired
  predictions: knowledge/runs/run-i801-a2-blcs-reference-paired/pred_test.npz
  reference_counterfactual: knowledge/runs/run-i801-a2-blcs-reference-paired/reference_counterfactual.json
  log: .training_queue/logs/1787643447709299464_3887229_i801_attempt2_blcs_reference_paired_v6_final.log
parents:
- run-i801-d-reference-seeded
relations:
- to: run-i801-a2-blcs-selector-zero-paired
  rel: compares
tags:
- blcs
- reference-camera
- counterfactual
- paired-evaluation
- selector
- ablation-d
- camera-view-v2
---

## 考察 / Findings

### 要約

BLCSの学習済みD reference-selector checkpointを、同じ100 test scenesの`cam_2`（same side）と`cam_0`（opposite side）でcheckpoint-only評価した。Y-sign accuracyは`0.7531`対`0.7668`、physical frameへ戻した予測間position consistencyは`3.0581m`であり、camera sideを変えても一意なphysical軌道へ収束するところまでは達していない。

### アーキテクチャ詳細

親run `run-i801-d-reference-seeded`のD（shared FFN / layer-end mHC）checkpointを用いる。queryと指定camera objectの第三RoPE座標を`0`、それ以外を`1`にする`reference` selector、camera-view CourtKP20、reference-camera target frameを維持し、fitを呼ばずにT128/V6の2 passだけを実行した。

### メトリクスの解釈

same-sideのX/Y/Z MAEは`2.4720/5.0979/0.9446m`、opposite-sideは`2.5352/5.1263/0.9352m`で、どちらもY誤差が支配的である。reference local index別position errorはindex 2で`6.2680m`、index 0で`6.3120m`。評価runのため収束曲線はない。

### アーキテクチャ⇄メトリクスの因果考察

side間のreference-frame誤差は近いが、physical-restored prediction consistencyが`3.0581m`残る。selector信号だけでは、同じphysical軌道を`I | Rz(pi)`のどちらから要求しても同一点へ戻す制約を十分に学習できていない可能性がある。これはpaired predictionの観測であり、単独test metricだけからは見えない。

### 既存実験との比較

matched対照の`run-i801-a2-blcs-selector-zero-paired`よりphysical consistencyは`0.2678m`良い。same-sideではXYZすべて良い一方でY-signは`0.00544`低く、opposite-sideではY-signとZが良いがX/Yが悪い。したがってselectorの寄与は指標・sideで混在する。

### 次に有効な実験

production defaultはv1を維持する。継続する場合は、physical-restored consistencyを直接lossへ入れるpaired-reference trainingと複数seedを比較する。
