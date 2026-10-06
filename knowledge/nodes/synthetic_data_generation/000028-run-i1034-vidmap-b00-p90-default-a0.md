---
id: run-i1034-vidmap-b00-p90-default-a0
type: run
task: synthetic_data_generation
sequence: 28
recorded_at: '2026-10-07'
title: 'VidMap既定構成のB00共通90枚比較: 31 keyframe・10,280点'
issue: 1034
provider: codex
session: 01a1106e-187c-7163-be4e-9f1673767ad9
date: '2026-10-07'
status: done
config:
  model: VidMap@1a48f2a1c9b59ba1e28bf40eeba1777f5d34ebb1
  data: B00先頭90秒・NHT既定1 fps/quality filterで凍結した共通90枚
  frontend: uncalib/base
  mapping: uncalib/base
  global_positioning_seed: 1
  keypoint_seed: 42 + keyframe_id
  input_resolution: 1920x1080
  pycolmap: 4.3.0 source
  torch: 2.14.0+cu130
  cache: 12-image sanity compiled graphs reused; additional compilation included
metrics:
  elapsed_seconds: 105.863142
  device_peak_memory_mib: 15860.0
  sfm/input_images: 90.0
  sfm/registered_images: 31.0
  sfm/registration_ratio: 0.344444
  sfm/supported_registered_images: 31.0
  sfm/supported_registration_ratio: 0.344444
  sfm/minimum_registered_points_per_image: 1302.0
  sfm/p05_registered_points_per_image: 1340.5
  sfm/median_registered_points_per_image: 1546.0
  sfm/sparse_points: 10280.0
  sfm/points_per_supported_camera: 331.612903
  sfm/point_cloud_coverage/voxel_occupancy_fraction: 0.284144
  sfm/mean_reprojection_error_px: 1.179721
  sfm/median_reprojection_error_px: 1.066975
  sfm/p95_reprojection_error_px: 2.415333
  sfm/mean_track_length: 4.63716
  sfm/median_track_length: 3.0
  sfm/p95_track_length: 13.0
  sfm/trajectory/median_step: 0.566343
  sfm/trajectory/p95_step: 0.726156
  sfm/trajectory/maximum_step: 0.73766
  sfm/trajectory/maximum_step_to_median: 1.302497
  sfm/trajectory/step_outlier_count: 0.0
  sfm/trajectory/median_rotation_step_deg: 3.155536
  sfm/trajectory/p95_rotation_step_deg: 8.618893
  sfm/trajectory/maximum_rotation_step_deg: 11.518439
  sfm/trajectory/planarity_ratio: 0.002565
  sfm/trajectory/p05_step: 0.361037
  sfm/trajectory/near_duplicate_step_count: 0.0
  sfm/trajectory/near_duplicate_fraction: 0.0
  sfm/cameras/0/width: 1920.0
  sfm/cameras/0/height: 1080.0
  sfm/cameras/0/focal_to_width: 0.548154
  sfm/mapping_components: 1.0
  sfm/triangulation/sampled_points: 10280.0
  sfm/triangulation/median_angle_deg: 25.627732
  sfm/triangulation/p05_angle_deg: 2.024694
  sfm/intrinsics_stability/camera_count: 1.0
  sfm/intrinsics_stability/focal_to_width_median: 0.548154
  sfm/intrinsics_stability/focal_length_coefficient_of_variation: 0.0
  sfm/intrinsics_stability/p95_adjacent_relative_focal_change: 0.0
  sfm/minimum_focal_to_width: 0.548154
  sfm/maximum_focal_to_width: 0.548154
repro:
  commit: b113500e37bd391125a39a6a1862a4e73fabf11a
  branch: research/sfm-night-20261006
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: /home/kamimura/projects/tennis-lab/.claude/worktrees/sfm-night-20261006/.venv/bin/python
    -m experiments.sfm_comparison.run_command --campaign /home/kamimura/projects/tennis-lab/outputs/aris/i1034-sfm-night-20261006
    --run-id vidmap-b00-p90-default-a0 --stage sfm --seconds 5400 --required-path
    /home/kamimura/projects/tennis-lab/outputs/aris/i1034-sfm-night-20261006/vidmap-p90-default-a0/evaluation/metrics.json
    --metrics-source /home/kamimura/projects/tennis-lab/outputs/aris/i1034-sfm-night-20261006/vidmap-p90-default-a0/evaluation/metrics.json
    -- env HF_HOME=/home/kamimura/projects/tennis-lab/.claude/worktrees/sfm-night-20261006/.cache/vidmap-model-cache/huggingface
    TORCH_HOME=/home/kamimura/projects/tennis-lab/.claude/worktrees/sfm-night-20261006/.cache/vidmap-model-cache/torch
    HF_HUB_DISABLE_XET=1 TORCHINDUCTOR_CACHE_DIR=/home/kamimura/projects/tennis-lab/.claude/worktrees/sfm-night-20261006/.cache/vidmap-compile
    TORCHINDUCTOR_COMPILE_THREADS=2 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
    /home/kamimura/projects/tennis-lab/.claude/worktrees/sfm-night-20261006/.cache/runtimes/vidmap-native/bin/python
    -m experiments.sfm_comparison.run_vidmap --images /home/kamimura/projects/tennis-lab/outputs/aris/i1034-sfm-night-20261006/baseline-a1/frames/images
    --manifest /home/kamimura/projects/tennis-lab/outputs/aris/i1034-sfm-night-20261006/inputs/manifest-a1.json
    --output /home/kamimura/projects/tennis-lab/outputs/aris/i1034-sfm-night-20261006/vidmap-p90-default-a0
    --evaluator-root /home/kamimura/projects/tennis-lab/.claude/worktrees/sfm-night-20261006/.cache/nht
artifacts:
  run_dir: knowledge/runs/run-i1034-vidmap-b00-p90-default-a0
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791312063000272170_903266_i1034-vidmap-b00-p90-default-a0.log
parents:
- run-i1034-vidmap-b00-sanity12-a0
relations:
- to: run-i1034-nht-b00-p90-s42-a1
  rel: compares
papers: []
tags:
- sfm
- vidmap
- aris
- common-input
---

## 観測と判断

SIFTと同じ凍結90枚を入力し、VidMap既定uncalib/baseが31 keyframeを選択、31/31を登録して10,280点を出力した。元入力全体に対するpose供給範囲は31/90（34.4%）。59枚のposeはrecに含まれない。この構成を、全90枚へposeを供給する現行NHTのSfMへそのまま置き換えない。

queue実行は105.863秒、うちVidMap本体104.776秒、CPU出力評価0.977秒。装置全体の2秒間隔サンプル最大は15,860 MiB（約15.49 GiB）。RTX 5060 Ti 16GBでこの入力は完走したが余裕は小さく、長尺・高解像度・並列half jobに適合するとは結論できない。

## 共通評価

全画像hash・登録画像名・原寸camera・有限geometryを確認し、両手法で同じNHT evaluatorとPyCOLMAP 4.3.0を使った。最終geometryからpoint.errorを再計算し、今回のVidMapでも保存値との差は全点0だった。点ごとのtrack平均pixel残差を点集合で集計したmeanは1.17972 px、medianは1.06697 px、p95は2.41533 px。median track lengthは3、連結成分は1、支持点100以上のcameraは31枚、軌跡step outlierは0。

SIFTの90/90・44,993点・p95 1.98720 pxに対し、出力camera/点数が少なく、今回の内部残差の値も低下していない。ただし特徴点集合、camera model（VidMap PINHOLE、SIFT OPENCV）、最適化と出力camera密度が異なるため、この値だけで絶対精度や描画品質の優劣を断定しない。両方shared cameraなのでfocal変動がほぼ0であることも校正精度の証拠ではない。

## 比較の限界

SIFTはCPU実装で、VidMapはGPU frontendとCPU mapperを使う。VidMapではsanityのcompiled graph cacheを再利用し、SIFT測定時は環境準備も並行していた。105.863対480.068秒は今回の観測値で、統制された高速化率ではない。GPU memoryはprocess専有量・連続測定の真のpeakではなく装置全体のサンプル最大。

共通CLIのgeneric gatesは元のshort-clip gatesと異なる。VidMapは元入力への登録率/支持率とgeneric sparse-point閾値で不合格になるが、31枚をkeyframeとして選んだ挙動を31/31登録失敗とは扱わない。独立pose/ground GT、地面ドリフト、新視点RGB/白線品質、複数scene・seedは未評価。学習曲線はない。

## 後段・再現

PyCOLMAP 4.1.1でも同じ31画像・10,280点・1cameraを読み込み、camera centerは4.3と全成分一致した。NHT trainerの旧SceneManagerとstandard scene exportは未実行。NHT候補形式へのbridgeと共通eval viewの固定が必要である。

raw queue repro、入力manifest、resolved config、実行log、共通metrics/audit、phase times、source/weightsの固定情報をbundleに保存した。外部native runtimeは`experiments/sfm_comparison/ENVIRONMENT.md`の手順で復元する。raw commandのcheckoutだけでモデル重み・native環境まで復元できるという主張ではない。
