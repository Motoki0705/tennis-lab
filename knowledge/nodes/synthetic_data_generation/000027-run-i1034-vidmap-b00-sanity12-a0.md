---
id: run-i1034-vidmap-b00-sanity12-a0
type: run
task: synthetic_data_generation
sequence: 27
recorded_at: '2026-10-07'
title: 'VidMap既定構成がB00 12枚sanityを完走: 7 keyframe・2,648点'
issue: 1034
provider: codex
session: 01a1106e-187c-7163-be4e-9f1673767ad9
date: '2026-10-07'
status: done
config:
  model: VidMap@1a48f2a1c9b59ba1e28bf40eeba1777f5d34ebb1
  data: B00先頭12枚・manifest-sanity12.json; runtime sanity only
  frontend: uncalib/base
  mapping: uncalib/base
  global_positioning_seed: 1
  keypoint_seed: 42 + keyframe_id
  input_resolution: 1920x1080
  pycolmap: 4.3.0 source
  torch: 2.14.0+cu130
metrics:
  elapsed_seconds: 350.425862
  device_peak_memory_mib: 9060.0
  input_images: 12
  registered_images: 7
  registration_ratio: 0.5833333333333334
  supported_registered_images: 7
  sparse_points: 2648
  mean_reprojection_error_px: 0.9477977956851569
  median_reprojection_error_px: 0.7603471971608171
  p95_reprojection_error_px: 2.3534708809713907
  mean_track_length: 4.139350453172206
  median_track_length: 4.0
  mapping_components: 1
repro:
  commit: bed9964eee2d2de0e818e76a4a56fe5cf6b91c67
  branch: research/sfm-night-20261006
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: /home/kamimura/projects/tennis-lab/.claude/worktrees/sfm-night-20261006/.venv/bin/python
    -m experiments.sfm_comparison.run_command --campaign /home/kamimura/projects/tennis-lab/outputs/aris/i1034-sfm-night-20261006
    --run-id vidmap-b00-sanity12-a0 --stage sanity --seconds 1800 --required-path
    /home/kamimura/projects/tennis-lab/outputs/aris/i1034-sfm-night-20261006/vidmap-sanity12-a0/rec/images.bin
    -- env HF_HOME=/home/kamimura/projects/tennis-lab/.claude/worktrees/sfm-night-20261006/.cache/vidmap-model-cache/huggingface
    TORCH_HOME=/home/kamimura/projects/tennis-lab/.claude/worktrees/sfm-night-20261006/.cache/vidmap-model-cache/torch
    HF_HUB_DISABLE_XET=1 TORCHINDUCTOR_CACHE_DIR=/home/kamimura/projects/tennis-lab/.claude/worktrees/sfm-night-20261006/.cache/vidmap-compile
    TORCHINDUCTOR_COMPILE_THREADS=2 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
    /home/kamimura/projects/tennis-lab/.claude/worktrees/sfm-night-20261006/.cache/runtimes/vidmap-native/bin/python
    -m vidmap.run --input_data /home/kamimura/projects/tennis-lab/outputs/aris/i1034-sfm-night-20261006/inputs/sanity12
    --output /home/kamimura/projects/tennis-lab/outputs/aris/i1034-sfm-night-20261006/vidmap-sanity12-a0
    --device cuda
artifacts:
  run_dir: knowledge/runs/run-i1034-vidmap-b00-sanity12-a0
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1791306572233754420_305168_i1034-vidmap-b00-sanity12-a0.log
parents:
- run-i1034-vidmap-cuda-probe-a0
relations: []
papers: []
tags:
- sfm
- vidmap
- aris
- sanity
---

## 観測

固定したB00先頭12枚を公式VidMap既定uncalib/baseへ入力し、共有queueで350.426秒で完走した。7枚をkeyframeとして選択し、7枚すべてを最終recへ登録した。元入力に対するpose供給範囲は7/12、疎点は2,648。小さなCUDA probeに加え、事前学習重みを使うfrontendからnative mapper、COLMAP出力まで実行できた。

装置全体の2秒間隔サンプル最大は9,060 MiB。初回RoMaV2/DA3コンパイルとDINOv3 source取得を含む時間であり、warm-cacheの速度benchmarkではない。画像は変更していない。

## 出力検証

12枚のhash、登録画像名、原寸camera、有限geometryを検査した。最終geometryから残差を再計算し、保存済みpoint.errorとの差は全点0だった。点ごとのtrack平均pixel残差を点集合で集計したp95は2.35347 px、meanは0.94780 px、median track lengthは4、連結成分は1。PyCOLMAP 4.1.1側でも同じ7画像・2,648点・1cameraを読み込めた。gsplat trainerの旧SceneManager・NHT学習/exportの互換性は未検証。

## 解釈と次の実験

12枚での実行可能性を確認したsanityであり、90枚SIFTとの精度・速度の順位付けには使わない。VidMapは非keyframe poseをこのrecへ出力しないため、7/12をkeyframe登録失敗率とも解釈しない。元画像全体へposeを提供する既存NHT workflowへの採用には別の受け渡し処理が必要。

独立pose/ground GTはなく、絶対精度や地面ドリフト改善は未測定。次はSIFTと同じ凍結90枚で既定構成を評価する。後段NHT学習と学習曲線はない。

## 保存・再現の範囲

実際のconfig、入力manifest、command.log、実行記録、共通metrics/audit、source/weightsの固定情報をbundleに保存した。raw queue reproは保持している。外部`vidmap.run`とnative runtimeはrepo外の環境で、現行repro path checkerはこの外部moduleを解決できない。復元手順は`experiments/sfm_comparison/ENVIRONMENT.md`を参照する。strictな自動再現検査の成功とは扱わない。
