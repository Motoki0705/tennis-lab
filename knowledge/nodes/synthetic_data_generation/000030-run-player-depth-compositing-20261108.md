---
id: run-player-depth-compositing-20261108
type: run
task: synthetic_data_generation
sequence: 30
recorded_at: '2026-11-08'
title: NHT背景と不透明前景の奥行き合成の情報不足とshader整合を検証する
issue: 1728
provider: codex
date: '2026-11-08'
status: done
summary_impact: updated
config:
  model: existing_nht_shader_no_retraining
  background: B00
  court: court-000
  resolution:
  - 256
  - 144
  camera_count: 3
  batch_size: 1
  known_sheet_cases: 4
  blender_version: 4.5.3 LTS
  blender_engine: CYCLES
  blender_device: CPU
  blender_resolution:
  - 32
  - 18
  nht_commit: bf05cd79b54475752fee8d8ebd19d743c5fa542f
  gsplat_commit: c8d5acd22314c1c1144a30e635e08d9c738a8b39
metrics:
  cpu_tests_passed: 35
  real_camera_count: 3
  gaussian_count: 1000000
  known_sheet_max_abs_error: 1.2874603272372553e-08
  public_three_camera_seconds_including_load: 3.799702899000067
  resident_load_seconds: 0.17952324900011263
  steady_full_min_seconds: 0.006339361999835091
  steady_full_max_seconds: 0.01419369099994583
  prefix_min_seconds: 0.002796419999867794
  prefix_max_seconds: 0.005168995999838444
  cuda_peak_allocated_bytes: 518069248
  cuda_peak_reserved_bytes: 541065216
  decoder_zero_features_min_rgb: 0.23193359375
  decoder_zero_features_max_rgb: 0.70263671875
  decoder_affinity_max_abs_error: 0.19012451171875
  blender_z_vs_camera_z_max_error_m: 0.0
  blender_z_vs_ray_distance_max_error_m: 0.7011399269104004
  blender_linear_premult_max_error: 0.0
  blender_pixel_center_ray_max_error_px: 0.0005300045013423293
repro:
  commit: 49eb7b4ab5ad04e2f2592cf90c5b2d777e02a567
  command: 'bash -lc ''set -euo pipefail

    export PYTHONPATH=.

    export PYTHONDONTWRITEBYTECODE=1

    /root/.local/share/uv/tools/nht/bin/python knowledge/runs/run-player-depth-compositing-20261108/gpu_probe.py
    --asset-root tmp/player-depth-assets/scenes/B00 --court court-000 --output outputs/synthetic_data_generation/profile/player-depth-study/probe'''
  cwd: .
artifacts:
  run_dir: knowledge/runs/run-player-depth-compositing-20261108
  drive_record: operations/colab/player-depth-study-20261108/probe
  drive_outputs:
  - outputs/synthetic_data_generation/profile/player-depth-study/probe
  - outputs/synthetic_data_generation/profile/player-depth-study/blender
  - outputs/synthetic_data_generation/profile/player-depth-study/inputs
  blender_drive_record: operations/colab/player-depth-study-20261108/blender-calibration
  setup_drive_record: operations/colab/player-depth-study-20261108/setup
  input_drive_record: operations/colab/player-depth-study-20261108/inputs-and-runtime-check
parents:
- run-court-assets-colab-l4-20261022
relations: []
papers: []
tags:
- compositing
- nht
- colab
- negative-result
---

## 結論

現行NHTのRGB・alpha・expected depthだけを使う前景合成は採用しない。
同じRGB/alpha/EDでも正しい遮蔽色が異なる反例をCPUで構成し、通常eval3dの4つの既知層を
GPUでも最大差1.29e-8で確認した。NHTの非線形shaderは部分特徴量をRGB寄与へ変換する
契約を持たず、空特徴でもRGB=.2319〜.7026、affine混合との差は最大.1901だった。
alphaを掛け直す候補と特徴正規化の候補の最大差.1263は、正解との誤差や誤差上限ではない。

B00の3 cameraを256×144で描き、全背景alphaの平均が.9998でも前景までのopacityは
それより低いことを確認した。保存コートから投影したネット帯の全背景alpha平均は.999878、
6 m prefixは.049885、10 m prefixは.401645。帯はsemantic maskではなく、色の正解を
取得したという主張でもない。白い帯・ぼけは既存NHTの表示に残る。

前景までのlinear RGB寄与・透過率を返す必要な目標契約を固定し、単位・座標・色・alphaと
公開API変更案を[研究bundle](../../runs/run-player-depth-compositing-20261108/README.md)へまとめた。
局所linear RGBを取得するAPI/モデルは未実装であり、C5の本番合成は未達。
現在のcheckpointを保つ近似の受容、または局所radiance decoderの学習・蒸留を許す追加範囲を、
親と人間が判断する必要がある。背景再学習・NHT本体変更・C5実装は実施していない。

## 単位・色・性能の確認

B00〜B03すべての採用alignmentの内容digestと相互変換・回転を確認した。
実GPU描画はB00だけで、各シーンのunits/mを共通値へ置き換えていない。
Blender 4.5.3 / Cycles CPUの既知平面ではZはcamera zだった（最大差0 m）。
ray距離とみなすと最大.70114 mずれた。部分coverageを含むCombined RGBは
premultiplied linearの期待値に最大差0で一致した。別engine/versionには未検証である。

L4で現行NHT full RGB+EDは6.34〜14.19 ms/frame、mean-z prefixは2.80〜5.17 ms/frame。
PyTorch peak allocatedは494.07 MiB。これは新しい物理合成器の性能保証でも総VRAM測定でもない。
必要tensorの容量計算、全量配列を保持しない方針と未測定の本番条件をbundleへ記録した。

## 実行と残る制約

CPU小配列14件＋既存NHT境界21件の35件が成功（3.88 s、-n 2）。収集35件、最大展開6件。
GPU/Blender描画を通常pytestへ追加していない。初回Colab setupはnative build後のPATH検査で
失敗し、PATH明示で復旧した。以後の入力検証・GPU probe・Blender校正はreturncode=0で、
失敗jobを含む全jobのDrive保存を確認した。session経過上限30.37分でstop済み。
TensorBoard・training curvesは学習を行っていないため該当しない。

調査runのstatus=doneはこの負の結果と契約の調査完了を表し、合成方式の採用成功を表さない。
動的前景合成の未対応範囲と必要な判断を新しく明らかにしたためsummary_impactをupdatedとし、
summaryへリンクした。既存Court生成recipe・採用checkpointは維持する。
C3の人間受入、任意前景のper-pixel gate、物理的な局所RGB、実人物/球の本番合成、
本番解像度での時間・総VRAMは未確認であり、C5開始条件をこの調査で解除しない。
