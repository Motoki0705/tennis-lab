---
id: run-ball-checkpoint-normalization-meiji-20260923
type: run
task: ball_detection
sequence: 18
recorded_at: '2026-09-23'
title: 'Ball推論: 保存されたRGB正規化の復元と実映像CPU比較'
provider: codex
status: done
config:
  checkpoint_sha256: cd7927ad27e53ddd6aa77df28eca3c5e674552461ccda083a41e99e629857892
  device: cpu
  normalization:
    enabled: true
    mean:
    - 0.485
    - 0.456
    - 0.406
    std:
    - 0.229
    - 0.224
    - 0.225
  frames_per_window: 8
  window_starts:
  - 0
  - 96
  - 232
  - 400
  - 504
  - 528
  - 768
  - 824
  - 992
  worktree: /home/kamimura/projects/tennis-lab/.claude/worktrees/tennis-scene-ball-preprocessing
metrics:
  meiji_selected_windows:
    cam0:
      missing_normalization:
        observed: 55
        detected_on_observed: 14
        missing_rate: 0.7454545454545455
        distance_px:
          mean: 191.13477032279474
          median: 238.24751405352907
          p95: 338.1685482943887
      saved_normalization:
        observed: 55
        detected_on_observed: 18
        missing_rate: 0.6727272727272727
        distance_px:
          mean: 41.993790356247914
          median: 6.583340740949266
          p95: 206.15950146014322
    cam1:
      missing_normalization:
        observed: 71
        detected_on_observed: 43
        missing_rate: 0.3943661971830986
        distance_px:
          mean: 326.1350869942494
          median: 7.558165262223937
          p95: 761.4654587271341
      saved_normalization:
        observed: 71
        detected_on_observed: 42
        missing_rate: 0.4084507042253521
        distance_px:
          mean: 60.15796960673705
          median: 4.8767200754253395
          p95: 323.1498294214242
    cam2:
      missing_normalization:
        observed: 62
        detected_on_observed: 50
        missing_rate: 0.19354838709677424
        distance_px:
          mean: 109.8701069143359
          median: 3.497922186424942
          p95: 759.2660414379559
      saved_normalization:
        observed: 62
        detected_on_observed: 48
        missing_rate: 0.22580645161290325
        distance_px:
          mean: 3.783591637483692
          median: 3.8120224963869864
          p95: 7.323245013094976
  tracknet_4px_matches_before_after:
  - 5
  - 4
  tracknet_model_input_max_abs_difference_vs_dataset: 0.0
  unit_tests_passed: 241
  mypy_files: 49
artifacts:
  run_dir: knowledge/runs/run-ball-checkpoint-normalization-meiji-20260923
parents:
- run-tennis-scene-meiji-raw-ball-baseline-20260923
relations: []
papers: []
tags:
- cpu_probe
- preprocessing
- checkpoint_contract
date: '2026-09-23'
session: 01a0c8c3-d190-7c51-b671-ded223340a2f
---

## 修正対象と独立確認

現checkpointはImageNet正規化enabled=true、mean=[0.485,0.456,0.406]、std=[0.229,0.224,0.225]を保存している。公開APIは生RGB[0,1]を要求するため、呼出し側の正規化を無効にするだけでは学習時のモデル入力にならなかった。生RGBの検証後、保存された変換をRGB/MDD変換前に一度だけ適用する構成へ修正した。学習・評価のdataset側で変換済みの入力は、その宣言範囲を検証して二重適用を避ける。保存宣言の欠落や設定不一致は明示エラーとした。

進行中のGPU runのコードを変えず、別の専用worktreeで修正・CPU検証を行った。RGB/MDD、両layout、境界値、checkpoint/UI/pipelineの経路を確認した。TrackNetの実8frameでは、既存datasetのNumPy前処理を独立に適用した場合と新しいTorch前処理のモデル入力が完全一致し、復号座標も一致した。

## 実映像診断

Meijiの3camera・各9窓×8frameをCPUで同じ重み・同じ画像について比較した。難しい時点を含む選定窓であり、全区間や独立test精度ではない。重複window集約・trajectory gate前の比較である。

検出できたobserved frameの平均画素距離は、保存正規化の復元でcam0 191.1→42.0、cam1 326.1→60.2、cam2 109.9→3.8となった。ただし検出数は14→18 / 43→42 / 50→48で、一律にrecallが上がるわけではない。全区間の欠損と3D軌道を再評価する必要がある。

TrackNetの既存8frame検査は4px以内の数が5→4、中央値2.87→3.71pxとなった。元テストの「5件以上」は未正規化入力に結び付いた経験値であり、保存設定復元の仕様とは分ける。旧テストの失敗ログを残し、独立したdataset前処理との全座標一致と既存のGT中央値上限を検査するテストへ改めた。閾値・GT・checkpointを調整して結果を合わせてはいない。修正後の関連241件、Ruff、mypy 49fileは通過した。

この段階はCPU診断と実装検証までで、主worktreeへの統合・修正版の全GPU clip/動画評価・validatorは未完了。学習ではなくTensorBoard曲線はない。
