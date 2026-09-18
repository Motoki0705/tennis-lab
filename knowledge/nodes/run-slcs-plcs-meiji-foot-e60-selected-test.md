---
id: run-slcs-plcs-meiji-foot-e60-selected-test
type: run
title: Meiji PLCS validation選定重みのCPU test
provider: codex
date: '2026-09-18'
status: done
config:
  model: multiview_axial_foot_residual
  data: camera_view_real_rgb_ft_v1
  precision: 32-true
  device: cpu
metrics:
  position_error_m: 0.2452688217163086
  angular_error_deg: 5.331236362457275
  position_accuracy_0.5m: 0.9361119866371155
  angle_accuracy_15deg: 0.9721306562423706
  canonical_mpjpe_m: 0.14431359147233117
  canonical_pck_0.1m: 0.41026804500960573
repro:
  command: bash outputs/plcs/analyze/meiji_foot_final/s42-001/evaluate_selected.sh
artifacts:
  run_dir: knowledge/runs/run-slcs-plcs-meiji-foot-e60-selected-test
  predictions: knowledge/runs/run-slcs-plcs-meiji-foot-e60-selected-test/pred_test.npz
  evaluation: knowledge/runs/run-slcs-plcs-meiji-foot-e60-selected-test/evaluation.json
  output_dir: outputs/plcs/analyze/meiji_foot_final/s42-001/selected_test
parents:
- run-slcs-plcs-meiji-foot-e60-resume-v3
relations: []
tags:
- slcs
- plcs
- meiji
- evaluation
---

## 考察 / Findings

### 要約
validationのみで選んだepoch57重みをCPUで独立にtest評価。位置誤差0.245268822m、回転誤差5.33123636deg、canonical MPJPE0.144313589m。

### アーキテクチャ詳細
学習時と同じPLCS config/モデル/metric/data loaderを使用。CUDA_VISIBLE_DEVICES空、OMP/MKL各2、precision=32-true、num_workers=0。選定checkpointをstrict loadした。重みSHA256と再実行スクリプトはbundle参照。

### メトリクスの解釈
200 sceneのsource-motion分離合成test。終端epoch59のbundleに対し、scene IDs・全教師・padding mask・camera/reference情報を含む全非予測arrayが完全一致。幾何/合成教師との一致であり、独立実測3D精度ではない。

### アーキテクチャ⇄メトリクスの因果考察
CPU FP32評価を使用した。評価のみのrunなので学習曲線は無い。kg_curvesの近似fingerprintは終端train runを誤対応したため、その自動生成曲線は削除し親runの曲線を参照する。終端testはGPU bf16-mixedなので、両者の数値差をcheckpoint改善だけに帰属できない。

### 既存実験との比較
終端epoch59は位置0.250760436m、回転5.34375deg、canonical MPJPE0.142817244m。選定重みは位置・回転が小さくcanonical poseは大きいが、選定基準はtestではなくvalidation位置誤差のみ。

### 次に有効な実験
選定済み教師を実RGBへ適用する。独立3D GTの収集後に実世界での絶対誤差を評価する。
