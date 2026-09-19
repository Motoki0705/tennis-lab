---
id: run-slcs-ball-velocity-train-scale-v1
type: run
title: 'SLCS速度loss準備: train-only教師速度中央値11.2595m/s'
provider: codex
session: 01a0b71f-c516-75b2-ac58-0c92e8a4f098
date: '2026-09-19'
status: done
config:
  model: none, label-only CPU analysis
  loss: planned supervised ball velocity
  data: slcs/real_rgb_v1 train only
  confidence_threshold: 0.5
metrics:
  train_windows: 466
  unique_positive_weight_camera_frame_pairs: 25694
  selected_high_confidence_pairs: 23493
  selected_scale_mps: 11.259468485469933
  high_confidence_speed_p95_mps: 26.425038598234046
  high_confidence_speed_p99_mps: 35.37607360681939
  high_confidence_speed_max_mps: 63.862420299756614
repro:
  commit: 559c04189770ade585e554c82a25bb3c39a8f423
  branch: codex/slcs-real-rgb
  command: >-
    env CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv/bin/python knowledge/runs/run-slcs-ball-velocity-train-scale-v1/analyze.py /home/kamimura/projects/tennis-lab/outputs/slcs/train/real_rgb_no_ball_smooth/s42-takeover-003/config.yaml
artifacts:
  run_dir: knowledge/runs/run-slcs-ball-velocity-train-scale-v1
  log: knowledge/runs/run-slcs-ball-velocity-train-scale-v1/report.json
parents:
- run-slcs-full-no-smooth-gap-rgb-val-v2
relations: []
tags:
- slcs
- real-rgb
- train-only
- velocity
- calibration
---

## 考察 / Findings

### 要約

教師速度lossの単位スケールをvalidation/testを見ずに決めるため、保存configと同じtrain窓の教師をCPU集計した。
両端confidenceの最小値が0.5以上のunique camera-frame pairの速度中央値は11.2594684855m/sだった。
この値を次の速度loss実験のscaleへ固定し、loss重み自体は別のtrain勾配probeで決める。

### アーキテクチャ詳細

productionのsplit・quality・windowを保持したSLCSWindowDatasetを用い、augmentationを無効化した。
教師だけのCPU解析なのでrequire_dino=Falseを明示し、特徴cacheやmodelには触れていない。
非padding・両端valid・frame index差1・両端confidence正のpairを抽出し、normalized差分をcourt scaleと実clip FPSでm/sへ変換した。
video/clip/camera/始点frameで重複排除し、重複する速度とconfidenceが完全一致することをassertした。

### メトリクスの解釈

trainはvideo_000、broadcast_shanghai、broadcast_washingtonのみ。466窓、正weight unique pair25694。
confidence>=0.5の23493pairでは速度p95=26.4250、p99=35.3761、max=63.8624m/s。
全正weightの中央値は11.1349m/sで、threshold変更による値の差もreportへ保持した。
confidence閾値はscale選定だけに使い、学習の教師をこの閾値で除外する方針ではない。学習曲線は対象外。

### アーキテクチャ⇄メトリクスの因果考察

教師の高速区間は実在するため、予測速度をゼロへ罰することや教師最大値でclipすることは目的に合わない。
速度残差をtrain由来の代表速度で無次元化する工学的設定であり、論文の推奨定数ではない。
これだけで教師の3D精度や時間的正しさは保証しない。

### 既存実験との比較

親のgap評価に見られる巨大予測速度を教師との整合lossで抑える準備であり、model改善の結果ではない。
別条件の教師・splitは作らず、既存60epoch controlの学習入力を保持した。

### 次に有効な実験

同じ初期化と固定train mini-batchでball supervised項と追加velocity項の出力勾配normを測り、
追加項を約10%にする重みを一度固定する。これは探索上の初期値で、testから調整しない。
その後60epochの単一loss追加比較を行い、位置平均/裾・速度誤差・高速教師区間・playerを確認する。
