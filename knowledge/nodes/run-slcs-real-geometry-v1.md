---
id: run-slcs-real-geometry-v1
type: run
title: 'Meiji: 多視点観測による教師ラベル補正'
provider: codex
date: '2026-09-18'
status: done
config:
  model: fixed PLCS/BLCS + geometry refinement
  loss: no training
  data: Meiji clip000 development
metrics:
  observable/ball_reprojection_median_px: 3.192934058942943
  observable/player_reprojection_median_px: 16.405216900073224
  evidence/ball_fraction: 0.9801980198019802
artifacts:
  run_dir: knowledge/runs/run-slcs-real-geometry-v1
  output_dir: outputs/tennis_scene/evaluate/meiji_geometry/s42-001
parents:
- run-slcs-real-infer-pilot-v1
relations: []
tags:
- slcs
- real-rgb
repro:
  commit: 4348077d8f7a0b35ab58350232cf4ccb607990f1
  branch: codex/slcs-real-rgb
  command: .venv/bin/python -m scripts.analysis.evaluate_refinement --scene outputs/tennis_scene/generate/meiji_rgb_v1/s42-002/video_000/clip_000/scene.npz
    --court outputs/tennis_scene/precompute/meiji_dino_vitpose/s42-001/video_000/clip_000/court.npz
    --output outputs/tennis_scene/evaluate/meiji_geometry/s42-001
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
---

## 考察 / Findings

### 要約
開発clipでボール再投影中央値82.233→3.193px、人物関節60.284→16.405pxとなった。モデル自体の3D精度改善ではなく、観測を用いた教師ラベル補正の結果である。

### アーキテクチャ詳細
2視点以上で三角測量し、再投影・高さ・コート範囲・速度で棄却。短い内挿は別source code、長い未支持区間はモデルpriorを保存し教師重み0。人物はhip rootのみ補正し、yaw/canonical poseはPLCS由来。

### メトリクスの解釈
教師として支持された割合: ball98.02%、player0 86.63%、player1 99.21%。同じ観測でfitと評価を行うため再投影の改善は独立3D精度の証明にならない。

### アーキテクチャ⇄メトリクスの因果考察
観測拘束で位置の縮みが減った一方、近似校正と2D注釈の系統誤差・yaw誤差は残る。max residualは未支持priorを含むため、集計値だけで全フレームを正解扱いしない。

### 既存実験との比較
親runと同じDINO/ViTPose/Court/ball入力と同じモデル出力を使用し、補正処理のみ比較した。

### 次に有効な実験
録画単位で分割した幾何疑似ラベルをBLCSへ学習し、未使用録画でモデル生出力の改善も確認する。
