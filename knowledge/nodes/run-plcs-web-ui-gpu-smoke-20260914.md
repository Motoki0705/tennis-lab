---
id: run-plcs-web-ui-gpu-smoke-20260914
type: run
title: PLCS Web UIの共有キュー経由CUDA推論確認
provider: codex
session: 01a09e9c-5c30-7bf3-8e7b-9d2f9d6d3bd8
date: '2026-09-14'
status: done
config:
  model: plcs_multiview_axial_reference
  data: single_object_camera_view_v2/scene_000006
metrics:
  http_status: 200
  window_frames: 16
  position_error_m: 0.06978401109151
repro:
  commit: 782fc137d85c054d0f5ef775fe5edbf8cf52f00f
  branch: codex/dataset-scene-review
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: /home/kamimura/projects/tennis-lab/.venv/bin/python -m src.tasks.base.visualization.inference_queue
    /home/kamimura/projects/tennis-lab/.training_queue/ui_requests/plcs-73lm5pmb/request.json
artifacts:
  run_dir: knowledge/runs/run-plcs-web-ui-gpu-smoke-20260914
parents: []
relations: []
tags: [plcs, web-ui, gpu, smoke]
---

## 考察 / Findings

### 要約
`POST /api/predict` から共有training queueのall予約を通し、CUDA推論のバイナリ応答を取得した。ワーカーはdoneとなり終了した。

### アーキテクチャ詳細
既存の `plcs-axial-reference-corners-v3-4-t128-seed42-epoch47.ckpt` を使用。カメラ0・1・2、参照camera_0、開始0・長さ16、canonical poseはGT。UI専用loaderは学習用設定を補完せず、保存済みモデル設定と重みを厳密に復元する。

### メトリクスの解釈
単一シーンの短い窓による機能確認であり、test split全体の性能評価ではない。GTとpred各16フレームを返し、位置誤差meanは0.069784 m。学習していないため収束曲線はない。

### アーキテクチャ⇄メトリクスの因果考察
数値からモデル改善の因果は判断しない。CPUの同一要求もHTTP200であり、CUDA要求が別プロセスで実行・返却される経路を確認した。

### 既存実験との比較
学習実験との比較は実施していない。repro bundleのpatchは未追跡の新規ソースを含まないため、この変更一式のworktree版も再現に必要。要求と結果の原本は共有キューのui_requestsに保持される。

### 次に有効な実験
学習済みmulti-objectチェックポイントが用意された時点で、同じWeb経路の実重み検証を追加する。
