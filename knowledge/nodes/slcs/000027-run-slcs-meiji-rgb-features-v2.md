---
task: slcs
sequence: 27
recorded_at: 2026-09-18
date_source: experiment_date
papers: []
id: run-slcs-meiji-rgb-features-v2
type: run
title: Meiji全57クリップのDINOv3 RGB特徴生成
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-18'
status: done
config:
  model: dinov3_vitb16
  data: meiji_3cam
metrics:
  clips_with_feature_marker: 57
  cameras_per_clip: 3
repro:
  commit: 4348077d8f7a0b35ab58350232cf4ccb607990f1
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 .venv/bin/python
    -m src.tennis_scene.scripts.build_slcs_dataset stage=features output_dir=tennis_scene/precompute/meiji_rgb_features/s42-gpu-002
    paths.external_asset_root=/home/kamimura/projects/tennis-lab/third_party
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-rgb-features-v2
parents: []
relations: []
tags: [slcs, meiji, rgb, features]
---

## 考察 / Findings

### 要約
全57クリップ・各3カメラのRGB patch特徴生成が完了。これは教師3D生成とは独立した前処理であり、SLCS学習完了を意味しない。

### アーキテクチャ詳細
固定DINOv3 ViT-B/16、画像256×448、patch16、特徴768次元、frame_stride10。特徴をfloat16で保存し、clip内frame index、入力動画とcheckpointのSHA-256を記録する。

### メトリクスの解釈
57はannotations/dino_v3/annotation.jsonが完成したクリップ数。推論精度指標ではない。特徴はdata/slcs/meiji_rgb_v4に保存し、後続データ版ではmanifestと特徴identityを照合して不変配列を再利用できる。

### アーキテクチャ⇄メトリクスの因果考察
RGBと2D検出を分離して保存するため、人物選択や教師モデルの修正でRGB特徴を再計算する必要はない。SLCSがこの特徴を有効活用できるかは別途ablation評価を要する。

### 既存実験との比較
先行GPU試行は既存特徴のidentity不一致で停止したが、同じ既存3クリップの再検証では不一致を再現できなかった。照合を無効化せず、差分項目を表示する診断を追加した本試行で全件完了。原因未確定のため旧失敗を解決済みとは扱わない。

### 次に有効な実験
人物観測の品質修正と擬似教師生成を完了させ、固定splitでSLCSを60epoch学習し、RGBあり・なし・検出欠損条件を比較する。
