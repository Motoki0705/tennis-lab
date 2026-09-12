---
id: run-i878-b00-rendered-line-projection
type: run
title: B00 NHTレンダリングRGBライン投影検証 (#878)
issue: 878
provider: codex
session: 01a094f3-f431-7871-b55d-5c34ca6f48dc
date: '2026-09-12'
status: done
config:
  model: court_line_dinov3_vitb16_lora_epoch19
  loss: inference_only
  data: B00_nht_rendered_rgb_48_observed_cameras
metrics:
  selected_camera_count: 48.0
  fit_camera_count: 32.0
  holdout_camera_count: 16.0
  rendered_inputs_different_from_captured_count: 48.0
  projected_line_point_count_total: 94266.0
  projection_nonzero_cell_count: 37173.0
  projection_evidence_sum: 12538.386719
  rendered_vs_captured_median_absolute_rgb_difference: 4.271098
repro:
  commit: c4bb0e63b43ae4492f190bfac485a4dae087af5d
  branch: feat/issue-878-rendered-line-input
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: TENNIS_RUN_B00_RENDERED_ALIGNMENT_GPU_TEST=1 TENNIS_B00_FIXTURE_REPOSITORY=/home/kamimura/projects/tennis-lab
    PYTHONPATH=. .venv/bin/pytest -q -n 0 -s tests/integration/synthetic_data_generation/test_alignment_rendered_input_gpu.py
artifacts:
  run_dir: knowledge/runs/run-i878-b00-rendered-line-projection
  legacy_projection: assets/synthetic_data_generation/issue-878/legacy-captured-weighted-projection.png
  rendered_projection: assets/synthetic_data_generation/issue-878/nht-rendered-weighted-projection.png
  comparison: assets/synthetic_data_generation/issue-878/weighted-projection-comparison.png
parents: []
relations: []
tags: [court-alignment, nht-render, line-detection, b00]
---

## 考察 / Findings

### 要約

B00 の既存 3DGS/NHT と固定48視点を用い、NHTレンダリングRGBからライン検出、同じ公開カメラによる地面逆投影、投影ヒートマップ生成まで完走した。48入力すべてが撮影RGBと非同一で、32 fit／16 holdoutの全視点が観測可能だった。

### アーキテクチャ詳細

ライン検出器は既存の `court_line_dinov3_vitb16_lora_epoch19` を変更せず、入力だけを `camera.image_path` の撮影RGBから `NHTRenderClient` が固定 observed camera で生成したRGBへ置き換えた。レンダリングに用いた姿勢・内部パラメータ・959×539解像度の同じ `SceneCamera` を逆投影にも使用し、旧B00と同じカメラ選択およびfit/holdout分割で評価した。

### メトリクスの解釈

レンダリングRGBと撮影RGBの平均絶対画素差は視点中央値で4.271/255であり、48視点すべてが非同一だった。検出された103,841画素のうち94,266点を有効に地面へ投影でき、999×908の共通グリッドに37,173個の非ゼロセルを得た。fit 32視点、holdout 16視点から除外されたカメラはない。推論のみのrunなので収束曲線は対象外である。

### アーキテクチャ⇄メトリクスの因果考察

入力と逆投影が同一の推定カメラ条件に束縛されたため、撮影時の真の姿勢とSfM推定姿勢の差を画像・レイ間の不整合として持ち込まない経路になった。94,266点の投影成立と全48視点の観測可能性は、この入力変更後もライン証拠量が不足していないことを示す。一方、投影ヒートマップの見た目だけから最終コートfit精度が改善したとは断定できない。

### 既存実験との比較

親runは未登録のため `parents` は空である。既存B00の撮影RGB由来ヒートマップと同一グリッドで比較すると、主要な複数コート境界は維持されつつ、中央および下側コート内部の証拠分布が変化した。これは48視点すべてで確認した入力RGB差と整合する。

### 次に有効な実験

B00〜B03で新入力による完全なコートfit/holdout受理まで再生成し、最終コート変換と下流ラベルの差を定量化する。特に撮影RGB経路との差を、holdout対応誤差と3DGS描画ラインへの距離で評価するのが有効である。
