---
task: slcs
sequence: 43
recorded_at: 2026-09-18
date_source: experiment_date
papers: []
id: run-slcs-real-court-probe-v3
type: run
title: 'Meiji court: 9フレーム集約と対象コート切り出しの比較'
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-18'
status: done
config:
  model: outputs/court_detection checkpoints
  loss: inference only
  data: Meiji video_000/clip_000 (development only)
metrics:
  development/cam0_full_raw_median_px: 239.6070084737347
  development/cam0_selected_fit_median_px: 10.137266229537637
  development/cam1_selected_fit_median_px: 4.65720390570091
  development/cam2_selected_fit_median_px: 3.3028425085614
repro:
  commit: 4348077d8f7a0b35ab58350232cf4ccb607990f1
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python
    -m scripts.analysis.meiji_court_probe --config scripts/analysis/meiji_court_probe.yaml
artifacts:
  run_dir: knowledge/runs/run-slcs-real-court-probe-v3
  output_dir: outputs/tennis_scene/evaluate/court_meiji/probe-v3-context
parents:
- run-slcs-real-court-probe-v2
relations: []
tags:
- slcs
- real-rgb
- court
- development
---

## 考察 / Findings

### 要約
9フレーム集約と対象コート切り出しの比較。学習済みCourt checkpointだけから検出し、既存手動CourtKP14を開発評価専用に使用した。

### アーキテクチャ詳細
DINOv3を用いたCourtモデルを厳密に復元。全画面と、outsourced annotationのobserved ball座標の1–99%範囲に余白を加えた切り出しを比較した。切り出しに手動Court点は用いない。チェックポイント、入力条件、各条件の全指標はrun directoryのYAML/JSONを参照。

### メトリクスの解釈
multiscale-depth3 checkpointの選択条件（cam0/1: margin=0.25、cam2: full）の手動Court点に対するホモグラフィ適用後中央値: cam0 10.137px, cam1 4.657px, cam2 3.303px。これは1クリップの2D開発指標であり、3D正解精度や独立test性能ではない。手動点自体にもホモグラフィ残差の中央値2–5pxがある。訓練を伴わないため収束曲線はない。

### アーキテクチャ⇄メトリクスの因果考察
cam0では複数コートが映る全画面入力が大きな誤りを生む。ballで対象コートを特定する切り出しは改善した。一方、狭すぎる切り出しは基線点を落とす。v2以降はcheckpointのpose検証と同じ長辺リサイズ・右下パッチpaddingを使用する。pose head投影は依然大きな誤差を示したため、3D教師に使用しない。

### 既存実験との比較
親run v2の結果と、同じ手動開発ラベル上で比較。条件ごとの結果をmetrics.jsonへ残した。

### 次に有効な実験
Court点の支持数を10以上に制限し、全クリップで品質ゲートを適用する。人物観測を抽出し、PLCS/BLCSの再投影と多視点整合性を評価する。
