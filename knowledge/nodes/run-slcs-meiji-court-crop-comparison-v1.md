---
id: run-slcs-meiji-court-crop-comparison-v1
type: run
title: 同Court重みの2pass crop候補がbaseline局所ずれを改善
provider: codex
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
date: '2026-09-19'
status: done
config:
  checkpoint: b863df1f01f00d2ff00a21d56a461879e3c5af193a9f32cec47a88d2842f8383
  clips:
  - video_000/clip_000
  - video_001/clip_003
  - video_002/clip_017
  cameras:
  - cam0
  - cam1
  variants:
  - ball_margin_025
  - ball_margin_050
  - initial_court_extent_union_20px
  fit_thresholds_unchanged: true
metrics:
  baseline_exact_views: 6
  compared_variants: 18
  fit_returned: 17
  rejected_variants: 1
  reviewed_frame_views: 18
  stable_input_files: 950
  baseline_local_abs_max_px: 26.688476545266894
  candidate_local_abs_max_px: 1.5072871008885613
  candidate_local_vertical_mean_px: -0.6794394284949148
  manual_cam0_mean_before_px: 15.097524177825226
  manual_cam0_mean_after_px: 10.180959920175422
  manual_cam1_mean_before_px: 8.717385127043071
  manual_cam1_mean_after_px: 6.941919553082247
repro:
  commit: 41020d13080c5c1765c218924bbbb4d2aee2ef84
  branch: codex/slcs-real-rgb
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash knowledge/runs/run-slcs-meiji-court-crop-probe-v1/repro.sh
    --output /home/kamimura/projects/tennis-lab/outputs/tennis_scene/evaluate/meiji_court_crop_probe/s42-001
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-court-crop-comparison-v1
  output_dir: outputs/tennis_scene/evaluate/meiji_court_crop_probe/s42-001
  diagnostics: knowledge/runs/run-slcs-meiji-court-crop-comparison-v1/summary.json
  visual_review: knowledge/runs/run-slcs-meiji-court-crop-comparison-v1/visual_review.json
parents:
- run-slcs-meiji-baseline-line-audit-v2
relations: []
tags:
- slcs
- meiji
- court
- crop
- calibration
---

## 考察 / Findings

### 要約
同じCourt checkpointとfit閾値で3収録×cam0/1×3cropを比較した。元cropのraw・score・Hは6viewすべて保存値と完全一致。初期HのCourt範囲を含め20px拡大するC候補は6viewすべてfitを返し、第3収録cam0の既存局所白線参照との差を最大26.69pxから1.51pxへ減らした。productionでの同値確認前で、完成教師への採用ではない。

### アーキテクチャ詳細
Aはoutsource ballの1–99percentile範囲へmargin0.25、Bは0.5、CはAとAの保存Hが写すCourt14点bboxのunionへ20pxを足して画像内へclampした。各view9frame、score0.15、min_points10、RANSAC20px、median fit上限15pxを固定。cam2は元のfull画像推論を維持する。既存第1収録の手動Court注釈は評価専用、crop作成には使わない。

### メトリクスの解釈
17/18variantがfitを返し、Bの第2収録cam0だけinlier不足で拒否された。Cの第3収録cam0のnear-baseline inlierは0から3、固定9画素サンプルの平均上下差は24.677pxから-0.679px（範囲-1.507〜0.056）になった。第1収録の既存注釈に対する平均誤差はcam0 15.098→10.181px、cam1 8.717→6.942px、p95も双方改善。一方cam1 medianは4.657→6.485pxと微増し、全面的な指標改善とは主張しない。全6sheet18frame-view54overlayを親が確認し、画像SHAと選択理由をvisual_review.jsonに保存した。

### アーキテクチャ⇄メトリクスの因果考察
初期ball cropが近側baselineを切り落とし、Hが支持点のない領域へ外挿していたという仮説を支持する。Cで近側支持が増え、事前に固定した局所白帯参照との差が減った。ただしレンズ歪み・検出誤差の寄与を分離しておらず、局所画素診断をコート全体・実測3D精度と解釈しない。残差p95はRANSAC除外点を含むため採否のGT基準ではない。

### 既存実験との比較
親runは保存HとRGB白帯の差を局所的に測定した。本runは同じ重み・閾値でcropだけ比較した。単純なmargin倍増Bは一部fit失敗と手動点medianの悪化があるため全view共通規則として採用しない。Cを次のproduction候補とし、元Aと実行入力の前後一致を確認した。学習は行っていない。

### 次に有効な実験
productionで初期ball cropからHを作り、そのCourt extentを含む2passを明示設定として実装する。新しいMeiji v9/観測rootへ分離し、3校正clipで本run Cのraw・score・Hと完全一致するか確認する。cam2は旧結果との一致を確認する。Courtが変わる人物選択は再計算し、古いHのreceiptを書き換えて再利用しない。
