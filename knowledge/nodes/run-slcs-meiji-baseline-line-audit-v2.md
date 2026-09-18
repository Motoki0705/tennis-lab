---
id: run-slcs-meiji-baseline-line-audit-v2
type: run
title: Meiji baseline白線と保存Courtの局所画素差
provider: codex
date: '2026-09-19'
status: done
config:
  device: cpu
  data: video_002/clip_017 cam0
  frames:
  - 0
  - 453
  - 907
  roi_xyxy:
  - 550
  - 495
  - 1040
  - 555
  automatic_adoption: false
metrics:
  input_files_unchanged: 7
  saved_images: 12
  reviewed_pair_sheets: 3
  local_pixel_samples: 9
  local_vertical_difference_min_px: 22.572385917031625
  local_vertical_difference_max_px: 26.688476545266894
  local_vertical_difference_mean_px: 24.676876527476892
artifacts:
  run_dir: knowledge/runs/run-slcs-meiji-baseline-line-audit-v2
  output_dir: outputs/tennis_scene/analyze/meiji_baseline_line_audit/s42-002
  diagnostics: knowledge/runs/run-slcs-meiji-baseline-line-audit-v2/results.json
  visual_review: knowledge/runs/run-slcs-meiji-baseline-line-audit-v2/visual_review.json
parents:
- run-slcs-meiji-baseline-line-audit-v1
relations: []
tags:
- slcs
- meiji
- court
- pixel-diagnosis
session: 01a0b314-89b5-7902-b0a1-3141a9323ea9
repro:
  commit: 10a8a73edf38348eb6322b901d05329b6b2c7f5d
  branch: codex/slcs-real-rgb
  command: bash knowledge/runs/run-slcs-meiji-baseline-line-audit-v2/repro.sh /new/unique/output
---

## 考察 / Findings

### 要約
選定した3時刻の右側baseline白帯と保存Court線の間に、同じ画像xで22.57–26.69pxの上下差があった。RGBの局所画素診断であり、人手GT・コート全体の誤差・実測3D精度ではない。

### アーキテクチャ詳細
v1のLSD配列shapeだけを修正し、固定ROI・候補閾値を維持して1回実行した。全LSD候補71/63/46、白帯pair候補4/5/7を保存。Hは候補抽出に使わず、各pairの中央線と比較するだけである。親が3枚のpair sheetと元RGBを確認し、白帯を挟む右側のpairをframe0/453/907で0/0/1と明示選択した。親はH重ね合わせも見ておりblind reviewではない。暗い影のpairは白帯として採用しない。

### メトリクスの解釈
選択pairの共通x区間939.435–991.816pxの端・中央で3時刻を比較した9点の上下差は22.572–26.688px、平均24.677px。正は推定baselineが白帯より画像下側にあることを示す。帯幅・線分抽出の不確かさがあり、校正された誤差区間ではない。7入力の前後dual SHA、動画/clip identity、12画像hashを照合した。実行時のpending結果を変えず、親選択と再計算をvisual_review.json/review_reference.pyへ分離した。学習曲線は無い。

### アーキテクチャ⇄メトリクスの因果考察
近側baselineにinlierが無いという先行所見に加え、実白線との局所的なずれを数値化した。点不足による外挿誤差は仮説であり、レンズ歪みや検出誤差との寄与を分離していない。保存p95はRANSAC除外点を含む検出残差であり、本画素差とは別物である。

### 既存実験との比較
v1は実装例外で画像を得られなかった。本runは同じ候補条件で完了した。Court visual QCの定性的所見を補うが、他camera・全コート・全frameへの一般化はしない。元観測や校正は変更していない。

### 次に有効な実験
同じCourt checkpointとfit閾値のまま、ball crop拡大および保存Hが示すCourt範囲を含むcropを比較する。元cropの再推論一致を確認し、近側inlier、画像白線、既存第1収録の注釈との対応を併せて判断する。
