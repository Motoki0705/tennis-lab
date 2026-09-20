---
id: run-court-prosac-paper-20260920
type: run
task: court_detection
sequence: 29
recorded_at: '2026-09-20'
title: 指定写真4枚のKP信頼度順PROSAC再推定
provider: codex
date: '2026-09-20'
status: done
config:
  model: existing epoch17 saved raw_kp and kp_scores; no neural forward
  data: all four supplied tennis_court photographs
  postprocess: confidence-ranked PROSAC + MSAC + inlier-only LS/LM
  threshold_diagonal_ratio: 0.005
  opencv: 5.0.0
metrics:
  images: 4
  homographies_found: 4
  inliers:
    local01: 5
    local02: 7
    local03: 5
    local04: 7
  inlier_rms_px:
    local01: 0.3283226979442124
    local02: 1.5849813951287364
    local03: 0.7186675713221305
    local04: 0.7849532630997869
artifacts:
  output_dir: paper/court_robustness/evidence/homography
  predictions: paper/court_robustness/evidence/homography/results.json
parents:
- run-court-supplied-photos-paper-20260918
relations: []
tags:
- court-detection
- paper
- homography
- prosac
- cpu
session: 01a0b44d-2642-7040-8df3-0cad69ce7cd8
repro:
  command: .venv/bin/python paper/court_robustness/homography_evidence.py
  branch: codex/court-robustness-report
---

## 考察 / Findings

### 要約
全4写真で信頼度順PROSACによるHを生成できたが、実コートとのずれは残る。Top-4固定の退化は実例Bで確認された。

### アーキテクチャ詳細
ニューラルネットは再実行せず、親runの保存座標・スコアを再利用。共通実装、設定、点ごとの診断はartifactsの証拠を正本とする。

### メトリクスの解釈
inliersは最終Hの閾値内点数、RMSはその推論点への自己整合でありGT精度ではない。学習runではないため収束曲線はない。

### アーキテクチャ⇄メトリクスの因果考察
写真Bの上位4点は参照上で共線のため固定4点では解けない。順位付き候補拡張で推定は成立した。高信頼度でも意味対応が誤っている可能性は残り、幾何的一貫性だけでは正解を保証しない。

### 既存実験との比較
親runの固定12組・重みなし候補評価から変更した。元の推論NPZ・checkpoint記録・LINE出力とTCD公式処理は保持し、Hと図だけを更新。全点残差と選択インライア残差は母集団が異なるため改善率は主張しない。

### 次に有効な実験
独立な人手GT付き画像で旧方式とPROSACの再投影精度・失敗率を同条件で比較し、信頼度順位とKPの正しさの関係も検証する。
