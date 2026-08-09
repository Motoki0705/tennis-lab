---
id: proposal-court-pnl-line-refinement
type: proposal
title: court homographyを点・線共同残差でrefineする
curator: chatgpt-schedule
date: 2026-08-07
status: candidate
task: court_detection
repo_paths:
  - src/tasks/court_detection/geometry/postprocess.py
  - src/tasks/court_detection/evaluation/homography_quality.py
  - src/tasks/court_detection/evaluation/image_evidence.py
hypothesis:
  statement: 既存の点RANSAC homographyへcourt line residualを追加すると、部分遮蔽条件のkeypoint誤差とline supportが改善する
  expected_effect: median keypoint errorを10%以上低下させ、line_edge_supportを0.05以上増加させる
  failure_condition: NaNまたは発散5%以上、median keypoint error 5%以上悪化、valid率2 point超低下、または追加50 ms/frame超
evaluation:
  metrics: [median_keypoint_error_px, all_residual_rms_normalized, line_edge_support, geometry_valid_rate, wall_time_ms]
  baseline_nodes: []
  seeds: 3
  acceptance: 3 seeds平均でmedian keypoint errorを10%以上低下し、line_edge_supportを0.05以上増加し、valid率低下2 point以内、追加50 ms/frame以内にする
evidence_runs: []
parents: []
relations:
  - to: paper-doi-10-1016-j-cviu-2026-104712
    rel: derived-from
tags: [literature, court-detection, homography]
---

## 背景

現行実装は14 keypointでhomographyをrefitし、line evidenceを事後品質指標に留める。formal graph上で同一条件のbaseline runを特定できないため`candidate`とする。

## 現行実装との差分

既存homographyを初期値に固定し、line maskから線分を抽出して点再投影誤差とcourt line距離を共同最小化する。detector再学習や完全camera calibrationは同時に行わない。

## 最小検証

full court、partial court、選手遮蔽を含む手動14点付き200 frameを固定し、point-only baselineとpoint+line treatmentへ同一予測を入力する。

## 比較対象

baselineは現行のpoint RANSAC/refit経路である。正式baseline runを登録後に`baseline_nodes`と`parents`を設定する。

## 合格条件と停止条件

frontmatterのacceptanceを満たせば動画評価へ進む。線支持だけ改善して点誤差が悪化する場合、NaN、発散、または誤線対応が頻発する場合は停止する。

## リスク

court以外の直線を誤対応し得る。公式codeはGPL-2.0のため、直接移植前にlicense影響を確認する。
