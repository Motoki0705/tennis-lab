---
id: run-i2633-court-viewpoint-comparison
type: run
task: court_detection
sequence: 83
recorded_at: '2026-11-27'
title: 視点分布を変えた4学習runの実画像集計
issue: 2633
provider: codex
date: '2026-11-27'
status: done
summary_impact: none
config:
  registry: real-video-v1
  conditions: [narrow, broad]
  training_seeds: [42, 43]
  updates: 6000
  bootstrap_unit: video
  bootstrap_replicates: 2000
  bootstrap_seed: 42
metrics:
  test/kp_mean_px512_broad_minus_narrow: 7.568396
  test/h_formed_pct_broad_minus_narrow: -10.224154
  test/challenging_kp_mean_px512_broad_minus_narrow: 8.678363
  test/challenging_h_formed_pct_broad_minus_narrow: -10.933941
repro:
  commit: 7dac77db461e5054579d9a6dfea7ecd48454d3aa
  command: >-
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python
    knowledge/runs/run-i2633-court-viewpoint-comparison/report.py
    --campaign outputs/court_detection/campaign/viewpoint-diversity-v1
    --output outputs/court_detection/campaign/viewpoint-diversity-v1/report-lf
artifacts:
  run_dir: knowledge/runs/run-i2633-court-viewpoint-comparison
  comparison: knowledge/runs/run-i2633-court-viewpoint-comparison/comparison.json
  metrics_csv: knowledge/runs/run-i2633-court-viewpoint-comparison/metrics.csv
  figure: knowledge/runs/run-i2633-court-viewpoint-comparison/test-effects.png
  queue_evidence: knowledge/runs/run-i2633-court-viewpoint-comparison/queue-evidence.json
  preparation: knowledge/runs/run-i2633-court-viewpoint-comparison/preparation.json
  registry: knowledge/runs/run-i2633-court-viewpoint-comparison/registry.json
parents:
- run-i2633-viewpoint-narrow-s42
- run-i2633-viewpoint-broad-s42
- run-i2633-viewpoint-broad-s43
- run-i2633-viewpoint-narrow-s43
relations: []
papers: []
tags: [viewpoint-diversity, real-video-v1]
---

各runの最終checkpointを読み直し、6,000更新、seed、学習画像IDの順序、共通合成val、実画像registry、設定とcheckpointのSHA-256が保存記録に一致した。設定は学習subset、seed、出力先の3点を除いて一致した。実画像8,840枚×4 runの採点行に欠損・重複がなく、固定の層別と照合できた。 [再現手順](../../runs/run-i2633-court-viewpoint-comparison/README.md) · [比較JSON](../../runs/run-i2633-court-viewpoint-comparison/comparison.json)

val/test、各層、2条件×2 seedの指標56行を出力した。空の層では画像数と動画数が0で、指標を空欄にした。差は広分布−狭分布で、seedを対応させた。動画単位の2,000回の再抽出では各動画の全採点行と両seedの対応を保ち、2 seed平均の差の95%区間を求めた。 [指標CSV](../../runs/run-i2633-court-viewpoint-comparison/metrics.csv) · [testの図](../../runs/run-i2633-court-viewpoint-comparison/test-effects.png)

独立した計算で指標56行と効果14行を照合した。各runの学習曲線には、合成64枚と実画像val 4,468枚を併せたfit時の監視値を収録した。比較表の実画像指標は、最終checkpointの実画像だけの採点行を集計した。

4本のqueue終端 `done`、実行時の完全HEAD、stdoutと完了receiptの一致を照合した。学習・採点の実測合計は20,951.9秒（約5時間49分）。queueの再現記録生成から終端までの合計は約5時間50分だった。 [queueの照合結果](../../runs/run-i2633-court-viewpoint-comparison/queue-evidence.json)

summary_impact は `none`。実験の判定は[比較group](000084-group-i2633-court-viewpoint-diversity.md)からsummaryへ反映する。
