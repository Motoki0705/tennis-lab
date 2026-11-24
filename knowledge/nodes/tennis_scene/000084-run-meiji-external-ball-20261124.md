---
id: run-meiji-external-ball-20261124
type: run
task: tennis_scene
sequence: 84
recorded_at: '2026-11-24'
title: Meiji 4camの外部ボール位置を既存の69シーンへ反映
issue: 2458
provider: codex
date: '2026-11-24'
status: done
summary_impact: none
config:
  dataset: meiji_4cam
  video: video_000
  input: external_clip_positions
  ball_fit: equal_weight_all_available_views_dlt_pixel_least_squares
  source_code_sha256: e3da6650fc904094b102f76ffee22ab647bee41edb5959c1c810767226d650cb
metrics:
  published_clips: 69
  saved_alignment_clips: 55
  shared_court_side_clips: 14
  frames: 46228
  valid_3d_frames: 45958
  valid_3d_rate: 0.9941593839
  insufficient_views: 270
  invalid_multiview_geometry: 0
  all_clips_wall_seconds: 1132.18
  all_clips_peak_rss_kib: 2583600
repro:
  commit: 5f447e4b851ce015feee05ea6121c5e479d39387
  command: bash knowledge/runs/run-meiji-external-ball-20261124/reproduce.sh "$dataset" outputs/external-ball-reproduction
artifacts:
  run_dir: knowledge/runs/run-meiji-external-ball-20261124
  clips: knowledge/runs/run-meiji-external-ball-20261124/clips.csv
  summary: knowledge/runs/run-meiji-external-ball-20261124/summary.json
parents:
- run-i2363-trt-player-video
- run-court-side-video-shared-20261118
relations: []
papers: []
tags:
- ball3d
- fixed-geometry
- cpu
---

## 考察

外部JSONの全4ステータスを画面内で採用し、固定カメラで全69 clipを三角測量した。
3Dが得られたのは45,958/46,228 frame（99.42%）で、両参照から新版を読み戻せた。
既存の55 clipの選手・身体配列はバイト単位で一致し、残る14 clipは選手軸を空にして、
保存済み校正とボール由来の共有向きでボール3Dを得た。
[照合結果・処理時間・再投影](../../runs/run-meiji-external-ball-20261124/README.md)

入力点の一致度と3Dの生成率を評価した。独立した3D正解に対する精度は未計測。
このrunは保存済み評価データセットへの適用を扱う。標準pipelineの設定やTensorRT FP16の
採否は、既存の比較と全件評価に基づいて判断する。
