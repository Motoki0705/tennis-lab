---
id: run-i2440-player-selection
type: run
task: person_tracking
sequence: 14
recorded_at: '2026-11-24'
title: アレーと短い追跡断片に対する選手選別の比較
issue: 2440
provider: codex
date: '2026-11-24'
status: done
summary_impact: updated
config:
  doubles: meiji_4cam/video_000
  singles: meiji_3cam dev
  core_half_widths_m: [4.115, 5.485]
  min_presence_fractions: [0.25, 0.125]
metrics:
  loss_analysis_camera_frames: 20804
  loss_analysis_rejected_by_selection: 10853
  doubles_clips: 69
  baseline_player_frames_3d: 133887
  wide_player_frames_3d: 147105
  short_player_frames_3d: 136212
  wide_short_player_frames_3d: 141398
  stopped_clips: {baseline: 14, wide: 11, short: 14, wide_short: 13}
  singles_false_positives: {baseline: 0, wide: 0, short: 212, wide_short: 212}
repro:
  commit: 4a75d8ecd1fd0d1f6fea070061f19f4f61dcc2f5
artifacts:
  run_dir: knowledge/runs/run-i2440-player-selection
  loss: knowledge/runs/run-i2440-player-selection/loss.json
  comparison: knowledge/runs/run-i2440-player-selection/comparison.json
  singles: knowledge/runs/run-i2440-player-selection/singles.json
parents: [run-i2363-trt-player-video, run-i964-default-merge-r11-20260930]
relations: []
papers: []
tags: [court-dwell, doubles, cpu-replay]
---

## 考察

55クリップで選別落ち10,853件を再現した。ダブルス幅・滞在25%では、三角測量で有効な選手フレームが13,218件増え、アレーに立つ選手が外れていた020・058を含む3クリップが復帰した。シングルス開発用4クリップの選別結果は同じだった。滞在率12.5%では新規停止が生じ、開発用4クリップで非選手boxが212件増えた。[集計、設定、再現手順](../../runs/run-i2440-player-selection/README.md)

ユーザーの判断でダブルス幅・25%を全形式の既定とした。4カメラの比較は保存成果物の出力数に基づき、誤選別率は未測定。既存のシーンへの反映は #2460、短い断片への対処は #2445 の3D軌跡で検証する。
