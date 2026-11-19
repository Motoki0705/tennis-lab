---
id: run-i1757-model-reuse-heartbeat-1
type: run
task: tennis_scene
sequence: 77
recorded_at: '2026-11-19'
title: Meiji3クリップモデル再利用の時間短縮と旧逐次とのボール出力差
issue: 1757
provider: codex
date: '2026-11-19'
status: done
summary_impact: updated
parents: [run-i1757-video-batch-meiji]
repro:
  commit: 795d63df1682d0d86f769d584dfbfa263419ad0d
  branch: codex/video-batch-benchmark
metrics:
  first_wall_including_startup_seconds: 1876.4076738799922
  warm_wall_seconds: [1797.1097804919991, 1808.406597519992]
  warm_sample_stdev_seconds: 7.9880559263173705
  allocated_peak_bytes: 4184557568
  reserved_peak_bytes: 4460642304
  rss_peak_bytes: 9441755136
  scenes_reloaded: 9
  recorder_errors: 0
  quality_acceptance_vs_serial: false
artifacts:
  run_dir: knowledge/runs/run-i1757-video-batch-meiji
  output_dir: outputs/tennis_scene/evaluate/video-batch-benchmark-20261119/reuse-heartbeat-1
  queue_job: 0-8204917572928512634_2026-11-19T10-00-27-071487365Z_1772136_i1757-reuse-heartbeat-1
  metrics: knowledge/runs/run-i1757-video-batch-meiji/reuse-heartbeat-1.json
---

3クリップ×3回の一括再構成が完走し、すべてのsceneを再読込した。warm中央値1802.76秒は、別commitの逐次基準2018.28秒より10.68%短かった。主要モデルは各回1回ロード。court_sideは全件ball参照と一致した。

固定許容誤差による逐次との比較は全組で不合格。最初の数値差はBallの検出evidenceで、sceneのボール座標、visibility、棄却コード、027と037の3D有効maskにも差が出た。summaryには時間短縮と品質の未達を記録する。 [時間内訳と差分](../../runs/run-i1757-video-batch-meiji/model-reuse-comparison.md)
