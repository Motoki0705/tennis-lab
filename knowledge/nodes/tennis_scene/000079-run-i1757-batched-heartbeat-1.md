---
id: run-i1757-batched-heartbeat-1
type: run
task: tennis_scene
sequence: 79
recorded_at: '2026-11-20'
title: Meiji3クリップ横断batchの時間内訳と保存結果の差
issue: 1757
provider: codex
date: '2026-11-19'
status: done
summary_impact: updated
parents: [run-i1757-video-batch-meiji, run-i1757-model-reuse-heartbeat-1]
repro:
  commit: fb6255a2853a6992e3d96287d7b989b681c2e336
  branch: codex/video-batch-benchmark
metrics:
  first_wall_including_startup_seconds: 1984.3342191349948
  warm_wall_seconds: [1859.5331290100003, 1890.4414682780043]
  warm_sample_stdev_seconds: 21.855496291620117
  allocated_peak_bytes: 3985212416
  reserved_peak_bytes: 4452253696
  rss_peak_bytes: 9582796800
  batch_working_set_peak_bytes: 2344016548
  scenes_reloaded: 9
  recorder_errors: 0
  quality_acceptance_vs_serial: false
  quality_acceptance_vs_reuse: false
artifacts:
  run_dir: knowledge/runs/run-i1757-batched-heartbeat-1
  output_dir: outputs/tennis_scene/evaluate/video-batch-benchmark-20261119/batched-heartbeat-1
  queue_job: 0-8204904851837375583_2026-11-19T13-32-28-162624416Z_2750129_i1757-batched-heartbeat-1
  metrics: knowledge/runs/run-i1757-batched-heartbeat-1/batched-heartbeat-1.json
---

3クリップの各3回が完走し、全9件を再読込した。warm中央値1874.99秒は別コミットのモデル再利用より4.01%長い。横断dispatchの充填率は各モデル99.90%以上で、DINOのforwardは約38秒短かった。recorderで増分が大きいのはdispatch、帰属先未計測の区間、モデル読込だった。

court_sideのplayer証拠は全件ball参照と一致した。旧逐次とモデル再利用との固定許容差比較は全組不合格で、ボール3Dと人物のroot・関節に差が出た。時間だけで横断batchを採用できる根拠は得られなかった。 [時間内訳・forward shape・差分](../../runs/run-i1757-batched-heartbeat-1/result.md)
