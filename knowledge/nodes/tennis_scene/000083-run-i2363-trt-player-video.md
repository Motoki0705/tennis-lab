---
id: run-i2363-trt-player-video
type: run
task: tennis_scene
sequence: 83
recorded_at: '2026-11-23'
title: TensorRT FP16で全69クリップのプレーヤー3D再構築を評価
issue: 2363
provider: codex
date: '2026-11-22'
status: failed
summary_impact: updated
config:
  dataset: meiji_4cam
  video: video_000
  precision: TensorRT FP16
  batch_sizes: {vitpose_crops: 8, hmr2: 16, dino: 1, clip_reid: 16, gvhmr: 4}
  gvhmr_length_profile: [2, 177, 1055]
  players_per_side: 2
  ball_enabled: false
  court: video_shared, clip-level player side evidence
  capture_sample_interval_seconds: 0.5
  capture_cuda_synchronize: false
metrics:
  selected_clips: 69
  generated_clips: 55
  failed_clips: 14
  blocked_clips: 0
  cli_exit_code: 1
  cli_wall_seconds: 43858.458501909
  peak_vram_gib: 1.7618141174316406
  peak_rss_gib: 6.161018371582031
  gvhmr_ort_bit_equal_observed_shapes: 64
  gvhmr_finite_gpu_boundary_shapes: 6
  gpu_idle_triangulation_thread_cpu_seconds: 4371.734192337
  smpl_valid_count: 133235
  smpl_eligible_count: 139088
repro:
  commit: fef1d626dd0c40ea8ce69705eda777fec4801021
  queue_job: 0-8204644184801783846_2026-11-22T13-56-55-198216153Z_507741_i2363-trt-all-v1.job
artifacts:
  run_dir: knowledge/runs/run-i2363-trt-player-video
  receipt: knowledge/runs/run-i2363-trt-player-video/full-video.json
  timings: knowledge/runs/run-i2363-trt-player-video/performance.json
  clips: knowledge/runs/run-i2363-trt-player-video/clips.csv
  videos: knowledge/runs/run-i2363-trt-player-video/videos.json
  output_dir: outputs/i2363/i2363-all-v1
parents:
- run-i2362-tensorrt-pipeline
- run-i2336-vision-batch-sweep
- run-i2355-gvhmr-pred-cam-clamp
- run-i2335-trt-batch-sweep
- run-court-video-drift-lines-20261117
relations: []
papers: []
tags: [tensorrt, fp16, video-batch, profiling]
---

## 考察

固定court・ボール無効の `video_000` 69件を、横断バッチを使い1 passで試行した。55件が生成され、人物対応で13件、コート向きで1件が停止した。生成したsceneの3D関節点の96.04%、SMPLの95.79%が有効。低coverageの6件では、人物IDが単一viewにしか割り当てられない区間が欠測につながった。 [全clipの結果と外れ値の診断](../../runs/run-i2363-trt-player-video/README.md#出力品質)

全件の再構築は12時間10分58秒、Peak VRAM 1.76 GiB、Peak RSS 6.16 GiB。GPU遊休に重なったCPU処理の首位はプレーヤーの三角測量で、推定4,371.73 CPU秒。代表clipではCUDA計時同期を追加するとwallが739.79秒から899.28秒になり、保存成果物のハッシュとexecution keyは一致した。 [時間・資源・改善候補](../../runs/run-i2363-trt-player-video/README.md#時間と資源)・[代表clipの同期計時](../../runs/run-i2363-trt-player-video/README.md#代表クリップの結果)

有限加算maskのGVHMRは、全件で使った64 shapeでONNX書換え前後がビット一致した。B=1・4、L=2・177・1055のGPU probeで出力の有限性とFP16層精度を確認した。停止の根拠と可視化は保存済み。TensorRT FP16の採否は判断待ち。 [実行したshape](../../runs/run-i2363-trt-player-video/gvhmr-all-video-shapes.json)・[代表と外れ値の動画](../../runs/run-i2363-trt-player-video/videos.json)
