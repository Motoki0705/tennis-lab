---
id: run-i1757-dino-gpu-health
type: run
task: tennis_scene
sequence: 73
recorded_at: '2026-11-19'
title: DINO fp16の停止とGPU定期照会の切り分け
issue: 1757
provider: codex
status: planned
summary_impact: none
config:
  clip: video_000/clip_027
  camera: cam0
  frames: 235
  precision: fp16
  batch_size: 1
  heartbeat_timeout_seconds: 60
  condition_limit_seconds: 480
  gpu_query_interval_seconds_in_b: 1
  gpu_query_timeout_seconds_in_b: 10
metrics: {}
artifacts:
  run_dir: knowledge/runs/run-i1757-dino-gpu-health
parents:
- run-i1757-video-batch-meiji
relations: []
papers: []
tags:
- inference
- diagnosis
---

## 考察 / Findings

逐次probeは2回ともDINO fp16の初回推論後に停止した。DINO単独で同じ入力の全235フレームを実行し、GPU定期照会を行わない条件Aと、別プロセスから1秒ごとに同じ照会を行う条件Bを比較する。条件Bは条件Aの完走を前提とする。[再現手順](../../runs/run-i1757-dino-gpu-health/README.md)
