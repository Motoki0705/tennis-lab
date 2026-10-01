---
id: run-ball-mix-v2-player-pose-gpu-smoke-20261002
type: run
task: player_pose_annotation
sequence: 1
recorded_at: '2026-10-02'
title: コートを使わないDINO・ViTPose・CLIPのGPU動作確認
provider: codex
session: 01a0f53f-bff2-7202-b7ee-5fa777ae6627
date: '2026-10-02'
status: done
config:
  store: ball-mix-v2
  court_policy: disabled
  tracking: strongsort_pp_pose
  review_model: gpt-6.1-sol
  presence_threshold_percent: 40
metrics:
  processed_frames: 1
  detected_people: 17
repro:
  commit: bd48b063d73778710f849e56178e07806343ec1e
  branch: codex/ball-mix-v2-player-pose
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: PYTHONPATH=/home/kamimura/projects/tennis-lab/outputs/ball_refiner/generate/context/i935-context-fullframe-pilot-r10-20260928/dino_extension/lib:/home/kamimura/projects/tennis-lab/.claude/worktrees/ball-mix-v2-player-pose
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    CUDA_LAUNCH_BLOCKING=1 /home/kamimura/projects/tennis-lab/.venv/bin/python -u
    /home/kamimura/projects/tennis-lab/.claude/worktrees/ball-mix-v2-player-pose/.runtime/gpu_smoke.py
artifacts:
  run_dir: knowledge/runs/run-ball-mix-v2-player-pose-gpu-smoke-20261002
  log: /home/kamimura/projects/tennis-lab/.training_queue/logs/1790869513048574210_48328_ball-mix-v2-player-pose-gpu-smoke-20261002.log
parents: []
relations: []
papers: []
tags:
- pose-input
- court-free
- functional-smoke
---

ball-mix-v2の1フレームでCOCO人物DINO、ViTPose-H、CLIP-ReIDの順に実行し、17人物の姿勢(17,17,3)と外観(17,1280)を得た。コートモデルは実行していない。これはCUDA/拡張/入出力の機能確認であり、検出・姿勢・IDの正解率を評価したものではない。

元のqueueコマンド・commit・ログはrun.jsonに保持した。元の未追跡smokeスクリプトをbundleへ保存した。repro.shはその原文スクリプトを使い、同じCUDA拡張を追跡済みビルド入口から再構築する。再構築した拡張のbyte hashは元と同一とは保証しない。元の重み・ball storeとCUDA環境が必要。TensorBoard/学習/precision-recallの測定は対象外。
