---
id: run-person-stage-precision-interrupted-20261118
type: run
task: tennis_scene
sequence: 61
recorded_at: '2026-11-18'
title: 共有training queueの終了シグナルで人物推論の全尺計測を中断
issue: 2238
provider: codex
session: 01a1ec52-c2ef-70e3-bd48-588f68ed717e
date: '2026-11-18'
status: failed
config:
  completed_cases:
  - - fp32
    - clip_037
  - - fp32
    - clip_000
  - - vitpose_bf16
    - clip_037
  - - vitpose_bf16
    - clip_000
  - - vitpose_fp16
    - clip_037
  interrupted_case:
  - vitpose_fp16
  - clip_000
metrics:
  completed_cases: 5
  queue_exit_code: 143
  last_vitpose_call_in_stage: 35
repro:
  commit: ba5d26a0767955563ed24a85bb0d9d8bddcc6c16
  branch: codex/person-stage-speed
  remote: git@github.com:Motoki0705/tennis-lab.git
  command: env PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    OPENBLAS_NUM_THREADS=1 TENNIS_LAB_DINO_EXTENSION=/home/kamimura/projects/tennis-lab/.claude/worktrees/video-batch-cli/outputs/tennis_scene/evaluate/video-batch-runtime-20261117/dino-extension-a2/lib/MultiScaleDeformableAttention.so
    .venv/bin/python -u knowledge/runs/run-person-stage-precision-meiji-20261118/gpu_guard.py
    --health outputs/precision/full/health knowledge/runs/run-person-stage-precision-meiji-20261118/benchmark.py
    --dataset data/tennis_multiview/processed/meiji_4cam/dataset --shared-root /home/kamimura/projects/tennis-lab/.worktree-outputs/court-video-shared
    --pin knowledge/runs/run-person-stage-precision-meiji-20261118/full-inputs.json
    --output outputs/precision/full --smoke outputs/precision/smoke/health/status.json
artifacts:
  run_dir: knowledge/runs/run-person-stage-precision-interrupted-20261118
  log: .training_queue/logs/0-8205031227420113411_2026-11-18T02-26-12-579886588Z_2339132_i2238-precision-meiji-full.log
  output_dir: .worktree-outputs/person-stage-speed/precision/full
parents:
- run-person-stage-precision-smoke-20261118
relations: []
papers: []
tags:
- inference-precision
- interrupted
summary_impact: none
---

# 中断の観測

2026-11-18 12:19 JST、Meiji 4camの全尺計測中に共有training queueのworkerがTERMを受け、
監督していたジョブを停止した。queueは `state=failed`・`wait=signal`、出力は `exit_code=143` を記録した。
最後のheartbeatは `person_tracking/cam1 vitpose call 35`。計測ログにCUDA例外とheartbeat timeoutの記録はなく、
guardの完了記録は生成されなかった。TERMの送信元は未確認。
[queue状態](../../runs/run-person-stage-precision-interrupted-20261118/terminal-state.txt)・[ログ末尾](../../runs/run-person-stage-precision-interrupted-20261118/terminal-log.txt)

完了した5ケースは、両クリップのFP32・ViTPose BF16と、clip_037のViTPose FP16。
未完了のclip_000のViTPose FP16は元の計測ファイルとともに退避した。
再開時は完了marker、入力メディア、重み、PyTorch・CUDA・GPU、pipelineコードを照合し、未完了のケースから実行する。
この中断は計測条件の採否を変えない（summary_impact: none）。
[全尺計測の固定入力](../../runs/run-person-stage-precision-meiji-20261118/full-inputs.json)
