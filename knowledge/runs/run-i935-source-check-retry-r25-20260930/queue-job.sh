#!/usr/bin/env bash
# name: i935-e9-anchored-source3cam-r25-20260930
# added: 2026-09-30T22:25:09+09:00
# provider: codex
# session: 01a0f263-c4b3-7972-a86a-90c1f4c56cc8
# issue: 935
# resource: all
# external_teardown_ack: 0
# prune_ckpt: 0
cd /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain
export TENNIS_RUN_ID=1790774709010605164_1159028_i935-e9-anchored-source3cam-r25-20260930
export TENNIS_REPRO_DIR=/home/kamimura/projects/tennis-lab/.training_queue/repro/1790774709010605164_1159028_i935-e9-anchored-source3cam-r25-20260930
export TENNIS_GPU_RESOURCE=all
bash /home/kamimura/projects/tennis-lab/.claude/skills/training-queue/scripts/training_queue.sh __capture-repro --dir /home/kamimura/projects/tennis-lab/.training_queue/repro/1790774709010605164_1159028_i935-e9-anchored-source3cam-r25-20260930 --name i935-e9-anchored-source3cam-r25-20260930 --provider codex --session 01a0f263-c4b3-7972-a86a-90c1f4c56cc8 --issue 935 --cmd timeout\ --signal=TERM\ --kill-after=15s\ 1185s\ env\ PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain\ OMP_NUM_THREADS=2\ MKL_NUM_THREADS=2\ OPENBLAS_NUM_THREADS=2\ TORCHINDUCTOR_COMPILE_THREADS=2\ MAX_JOBS=2\ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\ /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain/.venv/bin/python\ -u\ /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain/knowledge/runs/run-i935-source-check-retry-r25-20260930/launch_check.py\ --plan\ /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain/knowledge/runs/run-i935-source-check-retry-r25-20260930/plan.json
timeout --signal=TERM --kill-after=15s 1185s env PYTHONPATH=/home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=2 MAX_JOBS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain/.venv/bin/python -u /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain/knowledge/runs/run-i935-source-check-retry-r25-20260930/launch_check.py --plan /home/kamimura/projects/tennis-lab/.claude/worktrees/c930-i935-11-pilot-retrain/knowledge/runs/run-i935-source-check-retry-r25-20260930/plan.json
