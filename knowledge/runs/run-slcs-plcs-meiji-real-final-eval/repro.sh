#!/usr/bin/env bash
set -euo pipefail
cd /home/kamimura/projects/tennis-lab/.claude/worktrees/slcs-real-rgb
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 .venv/bin/python -m src.tennis_scene.scripts.build_slcs_dataset stage=infer device=cpu paths.external_asset_root=/home/kamimura/projects/tennis-lab/third_party 'dataset_clip_ids=[video_000/clip_000,video_001/clip_000]' 'clip_ids=[video_000/clip_000,video_001/clip_000]' dataset_output_directory=slcs/meiji_teacher_comparison_v1 output_dir=tennis_scene/generate/meiji_teacher_comparison/s42-001 observation_directory=tennis_scene/precompute/meiji_dino_vitpose/s42-002 people.court_half_width_m=5.8 people.long_gap_policy=error features.enabled=false
