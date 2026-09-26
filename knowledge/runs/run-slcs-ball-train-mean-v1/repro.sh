#!/usr/bin/env bash
# #931: the original command read these files from outputs/; they are saved in this bundle.
set -euo pipefail
# From repository root at the recorded commit, with original data and prediction bundles.
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -B knowledge/runs/run-slcs-ball-train-mean-v1/probe.py --training-config "$SCRIPT_DIR/config.yaml" --evaluation-dir outputs/slcs/evaluate/real_rgb_pilot_baseline_val_full/s42-002 --evaluation-dir outputs/slcs/evaluate/real_rgb_pilot_baseline_test_full/s42-002 --evaluation-dir outputs/slcs/evaluate/real_rgb_pilot_augmented_val_full/s42-002 --evaluation-dir outputs/slcs/evaluate/real_rgb_pilot_augmented_test_full/s42-002 --output-dir outputs/slcs/analyze/ball_train_mean/REPRO_NEW_RUN_ID
