#!/usr/bin/env bash
set -euo pipefail

run=knowledge/runs/run-i2634-player-selection-ablation
git_common=$(git rev-parse --path-format=absolute --git-common-dir)
main=$(dirname "$git_common")
output=${1:?Specify a new output directory}
shift
archive3=${MEIJI3_ARCHIVED_OUTPUTS:-"/mnt/wsl/tennis-data$main/outputs"}
archive4=${MEIJI4_I2363_RUN:-"$main/.worktree-outputs/parent-2278-issue-2363-primary/i2363/i2363-all-v1"}
dataset3=${MEIJI3_DATASET:-"data/tennis_multiview/processed/meiji_3cam/dataset"}

export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

.venv/bin/python -B "$run/replay.py" --dataset3 "$dataset3" --archive3 "$archive3" \
  --archive4 "$archive4" --output "$output" "$@"
.venv/bin/python -B "$run/history.py" "$output/historical-2d.csv"
