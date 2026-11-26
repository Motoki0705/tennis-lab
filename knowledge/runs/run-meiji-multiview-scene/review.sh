#!/usr/bin/env bash
set -euo pipefail

# Run from the evidence checkout. The output roots are separate from the dataset.
worktrees="${MEIJI_WORKTREES:-$(dirname "$PWD")}"
primary="$worktrees/parent-1739-issue-2446-primary"
i2363="$worktrees/parent-2278-issue-2363-primary/outputs/i2363/i2363-all-v1"
candidates="$worktrees/parent-1739-issue-2444-primary/outputs/i2444/refine/timing/meiji4"
tracks="$worktrees/parent-1739-issue-2445-primary/outputs/i2445/integrated/meiji4"
archive="${MEIJI_ARCHIVE:-/mnt/wsl/tennis-data/home/kamimura/projects/tennis-lab/outputs}"

run_dir="knowledge/runs/run-meiji-multiview-scene"
result="${MEIJI_REVIEW_OUTPUT:-outputs/i2446-evidence}"
page="${MEIJI_PAGE_OUTPUT:-$primary/outputs/pose-comparison-bodies}"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

case "${1:-all}" in
  scores)
    .venv/bin/python "$run_dir/compare_meiji3.py" \
      --dataset data/tennis_multiview/processed/meiji_3cam/dataset \
      --archive "$archive" --output "$result/meiji3.json"
    ;;
  verify)
    .venv/bin/python "$run_dir/verify_meiji4.py" --i2363-run "$i2363" \
      --scenes "$primary/outputs/i2446" --output "$result"
    ;;
  render)
    .venv/bin/python "$run_dir/render_bodies.py" \
      --page "$primary/outputs/pose-comparison" --i2363-run "$i2363" \
      --scenes "$primary/outputs/i2446" --candidates "$candidates" \
      --pose-record knowledge/runs/run-multiview-pose-link-recall-20261124/meiji4.json \
      --tracks "$tracks" \
      --track-record knowledge/runs/run-i2445-player-slot-integration/meiji4.json \
      --regressor ckpt/body_models/gvhmr/smpl_neutral_J_regressor.pt --output "$page"
    ;;
  all)
    bash "$run_dir/review.sh" scores
    bash "$run_dir/review.sh" verify
    bash "$run_dir/review.sh" render
    ;;
  *)
    printf 'Usage: %s [scores|verify|render|all]\n' "$0" >&2
    exit 2
    ;;
esac
