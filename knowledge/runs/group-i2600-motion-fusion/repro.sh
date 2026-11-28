#!/usr/bin/env bash
set -euo pipefail

bundle=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
main=$(dirname -- "$(git -C "$bundle" rev-parse --path-format=absolute --git-common-dir)")
implementation=${MOTION_IMPLEMENTATION:-"$main/.claude/worktrees/parent-1722-issue-2600-primary"}
expected=${MOTION_REPORT_HEAD:-3d8cec367e8a9fb2cb44788bdae3eb494e0aeb3f}
saved=${MOTION_RUN_ROOT:-"$main/.worktree-outputs/parent-1722-issue-2600-primary/ball_refiner_3d/post_training/coverage-r2"}
output=${1:?Supply an absolute path to a new report directory}
if [[ "$output" != /* ]]; then
    exit 2
fi
cd -- "$implementation"
test "$(git rev-parse HEAD)" = "$expected"
test -z "$(git status --porcelain -- src)"
exec env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    PYTHONPATH=. .venv/bin/python "$bundle/compare.py" \
    --run-root "$saved" --output "$output"
