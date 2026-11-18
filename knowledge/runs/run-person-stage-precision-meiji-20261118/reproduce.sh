#!/usr/bin/env bash
set -euo pipefail

fail() {
    printf '[repro] %s\n' "$*" >&2
    exit 1
}

[[ $# -ge 2 ]] || fail "expected the run kind and the recorded commit"
kind="$1"
commit="$2"
shift 2
case "$kind" in
    smoke)
        [[ $# -eq 0 ]] || fail "smoke takes no arguments"
        ;;
    full)
        [[ $# -eq 0 || ( $# -eq 1 && "$1" == --resume ) ]] ||
            fail "full reproduction accepts only --resume"
        : "${TENNIS_PRECISION_SHARED_ROOT:?Set the root of the pinned shared court and drift stores}"
        ;;
    *)
        fail "unknown run kind $kind"
        ;;
esac
: "${TENNIS_RUN_ID:?Submit the command via the shared training queue}"
: "${TENNIS_REPRO_DIR:?Submit the command via the shared training queue}"
[[ "${TENNIS_GPU_RESOURCE:-}" == all ]] || fail "use --resource all in the shared training queue"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO="$(cd -- "${TENNIS_REPO:-.}" && pwd -P)"
cd "$REPO"
[[ "$(git rev-parse HEAD)" == "$commit" ]] ||
    fail "use a checkout at the recorded commit $commit"
git diff --quiet HEAD -- || fail "the recorded checkout has changes"

dataset="${TENNIS_PRECISION_DATASET:-$REPO/data/tennis_multiview/processed/meiji_4cam/dataset}"
output="${TENNIS_PRECISION_OUTPUT:-$REPO/outputs/precision-repro}"
measurements="$REPO/knowledge/runs/run-person-stage-precision-meiji-20261118"
if [[ "$kind" == smoke ]]; then
    pin="$measurements/smoke-inputs.json"
else
    pin="$measurements/full-inputs.json"
fi
TENNIS_LAB_DINO_EXTENSION="$(bash "$SCRIPT_DIR/rebuild-dino.sh" "$pin" "$output/native-dino")"
export TENNIS_LAB_DINO_EXTENSION
export PYTHONPATH="$REPO" PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

if [[ "$kind" == smoke ]]; then
    exec .venv/bin/python -u "$measurements/gpu_guard.py" \
        --health "$output/smoke/health" "$measurements/smoke.py" \
        --dataset "$dataset" --pin "$pin" --output "$output/smoke/metrics.json"
fi
exec .venv/bin/python -u "$measurements/gpu_guard.py" \
    --health "$output/full/health-$TENNIS_RUN_ID" "$measurements/benchmark.py" \
    --dataset "$dataset" --shared-root "$TENNIS_PRECISION_SHARED_ROOT" \
    --pin "$pin" --output "$output/full" --smoke "$output/smoke/health/status.json" "$@"
