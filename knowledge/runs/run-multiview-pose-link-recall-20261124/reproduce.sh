#!/usr/bin/env bash
set -euo pipefail

run=knowledge/runs/run-multiview-pose-link-recall-20261124
previous=knowledge/runs/run-multiview-pose-meiji-20261124
main=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
archives=${MEIJI_POSE_ARCHIVED_OUTPUTS:-"/mnt/wsl/tennis-data$main/outputs"}
mode=${1:-all}
output=${2:-outputs/i2444/refine/qualified}

case "$mode" in
    meiji3|meiji4) bash "$previous/reproduce.sh" "$mode" "$output" ;;
    all|trace|holdout-inputs)
        export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
        export PYTHONPATH="$run:$previous:knowledge/runs/run-i2444-multiview-pose-probe:."
        dataset=data/tennis_multiview/processed/meiji_3cam/dataset
        if [[ "$mode" == all || "$mode" == trace ]]; then
            .venv/bin/python -B "$run/trace.py" \
                --dataset "$dataset" --archives "$archives" --output "$output/losses.json"
        fi
        if [[ "$mode" == all || "$mode" == holdout-inputs ]]; then
            .venv/bin/python -B "$run/holdout_inputs.py" --dataset "$dataset" \
                --archive "$archives/person_tracking/evaluate/i964-unseen-r16-20261001" \
                --output "$output/holdout-inputs.json"
        fi
        if [[ "$mode" == all ]]; then
            bash "$previous/reproduce.sh" all "$output"
        fi
        ;;
    *)
        printf 'Usage: bash %s/reproduce.sh [all|trace|meiji3|meiji4|holdout-inputs] [output-directory]\n' "$run" >&2
        exit 2
        ;;
esac
