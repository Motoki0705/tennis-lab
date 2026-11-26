#!/usr/bin/env bash
set -euo pipefail

repro="${TENNIS_REPRO_DIR:?}"
input="${1:?provide the #2497 input directory}"
scripts=outputs/fasthmr-setup/speed-pt16
export PYTHONPATH="$PWD:$PWD/knowledge/runs/run-person-model-speed-baseline:$PWD/outputs/fasthmr-setup/python"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cp "$scripts"/{benchmark_fasthmr.py,run.sh,source.sha256} "$repro/"
mkdir -p "$repro/predictions"
status=0
for batch in 1 2 4; do
    if ! .venv/bin/python -B -u "$repro/benchmark_fasthmr.py" --input-dir "$input" --batch "$batch" --output "$repro/predictions/pt16-b$batch.json"; then
        status=1
    fi
done
exit "$status"
