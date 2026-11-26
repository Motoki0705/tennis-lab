#!/usr/bin/env bash
set -euo pipefail

cp outputs/fasthmr-setup/smoke_fixed_body_inputs.py "${TENNIS_REPRO_DIR:?}/fasthmr-smoke.py"
export PYTHONPATH="$PWD:$PWD/outputs/fasthmr-setup/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
smoke_status=0
for backend in pt32 pt16; do
    if ! .venv/bin/python -u "$TENNIS_REPRO_DIR/fasthmr-smoke.py" "$backend"; then
        smoke_status=1
    fi
done
exit "$smoke_status"
