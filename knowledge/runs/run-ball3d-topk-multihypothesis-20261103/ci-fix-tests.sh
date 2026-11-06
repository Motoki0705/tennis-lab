#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
TOPK_CI_RUN=knowledge/runs/run-ball3d-topk-multihypothesis-20261103
# Reuse the recorded related suite without rewriting its historical receipts.
readarray -t TOPK_CI_TARGETS < <(.venv/bin/python - <<'PY'
import json
from pathlib import Path
receipt = json.loads(Path('knowledge/runs/run-ball3d-topk-multihypothesis-20261103/test-results.json').read_text())
print('\n'.join(dict.fromkeys(node.split('::')[0] for node in receipt['nodes'])))
PY
)
TOPK_CI_TARGETS+=(
  tests/unit/utils/configuration/test_contracts.py::test_adapter_fields_have_truthful_default_precedence_and_absence_policy
  tests/unit/submodules/models/dino/test_extension.py
)
.venv/bin/python "$TOPK_CI_RUN/measure_tests.py" --output "$TOPK_CI_RUN/ci-fix-collection.json" -- \
  -n 0 --collect-only -q "${TOPK_CI_TARGETS[@]}" > "$TOPK_CI_RUN/ci-fix-collection.txt"
.venv/bin/python "$TOPK_CI_RUN/measure_tests.py" --output "$TOPK_CI_RUN/ci-fix-related-tests.json" -- \
  -n 0 --durations=0 --durations-min=0 "${TOPK_CI_TARGETS[@]}" > "$TOPK_CI_RUN/ci-fix-related-tests.txt"
tail -3 "$TOPK_CI_RUN/ci-fix-related-tests.txt"
