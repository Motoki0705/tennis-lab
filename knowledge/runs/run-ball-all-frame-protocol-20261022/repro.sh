#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_REPO_ROOT:?Set SOURCE_REPO_ROOT to the repo containing the pinned read-only snapshot}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SURVEY_OUTPUT="${SURVEY_OUTPUT:-$PWD/tmp/ball-all-frame-protocol-repro}"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python "$SCRIPT_DIR/all_frame_probe.py" \
  --source-repo "$SOURCE_REPO_ROOT" --output "$SURVEY_OUTPUT"
.venv/bin/python - "$SCRIPT_DIR" "$SURVEY_OUTPUT" <<'PY'
import sys
from pathlib import Path

expected, actual = map(Path, sys.argv[1:])
checked = []
for path in sorted(actual.iterdir()):
    if path.name == "execution.json":
        continue  # runtime/RSS and the enclosing Git commit are not metric evidence
    if path.read_bytes() != (expected / path.name).read_bytes():
        raise ValueError(f"Reproduction differs from the recorded proposal: {path.name}")
    checked.append(path.name)
print(f"Exact reproduction: {len(checked)} evidence files match (execution metadata excluded)")
PY
