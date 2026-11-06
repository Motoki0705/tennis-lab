#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${1:-$(mktemp -d /tmp/ball3d-dense-ci-fix-tests.XXXXXX)}"
echo "Test logs: $LOG_DIR"
nodes=(
  tests/unit/utils/configuration
  tests/unit/synthetic_data_generation/visualization/test_contracts.py
  tests/unit/submodules/models/dino/test_extension.py
  tests/unit/tennis_scene/ball_likelihood
  tests/integration/tennis_scene/test_ball3d_dense_likelihood.py
  tests/unit/tennis_scene/evaluation/ball3d
  tests/unit/tennis_scene/test_configuration.py
  tests/unit/tennis_scene/pipeline/test_orchestrator.py
  tests/unit/tennis_scene/pipeline/components/test_ball_points.py
  tests/unit/tennis_scene/pipeline/components/test_ball_evidence.py
  tests/unit/tennis_scene/pipeline/test_artifact_codec.py
  tests/unit/tennis_scene/visualization/test_ball3d_comparison.py
  tests/unit/tennis_scene/visualization/test_ball3d_tables.py
  tests/integration/tennis_scene/test_ball3d_bundle.py
  tests/integration/tennis_scene/test_ball3d_media.py
)
if [[ $# -eq 0 ]]; then
  .venv/bin/python -m pytest -n 0 -q --collect-only "${nodes[@]}" > "$LOG_DIR/collection.txt" 2>&1
  .venv/bin/python -m pytest -n 0 -q --durations=0 --durations-min=0 \
    --junitxml="$LOG_DIR/results.xml" "${nodes[@]}" > "$LOG_DIR/durations.txt" 2>&1
fi
.venv/bin/python - "$DIR" "$LOG_DIR" <<'PY'
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

target, logs = map(Path, sys.argv[1:])
collection = (logs / "collection.txt").read_text()
raw = (logs / "durations.txt").read_text()
stages = {}
for value, stage, node in re.findall(r"([0-9.]+)s\s+(setup|call|teardown)\s+(tests/[^\n]+)", raw):
    stages.setdefault(node, {})[stage] = float(value)
rows = []
for case in ET.parse(logs / "results.xml").getroot().iter("testcase"):
    node = case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"]
    assert all(case.find(k) is None for k in ("failure", "error")), node
    skipped = case.find("skipped")
    row = {"node": node, "outcome": "passed" if skipped is None else "skipped",
           "seconds_setup_call_teardown": float(case.attrib["time"]), "phases_rounded_s": stages[node]}
    if skipped is not None:
        row["skip_reason"] = skipped.attrib["message"]
    rows.append(row)
c = re.search(r"(\d+) tests collected in ([\d.]+)s", collection)
r = re.search(r"(\d+) passed(?:,.*)? in ([\d.]+)s", raw)
assert int(c[1]) == len(rows)
passed = sum(row["outcome"] == "passed" for row in rows)
assert int(r[1]) == passed
receipt = {
    "command": "bash knowledge/runs/run-ball3d-dense-likelihood-20261103/ci-fix-tests.sh",
    "pytest_options": ["-n", "0", "-q", "--durations=0", "--durations-min=0"],
    "collected": len(rows), "passed": passed, "skipped": len(rows) - passed, "collection_seconds": float(c[2]),
    "pytest_elapsed_seconds": float(r[2]), "raw_logs_dir": str(logs),
    "raw_logs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in logs.iterdir()},
    "cases": rows,
}
(target / "ci-fix-tests.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
print({k: v for k, v in receipt.items() if k != "cases"})
PY
