#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
TOPK_COMPOSITION_RUN=knowledge/runs/run-ball3d-topk-multihypothesis-20261103
TOPK_COMPOSITION_TMP=$(mktemp -d /tmp/ball3d-topk-composition-tests.XXXXXX)
readarray -t TOPK_COMPOSITION_TARGETS < <(.venv/bin/python - <<'PY'
import json
from pathlib import Path
receipt = json.loads(Path('knowledge/runs/run-ball3d-topk-multihypothesis-20261103/test-results.json').read_text())
print('\n'.join(dict.fromkeys(node.split('::')[0] for node in receipt['nodes'])))
PY
)
TOPK_COMPOSITION_TARGETS+=(
  tests/unit/utils/configuration/
  tests/unit/synthetic_data_generation/visualization/test_contracts.py
  tests/unit/submodules/models/dino/test_extension.py
)
.venv/bin/python "$TOPK_COMPOSITION_RUN/measure_tests.py" --output "$TOPK_COMPOSITION_TMP/collection.json" -- \
  -n 0 --collect-only -q "${TOPK_COMPOSITION_TARGETS[@]}" > "$TOPK_COMPOSITION_TMP/collection.txt"
.venv/bin/python "$TOPK_COMPOSITION_RUN/measure_tests.py" --output "$TOPK_COMPOSITION_TMP/tests.json" -- \
  -n 0 --durations=0 --durations-min=0 "${TOPK_COMPOSITION_TARGETS[@]}" > "$TOPK_COMPOSITION_TMP/tests.txt"
.venv/bin/python - "$TOPK_COMPOSITION_TMP" "$TOPK_COMPOSITION_RUN" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
source, target = map(Path, sys.argv[1:])
collection = json.loads((source / 'collection.json').read_text())
tests = json.loads((source / 'tests.json').read_text())
assert collection['nodes'] == tests['nodes']
nodes = {node: i for i, node in enumerate(tests['nodes'])}
for report in tests['reports']:
    report['node_index'] = nodes[report.pop('node')]
tests['full_stdout_local_path'] = str(source / 'tests.txt')
(target / 'composition-tests.json').write_text(json.dumps(tests, separators=(',', ':')) + '\n')
collection.pop('reports')
collection.pop('nodes')
collection['nodes_equal_execution_receipt'] = True
collection['execution_receipt_sha256'] = hashlib.sha256((target / 'composition-tests.json').read_bytes()).hexdigest()
(target / 'composition-collection.json').write_text(json.dumps(collection, indent=2) + '\n')
print(json.dumps({'collection_count': collection['collected_count'], 'collection_seconds': collection['wall_seconds'],
                  'execution_seconds': tests['wall_seconds'], 'exit_code': tests['exit_code'],
                  'full_stdout_local_path': tests['full_stdout_local_path']}, indent=2))
PY
tail -3 "$TOPK_COMPOSITION_TMP/tests.txt"
