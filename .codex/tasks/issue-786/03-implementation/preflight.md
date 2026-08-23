# Production preflight

- Issue: #786
- Attempt: 2
- Test cycle: 2
- Status: COMPLETE
- Candidate SHA-256: `sha256:0d52bf2f739ab9f989ed8d64df18e5e7d5d53dc4b386cd493dc4633f710b87bb`

## Candidate identity

State was read before substantive review. It records `phase = "implementation"`,
`preflight_verdict = "RETURN"`, `preflight_cycle = 2`, and `test_cycle = 1`,
so this is the required closure review. The existing RETURN artifact was read
first; its F-001 and original merge-local worklist are the complete frozen
scope. No discovery mutation category was added.

- Branch/head: `feat/issue-786-normalization-v2` /
  `2f041e48d1cfb6bde3cce47db2ba365c109a062c` (`fix: close merged ablation
  mypy regression`).
- Frozen base: `59e3b166c2d010d5e62be52c2be76d98a94af0e0`.
- `candidate-fingerprint .codex/tasks/issue-786` was recomputed as
  `sha256:0d52bf2f739ab9f989ed8d64df18e5e7d5d53dc4b386cd493dc4633f710b87bb`,
  matching the assigned candidate identity.
- The complete Issue non-workflow diff from the frozen base is 323 paths;
  the PR-shaped diff from `origin/main` is 256 paths (27 knowledge, 164
  `src/**`, 64 `tests/**`, and `.gitignore`). Both diff checks pass. The
  reviewer changed no source, tests, plan, state, Issue, or implementation
  artifact; only this preflight artifact and manager-owned preflight results
  and logs are in the reviewer's scope.

## Changed scope

- F-001 is repaired narrowly in
  `tests/unit/tasks/blcs/models/test_blcs_track_query_ablation_model.py`:
  `cast("object", BLCSTrackQueryAblationModel) is not
  BLCSTrackQueryModel` widens only mypy's static view. The runtime `is not`
  assertion, class names/modules, forward signature, strict checkpoint
  incompatibility, and baseline state-key inventory remain unchanged.
- The three merge overlaps remain coherent: `src/tasks/blcs/configuration.py`,
  `tests/unit/tasks/blcs/inference/test_tracking_predictor.py`, and
  `tests/unit/tasks/plcs/inference/test_tracking_predictor.py`. The BLCS
  parser accepts both `blcs_track_query` and
  `blcs_track_query_ablation`, retains strict generator/loss normalization
  validation, and the merged predictor tests retain default-v1/v2 metre-scale
  assertions plus exact ablation model/adapter checkpoint restoration.
- Normalization coverage and upstream track-query-ablation coverage coexist in
  the final tree. The merge has no unmerged index entries and no conflict
  markers in `src` or `tests`.

## Deterministic policy checks

- Closure scope was limited to F-001, the canonical preflight checks, the
  three conflict resolutions, normalization/upstream-ablation coexistence,
  conflict/index integrity, PR-shaped diff sanity, and direct repair
  regressions. The frozen Issue, exploration, plan, checks manifest,
  implementation handoff, repository guidance, state, prior RETURN, current
  code, and complete frozen-base/PR-shaped diffs were inspected.
- `git diff --check 59e3b166c2d010d5e62be52c2be76d98a94af0e0 -- .
  ':(exclude).codex/tasks'` passed.
- `git diff --check origin/main...HEAD -- . ':(exclude).codex/tasks'`
  passed.
- `git ls-files -u` returned zero entries; the marker scan over `src` and
  `tests` returned no `<<<<<<<`, `=======`, or `>>>>>>>` lines.
- The focused merge checks exercised both normalization versions, strict
  nested BLCS schemas, generator-section validation for both model families,
  default/v2 physical outputs, and exact upstream ablation restoration.

## Focused checks

- F-001 repair regression:
  `.venv/bin/python -m pytest -q
  tests/unit/tasks/blcs/models/test_blcs_track_query_ablation_model.py` →
  **PASS**, 8 passed. This includes the exact distinct-class identity
  assertion and strict baseline/ablation checkpoint incompatibility.
- Frozen merge-local worklist:
  `.venv/bin/python -m pytest -q
  tests/integration/tasks/blcs/test_model_configs.py
  tests/integration/tasks/plcs/test_model_configs.py
  tests/integration/tasks/blcs/test_track_query_ablation.py
  tests/integration/tasks/plcs/test_track_query_ablation.py
  tests/unit/tasks/blcs/inference/test_tracking_predictor.py
  tests/unit/tasks/plcs/inference/test_tracking_predictor.py` → **PASS**,
  51 passed. This verifies all three conflict resolutions and the
  normalization/upstream-ablation coexistence boundary.

## Canonical command results

Every required preflight-stage check in `02-planning/checks.json` was executed
through `manage_issue_task.py run-check` for this candidate:

- `run-check .codex/tasks/issue-786 preflight preflight-regression` →
  **PASS**, exit 0, 127 passed. Result is bound in
  `03-implementation/preflight-checks.json` with invocation digest
  `sha256:3a4023b90399d612468e51a82fa1078582859b7eb78a66691ebe55fba8eb3686`.
- `run-check .codex/tasks/issue-786 preflight precommit-all` → **PASS**, exit
  0. Ruff, mypy, and task-script reviewer all passed. Result is bound in
  `03-implementation/preflight-checks.json` with invocation digest
  `sha256:a8c9e12dd2478c4cf43f586c9830d54e6b66c561ae1beb54ba56bd1bbfd2b61b`.

The generated results and raw logs bind both checks to the current candidate
fingerprint.

## Baseline comparison

The prior closure worklist had one required failure: `precommit-all` rejected
the upstream ablation test's direct class-identity comparison with mypy
`comparison-overlap`. The current candidate changes only that comparison's
static typing view and retains the same runtime assertion. The fresh canonical
preflight run now passes mypy and all other hooks, while the preflight
regression remains green at 127 passed.

The three merge-overlap paths preserve both the Issue's normalization behavior
and upstream-main ablation behavior, as shown by the 51 focused passes and the
8 direct repair tests. No new semantic finding or unrelated regression was
encountered, and no source or test repair was made by this reviewer.

## Commands and exact outcomes

- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py candidate-fingerprint .codex/tasks/issue-786`
  → `sha256:0d52bf2f739ab9f989ed8d64df18e5e7d5d53dc4b386cd493dc4633f710b87bb`.
- `.venv/bin/python -m pytest -q tests/unit/tasks/blcs/models/test_blcs_track_query_ablation_model.py`
  → **PASS**, 8 passed.
- `.venv/bin/python -m pytest -q tests/integration/tasks/blcs/test_model_configs.py tests/integration/tasks/plcs/test_model_configs.py tests/integration/tasks/blcs/test_track_query_ablation.py tests/integration/tasks/plcs/test_track_query_ablation.py tests/unit/tasks/blcs/inference/test_tracking_predictor.py tests/unit/tasks/plcs/inference/test_tracking_predictor.py`
  → **PASS**, 51 passed.
- `git diff --check` against both the frozen base and `origin/main` → **PASS**;
  `git ls-files -u` → zero; conflict-marker scan → none.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py artifact-check .codex/tasks/issue-786 preflight`
  → **PASS** (`ok`).

## Final production preflight verdict

PASS

## RETURN implementation findings

None
