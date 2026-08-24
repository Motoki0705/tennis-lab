# Final candidate seal

- Issue: #786
- Attempt: 6
- Test cycle: 1
- Status: COMPLETE
- Frozen issue SHA-256: `6279b189d4b3c0a7c11da3e605fbc252624f5a60ec808db2c476e061f55fa6a9`
- Frozen acceptance checklist SHA-256: `95bcebf4388fdba9773e3c538c9e22caf82b6e4a413ec1241e9a58b0c4483032`
- Candidate SHA-256: `sha256:6ca1bf6d8eaf5a619a0d923c0568806510c5a35ea663d3332b2c0cff23492b35`

## Candidate identity

The Tester PASS candidate is exactly
`sha256:6ca1bf6d8eaf5a619a0d923c0568806510c5a35ea663d3332b2c0cff23492b35`.
The Tester `tests.md`, `test-checks.json`, and `state.toml` test binding all
record this same identity and `test_verdict = "PASS"`. A fresh
`candidate-fingerprint` at the start of Seal and again after all Seal checks
returned the same hash. The frozen base revision is
`59e3b166c2d010d5e62be52c2be76d98a94af0e0`.

No candidate-bound `AT-*` probes are present because this task is in the
frozen `LEGACY` adversarial-testing mode; the Tester evidence records that
the canonical checks are the authorized repair-local evidence.

## Changed-since-test inspection

There is no changed-since-test candidate content. The pre-check and post-check
fingerprints are both the Tester PASS hash above, including untracked candidate
tests. The only workflow-owned files changed while this candidate was being
sealed are under `.codex/tasks/issue-786/` and the generated Seal result/log
artifacts; those paths are excluded from the candidate fingerprint by the
workflow contract. The Seal Reviewer did not edit source, tests, fixtures,
documentation, planning, Issue, or state.

The complete working-tree content delta relative to `HEAD` was inspected and
was already part of the Tester-bound candidate:

- Attempt-6's exact repair is the local `text: str` bridge in
  `tests/integration/tasks/test_court_coordinate_normalization_documentation.py::_read`.
  The UTF-8 `Path.read_text` expression, path, exception propagation, and all
  four assertions are unchanged.
- The earlier approved documentation/config inventory is
  `src/synthetic_data_generation/dataset/plcs/README.md`,
  `src/tasks/base/README.md`, both `src/tasks/base/configs/court_coordinate_normalization/*.yaml`,
  `src/tasks/blcs/README.md`, `src/tasks/blcs/configs/run/train.yaml`,
  `src/tasks/plcs/README.md`, both
  `src/tasks/plcs/configs/court_coordinate_normalization/*.yaml`,
  `src/tasks/plcs/configs/{generate_dataset_norm_v2,train_norm_v2}.yaml`,
  `src/tasks/slcs/README.md`, and
  `src/tennis_scene/{README.md,generate_dataset/README.md}`.
- The earlier approved AC-017 implementation paths are
  `src/tasks/plcs/artifact_paths.py` and `src/tasks/plcs/training/runner.py`.
  The approved Test Writer paths are
  `tests/integration/tasks/plcs/test_artifact_publication.py`,
  `tests/integration/tasks/test_court_coordinate_normalization_documentation.py`,
  and `tests/unit/tasks/plcs/generate_dataset/io/test_dataset_io.py`, together
  with the existing normalization smoke, PLCS predictor, and PLCS configuration
  tests.

The three untracked paths are exactly those three approved test paths; no other
untracked path is present outside workflow artifacts.

## Canonical command results

Every required Seal-stage check from `02-planning/checks.json` was executed
through `manage_issue_task.py run-check .codex/tasks/issue-786 seal <check-id>`.
All 14 machine results bind the Tester candidate and have exit code 0:

| Check ID | Exit | Result | Exact outcome | Raw log |
|---|---:|---|---|---|
| `unit-contract` | 0 | PASS | 56 passed | `logs/canonical-seal-unit-contract.log` |
| `plcs-artifact-preservation` | 0 | PASS | 77 passed | `logs/canonical-seal-plcs-artifact-preservation.log` |
| `legacy-v1-checkpoint-parity` | 0 | PASS | 8 passed | `logs/canonical-seal-legacy-v1-checkpoint-parity.log` |
| `unit-blcs` | 0 | PASS | 39 passed | `logs/canonical-seal-unit-blcs.log` |
| `unit-plcs` | 0 | PASS | 27 passed | `logs/canonical-seal-unit-plcs.log` |
| `unit-slcs` | 0 | PASS | 35 passed | `logs/canonical-seal-unit-slcs.log` |
| `integration-normalization` | 0 | PASS | 14 passed | `logs/canonical-seal-integration-normalization.log` |
| `preflight-regression` | 0 | PASS | 167 passed | `logs/canonical-seal-preflight-regression.log` |
| `knowledge-graph` | 0 | PASS | 181 nodes, 0 errors, 4 warnings | `logs/canonical-seal-knowledge-graph.log` |
| `normalization-documentation` | 0 | PASS | 4 passed | `logs/canonical-seal-normalization-documentation.log` |
| `candidate-python-mypy` | 0 | PASS | no issues in 1124 source files | `logs/canonical-seal-candidate-python-mypy.log` |
| `documentation-test-mypy` | 0 | PASS | no issues in 1 source file | `logs/canonical-seal-documentation-test-mypy.log` |
| `precommit-all` | 0 | PASS | ruff, mypy, and task script reviewer passed | `logs/canonical-seal-precommit-all.log` |
| `full-pytest` | 0 | PASS | 3393 passed, 78 skipped, 18 warnings | `logs/canonical-seal-full-pytest.log` |

The four knowledge-graph warnings are the pre-existing unrelated nodes named
in the raw log and are not failures. The two distinct mypy checks both pass:
the broad hook-equivalent `src tests` run and the isolated changed-file run.

## Complete scope inspection

The frozen-base inventory contains 326 tracked changed paths plus the three
approved untracked Test Writer paths (329 candidate paths total). Every path
was classified against the plan's Issue ownership and merge provenance:

- 247 paths are first-parent Issue-owned normalization implementation,
  configuration, documentation, tests/fixtures, or knowledge evidence.
  The 27 knowledge paths are the four run bundles and five v1/v2 baseline
  nodes/group record required by AC-019.
- Eight merge-resolution paths are the explicitly reconciled paths
  `.gitignore`, `src/tasks/blcs/README.md`,
  `src/tasks/blcs/configuration.py`, `src/tasks/blcs/model_io/adapters.py`,
  `src/tasks/plcs/README.md`, `src/tasks/plcs/configuration.py`,
  `tests/unit/tasks/blcs/inference/test_tracking_predictor.py`, and
  `tests/unit/tasks/plcs/inference/test_tracking_predictor.py`; each is
  covered by the Issue implementation or the frozen merge/support inventory.
- 68 paths are second-parent-only upstream provenance: the 16 workflow skill
  files, two `.codex/agents` definitions, `AGENTS.md`, the track-query
  ablation model/config/factory/composition paths, shared model/CUDA operator
  paths, and their architecture/workflow e2e and integration/unit tests. They
  are retained as upstream baseline and are not Issue-authored scope.
- The six exact repository-gate support paths approved by the plan are
  `.gitignore`, `src/automation/chatgpt_mcp/jobs.py`,
  `src/tasks/ball_detection/data/components/staged_sampler.py`,
  `tests/e2e/colab/test_training_path_contracts.py`,
  `tests/e2e/synthetic_data_generation/test_b00_gpu_acceptance.py`, and
  `tests/e2e/synthetic_data_generation/test_removed_architecture.py`.
- The post-merge AC-017 paths and the Attempt-4/5/6 documentation/test paths
  listed in Changed-since-test inspection are all within the exact later
  attempt scopes in `plan.md`; no path outside those scopes remains.

The complete tracked inventory has no unmapped path, no unapproved generated
artifact, and no production edit attributable to the Test Writer or Seal
Reviewer. `git diff --check` is clean. Repository guidance is satisfied by
the passing pre-commit hooks, including Ruff, strict mypy, and task-script
reviewer; the isolated and broad mypy checks additionally prove the repaired
test boundary without a cast, ignore, fallback, or configuration weakening.

## Commands and exact outcomes

The canonical invocations, cwd, environment, invocation digests, candidate
bindings, exit codes, and raw logs are machine-recorded in
`03-implementation/seal-checks.json`; no argv, cwd, or environment was
reconstructed from prose. The result file contains exactly 14 required rows,
all with candidate
`sha256:6ca1bf6d8eaf5a619a0d923c0568806510c5a35ea663d3332b2c0cff23492b35`,
exit code `0`, and verdict `PASS`. The final fingerprint recomputation after
these commands is unchanged.

Evidence is internally complete: the frozen Issue/checklist hashes agree
across Issue/state/plan/tests; the Tester PASS artifact and all 14 test-stage
results bind the same candidate; the Seal result binds that candidate; and
the required canonical logs exist. No Seal state transition was requested or
performed.

## Final candidate seal verdict

PASS

## RETURN implementation findings

None
