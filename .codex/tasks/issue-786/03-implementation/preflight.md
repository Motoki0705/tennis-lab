# Production preflight

- Issue: #786
- Attempt: 1
- Test cycle: 1
- Status: COMPLETE
- Candidate SHA-256: `sha256:0b141aeead98e5cfcdf04f55132ed10a43acb5c95e676ad5823061104904a7e6`

## Candidate identity

- Review mode: **Discovery**. `state.toml` has `preflight_verdict = ""` and `preflight_cycle = 0`; this is the first pending preflight cycle.
- Branch/head: `feat/issue-786-normalization-v2` / `2e77ed5dc98aba36a69559f3ddbfce99beb363c2`.
- Frozen base: `59e3b166c2d010d5e62be52c2be76d98a94af0e0`.
- Recomputed candidate fingerprint: `sha256:0b141aeead98e5cfcdf04f55132ed10a43acb5c95e676ad5823061104904a7e6` (matches `implementation.md`).
- Review scope is the complete candidate diff against the frozen base: 193 paths, 5,385 additions, and 425 deletions, excluding `.codex/tasks/` workflow artifacts.

## Changed scope

- The candidate adds the shared immutable v1/v2 court-coordinate resolver, base dataset/checkpoint metadata contracts, non-overwriting BLCS/PLCS materialization, and shared Hydra configuration.
- BLCS, PLCS, SLCS, and tennis-scene generation, loading, losses/metrics, inference, projection/rendering, checkpoint propagation, and provenance are threaded through the selected contract. Documentation, configuration inventory, four BLCS visibility tests, knowledge nodes, and reproducibility bundles are also in the diff.
- The physical court constants, UV/camera conventions, root-relative canonical pose metres, public tennis-scene metre arrays, and legacy v1 aliases remain unchanged. No workflow artifact other than this preflight record and canonical generated results is in the candidate scope.

## Deterministic policy checks

- Candidate identity and base binding are valid. `candidate-fingerprint` recomputed the SHA-256 above.
- `git diff --check 59e3b166c2d010d5e62be52c2be76d98a94af0e0 -- . ':(exclude).codex/tasks'`: PASS (no whitespace errors).
- The fixed-scale inventory found `COURT_COORD_SCALE_*` only as documented v1 aliases in `src/utils/schema/court.py` and in existing tests. Remaining `HALF_*`/`NET_HEIGHT_POST` references are physical court geometry, rendering, sampling, or the resolver definition, not active normalization consumers.
- The changed script paths have module docstrings; the canonical pre-commit task-script reviewer passed. The complete diff contains no physical-court, UV, camera, canonical-pose, or SceneResult-unit changes.

## Focused checks

- Config-composition matrix (the only configuration category frozen in `plan.md`): BLCS `train`, `train_tracking`, `generate_dataset`, and `visualize` v1/v2 passed; PLCS `train`, `train_tracking`, and version-qualified train roots v1/v2 passed; the PLCS visualization boundary validator passed for v1/v2; SLCS `train`, `evaluate`, and `predict_clip` v1/v2 passed; and tennis-scene `pipeline`/`generate_dataset` v1/v2 passed. PLCS generation composition is blocked by the external-data symlink failure recorded below.
- Unit round-trip matrix: NumPy and Torch position/velocity `(..., 3)` examples round-tripped in v1/v2 with maximum absolute error at or below `1e-5`; unknown `v3` and invalid trailing shapes were explicitly rejected.
- Metadata mutation matrix: all-missing metadata was accepted only for explicit v1 and rejected for v2; complete v1/v2 metadata was accepted; root-only, scene-only, mixed, runtime-mismatch, and unknown-version mutations were rejected.
- Checkpoint mismatch matrix: metadata-free checkpoints were accepted only with explicit v1 runtime and rejected for v2; matching metadata was accepted, mismatches were rejected, metadata restoration worked, and absent metadata with no runtime or v2 was rejected.
- Materialization smoke: a temporary legacy BLCS v1 dataset was copied to a separate `norm-v2` root with source preservation, normalized landmark values `x=0.461506...`, `y=1`, `z=0.090029...`, round-trip error `2.38e-7m`, and overwrite refusal. An additional legacy fixture with a valid scene header but no root `meta.json` exposed the implementation finding in `## RETURN implementation findings`.

## Canonical command results

- `preflight-regression`: **FAIL**, exit 1; `107 passed, 1 failed`. The failure is `tests/unit/tasks/plcs/test_configuration.py::test_generation_output_is_explicitly_data_root_relative`; `PLCSGenerationConfig.from_config` rejects `data/ACCAD/Female1Running_c3d` because the worktree `data/ACCAD` symlink resolves to `/home/kamimura/projects/tennis-lab/data/ACCAD`, outside the worktree data root. Exact log: `.codex/tasks/issue-786/logs/canonical-preflight-preflight-regression.log`.
- `precommit-all`: **FAIL**, exit 1. Ruff and the task-script reviewer passed. Mypy reported eight errors in unchanged files: three in `src/tasks/ball_detection/data/components/staged_sampler.py`, two in `src/automation/chatgpt_mcp/jobs.py`, one redundant cast in `src/synthetic_data_generation/dataset/plcs/execution.py`, and two in `tests/e2e/colab/test_training_path_contracts.py`. Exact log: `.codex/tasks/issue-786/logs/canonical-preflight-precommit-all.log`.
- The generated `.codex/tasks/issue-786/03-implementation/preflight-checks.json` binds both required results to the recomputed candidate fingerprint and records both required checks as FAIL.

## Baseline comparison

- The PLCS regression failure is environment/baseline classified, not a changed-file regression: the failing test and `src/tasks/plcs/configuration_contracts.py` are unchanged from the frozen base, while the unchanged path resolver correctly rejects the symlink escape. The worktree's `data/ACCAD` symlink target is outside the worktree.
- All eight mypy diagnostics are in unchanged paths and none is in the candidate's changed Python files. Ruff and the task-script reviewer are clean. These classifications do not convert the required canonical results to PASS; the gate remains open until the environment/baseline is made reproducible and the checks are rerun.
- Independently of those baseline/environment failures, the materialization smoke found a candidate defect: the validator explicitly permits an absent root header for legacy v1, but the materializer cannot publish such a source.

## Commands and exact outcomes

- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py candidate-fingerprint .codex/tasks/issue-786`: PASS; `sha256:0b141aeead98e5cfcdf04f55132ed10a43acb5c95e676ad5823061104904a7e6`.
- `git diff --check 59e3b166c2d010d5e62be52c2be76d98a94af0e0 -- . ':(exclude).codex/tasks'`: PASS.
- The bounded config-composition, unit round-trip, metadata-mutation, checkpoint-mismatch, materialization-smoke, and fixed-scale-inventory diagnostics above completed as recorded; all passed except the separately identified missing-root materialization case and the PLCS generation case blocked by the external symlink.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 preflight preflight-regression`: exact outcome **FAIL**, exit 1, 107 passed/1 failed.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 preflight precommit-all`: exact outcome **FAIL**, exit 1; Ruff/task-script reviewer passed, mypy failed with 8 errors.

## Final production preflight verdict

RETURN

## RETURN implementation findings

- **AC-015, AC-018 — missing legacy root metadata breaks materialization.** `validate_dataset_court_coordinate_contract` intentionally accepts a metadata-free v1 source when the root `meta.json` is absent, but `_write_contract_metadata` in `src/tasks/base/data/court_coordinate_materializer.py` unconditionally calls `_load_json_object(root_path)` (lines 312–319). A bounded fixture with `scenes/scene_0/meta.json` present, no root `meta.json`, and valid v1 BLCS arrays passes source validation and then fails during staging with `FileNotFoundError` for the staged root `meta.json`; no v2 artifact is published. Repair bundle: treat an absent validated legacy root document as `{}` before injecting the target contract, retain strict scene-header validation, add one regression case to the materialization smoke, then rerun both required preflight checks in a worktree-contained external-data environment and verify the root/scene metadata and round-trip evidence again.
