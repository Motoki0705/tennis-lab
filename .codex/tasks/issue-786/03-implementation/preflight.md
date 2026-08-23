# Production preflight

- Issue: #786
- Attempt: 1
- Test cycle: 1
- Status: COMPLETE
- Candidate SHA-256: `sha256:5e1143f697800fbf6958b465f5e4d080088037c019956be639845cc4741f04f3`

## Candidate identity

- Review mode: **Closure**. `state.toml` records the first Preflight `RETURN` for `preflight_cycle = 1` and the pending test cycle; this review is restricted to the frozen RETURN findings, canonical preflight checks, and direct repair-local regressions.
- Branch/head: `feat/issue-786-normalization-v2` / `ed8415a4a5e63367436f3755ba64b05fde2640c1`.
- Frozen base: `59e3b166c2d010d5e62be52c2be76d98a94af0e0`.
- Recomputed candidate fingerprint: `sha256:5e1143f697800fbf6958b465f5e4d080088037c019956be639845cc4741f04f3` (matches `implementation.md` and both generated canonical results).
- Complete candidate diff against the frozen base is 197 paths, 5,420 additions, and 441 deletions, excluding `.codex/tasks/` workflow artifacts.

## Changed scope

- The frozen implementation repair adds the validated-legacy-root branch in `src/tasks/base/data/court_coordinate_materializer.py`, preserving strict scene-header validation and non-overwrite publication.
- The frozen environment repair makes the licensed ACCAD tree worktree-contained for the PLCS path regression; it does not alter the path-contract implementation.
- The frozen mypy repair is limited to type narrowing/casts and targeted decorator annotations in the four paths named by the prior Preflight finding. No reviewer production or test changes were made.

## Deterministic policy checks

- `candidate-fingerprint .codex/tasks/issue-786`: PASS; the recomputed identity above is current.
- `git diff --check 59e3b166c2d010d5e62be52c2be76d98a94af0e0 -- . ':(exclude).codex/tasks'`: PASS; no whitespace errors.
- State, the prior `preflight.md`, the frozen plan/check manifest, implementation handoff, repository guidance, and the complete candidate diff were read before closure review. No discovery category was added.

## Focused checks

- **Missing-root legacy-v1 materialization (AC-015, AC-018):** A temporary BLCS fixture with no root `meta.json`, a metadata-free legacy-v1 scene header, and valid `ball_pos_world.npy`/`ball_pos_norm.npy` was accepted only under v1 and materialized into a separate `norm-v2` root. The published root and scene headers both carry identical v2 metadata, including scale `(11.885, 11.885, 11.885)` and units. Physical reconstruction error was `1.1920929e-07m` (at most `1e-5m`). Source world/normalized arrays, source scene header, and an unrelated source file were byte/content-preserved. Repeating the operation refused overwrite. An explicit missing scene path and a scene directory missing `meta.json` were rejected before output publication.
- **Worktree-contained ACCAD path regression:** `data/ACCAD` resolves to `/home/kamimura/projects/tennis-lab/.claude/worktrees/issue-786-normalization-v2/data/ACCAD`, is not a symlink, and is contained below the worktree data root. The former PLCS generation path case and the complete direct repair-local test set pass.
- **Exact prior mypy finding set and repair-local regressions:** The frozen RETURN reported eight diagnostics: three in `src/tasks/ball_detection/data/components/staged_sampler.py`, two in `src/automation/chatgpt_mcp/jobs.py`, one in `src/synthetic_data_generation/dataset/plcs/execution.py`, and two in `tests/e2e/colab/test_training_path_contracts.py`. The repaired direct suites (`test_staged_sampler.py`, `test_jobs.py`, `test_execution.py`, `test_training_path_contracts.py`, and `tests/unit/tasks/plcs/test_configuration.py`) passed `58` tests.

## Canonical command results

- `preflight-regression`: **PASS**, exit 0; `108 passed`. The generated result is bound to candidate `sha256:5e1143f697800fbf6958b465f5e4d080088037c019956be639845cc4741f04f3` and is recorded in `logs/canonical-preflight-preflight-regression.log`.
- `precommit-all`: **PASS**, exit 0; Ruff, mypy, and task-script reviewer each passed. The generated result is bound to the same candidate and is recorded in `logs/canonical-preflight-precommit-all.log`.
- `preflight-checks.json` contains both required preflight-stage checks, each with exit code 0, current candidate identity, invocation digest, and verdict `PASS`.

## Baseline comparison

- The prior candidate fingerprint `sha256:0b141aeead98e5cfcdf04f55132ed10a43acb5c95e676ad5823061104904a7e6` had the frozen PLCS ACCAD path failure (`107 passed, 1 failed`) and the eight baseline mypy diagnostics. The current candidate fingerprint is `sha256:5e1143f697800fbf6958b465f5e4d080088037c019956be639845cc4741f04f3`; the same canonical checks now pass and the direct repair-local suites pass.
- The prior AC-015/AC-018 missing-root materialization failure is closed: a validated metadata-free v1 source with an absent root header now publishes v2 root/scene metadata while preserving source data, enforcing the physical round trip, refusing overwrite, and rejecting missing scene metadata.
- No frozen closure finding remains open, and no materially new category was encountered.

## Commands and exact outcomes

- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py candidate-fingerprint .codex/tasks/issue-786`: PASS; `sha256:5e1143f697800fbf6958b465f5e4d080088037c019956be639845cc4741f04f3`.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 preflight preflight-regression`: PASS; exit 0, `108 passed`.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 preflight precommit-all`: PASS; exit 0, Ruff/mypy/task-script reviewer passed.
- `.venv/bin/python -m pytest -q tests/unit/tasks/ball_detection/data/test_staged_sampler.py tests/unit/automation/chatgpt_mcp/test_jobs.py tests/unit/synthetic_data_generation/dataset/plcs/test_execution.py tests/e2e/colab/test_training_path_contracts.py tests/unit/tasks/plcs/test_configuration.py`: PASS; `58 passed in 12.17s`.
- Bounded temporary-fixture materialization diagnostic: PASS; missing-root legacy-v1 source produced matching root/scene v2 metadata, `1.1920929e-07m` physical round-trip error, source preservation, overwrite refusal, and missing-scene rejection.

## Final production preflight verdict

PASS

## RETURN implementation findings

None
