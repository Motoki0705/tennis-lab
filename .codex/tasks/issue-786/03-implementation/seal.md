# Final candidate seal

- Issue: #786
- Attempt: 2
- Test cycle: 2
- Seal cycle: 2
- Status: COMPLETE
- Frozen acceptance checklist SHA-256: `95bcebf4388fdba9773e3c538c9e22caf82b6e4a413ec1241e9a58b0c4483032`
- Base revision: `59e3b166c2d010d5e62be52c2be76d98a94af0e0`
- Candidate SHA-256: `sha256:30e9ef4b33bc6ffb35e756376de425b67b7c3c08f8d72f2875fd8042c1a5aea9`

## Candidate identity

The required first identity check was recomputed with `candidate-fingerprint
.codex/tasks/issue-786` and returned exactly
`sha256:30e9ef4b33bc6ffb35e756376de425b67b7c3c08f8d72f2875fd8042c1a5aea9`.
It equals `state.toml` `test_candidate_sha256`, the Candidate SHA-256 in
`tests.md`, and the candidate bound by every PASS result in `test-checks.json`.
The frozen base is `59e3b166c2d010d5e62be52c2be76d98a94af0e0`; candidate
fingerprinting excludes only `.codex/tasks/` workflow storage.

State records attempt 2, `preflight_cycle = 2`, `preflight_verdict = PASS`,
`test_cycle = 2`, `test_verdict = PASS`, `test_return_count = 0`, and legacy
Tester mode. The Tester PASS is therefore the authoritative post-test identity
for this seal. The merge commit `64ea1b5a99bacd5ec7f8ab4f356333835eaa9de9`
has parents `2661f3a80b56d5b2e1d44106162ba199cfaf45b0` and
`179dac756aef137c9a35b1025ce76f0a31023648`; the final Tester candidate is
`9b50abfdab08e7a4e1ed7b366cc6c980248538ff`.

## Changed-since-test inspection

`git status --short --untracked-files=all` is empty. The final candidate is
the Tester commit `9b50abfd` and `git log 9b50abfd..HEAD` is empty, so no
source, test, or fixture content was changed after Tester PASS. The only
post-test writes in this review are this owned artifact and manager-generated
seal result/raw-log files under `.codex/tasks/`; they are excluded from the
candidate fingerprint. Recomputing the fingerprint after each inspection
remains equal to the Tester identity.

The final post-test commit changed only `tests/integration/tasks/blcs/test_model_configs.py`,
`tests/unit/tasks/blcs/inference/test_tracking_predictor.py`,
`tests/unit/tasks/plcs/inference/test_tracking_predictor.py`, and workflow
evidence. No production or fixture path is in `HEAD^..HEAD`.

The PR-shaped `origin/main...HEAD` inventory is 268 paths: 11 frozen workflow
artifacts under `.codex/tasks/issue-786/`, 27 formal `knowledge/nodes` and
`knowledge/runs` evidence paths, 164 `src/**` production/configuration/docs
paths in the Issue's normalization/base/BLCS/PLCS/SLCS/scene/synthetic-data
scope, 65 `tests/**` unit/integration/e2e/fixture paths, and one `.gitignore`
training-artifact policy line. There are no other paths and no untracked
paths. The complete frozen-base inventory is 334 paths (323 non-workflow),
with upstream-main additions visible only in the frozen-base comparison; the
PR-shaped comparison is the delivery scope used here.

## Canonical command results

All ten required seal-stage checks from `02-planning/checks.json` are executed
through `manage_issue_task.py run-check .codex/tasks/issue-786 seal <check-id>`.
The resulting `seal-checks.json` is schema version 1, stage `seal`, and every
row must bind the exact Tester candidate above, the manifest argv/cwd/env, exit
code 0, and verdict PASS.

| Check ID | Required | Result |
|---|---:|---|
| `unit-contract` | yes | PASS, exit 0; 56 passed in 10.65s; invocation `sha256:3ed4ae83b3de4124b06a061a25a86d03951e6e2af05581134fdee52d26573d76` |
| `legacy-v1-checkpoint-parity` | yes | PASS, exit 0; 8 passed in 11.59s; invocation `sha256:f0fbee638ef03518f51cf6b653eff143032562501537fd1a3430fa218c864937` |
| `unit-blcs` | yes | PASS, exit 0; 39 passed in 16.82s; invocation `sha256:91c5453fbdb390f7c8cbb1295463de9d7082414a9651abb0e1a829f1e0c7b2d9` |
| `unit-plcs` | yes | PASS, exit 0; 27 passed in 10.79s; invocation `sha256:729094dc11098d6875294d4c0a8e329cd32e682e4a6fd08b462bbe911cdef0ae` |
| `unit-slcs` | yes | PASS, exit 0; 35 passed in 11.19s; invocation `sha256:f3730fcb1216ec4c1a68a737cf4ef7739e923c0ccace263857cd5cbc6889a21f` |
| `integration-normalization` | yes | PASS, exit 0; 14 passed in 12.33s; invocation `sha256:393e115292ad3ecd19c97c70d78bd6d9001264cfd88d03b7cfbf7126579bb61c` |
| `preflight-regression` | yes | PASS, exit 0; 127 passed in 11.45s; invocation `sha256:3a4023b90399d612468e51a82fa1078582859b7eb78a66691ebe55fba8eb3686` |
| `knowledge-graph` | yes | PASS, exit 0; 181 nodes, 0 errors, 4 warnings; invocation `sha256:fc39d3529b5dafe9405f680ef63607bb0bbde8d72e77c1b439cfae69879e8446` |
| `precommit-all` | yes | PASS, exit 0; Ruff, mypy, and task-script reviewer passed; invocation `sha256:a8c9e12dd2478c4cf43f586c9830d54e6b66c561ae1beb54ba56bd1bbfd2b61b` |
| `full-pytest` | yes | PASS, exit 0; 3331 passed, 78 skipped, 18 warnings in 846.62s; invocation `sha256:fedc9bb174546f8c9935561e4733f1372d07168fc0dd38846a0defd3ee52390b` |

Legacy Tester evidence is internally complete: `tests.md` has one PASS row for
all 22 ACs, records all ten test-stage canonical checks as PASS, and records
no AT probes because `state.toml` deliberately retains `adversarial_testing_mode
= LEGACY`. `test-checks.json` has all ten candidate-bound PASS rows. The
committed representative checkpoint names end in `.ckpt.bin`, not a forbidden
model suffix, and the fixture manifest binds both checkpoint/golden bytes and
the BLCS/PLCS dataset trees.

## Complete scope inspection

The Issue scope and frozen plan authorize the versioned normalization contract,
shared/base configuration and metadata gates, BLCS/PLCS/SLCS and tennis-scene
propagation, materialization, documentation, formal baseline evidence, and
mirrored tests/fixtures. Every PR-shaped path falls in one of those categories
or is the explicitly required repository artifact policy line. No third-party
tree, generated large output, unrelated task, or production checkpoint was
added.

The merge commit records exactly three conflict resolutions:

1. `src/tasks/blcs/configuration.py` retains the normalization resolver,
   v1/v2 beta and gravity helpers, strict normalization fields, while adding
   the upstream `blcs_track_query_ablation` typed parser and generator-boundary
   validation.
2. `tests/unit/tasks/blcs/inference/test_tracking_predictor.py` retains
   normalization v1/v2 physical-scale and checkpoint metadata assertions while
   retaining exact upstream ablation model/adapter dispatch.
3. `tests/unit/tasks/plcs/inference/test_tracking_predictor.py` retains the
   same v1/v2 scale and metadata assertions alongside exact upstream ablation
   model/adapter restoration.

The merge has zero `git ls-files -u` entries and no conflict markers in `src`
or `tests`. The bounded F-001 repair is test-only and typing-only: in
`tests/unit/tasks/blcs/models/test_blcs_track_query_ablation_model.py`,
`cast("object", BLCSTrackQueryAblationModel)` preserves the exact distinct
class-identity assertion while satisfying strict mypy. No production or
fixture behavior changed for that repair.

Repository/test/script boundaries are satisfied by the canonical `precommit-all`
and `full-pytest` PASS results. The fixture tree contains no path ending in
`.ckpt`, `.pt`, `.pth`, `.pkl`, `.pickle`, `.onnx`, or `.safetensors`; the
PR-safe `.ckpt.bin` names are the only checkpoint-like committed fixture
names. `git diff --check` is clean for the complete non-workflow diff.
The sole non-normalization `src/**` path in the PR-shaped inventory,
`src/automation/chatgpt_mcp/jobs.py`, contains only the narrow mypy typing
annotations carried by the repository-rule repair; it has no behavior change
and is covered by the passing all-files hook.

The direct `git diff --check` over the PR-shaped tree reports one blank line at
EOF in the generated frozen `.codex/tasks/issue-786/issue.md`; the complete
non-workflow source/test/fixture diff is clean. This is frozen workflow
rendering, not a candidate content change or a seal-stage defect.

## Commands and exact outcomes

- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py candidate-fingerprint .codex/tasks/issue-786` → `sha256:30e9ef4b33bc6ffb35e756376de425b67b7c3c08f8d72f2875fd8042c1a5aea9`.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal unit-contract` → PASS, exit 0; 56 passed.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal legacy-v1-checkpoint-parity` → PASS, exit 0; 8 passed.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal unit-blcs` → PASS, exit 0; 39 passed.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal unit-plcs` → PASS, exit 0; 27 passed.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal unit-slcs` → PASS, exit 0; 35 passed.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal integration-normalization` → PASS, exit 0; 14 passed.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal preflight-regression` → PASS, exit 0; 127 passed.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal knowledge-graph` → PASS, exit 0; 181 nodes, 0 errors, 4 warnings.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal precommit-all` → PASS, exit 0; Ruff, mypy, and task-script reviewer passed.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal full-pytest` → PASS, exit 0; 3331 passed, 78 skipped, 18 warnings.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py artifact-check .codex/tasks/issue-786 seal` → `ok`.

## Final candidate seal verdict

PASS

## RETURN implementation findings

None
