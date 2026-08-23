# Final candidate seal

- Issue: #786
- Attempt: 1
- Test cycle: 2
- Status: COMPLETE
- Frozen acceptance checklist SHA-256: `95bcebf4388fdba9773e3c538c9e22caf82b6e4a413ec1241e9a58b0c4483032`
- Candidate SHA-256: `sha256:12a84769a34297c0a735714ff507dde2523bbdda6465f5e12a7df9c4d7521d40`

## Candidate identity

- Frozen base revision: `59e3b166c2d010d5e62be52c2be76d98a94af0e0`.
- The Tester PASS candidate recorded in `state.toml`, `tests.md`, and `test-checks.json` is exactly `sha256:12a84769a34297c0a735714ff507dde2523bbdda6465f5e12a7df9c4d7521d40`.
- Independent recomputation before seal checks: `python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py candidate-fingerprint .codex/tasks/issue-786` -> `sha256:12a84769a34297c0a735714ff507dde2523bbdda6465f5e12a7df9c4d7521d40`.
- The candidate equals the Tester PASS identity; no identity mismatch was found.

## Changed-since-test inspection

- The recomputed fingerprint matched the Tester PASS fingerprint before canonical seal execution and after all nine seal checks.
- `git status --porcelain=v2 --untracked-files=all` was empty. `git ls-files --others --exclude-standard` was also empty.
- The complete base-to-candidate diff has 232 tracked paths (7,976 insertions and 456 deletions); 221 are candidate content paths and 11 are workflow artifacts under `.codex/tasks/issue-786/`. No untracked path is present.
- No source, configuration, documentation, knowledge, test, plan, Issue, state, or other candidate content changed during this seal review. The only authored artifact is this replacement of `seal.md`; generated `seal-checks.json` and canonical seal logs are the permitted stage evidence.
- `git diff --check` is clean for all non-workflow candidate content. Its only output is the generated frozen Issue file's trailing blank line at `.codex/tasks/issue-786/issue.md:148`; the frozen Issue was not modified.

## Canonical command results

All commands below were executed through the manifest-authoritative form `python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal <check-id>`. Every result has exit code 0, verdict `PASS`, the Tester candidate fingerprint, and a present raw log in `.codex/tasks/issue-786/logs/`.

| Seal check ID | Exact outcome | Machine log |
|---|---|---|
| `unit-contract` | PASS — 56 passed in 9.05s | `logs/canonical-seal-unit-contract.log` |
| `unit-blcs` | PASS — 33 passed in 9.37s | `logs/canonical-seal-unit-blcs.log` |
| `unit-plcs` | PASS — 25 passed in 9.20s | `logs/canonical-seal-unit-plcs.log` |
| `unit-slcs` | PASS — 35 passed in 9.53s | `logs/canonical-seal-unit-slcs.log` |
| `integration-normalization` | PASS — 14 passed in 10.27s | `logs/canonical-seal-integration-normalization.log` |
| `preflight-regression` | PASS — 125 passed in 10.85s | `logs/canonical-seal-preflight-regression.log` |
| `knowledge-graph` | PASS — 181 nodes, 0 errors, 4 warnings | `logs/canonical-seal-knowledge-graph.log` |
| `precommit-all` | PASS — ruff, mypy, task-script-reviewer | `logs/canonical-seal-precommit-all.log` |
| `full-pytest` | PASS — 3,230 passed, 53 skipped, 19 warnings in 761.31s | `logs/canonical-seal-full-pytest.log` |

The nine required manifest IDs are exactly the nine results in `seal-checks.json`; all result candidate values equal the Tester PASS candidate.

## Complete scope inspection

- Scope authority is the frozen Issue, the parent plan, and `checks.json`. The complete `git diff --name-status 59e3b166c2d010d5e62be52c2be76d98a94af0e0` inventory was inspected. The 221 non-workflow paths are listed below in full.
- `.gitignore` adds only the repository's generated `artifacts/` exclusion needed to keep training outputs out of the versioned evidence tree.
- The 27 `knowledge/` paths are the four v1/v2 baseline run bundles and comparison group required by AC-019, with knowledge validation passing.
- The 164 `src/` paths are limited to the shared normalization/schema/materialization/configuration boundary, BLCS, PLCS (including compact synthetic-data PLCS), SLCS, tennis-scene integration, geometry/schema, and behavior-preserving repository typing-boundary fixes. The two ancillary typing-only files (`src/automation/chatgpt_mcp/jobs.py` and `src/tasks/ball_detection/data/components/staged_sampler.py`) contain no behavior or interface change and were covered by the repository-wide hooks.
- The 29 `tests/` paths are all test additions or modifications under the approved mirrored unit/e2e locations and the single normalization integration smoke. No test path is outside `tests/`.
- The 11 workflow paths are task artifacts only and are excluded from the candidate fingerprint; they are not production or test scope.

Workflow artifact inventory (all task-local and excluded from the candidate fingerprint):

```text
A .codex/tasks/issue-786/00-feasibility/feasibility.md
A .codex/tasks/issue-786/01-exploration/exploration.md
A .codex/tasks/issue-786/02-planning/plan.md
A .codex/tasks/issue-786/03-implementation/implementation.md
A .codex/tasks/issue-786/03-implementation/preflight.md
A .codex/tasks/issue-786/03-implementation/seal.md
A .codex/tasks/issue-786/03-implementation/tests.md
A .codex/tasks/issue-786/04-validation/validation.md
A .codex/tasks/issue-786/05-packaging/packaging.md
A .codex/tasks/issue-786/issue.md
A .codex/tasks/issue-786/state.toml
```

Complete non-workflow path inventory (status and path):

```text
M .gitignore
A knowledge/nodes/group-i786-normalization-v1-v2.md
A knowledge/nodes/run-i786-blcs-norm-v1-b64-w16.md
A knowledge/nodes/run-i786-blcs-norm-v2-b64-w16.md
A knowledge/nodes/run-i786-plcs-norm-v1.md
A knowledge/nodes/run-i786-plcs-v2-resume-b24-r2.md
A knowledge/runs/run-i786-blcs-norm-v1-b64-w16/curves.png
A knowledge/runs/run-i786-blcs-norm-v1-b64-w16/git_status.txt
A knowledge/runs/run-i786-blcs-norm-v1-b64-w16/repro.sh
A knowledge/runs/run-i786-blcs-norm-v1-b64-w16/run.json
A knowledge/runs/run-i786-blcs-norm-v1-b64-w16/uncommitted.patch
A knowledge/runs/run-i786-blcs-norm-v2-b64-w16/curves.png
A knowledge/runs/run-i786-blcs-norm-v2-b64-w16/git_status.txt
A knowledge/runs/run-i786-blcs-norm-v2-b64-w16/metrics.json
A knowledge/runs/run-i786-blcs-norm-v2-b64-w16/pred_test.npz
A knowledge/runs/run-i786-blcs-norm-v2-b64-w16/repro.sh
A knowledge/runs/run-i786-blcs-norm-v2-b64-w16/run.json
A knowledge/runs/run-i786-blcs-norm-v2-b64-w16/uncommitted.patch
A knowledge/runs/run-i786-plcs-norm-v1/curves.png
A knowledge/runs/run-i786-plcs-norm-v1/git_status.txt
A knowledge/runs/run-i786-plcs-norm-v1/repro.sh
A knowledge/runs/run-i786-plcs-norm-v1/run.json
A knowledge/runs/run-i786-plcs-norm-v1/uncommitted.patch
A knowledge/runs/run-i786-plcs-v2-resume-b24-r2/curves.png
A knowledge/runs/run-i786-plcs-v2-resume-b24-r2/git_status.txt
A knowledge/runs/run-i786-plcs-v2-resume-b24-r2/repro.sh
A knowledge/runs/run-i786-plcs-v2-resume-b24-r2/run.json
A knowledge/runs/run-i786-plcs-v2-resume-b24-r2/uncommitted.patch
M src/automation/chatgpt_mcp/jobs.py
M src/synthetic_data_generation/dataset/plcs/README.md
M src/synthetic_data_generation/dataset/plcs/assembler.py
M src/synthetic_data_generation/dataset/plcs/execution.py
M src/synthetic_data_generation/dataset/plcs/handler.py
M src/synthetic_data_generation/dataset/plcs/validation.py
M src/tasks/ball_detection/data/components/staged_sampler.py
M src/tasks/base/__init__.py
A src/tasks/base/configs/__init__.py
A src/tasks/base/configs/court_coordinate_normalization/v1.yaml
A src/tasks/base/configs/court_coordinate_normalization/v2.yaml
A src/tasks/base/configs/materialize_court_coordinate_normalization.yaml
M src/tasks/base/configuration.py
M src/tasks/base/data/__init__.py
A src/tasks/base/data/court_coordinate_contract.py
A src/tasks/base/data/court_coordinate_materializer.py
M src/tasks/base/data/dataset_writer.py
M src/tasks/base/data/scene_dataset.py
M src/tasks/base/model_io/__init__.py
A src/tasks/base/model_io/court_coordinate_contract.py
A src/tasks/base/scripts/__init__.py
A src/tasks/base/scripts/materialize_court_coordinate_normalization.py
M src/tasks/blcs/README.md
M src/tasks/blcs/configs/api_server.yaml
A src/tasks/blcs/configs/data/broadcast_norm_v1.yaml
A src/tasks/blcs/configs/data/broadcast_norm_v2.yaml
M src/tasks/blcs/configs/generate_dataset.yaml
M src/tasks/blcs/configs/loss/tracking.yaml
M src/tasks/blcs/configs/preview_augmentation.yaml
M src/tasks/blcs/configs/run/generate_dataset.yaml
M src/tasks/blcs/configs/run/train.yaml
M src/tasks/blcs/configs/train.yaml
M src/tasks/blcs/configs/train_chunked.yaml
M src/tasks/blcs/configs/train_chunked_gan.yaml
A src/tasks/blcs/configs/train_normalization_v1.yaml
A src/tasks/blcs/configs/train_normalization_v2.yaml
M src/tasks/blcs/configs/train_tracking.yaml
M src/tasks/blcs/configs/train_tracking_chunked.yaml
M src/tasks/blcs/configs/training/default.yaml
M src/tasks/blcs/configs/visualize.yaml
M src/tasks/blcs/configuration.py
M src/tasks/blcs/data/chunk_manager.py
M src/tasks/blcs/data/dataset.py
M src/tasks/blcs/data/tracking_dataset.py
M src/tasks/blcs/data/tracking_types.py
M src/tasks/blcs/data/types.py
A src/tasks/blcs/data/visibility.py
M src/tasks/blcs/generate_dataset/api_server/service.py
M src/tasks/blcs/generate_dataset/config.py
M src/tasks/blcs/generate_dataset/io/dataset_io.py
M src/tasks/blcs/generate_dataset/scene_generator.py
M src/tasks/blcs/generate_dataset/simulation/ball_physics.py
M src/tasks/blcs/generate_dataset/simulation/rally_simulator.py
M src/tasks/blcs/generate_dataset/utils/parallel_runner.py
M src/tasks/blcs/inference/predictor.py
M src/tasks/blcs/inference/tracking_predictor.py
M src/tasks/blcs/model_io/adapters.py
M src/tasks/blcs/model_io/checkpoints.py
M src/tasks/blcs/model_io/contracts.py
M src/tasks/blcs/models/components/differentiable_projection.py
M src/tasks/blcs/scripts/generate_dataset.py
M src/tasks/blcs/training/lightning_module.py
M src/tasks/blcs/training/losses.py
M src/tasks/blcs/training/metrics.py
M src/tasks/blcs/training/runner.py
M src/tasks/blcs/training/tracking_lightning_module.py
M src/tasks/blcs/training/tracking_losses.py
M src/tasks/blcs/training/tracking_matching.py
M src/tasks/blcs/training/tracking_metrics.py
M src/tasks/blcs/visualization/api/predict.py
M src/tasks/blcs/visualization/io/scene.py
M src/tasks/blcs/visualization/orchestrator.py
M src/tasks/plcs/README.md
M src/tasks/plcs/configs/analyze_angle_velocity.yaml
M src/tasks/plcs/configs/analyze_dataset_distribution.yaml
M src/tasks/plcs/configs/analyze_loss_dominance.yaml
M src/tasks/plcs/configs/analyze_rotation_error_samples.yaml
A src/tasks/plcs/configs/court_coordinate_normalization/v1.yaml
A src/tasks/plcs/configs/court_coordinate_normalization/v2.yaml
A src/tasks/plcs/configs/data/multiview_sequence_norm_v1.yaml
A src/tasks/plcs/configs/data/multiview_sequence_norm_v2.yaml
M src/tasks/plcs/configs/generate_dataset.yaml
A src/tasks/plcs/configs/generate_dataset_norm_v1.yaml
A src/tasks/plcs/configs/generate_dataset_norm_v2.yaml
M src/tasks/plcs/configs/loss/_base.yaml
M src/tasks/plcs/configs/loss/tracking.yaml
M src/tasks/plcs/configs/preview_augmentation.yaml
M src/tasks/plcs/configs/train.yaml
M src/tasks/plcs/configs/train_chunked.yaml
M src/tasks/plcs/configs/train_chunked_gan.yaml
A src/tasks/plcs/configs/train_norm_v1.yaml
A src/tasks/plcs/configs/train_norm_v2.yaml
M src/tasks/plcs/configs/train_tracking.yaml
M src/tasks/plcs/configs/train_tracking_chunked.yaml
M src/tasks/plcs/configs/visualize.yaml
M src/tasks/plcs/configuration.py
M src/tasks/plcs/data/chunk_manager.py
M src/tasks/plcs/data/dataset.py
M src/tasks/plcs/data/targets.py
M src/tasks/plcs/data/tracking_dataset.py
M src/tasks/plcs/generate_dataset/config.py
M src/tasks/plcs/generate_dataset/io/dataset_io.py
M src/tasks/plcs/generate_dataset/io/scene_loader.py
M src/tasks/plcs/generate_dataset/scene_generator.py
M src/tasks/plcs/inference/predictor.py
M src/tasks/plcs/inference/tracking_predictor.py
M src/tasks/plcs/model_io/__init__.py
A src/tasks/plcs/model_io/court_coordinate_checkpoint.py
M src/tasks/plcs/scripts/analysis/analyze_angle_velocity.py
M src/tasks/plcs/scripts/analysis/analyze_dataset_distribution.py
M src/tasks/plcs/scripts/analysis/analyze_loss_dominance.py
M src/tasks/plcs/scripts/analysis/visualize_rotation_error_samples.py
M src/tasks/plcs/scripts/generate_dataset.py
M src/tasks/plcs/training/lightning_module.py
M src/tasks/plcs/training/losses.py
M src/tasks/plcs/training/metrics.py
M src/tasks/plcs/training/runner.py
M src/tasks/plcs/training/tracking_lightning_module.py
M src/tasks/plcs/training/tracking_losses.py
M src/tasks/plcs/training/tracking_matching.py
M src/tasks/plcs/training/tracking_metrics.py
M src/tasks/plcs/visualization/adapters/render_inputs.py
M src/tasks/plcs/visualization/api/predict.py
M src/tasks/plcs/visualization/io/scene.py
M src/tasks/plcs/visualization/orchestrator.py
M src/tasks/plcs/visualization/rendering/scene_renderer.py
M src/tasks/slcs/README.md
M src/tasks/slcs/configs/evaluate.yaml
M src/tasks/slcs/configs/predict_clip.yaml
M src/tasks/slcs/configs/run/train.yaml
M src/tasks/slcs/configs/train.yaml
M src/tasks/slcs/configuration.py
M src/tasks/slcs/data/annotation.py
M src/tasks/slcs/data/dataset.py
M src/tasks/slcs/data/types.py
M src/tasks/slcs/evaluation/evaluate.py
M src/tasks/slcs/inference/predictor.py
M src/tasks/slcs/model_io/__init__.py
M src/tasks/slcs/model_io/adapter.py
A src/tasks/slcs/model_io/checkpoints.py
M src/tasks/slcs/model_io/factory.py
A src/tasks/slcs/normalization.py
M src/tasks/slcs/scripts/evaluate.py
M src/tasks/slcs/scripts/predict_clip.py
M src/tasks/slcs/scripts/train.py
M src/tasks/slcs/training/lightning_module.py
M src/tasks/slcs/training/metrics.py
M src/tasks/slcs/training/runner.py
M src/tennis_scene/README.md
M src/tennis_scene/configs/generate_dataset.yaml
M src/tennis_scene/configs/pipeline.yaml
M src/tennis_scene/configuration.py
M src/tennis_scene/generate_dataset/README.md
M src/tennis_scene/pipeline/components/blcs.py
M src/tennis_scene/pipeline/components/plcs.py
M src/tennis_scene/pipeline/orchestrator.py
M src/tennis_scene/schema.py
M src/tennis_scene/scripts/generate_dataset.py
M src/tennis_scene/scripts/run_pipeline.py
M src/utils/configuration/inventory.py
M src/utils/geometry/court_pose.py
M src/utils/schema/__init__.py
M src/utils/schema/court.py
A src/utils/schema/court_normalization.py
M tests/e2e/colab/test_training_path_contracts.py
M tests/e2e/synthetic_data_generation/test_b00_gpu_acceptance.py
M tests/e2e/synthetic_data_generation/test_removed_architecture.py
A tests/integration/tasks/test_court_coordinate_normalization_smoke.py
A tests/unit/tasks/base/data/test_court_coordinate_contract.py
A tests/unit/tasks/base/model_io/test_court_coordinate_contract.py
A tests/unit/tasks/blcs/data/test_dataset.py
M tests/unit/tasks/blcs/data/test_tracking_dataset.py
A tests/unit/tasks/blcs/data/test_visibility.py
A tests/unit/tasks/blcs/generate_dataset/simulation/test_ball_physics.py
M tests/unit/tasks/blcs/inference/test_predictor.py
M tests/unit/tasks/blcs/inference/test_tracking_predictor.py
M tests/unit/tasks/blcs/model_io/test_adapters.py
M tests/unit/tasks/blcs/model_io/test_checkpoints.py
M tests/unit/tasks/blcs/training/test_losses.py
M tests/unit/tasks/blcs/training/test_tracking_matching.py
M tests/unit/tasks/blcs/training/test_tracking_metrics.py
M tests/unit/tasks/plcs/data/test_targets.py
A tests/unit/tasks/plcs/generate_dataset/test_scene_generator.py
M tests/unit/tasks/plcs/inference/test_predictor.py
M tests/unit/tasks/plcs/inference/test_tracking_predictor.py
M tests/unit/tasks/plcs/training/test_losses.py
M tests/unit/tasks/slcs/inference/test_predictor.py
M tests/unit/tasks/slcs/model_io/test_adapter.py
A tests/unit/tasks/slcs/training/test_metrics.py
A tests/unit/tennis_scene/test_schema.py
M tests/unit/utils/configuration/test_contracts.py
M tests/unit/utils/geometry/test_court_pose.py
A tests/unit/utils/schema/test_court_normalization.py
```

Evidence completeness checks:

- Frozen Issue, plan, and Tester evidence each contain exactly 22 ordered AC rows. `tests.md` reports PASS for AC-001 through AC-022, records the final candidate, and reports no final-cycle failure.
- `checks.json` contains nine required checks authorized for `seal`; `seal-checks.json` contains all nine IDs exactly once, all PASS, all exit code 0, and all candidate-bound to the Tester identity.
- Repository guidance requires the `.venv`, pytest, Ruff, mypy, and task-script reviewer paths. `precommit-all` passed all three hooks, and `full-pytest` passed the GPU-visible repository baseline.
- The knowledge graph check reports 181 nodes and zero errors; its four warnings are non-failing warnings from the canonical validator.
- No implementation or preflight narrative was used as a conclusion, and no new semantic mutation, fuzzing, architecture exploration, or production-readiness campaign was performed during Seal.

## Commands and exact outcomes

```text
.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py candidate-fingerprint .codex/tasks/issue-786
sha256:12a84769a34297c0a735714ff507dde2523bbdda6465f5e12a7df9c4d7521d40

.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal unit-contract
PASS (exit 0; 56 passed)
.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal unit-blcs
PASS (exit 0; 33 passed)
.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal unit-plcs
PASS (exit 0; 25 passed)
.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal unit-slcs
PASS (exit 0; 35 passed)
.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal integration-normalization
PASS (exit 0; 14 passed)
.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal preflight-regression
PASS (exit 0; 125 passed)
.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal knowledge-graph
PASS (exit 0; 181 nodes, 0 errors)
.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal precommit-all
PASS (exit 0; Ruff, mypy, task-script-reviewer)
.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal full-pytest
PASS (exit 0; 3230 passed, 53 skipped, 19 warnings)

.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py artifact-check .codex/tasks/issue-786 seal
PASS
```

## Final candidate seal verdict

PASS

## RETURN implementation findings

None
