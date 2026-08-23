# Final candidate seal

- Issue: #786
- Attempt: 2
- Test cycle: 1
- Status: COMPLETE
- Frozen acceptance checklist SHA-256: `95bcebf4388fdba9773e3c538c9e22caf82b6e4a413ec1241e9a58b0c4483032`
- Candidate SHA-256: `sha256:ee83a05ecdca681a0e926948b581abe63aef5de6d7dd218ba93a108b057063a0`

## Candidate identity

- Frozen base revision: `59e3b166c2d010d5e62be52c2be76d98a94af0e0`.
- The Tester PASS binding in `state.toml`, `tests.md`, and `test-checks.json` is exactly `sha256:ee83a05ecdca681a0e926948b581abe63aef5de6d7dd218ba93a108b057063a0`.
- Independent recomputation before seal execution returned `sha256:ee83a05ecdca681a0e926948b581abe63aef5de6d7dd218ba93a108b057063a0`; the post-check recomputation returned the same value.
- Candidate identity therefore equals the Tester PASS candidate. This seal records identity/evidence only; no source or test repair was made.

## Changed-since-test inspection

- At review entry, the candidate fingerprint matched the Tester PASS binding. After every required seal-stage command, it still matched exactly.
- Before authoring this artifact, `git status --porcelain=v2 --untracked-files=all` was empty and `git ls-files --others --exclude-standard` was empty. After authoring, the only non-ignored worktree change is this replacement of `03-implementation/seal.md`.
- The complete base-to-candidate path inventory below was inspected. No source, configuration, documentation, knowledge, test, fixture, plan, frozen Issue, or state content changed during this seal review. Canonical `seal-checks.json` and `canonical-seal-*.log` files are the permitted generated seal evidence under `.codex/tasks/issue-786/`; they are excluded from the candidate fingerprint.
- Canonical checks were run without source/test/fixture edits. The final candidate fingerprint remains the Tester binding, proving no candidate-content change since Tester PASS.

## Canonical command results

Each required check was executed via the manifest-authoritative command:
`.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal <check-id>`.
The generated `seal-checks.json` has stage `seal`, exactly ten required results, and binds every result to the Tester candidate. Every result has exit code 0 and verdict `PASS`.

| Seal check ID | Exact outcome | Machine log |
|---|---|---|
| `unit-contract` | 56 passed in 18.15s | `logs/canonical-seal-unit-contract.log` |
| `legacy-v1-checkpoint-parity` | 7 passed in 10.25s | `logs/canonical-seal-legacy-v1-checkpoint-parity.log` |
| `unit-blcs` | 37 passed in 9.99s | `logs/canonical-seal-unit-blcs.log` |
| `unit-plcs` | 25 passed in 9.99s | `logs/canonical-seal-unit-plcs.log` |
| `unit-slcs` | 35 passed in 10.36s | `logs/canonical-seal-unit-slcs.log` |
| `integration-normalization` | 14 passed in 10.55s | `logs/canonical-seal-integration-normalization.log` |
| `preflight-regression` | 127 passed in 10.45s | `logs/canonical-seal-preflight-regression.log` |
| `knowledge-graph` | 181 nodes, 0 errors, 4 pre-existing warnings | `logs/canonical-seal-knowledge-graph.log` |
| `precommit-all` | Ruff, mypy, and task-script-reviewer all passed | `logs/canonical-seal-precommit-all.log` |
| `full-pytest` | 3245 passed, 53 skipped, 19 warnings in 672.82s | `logs/canonical-seal-full-pytest.log` |

The complete evidence set is candidate-bound: all ten machine result records contain the exact Tester SHA-256, matching invocation digests from `checks.json`, exit code 0, and present raw logs.

## Complete scope inspection

The frozen Issue, parent plan, `checks.json`, repository guidance, current code, and complete diff from the frozen base were inspected. The approved-scope conclusion is:

- The diff contains 266 tracked paths relative to the frozen base: 11 workflow paths under `.codex/tasks/issue-786/` and 255 candidate-content paths.
- The 27 `knowledge/` paths are the four v1/v2 experiment bundles and comparison node required by AC-019; `knowledge-graph` validates them with 0 errors.
- The 164 `src/` paths are within the shared court-normalization contract/materializer/configuration boundary, BLCS, PLCS (including synthetic-data PLCS), SLCS, tennis-scene integration, court geometry/schema, and repository-rule compatibility. The two ancillary source paths (`src/automation/chatgpt_mcp/jobs.py` and `src/tasks/ball_detection/data/components/staged_sampler.py`) contain only typing-boundary annotations/casts required by the repository hooks and no behavior/interface change.
- The 63 `tests/` paths are independent Test Writer additions or modifications under the approved mirrored unit/e2e locations, the committed frozen-base legacy fixtures, and the normalization integration smoke. No test path is outside `tests/`, and no production implementation was added under a test path.
- `.gitignore` adds only the `artifacts/` exclusion for generated training outputs. It does not alter runtime behavior or evidence content.
- Repository boundaries are satisfied: production changes remain under the approved task/shared integration areas, test changes remain test-only, the generated knowledge evidence is under `knowledge/`, and the only authored seal-stage file is this artifact. No untracked non-ignored path exists.

The exact workflow inventory is:

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

The exact non-workflow inventory (status and path), inspected against the Issue scope and plan's planned directories/symbols, is:

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
A tests/fixtures/issue_786/legacy_v1_representative/blcs_representative_legacy_v1.ckpt
A tests/fixtures/issue_786/legacy_v1_representative/blcs_representative_legacy_v1_golden.npz
A tests/fixtures/issue_786/legacy_v1_representative/datasets/blcs_legacy_v1/scenes/scene_representative_blcs/ball_pos_norm.npy
A tests/fixtures/issue_786/legacy_v1_representative/datasets/blcs_legacy_v1/scenes/scene_representative_blcs/ball_pos_world.npy
A tests/fixtures/issue_786/legacy_v1_representative/datasets/blcs_legacy_v1/scenes/scene_representative_blcs/ball_vel_world.npy
A tests/fixtures/issue_786/legacy_v1_representative/datasets/blcs_legacy_v1/scenes/scene_representative_blcs/cam_0_ball_uv.npy
A tests/fixtures/issue_786/legacy_v1_representative/datasets/blcs_legacy_v1/scenes/scene_representative_blcs/cam_0_ball_vis.npy
A tests/fixtures/issue_786/legacy_v1_representative/datasets/blcs_legacy_v1/scenes/scene_representative_blcs/cam_0_ball_visibility_ratio.npy
A tests/fixtures/issue_786/legacy_v1_representative/datasets/blcs_legacy_v1/scenes/scene_representative_blcs/cam_0_court_kp_uv.npy
A tests/fixtures/issue_786/legacy_v1_representative/datasets/blcs_legacy_v1/scenes/scene_representative_blcs/cam_0_court_kp_vis.npy
A tests/fixtures/issue_786/legacy_v1_representative/datasets/blcs_legacy_v1/scenes/scene_representative_blcs/cam_0_court_visibility_count.npy
A tests/fixtures/issue_786/legacy_v1_representative/datasets/blcs_legacy_v1/scenes/scene_representative_blcs/meta.json
A tests/fixtures/issue_786/legacy_v1_representative/datasets/blcs_legacy_v1/scenes/scene_representative_blcs/scalars.json
A tests/fixtures/issue_786/legacy_v1_representative/datasets/blcs_legacy_v1/test.txt
A tests/fixtures/issue_786/legacy_v1_representative/datasets/plcs_legacy_v1/scenes/scene_representative_plcs/cam_0_court_kp_uv.npy
A tests/fixtures/issue_786/legacy_v1_representative/datasets/plcs_legacy_v1/scenes/scene_representative_plcs/cam_0_court_kp_vis.npy
A tests/fixtures/issue_786/legacy_v1_representative/datasets/plcs_legacy_v1/scenes/scene_representative_plcs/cam_0_court_visibility_count.npy
A tests/fixtures/issue_786/legacy_v1_representative/datasets/plcs_legacy_v1/scenes/scene_representative_plcs/cam_0_human_kp_uv.npy
A tests/fixtures/issue_786/legacy_v1_representative/datasets/plcs_legacy_v1/scenes/scene_representative_plcs/cam_0_human_kp_vis.npy
A tests/fixtures/issue_786/legacy_v1_representative/datasets/plcs_legacy_v1/scenes/scene_representative_plcs/cam_0_human_visibility_ratio.npy
A tests/fixtures/issue_786/legacy_v1_representative/datasets/plcs_legacy_v1/scenes/scene_representative_plcs/canonical_pose_3d.npy
A tests/fixtures/issue_786/legacy_v1_representative/datasets/plcs_legacy_v1/scenes/scene_representative_plcs/human_kp_3d.npy
A tests/fixtures/issue_786/legacy_v1_representative/datasets/plcs_legacy_v1/scenes/scene_representative_plcs/meta.json
A tests/fixtures/issue_786/legacy_v1_representative/datasets/plcs_legacy_v1/scenes/scene_representative_plcs/position.npy
A tests/fixtures/issue_786/legacy_v1_representative/datasets/plcs_legacy_v1/scenes/scene_representative_plcs/rotation.npy
A tests/fixtures/issue_786/legacy_v1_representative/datasets/plcs_legacy_v1/scenes/scene_representative_plcs/scalars.json
A tests/fixtures/issue_786/legacy_v1_representative/datasets/plcs_legacy_v1/test.txt
A tests/fixtures/issue_786/legacy_v1_representative/generate_representative.py.txt
A tests/fixtures/issue_786/legacy_v1_representative/manifest.json
A tests/fixtures/issue_786/legacy_v1_representative/plcs_representative_legacy_v1.ckpt
A tests/fixtures/issue_786/legacy_v1_representative/plcs_representative_legacy_v1_golden.npz
A tests/integration/tasks/test_court_coordinate_normalization_smoke.py
A tests/integration/tasks/test_legacy_v1_checkpoint_parity.py
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
M tests/unit/tasks/plcs/generate_dataset/io/test_scene_loader.py
A tests/unit/tasks/plcs/generate_dataset/test_scene_generator.py
M tests/unit/tasks/plcs/inference/test_predictor.py
M tests/unit/tasks/plcs/inference/test_tracking_predictor.py
M tests/unit/tasks/plcs/training/test_losses.py
M tests/unit/tasks/plcs/visualization/io/test_scene.py
M tests/unit/tasks/slcs/inference/test_predictor.py
M tests/unit/tasks/slcs/model_io/test_adapter.py
A tests/unit/tasks/slcs/training/test_metrics.py
A tests/unit/tennis_scene/test_schema.py
M tests/unit/utils/configuration/test_contracts.py
M tests/unit/utils/geometry/test_court_pose.py
A tests/unit/utils/schema/test_court_normalization.py
```

Evidence completeness and internal consistency are also bounded and closed:

- Frozen Issue/state identity is intact: Issue #786, attempt 2, base `59e3b166c2d010d5e62be52c2be76d98a94af0e0`, Issue snapshot `6279b189d4b3c0a7c11da3e605fbc252624f5a60ec808db2c476e061f55fa6a9`, and checklist hash `95bcebf4388fdba9773e3c538c9e22caf82b6e4a413ec1241e9a58b0c4483032` agree across frozen artifacts and state.
- The Tester artifact is cycle 1, records the exact post-test candidate and PASS, and maps all 22 acceptance items. `test-checks.json` records all ten required test-stage results as PASS on that same candidate.
- Machine preflight evidence is present for both required preflight-stage checks, both PASS on the state-bound preflight candidate `sha256:cd14bed3320f21545ccb001a5b523eab8dd900cfe3cdff842843a72e62d9683f`. This seal does not rely on the preflight narrative.
- The canonical seal result set is complete and fresh for the Tester candidate. All required seal-stage IDs from `checks.json` are present exactly once.
- Repository guidance requires `.venv/bin/python`, pytest, Ruff/mypy, and task-script review; the canonical precommit and full-suite outcomes above satisfy those evidence requirements. No script-convention violation is reported.
- The current state remains parent-owned `phase = "implementation"`, `status = "in_progress"`, and empty seal binding; this reviewer does not mutate state or call `seal-verdict`.

## Commands and exact outcomes

- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py candidate-fingerprint .codex/tasks/issue-786` returned `sha256:ee83a05ecdca681a0e926948b581abe63aef5de6d7dd218ba93a108b057063a0` before and after all ten seal checks.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal unit-contract` -> exit 0; 56 passed.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal legacy-v1-checkpoint-parity` -> exit 0; 7 passed.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal unit-blcs` -> exit 0; 37 passed.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal unit-plcs` -> exit 0; 25 passed.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal unit-slcs` -> exit 0; 35 passed.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal integration-normalization` -> exit 0; 14 passed.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal preflight-regression` -> exit 0; 127 passed.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal knowledge-graph` -> exit 0; 181 nodes, 0 errors, 4 warnings.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal precommit-all` -> exit 0; Ruff, mypy, task-script-reviewer passed.
- `.venv/bin/python .agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py run-check .codex/tasks/issue-786 seal full-pytest` -> exit 0; 3245 passed, 53 skipped, 19 warnings.

## Final candidate seal verdict

PASS

## RETURN implementation findings

None
