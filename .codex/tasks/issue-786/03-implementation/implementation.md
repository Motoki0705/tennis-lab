# Implementation

- Issue: #786
- Attempt: 1
- Test cycle: 2
- Status: COMPLETE
- Candidate SHA-256: `sha256:708a562c5be1dc6c83bee32418680531d113d8fe55e4dfd2c570e82be7d6f90d`

## Assigned ownership

- `i786_impl_core` and `i786_impl_core_repair`: shared normalization schema, base dataset/checkpoint contracts, Hydra group, and non-destructive materializer.
- `i786_impl_blcs` and `i786_repair_blcs_legacy`: BLCS generation, loading, loss/gravity/projection, metrics, predictors, and checkpoint propagation.
- `i786_impl_plcs` and `i786_repair_plcs_legacy`: PLCS generation/loading, position-only scaling, losses, metrics, predictors, and rendering/court-pose integration.
- `i786_impl_slcs`: SLCS and tennis-scene propagation while preserving metre-valued public outputs.
- Parent integrator: cross-task configuration, legacy compatibility repairs, invisible-BLCS observation masking, materialization, baseline training, knowledge registration, and configuration-inventory integration.

## Files and symbols changed

- Shared authority: `src/utils/schema/court_normalization.py`, `src/utils/schema/court.py`, and `src/utils/geometry/court_pose.py` define and consume the immutable v1/v2 contract.
- Base contract and artifacts: `src/tasks/base/configuration.py`, `src/tasks/base/data/`, `src/tasks/base/model_io/`, shared Hydra configs, and `materialize_court_coordinate_normalization` implement version propagation, strict dataset/checkpoint compatibility, and non-overwriting conversion.
- BLCS: task configs plus generation/data/model-IO/training/inference/visualization consumers now receive the selected contract. `src/tasks/blcs/data/visibility.py` and dataset boundaries zero only invisible UV observations before strict normalized-coordinate validation.
- PLCS: task configs plus generation/data/model-IO/training/inference/visualization consumers apply versioned scale only to court position/translation; root-relative canonical pose stays in metres.
- SLCS and integration: SLCS configs/data/model-IO/training/evaluation/inference and tennis-scene schema/pipeline retain metre-valued `SceneResult` outputs while carrying normalization provenance.
- Configuration audit: `src/utils/configuration/inventory.py` registers the new materializer CLI and its fail-closed boundary validator.
- Evidence and documentation: task READMEs/config comments, four run nodes, one comparison group, and reproducibility bundles under `knowledge/`.

## Behavior implemented

- `v1` remains the explicit backward-compatible default with scale `(5.485, 11.885, 1.07)m`; `v2` is opt-in with common scale `(11.885, 11.885, 11.885)m`.
- Unknown versions, partial metadata, runtime/dataset/checkpoint mismatches, and metadata-free artifacts under v2 fail explicitly. Metadata-free legacy artifacts remain accepted only under v1.
- BLCS, PLCS, and SLCS normalize/decode using the same selected contract across generation, training, metrics, inference, projection/render boundaries, and checkpoint loading.
- v2 uses a common physical Smooth L1 transition and uniform default position/Hungarian axes; v1 preserves its historical behavior.
- v2 datasets were materialized beside, not over, legacy datasets and validated by normalized-to-metre round trips.
- BLCS v1/v2 and PLCS v1/v2 baselines were trained through the shared queue and recorded in physical metre metrics. BLCS v2 reduced mean position error from `2.405233m` to `2.338450m` under matched batch/worker settings.
- After discovery Preflight RETURN, the materializer now publishes an empty root document only when prior validation proved the source is metadata-free legacy v1 and its root `meta.json` is absent. Scene metadata, v2, mismatch, and overwrite checks remain strict.

## Plan deviations and rationale

- PLCS v2 was resumed from the epoch-62 checkpoint with `batch_size=24` on GPU 0 after measured heterogeneous two-GPU DDP was substantially slower; PLCS v1 used `batch_size=4`. Therefore the PLCS figures are operational baselines, not a normalization-only controlled comparison. The knowledge group states this limitation explicitly.
- GPU 1 was not retained for the final runs because its 4GB capacity made heterogeneous DDP throughput much worse than GPU 0 alone. BLCS used GPU 0, `batch_size=64`, and user-requested `data.num_workers=16` for both versions.
- Runtime prediction files for earlier completed runs had been overwritten by subsequent executions. Their logged metrics, curves, and reproducibility metadata were retained without fabricating replacement predictions; the final BLCS v2 prediction bundle is preserved.
- The required all-files mypy gate exposed eight diagnostics in unchanged baseline files. The bounded repair added only type narrowing/casts/targeted decorator annotations and removed one redundant cast; runtime behavior and test assertions are unchanged.
- Tester cycle 1 returned only because the canonical full-suite environment hid CUDA and omitted a private NHT config required by existing baseline tests. The plan/check authority now exposes GPU 0 only to `full-pytest`, worktree-local non-symlink NHT configuration is present, and three direct test defects were corrected without production changes.

## Commands and results

- Normalization and task suites before final integration: `912 passed`.
- Invisible BLCS observation repair: `26 passed`; Ruff and mypy passed on the focused delta.
- Configuration inventory/audit after materializer registration: `11 passed in 39.10s`.
- Full repository diagnostic before the inventory/policy closure: `3087 passed, 53 skipped, 42 failed, 2 errors`; remaining failures were classified as missing worktree-only licensed/external data links and repository allowlists that must be updated by the independent Test Writer. Worktree data links are now present.
- Bounded Preflight repair: all-files mypy passed across `1103` files and staged-file mypy passed across the owned delta; Ruff passed on all owned files; `30` focused unit tests and `3` focused Colab e2e tests passed; the dynamic missing-root materialization fixture passed root/scene v2 metadata, physical round trip, source preservation, overwrite refusal, and missing-scene rejection.
- Test-environment closure evidence before cycle-2 Preflight: all nine test-stage canonical checks passed on the repaired test candidate, including `3230 passed, 53 skipped` in the GPU-visible full suite. Because that run preceded the state machine's required cycle-2 Preflight, it is supporting evidence only and will be rerun by a fresh Test Writer after Preflight PASS.
- Knowledge validation: `0 errors`; four warnings are pre-existing graph warnings.
- Shared training queue: both BLCS jobs completed 100 epochs with `batch_size=64`, `data.num_workers=16`; PLCS v2 completed through epoch 101 on GPU 0.

## Known limitations and remaining risks

- The independent Test Writer must update existing configuration/architecture policy tests for the new shared schema/base boundary and add the planned v1/v2 acceptance coverage before the final full-suite verdict.
- PLCS v1/v2 metrics are not a single-variable comparison because restart and batch-size conditions differ.
- Local full-suite execution depends on ignored worktree-local hard-link trees to licensed/external datasets in the original repository; these are environment evidence, not committed artifacts and preserve strict worktree-root path semantics.

## Handoff

Production integration is complete and ready for an independent discovery Preflight Reviewer. The reviewer should use only the diagnostic categories frozen in `plan.md`, execute canonical preflight check IDs through `manage_issue_task.py run-check`, and write `preflight.md` without editing source or tests.
