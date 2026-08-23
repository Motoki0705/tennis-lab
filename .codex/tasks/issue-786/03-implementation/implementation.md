# Implementation

- Issue: #786
- Attempt: 2
- Test cycle: 1
- Status: COMPLETE
- Candidate SHA-256: `sha256:ef780395bce0ef3208e173408a2ef1ab4880604d10fee427197ffc8fe270314d`

## Assigned ownership

- `i786_impl_core` and `i786_impl_core_repair`: shared normalization schema, base dataset/checkpoint contracts, Hydra group, and non-destructive materializer.
- `i786_impl_blcs` and `i786_repair_blcs_legacy`: BLCS generation, loading, loss/gravity/projection, metrics, predictors, and checkpoint propagation.
- `i786_impl_plcs` and `i786_repair_plcs_legacy`: PLCS generation/loading, position-only scaling, losses, metrics, predictors, and rendering/court-pose integration.
- `i786_impl_slcs`: SLCS and tennis-scene propagation while preserving metre-valued public outputs.
- Parent integrator: cross-task configuration, legacy compatibility repairs, invisible-BLCS observation masking, materialization, baseline training, knowledge registration, and configuration-inventory integration.
- Attempt-2 `i786_impl_core_repair`: make the public PLCS scene-load contract mandatory and verify all production callers forward an already resolved normalization contract.
- Attempt-2 parent integrator: incorporate the Validator findings, replace the invalid archived-checkpoint migration premise with frozen-base representative parity evidence, and bind this artifact to the repaired candidate.

## Files and symbols changed

- Shared authority: `src/utils/schema/court_normalization.py`, `src/utils/schema/court.py`, and `src/utils/geometry/court_pose.py` define and consume the immutable v1/v2 contract.
- Base contract and artifacts: `src/tasks/base/configuration.py`, `src/tasks/base/data/`, `src/tasks/base/model_io/`, shared Hydra configs, and `materialize_court_coordinate_normalization` implement version propagation, strict dataset/checkpoint compatibility, and non-overwriting conversion.
- BLCS: task configs plus generation/data/model-IO/training/inference/visualization consumers now receive the selected contract. `src/tasks/blcs/data/visibility.py` and dataset boundaries zero only invisible UV observations before strict normalized-coordinate validation.
- PLCS: task configs plus generation/data/model-IO/training/inference/visualization consumers apply versioned scale only to court position/translation; root-relative canonical pose stays in metres.
- SLCS and integration: SLCS configs/data/model-IO/training/evaluation/inference and tennis-scene schema/pipeline retain metre-valued `SceneResult` outputs while carrying normalization provenance.
- Configuration audit: `src/utils/configuration/inventory.py` registers the new materializer CLI and its fail-closed boundary validator.
- Evidence and documentation: task READMEs/config comments, four run nodes, one comparison group, and reproducibility bundles under `knowledge/`.
- Attempt-2 PLCS boundary: `src/tasks/plcs/generate_dataset/io/scene_loader.py::load_scene` and `src/tasks/plcs/visualization/io/scene.py::load_scene_bundle` require `CourtCoordinateNormalization`; metadata validation always precedes payload reads.
- Attempt-2 authority: `.codex/tasks/issue-786/{01-exploration/exploration.md,02-planning/plan.md}` records why archived checkpoint architecture drift is outside this normalization Issue and freezes representative frozen-base parity as the AC-003/004 test oracle.

## Behavior implemented

- `v1` remains the explicit backward-compatible default with scale `(5.485, 11.885, 1.07)m`; `v2` is opt-in with common scale `(11.885, 11.885, 11.885)m`.
- Unknown versions, partial metadata, runtime/dataset/checkpoint mismatches, and metadata-free artifacts under v2 fail explicitly. Metadata-free legacy artifacts remain accepted only under v1.
- BLCS, PLCS, and SLCS normalize/decode using the same selected contract across generation, training, metrics, inference, projection/render boundaries, and checkpoint loading.
- v2 uses a common physical Smooth L1 transition and uniform default position/Hungarian axes; v1 preserves its historical behavior.
- v2 datasets were materialized beside, not over, legacy datasets and validated by normalized-to-metre round trips.
- BLCS v1/v2 and PLCS v1/v2 baselines were trained through the shared queue and recorded in physical metre metrics. BLCS v2 reduced mean position error from `2.405233m` to `2.338450m` under matched batch/worker settings.
- After discovery Preflight RETURN, the materializer now publishes an empty root document only when prior validation proved the source is metadata-free legacy v1 and its root `meta.json` is absent. Scene metadata, v2, mismatch, and overwrite checks remain strict.
- Every direct PLCS scene load now requires the caller to declare its resolved normalization contract. Root and selected-scene metadata are checked unconditionally before `meta.json` or array payloads are read; there is no metadata-free public bypass.
- The legacy v1 compatibility promise remains scoped to normalization metadata and numerical behavior. It does not silently migrate checkpoint architecture/configuration formats that were already unloadable at the frozen base revision.

## Plan deviations and rationale

- PLCS v2 was resumed from the epoch-62 checkpoint with `batch_size=24` on GPU 0 after measured heterogeneous two-GPU DDP was substantially slower; PLCS v1 used `batch_size=4`. Therefore the PLCS figures are operational baselines, not a normalization-only controlled comparison. The knowledge group states this limitation explicitly.
- GPU 1 was not retained for the final runs because its 4GB capacity made heterogeneous DDP throughput much worse than GPU 0 alone. BLCS used GPU 0, `batch_size=64`, and user-requested `data.num_workers=16` for both versions.
- Runtime prediction files for earlier completed runs had been overwritten by subsequent executions. Their logged metrics, curves, and reproducibility metadata were retained without fabricating replacement predictions; the final BLCS v2 prediction bundle is preserved.
- The required all-files mypy gate exposed eight diagnostics in unchanged baseline files. The bounded repair added only type narrowing/casts/targeted decorator annotations and removed one redundant cast; runtime behavior and test assertions are unchanged.
- Tester cycle 1 returned only because the canonical full-suite environment hid CUDA and omitted a private NHT config required by existing baseline tests. The plan/check authority now exposes GPU 0 only to `full-pytest`, worktree-local non-symlink NHT configuration is present, and three direct test defects were corrected without production changes.
- The Validator-selected archived BLCS and PLCS checkpoints cannot be loaded by frozen base `59e3b166c2d010d5e62be52c2be76d98a94af0e0`: BLCS contains 132 state-key architecture/name differences and incomplete current typed config; PLCS carries obsolete configuration roots. Attempt 2 therefore does not add an unrelated, identity-bound architecture migration or relax strict parsing.
- AC-003/004 will instead use small deterministic metadata-free checkpoints, dataset fixtures, and expected outputs produced by the frozen base itself. This directly tests that the Issue preserves the behavior of artifacts the base revision could actually create and load.

## Commands and results

- Normalization and task suites before final integration: `912 passed`.
- Invisible BLCS observation repair: `26 passed`; Ruff and mypy passed on the focused delta.
- Configuration inventory/audit after materializer registration: `11 passed in 39.10s`.
- Full repository diagnostic before the inventory/policy closure: `3087 passed, 53 skipped, 42 failed, 2 errors`; remaining failures were classified as missing worktree-only licensed/external data links and repository allowlists that must be updated by the independent Test Writer. Worktree data links are now present.
- Bounded Preflight repair: all-files mypy passed across `1103` files and staged-file mypy passed across the owned delta; Ruff passed on all owned files; `30` focused unit tests and `3` focused Colab e2e tests passed; the dynamic missing-root materialization fixture passed root/scene v2 metadata, physical round trip, source preservation, overwrite refusal, and missing-scene rejection.
- Test-environment closure evidence before cycle-2 Preflight: all nine test-stage canonical checks passed on the repaired test candidate, including `3230 passed, 53 skipped` in the GPU-visible full suite. Because that run preceded the state machine's required cycle-2 Preflight, it is supporting evidence only and will be rerun by a fresh Test Writer after Preflight PASS.
- Knowledge validation: `0 errors`; four warnings are pre-existing graph warnings.
- Shared training queue: both BLCS jobs completed 100 epochs with `batch_size=64`, `data.num_workers=16`; PLCS v2 completed through epoch 101 on GPU 0.
- Attempt-2 PLCS loader repair: Ruff PASS, mypy PASS, and 23 focused tests PASS. Four existing loader tests intentionally await Test Writer changes because the public function now requires an explicit contract.
- Frozen-base representative bundle generated and strictly replayed on CPU float32 with `atol=1e-5, rtol=0`: manifest `3854d8fd9cbc83af3456a295fc872d4a0afcaa859550bca85d0e15859e4a2047`, generator `956bef2a7fff8e375398a62c031a95eaeab3247a89ffe1e92b5d091639059358`, BLCS checkpoint/golden `69af21b3f8008ab7f53708e1d03346113aafa49c857a5465ad6e6da86f80a5e7` / `0f494e30e07c99e3f59feba113093530a87c25334a216eea9bee33e9a32397e6`, and PLCS checkpoint/golden `6c212eec6bbe616b498000928733318b19f76e4ad957a680963621c672ca841d` / `3fcc1205f96f531f7b731248ecf1f8cd48c8e0bc0091d3d0e3dc04cf5eb1b969`.
- Frozen-base worktree remained tracked-clean before and after generation; both generated predictors strictly reloaded their metadata-free checkpoints before producing the recorded outputs.

## Known limitations and remaining risks

- The independent Test Writer must commit portable representative fixtures, add the planned frozen-base v1 parity integration test, update existing PLCS loader tests for the mandatory argument, and replace the nominal NumPy/Identity smoke with real task chains before the final full-suite verdict.
- PLCS v1/v2 metrics are not a single-variable comparison because restart and batch-size conditions differ.
- Local full-suite execution depends on ignored worktree-local hard-link trees to licensed/external datasets in the original repository; these are environment evidence, not committed artifacts and preserve strict worktree-root path semantics.

## Handoff

Attempt-2 production integration and authority repair are complete and ready for a fresh independent discovery Preflight Reviewer. The reviewer should use only the diagnostic categories frozen in the updated `plan.md`, execute canonical preflight check IDs through `manage_issue_task.py run-check`, verify the representative frozen-base premise and mandatory PLCS loader boundary, and write `preflight.md` without editing source or tests.
