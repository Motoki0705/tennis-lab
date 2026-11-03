# Heavy test preparation and evaluation

[日本語](heavy-test-cost.ja.md) · [简体中文](heavy-test-cost.zh-CN.md)

For [#1522](https://github.com/Motoki0705/tennis-lab/issues/1522), under
[#1390](https://github.com/Motoki0705/tennis-lab/issues/1390), retain every existing
merge-gate and coordinates-training case. Add one test of seed isolation. Production
code, defaults, CPU selection and scheduling weights are unchanged.

## Changes and preserved contracts

The merge fixture prepares a reviewed scenario through real git and parent/review
CLIs once per variant in each pytest worker. A session-local cache retains a private
snapshot and a separate copy of its Python metadata. Every later test receives a
new filesystem copy, private GitHub/registry/review records, and its own git objects,
refs, indexes and worktrees. Absolute private paths are rebound, then real
`git worktree repair` checks the relocated repository. A private path inside binary
state fails explicitly. The cache is never passed to a test as writable state.
Every gate invocation still starts the actual merge CLI subprocess and fake GitHub
CLI; changed-head/base/diff collection and merge calls are not mocked or cached.

The training smoke keeps the real BLCS generator, four rallies and four cameras,
but caps simulation at 256 physics steps (64 output frames) with a 32-frame minimum.
The seed is fixed at 992 so both validation and test keep multiple flight segments, checked explicitly. The short windows remain 32 frames and long windows are capped at 64 frames. The existing 16-wide, one-layer
model, batch size two, four roundtrip updates, five resume updates, three Flow
sampling steps, two-update validation blocks and partial final epoch remain.
CPU trainer threads are explicitly one. Saved physics diagnostics use the real
nonlinear fitter with one field start, no segment-only warmup and one joint
iteration, injected only at this test's evaluator boundary. No prediction, loss,
gradient, optimizer, checkpoint, plot or metric output is replaced.

| Old case or interaction | Retained case / defect it detects / unique interaction |
|---|---|
| Reprojection and shared generation | Same reprojection test and roundtrip split checks: real generation, overwrite rejection, four rally-disjoint train/val/test records; identical 3D/events/time and changed visible 2D reprojection across cameras |
| Regression roundtrip | Same case: real normal fit, nonempty AdamW state at update four, finite weighted losses, monitored best/last checkpoint reload, TensorBoard events, exact seeded evaluation and inference file contracts |
| GAN and GAN-only roundtrips | Both cases: generator schedule 0/0.5/1/1, three discriminator updates, reconstruction schedule 1/0.5/0/0, component/total accounting and final GAN-only objective |
| Flow roundtrip | Same case: real Flow loss/backward/optimizer, three-step seeded sampling, best/last reload and exact prediction/event-probability replay |
| Physics-head roundtrip | Same case: all four positive physics objectives and weighted sum, integrated arrays, field/surface/segment outputs and inference schema; actual finite fitted residual and saved smoke-fit settings |
| Regression / Flow / GAN resume | Same three cases: interruption after update two, native checkpoint restoration to update five/epoch two, one-update final epoch, dropout, exact full/resumed weights, Torch/Flow RNG and learning-curve equality |
| Full numerical physics convergence | Existing `test_physics_evaluation.py` and `test_ball_fitting.py` remain unchanged: production multi-start protocol, exact simulated-flight fit, nonphysical residuals, multiple segments/surfaces, events and invalid inputs. These supply numerical accuracy; smoke diagnostics supply lifecycle wiring |
| Human holds, missing/interrupted/unconfirmed review, projection/head/base/diff integrity, draft/fork/conflict/unknown inputs | All original gate cases and CI-state combinations remain; they mutate only a private prepared copy and call the real merge CLI, preserving the distinct rejection reason and no-agent/no-merge assertions |
| Independent/dependent PRs, partial decisions, base advances, stale/failing/lost merges, fixes, head races and newly added PRs | Same three-PR variants and scenario bodies, with actual mutable git history and fresh CLI collection; seed reuse does not merge any interaction cases |
| Standalone issue, non-main default branch, shared paths, completed research dependency, invalid auto-merge and no targets | Same dedicated variants / policy / CLI cases; smallest existing scenario sizes are unchanged |
| `test_execution_basis.py` / `test_judgement_execution.py` | Unmodified downstream scenarios, including three-PR/shared variants and explicit `make_gate_env` calls |
| New isolation test | Copies a prepared scenario, commits and adds a worktree, edits GitHub state and removes review evidence; its real CLI blocks while the original dry run succeeds. Original refs/state/result and seed bytes stay identical, git-object inodes differ, and first-case metadata edits do not reach the template |

## Matched local observation

Historical [Actions run 40986313121](https://github.com/Motoki0705/tennis-lab/actions/runs/40986313121) identifies the remaining hot cases below. Merge-gate setup/call totals are 1129.879/713.591 s. Training roundtrips cost 95.1–96.1 s, the physics-head case 190.7 s, and resume cases 0.95–1.08 s. These observations rank work; they are not the local matched denominator.

| Actions case | Setup s | Call s |
|---|---:|---:|
| `test_all_targets_merge_in_one_run_with_verified_base_advances[True-independent]` | 11.159 | 44.416 |
| `test_all_targets_merge_in_one_run_with_verified_base_advances[False-independent]` | 11.290 | 43.712 |
| `test_new_child_pr_does_not_invalidate_reviewed_prs[multiple]` | 11.188 | 36.792 |

| File | Cases | Setup s | Call s | Summed s | Reduction |
| --- | --- | --- | --- | --- | --- |
| `test_merge_gate.py` | 162 → 163 | 1804.797 → 175.366 | 1047.166 → 942.883 | 2852.110 → 1118.353 | 60.8% |
| `test_coordinates_training.py` | 9 → 9 | 15.622 → 0.785 | 1326.628 → 25.534 | 1342.258 → 26.324 | 98.0% |

One complete local pair: 171 passed before; 172 passed after. Command wall time: 2257.648 → 595.396 s. No original case was removed.

All nine training-file cases have call time at most **6.372 s**.

These are summed pytest worker elapsed seconds, not CPU time or predicted CI wall
savings. The complete original inventory is retained. The after total includes the
new isolation test. The run uses two workers, `--dist=worksteal`, native thread
limits of one and CUDA hidden. Numerical fit timing is inside evaluation/training
and is not added again to phase time.

| Operation | Before | After |
| --- | --- | --- |
| test_merge_gate: `fixture_parent_env` | 157 | 158 |
| test_merge_gate: `fixture_gate_env` | 156 | 157 |
| test_merge_gate: `reviewed_gate_preparations` | 156 | 13 |
| test_merge_gate: `seed_restore` | 0 | 145 |
| test_merge_gate: `git` | 2734 | 1396 |
| test_merge_gate: `git_commit` | 601 | 219 |
| test_merge_gate: `worktree_add` | 268 | 28 |
| test_merge_gate: `parent_cli` | 1400 | 142 |
| test_merge_gate: `review_cli` | 168 | 25 |
| test_merge_gate: `gate_cli` | 169 | 171 |
| test_coordinates_training: `model_init` | 14 | 14 |
| test_coordinates_training: `optimizer_step` | 64 | 64 |
| test_coordinates_training: `torch_save` | 56 | 56 |
| test_coordinates_training: `evaluate` | 62 | 62 |
| test_coordinates_training: `physics_fit` | 24 | 24 |
| fake GitHub CLI | 64554 | 29446 |

Real nonlinear fit: 1291.517 → 13.374 s. `git` counts pytest-helper invocations; nested git inside CLI subprocesses is outside that counter. `model_init` counts training Lightning-module construction, not checkpoint/inference model allocations.

Copied historical fake-GitHub call-log lines are excluded from executed-call
counts. The baseline `parent_env` still initializes a separate writable environment
for each test. Only reviewed preparation is amortized. Model initialization,
optimizer updates, saved evaluations and checkpoint I/O remain real.

## Defect injection and verification

| Fault | Before | After |
| --- | --- | --- |
| `human_hold` | 3 AssertionError | 3 AssertionError |
| `projection_integrity` | 2 AssertionError | 2 AssertionError |
| `resume_rng` | 3 AssertionError | 3 AssertionError |
| `reconstruction_schedule` | 1 AssertionError | 1 AssertionError |

The related command passes **187 cases**: the unchanged execution-basis/judgement
scenarios (**71 + 12**), all ball-refiner unit tests (**97**), the shared numerical
fitter (**5**), and assignment/new-case checks (**2**). The production numerical
protocol therefore retains its multi-start/convergence checks. A separate
`--dist=loadfile -n 2` reproduction of the two timed-out scenarios, head races and
both fixed/unfixed base-advance cases passes **5 cases**, exercising cached copies.

```bash
.venv/bin/python -m pytest -n 2 --dist=worksteal -q --no-cov \
  tests/e2e/agents_ops/test_execution_basis.py \
  tests/e2e/agents_ops/test_judgement_execution.py \
  tests/unit/tasks/ball_refiner_3d \
  tests/unit/utils/physics/test_ball_fitting.py \
  tests/e2e/development/test_ci_shards.py::test_assignment_uses_cost_and_preserves_files_including_new_tests \
  tests/e2e/development/test_ci_shards.py::test_only_explicit_function_fixture_files_split_by_measured_case_cost
.venv/bin/python -m spin lint
.venv/bin/python -m mypy --follow-imports=skip \
  tests/e2e/agents_ops/test_merge_gate.py \
  tests/integration/tasks/ball_refiner_3d/test_coordinates_training.py \
  tests/support/agents_ops/directory_seed.py
```

Lint and mypy pass (three changed Python files). All four fault runs exit 1 on
both versions, with the same **3 / 2 / 3 / 1 assertion failures** and no
setup/collection errors. They respectively bypass human holds, bypass projection
integrity, reset resumed RNG, and force reconstruction weight to remain one.


Full CPU collection/assignment: **12783 unique cases**, four partitions **3389 / 3246 / 3186 / 2962**, missing/duplicate **0/0**. The new case receives the existing unknown-case estimate. Actual collection validates fixture scopes. This is a collection audit, not full-suite execution. The initial collection had six errors from missing jsonschema/psutil/requests; locked versions were installed into a separate collection-only overlay, leaving the paired benchmark environment unchanged.

Temporary pytest plugins inject faults in memory; production source is never
edited. The gate plugin compiles the current policy into the actual CLI subprocess.
The training plugin changes the live Lightning implementation. A fresh pytest
process rebuilds every seed for each fault; prepared state cannot mask a mutated
implementation. The exact plugins, commands, per-case results, phase data,
initialization/operation counts and unsuccessful attempts are in
[machine evidence](heavy-test-evidence.json).

## Reproduction and limits

Use the [measurement runner](README.md) on each fixed source with the same
environment and observer from the evidence JSON:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
  .venv/bin/python -m scripts.ci_metrics run --workers 2 --output /absolute/new/observation -- \
  tests/e2e/agents_ops/test_merge_gate.py \
  tests/integration/tasks/ball_refiner_3d/test_coordinates_training.py
```

For operation counts, extract `observer_source` into `heavy_observer.py` on a
temporary `PYTHONPATH`, set `PYTEST_PLUGINS=heavy_observer` and point
`HEAVY_OBSERVER_DIR` at a new directory. For faults, use `mutation_plugin_source`,
`PYTEST_PLUGINS=heavy_mutation`, `HEAVY_MUTATION` and `HEAVY_MUTATION_DIR`, with the
selectors recorded in the evidence. Keep source unchanged through command exit.

n=1 on a shared local WSL host (AMD Ryzen 5 7430U with Radeon Graphics, Python 3.12, Torch 2.15.0+cpu); cache condition is unknown. Both successful observations use the same temporary tool overlay because the shared virtualenv lacked spin/pytest-cov. Exact versions and host load are in the evidence. They differ from Actions, so the historical run is not used as the matched speedup denominator. The original trainer overrides Torch intraop threads to two; the optimized smoke explicitly selects one, with native-library environment limits unchanged. Earlier attempts missing spin/pytest-cov and an interrupted 37-pass/5-failure/129-incomplete observation during severe host contention are retained. A separate cProfile attempt was interrupted inside the long production physics fit; it is not a completed timing sample.

The first optimized full observation had 170 passed and two 90-second CLI timeouts under concurrent host load, with the intermediate one-segment/one-joint fit budget. Its artifacts are retained. The gate implementation is unchanged in the successful full rerun; the final smoke fit uses zero segment warmup and one joint iteration. Faults, the five-case cached reproduction and all numerical regression tests remain visible. Reviewed-preparation counts are fixture entries minus seed restores, adding back the isolation test’s extra restore.

This is not a passing full CPU baseline or a four-shard Actions comparison.
Each shard/worker rebuilds its own seeds. The parent must refresh scheduling
weights and measure the integrated Actions run; this change does not edit the
workflow or timing profile. No tests, failure checks or slow markers are excluded.
