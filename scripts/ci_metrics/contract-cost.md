# Input contract test consolidation

[日本語](contract-cost.ja.md) · [简体中文](contract-cost.zh-CN.md)

For [#1403](https://github.com/Motoki0705/tennis-lab/issues/1403), classify malformed
inputs once in the shared validator, then verify every public entry and concrete
model configuration reaches that validator before computation. Production code,
numeric tolerances and existing state/exclusivity checks are unchanged.

| Old cases | Retained detection and interaction | Decision |
|---|---|---|
| `test_block.py::test_dense_boundary_rejects_noncanonical_full_masks`: 5 errors × MHA/GQA × module/direct/update = 30 | `test_dense_validator_rejects_noncanonical_full_masks` covers dtype, rank, exact shape, batch/query broadcasting and device; `test_dense_boundary_validates_before_computation` covers all 6 entry/attention combinations with real validation and a computation guard | 11 cases, 30 → 6 block constructions |
| `test_direct_forward_matches_module_call_for_valid_arguments`: MHA, GQA, CSWA | Existing `test_direct_forward_matches_module_call_values_gradients_and_state` covers dense values, input/parameter gradients and state; `test_cswa_direct_forward_matches_module_call_for_all_valid_states` retains the separate CSWA all-valid branch | Remove 2 weaker dense cases; retain the CSWA branch |
| `test_predict.py::test_rejects_wrong_external_input_before_forward`: 6 errors × DPT/DeepQuery × channels 2/5 = 24 | `test_mdd_coordinates.py` covers the six original classes plus rank, width and valid minimum size; the predictor test retains all 4 real type/channel combinations, actual validator binding and forward-not-called | 13 cases, 24 → 4 detector constructions |

The full paths are `tests/unit/utils/models/components/test_block.py` and
`tests/unit/tasks/ball_detection/model_io/{test_predict,test_mdd_coordinates}.py`.
The existing stronger direct-forward test is in the first file. Its exact function
name and the live nodeids are retained in the measurement inventory.

## Evidence

The selected files pass before (114 cases) and after (82 cases): a net reduction of
32, not a relocation-only count. Across the candidate cases, block/detector
constructions decrease from 57 to 11. No mutable model is shared between tests.

Three temporary faults were injected before and after; each run exited 1 with
ordinary test failures and no collection/setup errors:

| Injected fault | Before failures | After failures |
|---|---:|---:|
| Accept float dense attention masks | 6 | 7 |
| Accept float decoded RGB inputs | 4 | 5 |
| Bypass the DPT public input validator | 12 | 2 |

The spy wraps the function actually bound at the tested entry. Replacing that
binding with a known-good validator would hide a wiring defect. Both channel
configurations still detect the DPT bypass.

[Machine-readable evidence](contract-evidence.json) records versions, counts,
phase costs and mutation results. Timing is one local observation per version,
with zero xdist workers, native thread limits of one and CUDA hidden. Other local
validation was active; startup/cache/load were not controlled. These phase costs
help describe this subset, but establish neither a CI wall-time reduction nor the
parent's warm-cache median/p95 target.

Reproduce with `python -m scripts.ci_metrics run --workers 0 --output NEW_DIRECTORY --`
followed by the two original paths, adding `test_mdd_coordinates.py` for the new
version. General observation semantics are in [README](README.md).
